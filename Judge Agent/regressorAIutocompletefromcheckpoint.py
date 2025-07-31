from torch.utils.data import DataLoader
from torch.optim import AdamW
import smtplib

from tqdm.auto import tqdm
import os
os.environ['USE_LIBUV'] = '0'
from transformers import AutoTokenizer
import json
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification
#from transformers import TrainingArguments, Trainer
import numpy as np
# Import mean_squared_error for our calculation
from sklearn.metrics import mean_squared_error
from transformers import DataCollatorWithPadding, get_scheduler
import torch
from accelerate import Accelerator


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA not available. Training will run on CPU (which is very slow for transformers).")

def load_list_from_json(filename="output_data.json"):
    """
    Loads a list of dictionaries from a JSON file.

    Args:
        filename (str): The name of the file to load from.

    Returns:
        list: The loaded list of dictionaries, or None if an error occurs.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"Successfully loaded data from {filename}")
        return data_list
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
        return None
    except IOError as e:
        print(f"Error reading data from {filename}: {e}")
        return None

def tokenize_function(examples):
    combined_texts = [
        f"previous context:{prev}  main conjecture:{main}"
        for prev, main in zip(examples["previous context"], examples["text"])
    ]
    return tokenizer(
        combined_texts,
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH
    )

def compute_metrics(preds, labels):
    """
    Computes regression metrics.
    - MSE (Mean Squared Error): Lower is better.
    - RMSE (Root Mean Squared Error): Lower is better, more interpretable units.
    """
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    return {
        'mse': mse,
        'rmse': rmse,
    }

# --- MODIFICATION START ---
# We still need to define the base model to start from.
# The actual trained weights will be loaded later with `load_state`.
MODEL_CHECKPOINT = "distilbert-base-uncased"
# This is the path to the folder containing the state you want to load.
# Make sure this points to the correct epoch folder from your previous run.
CHECKPOINT_TO_LOAD  = "AIutocomplete_Regressorv3/epoch_1"

# --- MODIFICATION END ---

MAX_LENGTH = 512
BATCH_SIZE = 16 # Per device batch size
NUM_EPOCHS = 8 # This will be the number of *additional* epochs to run
LEARNING_RATE = 4e-5
OUTPUT_DIR = "AIutocomplete_Regressor_Continousv2" # Can be the same or a new directory

accelerator = Accelerator(log_with="tensorboard", project_dir=OUTPUT_DIR)

print(f"PyTorch version: {torch.__version__}")
print(f"Accelerator state: {accelerator.state}")

conjectures_data_with_context = load_list_from_json('AIutocomplete_Clear_Continous.json')
dataset = Dataset.from_list(conjectures_data_with_context)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
raw_datasets = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})
# The tokenizer was saved in the epoch folder, so we load it from there.
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_TO_LOAD)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

columns_to_remove = ["id", "env", "type", "text", "previous context", "following context", "catagories"]
actual_columns_to_remove = [col for col in columns_to_remove if col in tokenized_datasets['train'].column_names]
tokenized_datasets = tokenized_datasets.remove_columns(actual_columns_to_remove)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE
)

print(raw_datasets)
print(f"Example from training set: {raw_datasets['train'][0]}")

# First, initialize the model and optimizer as before
num_labels = 1
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Prepare everything with accelerator BEFORE loading the state
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

# --- MODIFICATION ---
# Now, load the saved state from the checkpoint folder.
# This will load the weights into `model`, and also restore the state of
# `optimizer` and `lr_scheduler`.
print(f"Resuming training from checkpoint: {CHECKPOINT_TO_LOAD}")
accelerator.load_state(CHECKPOINT_TO_LOAD)
# --- END MODIFICATION ---

progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
best_rmse = float('inf')

# The rest of the training loop remains identical
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch['labels'] = batch['labels'].float()
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    avg_train_loss = total_train_loss / len(train_dataloader)
    # -- EVALUATION --
    model.eval()
    all_preds = []
    all_labels = []
    total_eval_loss = 0
    for step, batch in enumerate(eval_dataloader):
        batch['labels'] = batch['labels'].float()
        with torch.no_grad():
            outputs = model(**batch)

        total_eval_loss += outputs.loss.item()
        predictions = outputs.logits.squeeze(-1)

        gathered_preds = accelerator.gather_for_metrics(predictions)
        gathered_labels = accelerator.gather_for_metrics(batch["labels"])

        all_preds.append(gathered_preds.cpu().numpy())
        all_labels.append(gathered_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_eval_loss = total_eval_loss / len(eval_dataloader)

    if accelerator.is_main_process:
        eval_metrics = compute_metrics(all_preds, all_labels)
        log_string = ( f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | RMSE: {eval_metrics['rmse']:.4f} | MSE: {eval_metrics['mse']:.4f}" )
        accelerator.print(log_string)

        try:
            with open(os.path.join(OUTPUT_DIR, "AIutocomplete_training_log.txt"), "a") as f:
                f.write(log_string + "\n")
        except Exception as e:
            print(f"Couldn't write to log file: {e}")

        # --- IMPORTANT ---
        # So that your new epoch checkpoints have the config.json, let's fix the saving logic
        # 1. Save the resumable state
        epoch_output_dir = os.path.join(OUTPUT_DIR, f"resumed_epoch_{epoch + 1}")
        accelerator.save_state(epoch_output_dir)

        # 2. ALSO save a standalone model with `save_pretrained`
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(epoch_output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(epoch_output_dir)
        accelerator.print(f"Saved checkpoint for epoch {epoch + 1} to {epoch_output_dir}")
        # --- END OF FIX ---

        if eval_metrics['rmse'] < best_rmse:
            best_rmse = eval_metrics['rmse']
            accelerator.print(f"New best RMSE: {best_rmse:.4f}! Saving best model...")

            best_model_dir = os.path.join(OUTPUT_DIR, "best_model_resumed")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(best_model_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(best_model_dir)
            accelerator.print(f"Saved best model to {best_model_dir}")

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    print("Training finished.")
    print(f"Final checkpoints and best model saved in {OUTPUT_DIR}")