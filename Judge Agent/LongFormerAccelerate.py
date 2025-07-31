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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
            # json.load reads the JSON data from the file object f
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
    # Combine texts for the entire batch
    combined_texts = [
        f"Does this abstarct go with this theorms abstract:{prev}  main conjecture:{main}"
        for prev, main in zip(examples["abstract"], examples["text"])
    ]
    # Tokenize the whole batch
    return tokenizer(
        combined_texts,
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH
    )


def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


MODEL_CHECKPOINT = "allenai/longformer-base-4096"
MAX_LENGTH = 4096
BATCH_SIZE = 2 # Per device batch size
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
OUTPUT_DIR = "RelevanceAgentv3_abstacrtr"

accelerator = Accelerator(log_with="tensorboard", project_dir=OUTPUT_DIR)

print(f"PyTorch version: {torch.__version__}")
# ACCELERATE: accelerator.state gives you all the info about your setup
print(f"Accelerator state: {accelerator.state}")



#conjectures_data_with_context_train = load_list_from_json('reallylilrelvalnce.json')
#conjectures_data_with_context_val   = load_list_from_json('reallylilrelvalnce.json')
# Convert to Hugging Face Dataset Format
#dataset_train = Dataset.from_list(conjectures_data_with_context_train)
#dataset_val   = Dataset.from_list(conjectures_data_with_context_val)

#raw_datasets = DatasetDict({
#    'train': dataset_train,
#    'validation': dataset_val
#})
#tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Apply the tokenizer to all splits
#tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
conjectures_data_with_context = load_list_from_json('lilrelvancebutlikewithabstarctsyeahilikejson.json')
# Convert to Hugging Face Dataset Format
dataset = Dataset.from_list(conjectures_data_with_context)
# Split into training and validation sets
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
raw_datasets = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Apply the tokenizer to all splits
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Remove original text columns
columns_to_remove = ["id", "env", "type", "text", "previous context", "following context", "catagories","abstract"]
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

num_labels = 2 # Normal (0) and Lightly Corrupted (1) and Haevy corrupted (2)
# load model from checkpoint
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# Create a learning rate scheduler
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)


progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
best_f1 = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
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
        with torch.no_grad():
            outputs = model(**batch)

        total_eval_loss += outputs.loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        gathered_preds = accelerator.gather_for_metrics(predictions)
        gathered_labels = accelerator.gather_for_metrics(batch["labels"])

        all_preds.append(gathered_preds.cpu().numpy())
        all_labels.append(gathered_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_eval_loss = total_eval_loss / len(eval_dataloader)

    if accelerator.is_main_process:
        eval_metrics = compute_metrics(all_preds, all_labels)
        # --- PRINTING and LOGGING TO FILE---
        log_string = ( f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | F1: {eval_metrics['f1']:.4f} | Accuracy: {eval_metrics['accuracy']:.4f}" )
        accelerator.print(log_string)

        try:
            # Append the log string to a text file
            with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "a") as f:
                f.write(log_string + "\n")
            # --- CHECKPOINTING LOGIC ---
        except:
            print("couldnt output the data")

        # 1. Save a checkpoint at the end of EVERY epoch
        epoch_output_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
        accelerator.save_state(epoch_output_dir)
        # We also save the tokenizer separately in the main process
        tokenizer.save_pretrained(epoch_output_dir)
        accelerator.print(f"Saved checkpoint for epoch {epoch + 1} to {epoch_output_dir}")

        # 2. Keep track of and save the BEST model separately
        if eval_metrics['f1'] > best_f1:
            best_f1 = eval_metrics['f1']
            accelerator.print("New best F1 score! Saving best model...")

            best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(best_model_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(best_model_dir)
            accelerator.print(f"Saved best model to {best_model_dir}")





accelerator.wait_for_everyone()
if accelerator.is_main_process:
    print("Training finished.")
    print(f"Final checkpoints and best model saved in {OUTPUT_DIR}")
