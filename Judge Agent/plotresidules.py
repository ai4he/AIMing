import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
import os

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# --- Configuration: Set these to match your training script ---
# This should point to the directory where the 'best_model' folder is located.
OUTPUT_DIR = "AIutocomplete_Regressor_Continousv1"
# This should point to the specific folder with the best model files.
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model")
# The original dataset file used for training.
DATASET_JSON = 'AIutocomplete_Clear_Continous.json'

MAX_LENGTH = 512
BATCH_SIZE = 32 # Can be larger for inference
SEED = 42 # Use the same seed as training to get the same validation split!


def load_list_from_json(filename="output_data.json"):
    """Loads a list of dictionaries from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        return data_list
    except FileNotFoundError:
        print(f"Error: Dataset file {filename} not found.")
        return None

def tokenize_function(examples):
    """Tokenizes the text fields."""
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

# --- Main Script ---
if __name__ == "__main__":
    # 1. Setup Device and Load Model/Tokenizer
    print(f"Loading best model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found at {MODEL_PATH}. "
            "Please ensure you have run the training script and a 'best_model' folder was created."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval() # Set model to evaluation mode

    # 2. Recreate the Validation Dataset
    print(f"Loading and preparing dataset from: {DATASET_JSON}")
    conjectures_data = load_list_from_json(DATASET_JSON)
    if conjectures_data is None:
        exit()

    dataset = Dataset.from_list(conjectures_data)
    # Use the same test_size and seed to ensure we get the identical validation set
    train_test_split = dataset.train_test_split(test_size=0.2, seed=SEED)
    raw_datasets = DatasetDict({'validation': train_test_split['test']})

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    columns_to_remove = ["id", "env", "type", "text", "previous context", "following context", "catagories"]
    actual_columns_to_remove = [col for col in columns_to_remove if col in tokenized_datasets['validation'].column_names]
    tokenized_datasets = tokenized_datasets.remove_columns(actual_columns_to_remove)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False, # No need to shuffle for evaluation
        collate_fn=data_collator,
        batch_size=BATCH_SIZE
    )

    # 3. Perform Inference to Get Predictions
    print("Running model on validation data to get predictions...")
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Predicting"):
            # Move batch to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # For regression, the logit is the prediction. Squeeze to remove extra dimension.
            logits = outputs.logits.squeeze(-1)

            all_predictions.extend(logits.cpu().numpy())
            all_true_labels.extend(batch["labels"].cpu().numpy())

    # Convert lists to numpy arrays for calculation
    predictions = np.array(all_predictions)
    true_labels = np.array(all_true_labels)

    # 4. Calculate Residuals
    residuals = true_labels - predictions
    print(f"\nCalculated {len(residuals)} residuals.")

    # 5. Create and Save the Plot
    print("Generating residual plot...")
    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot of predicted values vs. residuals
    ax.scatter(predictions, residuals, alpha=0.5, edgecolors='k', s=40)

    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')

    # Add labels and title
    ax.set_title('Residuals vs Actual', fontsize=16)
    ax.set_xlabel('Predicted Corruption Level', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.legend()
    ax.grid(True)

    # Save the plot to a file
    plot_filename = os.path.join(OUTPUT_DIR, "residual_plot.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved successfully to: {plot_filename}")

    # Display the plot
    plt.show()