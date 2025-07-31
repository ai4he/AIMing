# FILE: add_text_to_parquet.py
# PURPOSE: To perform a one-time fix on the Parquet file by adding the missing 'text' column.

import pandas as pd
import json

# ===================================================================
# --- CONFIGURATION ---
# ===================================================================

# The original JSON file that contains the text data.
ORIGINAL_JSON_FILE = "thms_10%.json"

# The Parquet file that is missing the 'text' column.
INCOMPLETE_PARQUET_FILE = "theorems_with_DEEPSEEK_MATH_7B_maxpool.parquet"

# The name for the new, complete, and final output file.
COMPLETE_PARQUET_FILE = "theorems_DEEPSEEK_COMPLETE_MAXPOOL.parquet"

# ===================================================================

print("--- Data Correction Script ---")

# --- 1. Load both data sources ---
print(f"Loading original text data from '{ORIGINAL_JSON_FILE}'...")
try:
    with open(ORIGINAL_JSON_FILE, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
except Exception as e:
    print(f"FATAL ERROR: Could not load JSON file. Error: {e}")
    exit()

print(f"Loading incomplete Parquet file '{INCOMPLETE_PARQUET_FILE}'...")
try:
    df = pd.read_parquet(INCOMPLETE_PARQUET_FILE)
except Exception as e:
    print(f"FATAL ERROR: Could not load Parquet file. Error: {e}")
    exit()

# --- 2. Verify and Merge ---
print("Verifying data alignment and merging 'text' column...")

# Sanity check to ensure the files correspond to each other.
if len(original_data) != len(df):
    print("FATAL ERROR: The number of records in the JSON file and the Parquet file do not match.")
    print(f"JSON records: {len(original_data)}, Parquet records: {len(df)}")
    exit()

# Extract the texts and add them as a new column. The order is preserved.
texts = [record['text'] for record in original_data]
df['text'] = texts

# Optional but recommended: Reorder columns for better readability
# This brings the text right next to its ID.
cols_order = ['id', 'text', 'categories', 'env', 'type', 'embedding']
df = df[cols_order]
print("'text' column successfully merged and columns reordered.")

# --- 3. Save the Final, Complete File ---
print(f"Saving new, complete dataset to '{COMPLETE_PARQUET_FILE}'...")
df.to_parquet(COMPLETE_PARQUET_FILE)

print("\n--- DATA SURGERY COMPLETE ---")
print("Your new, final dataset is ready.")
print(f"You can now use '{COMPLETE_PARQUET_FILE}' for all future analysis.")