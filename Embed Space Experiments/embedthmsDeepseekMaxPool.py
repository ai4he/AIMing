import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def load_list_from_json(filename="output_data.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"Successfully loaded data from {filename}")
        return data_list
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"Error loading {filename}: {e}")
        return None


def max_pooling(model_output, attention_mask):
    """
    Performs max-over-time pooling on the token embeddings. It correctly
    handles padding by masking out those tokens before finding the maximums.
    This version is safe for float16 and other data types.
    """
    token_embeddings = model_output.last_hidden_state

    # Get the data type of the embeddings to use the correct minimum value
    dtype = token_embeddings.dtype

    # Create an expanded mask to match the embedding shape
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # --- THE FIX IS HERE ---
    # We get the minimum representable value for the specific dtype (e.g., float16)
    # This prevents the overflow error.
    min_val = torch.finfo(dtype).min
    token_embeddings[input_mask_expanded == 0] = min_val
    # --- END OF FIX ---

    # Take the max value over the sequence length dimension (dim=1)
    max_embeddings = torch.max(token_embeddings, dim=1)[0]

    return max_embeddings

# ===================================================================
# --- MODIFICATIONS AS COMMANDED ---
# ===================================================================

# 1. THE MODEL IS NOW DEEPSEEK MATH 7B.
MODEL_NAME = 'deepseek-ai/deepseek-math-7b-instruct'

# 2. LOAD THE TOKENIZER AND FIX PADDING.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Large models often don't have a pad token, which breaks batching. We fix this.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. LOAD THE 7B MODEL ONTO GPU 1, COMPRESSED TO FIT.
#    This replaces the old model loading and the .to(device) call.
print(f"Loading {MODEL_NAME} onto GPU 1 in 4-bit mode...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map={'': 1}  # This forces the entire model onto GPU index 1.
)
print("Model loaded successfully.")

# ===================================================================
# --- YOUR SCRIPT'S CORE LOGIC (UNCHANGED) ---
# ===================================================================

all_theorems_data = load_list_from_json('thms_10%.json')

if not all_theorems_data:
    exit("Exiting script because data could not be loaded.")

texts_to_embed = []
metadata = []
for record in all_theorems_data:
    texts_to_embed.append(record['text'])
    metadata.append({
        'id': record['id'],
        'env': record['env'],
        'type': record['type'],
        'text': record['text'],
        'categories': record.get('catagories', 'N/A')
    })

BATCH_SIZE = 8  # WARNING: 7B models are slower. Start with a smaller batch size.
embeddings_list = []

# The device for tensors is now handled by the model's device_map, but we can specify it
# for the encoded_input to be sure.
device = model.device

for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE), desc="Embedding Theorems"):
    batch_texts = texts_to_embed[i:i + BATCH_SIZE]

    # Tokenizer now sends tensors directly to the correct GPU.
    encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    batch_embeddings = max_pooling(model_output, encoded_input['attention_mask'])
    batch_embeddings_normalized = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

    # Move to CPU for numpy conversion and storage.
    embeddings_list.extend(batch_embeddings_normalized.cpu().numpy())

df = pd.DataFrame(metadata)
df['embedding'] = embeddings_list

# Save to a new file to distinguish from the old embeddings.
output_file = '../paralellReal/theorems_with_DEEPSEEK_MATH_7B_maxpool.parquet'
df.to_parquet(output_file)

print(f"\nDone. Processed {len(df)} records.")
print(f"Final data with DeepSeek Math 7B embeddings saved to '{output_file}'.")