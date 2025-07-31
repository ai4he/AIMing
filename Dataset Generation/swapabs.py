import json
import random

def load_list_from_json(filename="output_data.json"):
    """Loads a list of dictionaries from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"Successfully loaded data from {filename}")
        return data_list
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
        return None
    except IOError as e:
        print(f"Error reading data from {filename}: {e}")
        return None



def save_list_to_json(data_list, filename="corrupted_output_data.json"):
    """Saves a list of dictionaries to a JSON file."""
    if data_list is None:
        print(f"Cannot save to {filename} because data is None.")
        return
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
        print(f"\nSuccessfully saved {len(data_list)} items to {filename}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
    except TypeError as e:
        print(f"Serialization error: {e}")


thms = load_list_from_json('thms_abs.json')
wrong = load_list_from_json('relevanceAbstractData.json')
data = []
ind = 0
for w,t in zip(wrong,thms[:-100]):
    w['label'] = 1
    data.append(w)
    t['label'] = 0
    data.append(t)


save_list_to_json(data,'relevanceAbstractData2.json')


