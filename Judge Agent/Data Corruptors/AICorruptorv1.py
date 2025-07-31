from openai import OpenAI
import json


# --- Function to save the list to a JSON file ---
def save_list_to_json(data_list, filename="currupted_output_data.json"):
    """
    Saves a list of dictionaries to a JSON file.

    Args:
        data_list (list): The list of dictionaries to save.
        filename (str): The name of the file to save to.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # json.dump writes the Python object to the file object f
            # indent=4 makes the JSON file human-readable (optional)
            json.dump(data_list, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved data to {filename}")
    except IOError as e:
        print(f"Error saving data to {filename}: {e}")
    except TypeError as e:
        print(f"Error: Data might not be JSON serializable. {e}")

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


conjects = load_list_from_json('filtered_data_27k.json')
AIcorHeavy = load_list_from_json('AI_corrupt_heavy.json')
AIcorLight = load_list_from_json('AI_corrupt_light.json')
numb = 120


print(conjects[numb])
print('\n\n')
print("light corrupt")
print(AIcorLight[numb])
print('\n\n')
print("heavy corrupt")
#print(AIcorHeavy[numb])
print('\n\n')
datas = []

save_list_to_json(datas,'AI_corrupt_heavy_20k_start.json')

