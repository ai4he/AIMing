from openai import OpenAI
import json
import math


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


conjects = load_list_from_json('100k_cata_data.json')
AIcor = load_list_from_json('AIutocomplete_Continous.json')

nextup = len(AIcor)



deepseek_api_key = 'sk-4a3b7fa7c48649f78ac338653b42494d'
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")


corruption = 0.01

for x in range(nextup,nextup+300):
    con = conjects[x]['text']
    damn = con
    context = conjects[x]['previous context']
    typ = conjects[x]['type']


    # print(con)
    # print("\n\n\n")

    con = con[:int(len(con) * corruption)]



    # print('halfsies')
    # print(con)
    # print("\n\n\n")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant who always responds exactly in the format specified by the user"},
            {"role": "user",
             "content": f"The following is the first {corruption * 100} percent of a math conjecture formatted in Latex. Before the conjetcure there will be a small bit of the context the proceeded the conjecture."
                        f" Please write nothing, except a continuation and ending for the conjecture."
                        f"I will be appending your output to the conjecture directly so it is imparitive you output nothing but the text of your finishing half of the conjecture."
                        f"Here is your conjecture in context. conjecture type:{typ} context:{context} conjecture: {con}"},
        ],
        stream=False
    )

    # print(response.choices[0].message.content)
    newcon = con + response.choices[0].message.content
    conjects[x]['text'] = newcon
    factor = 10 ** 2
    conjects[x]['label'] = math.trunc(corruption * factor) / factor
    AIcor.append(conjects[x])

    if corruption < 1:
        corruption += 0.01
    else:
        corruption = 0.01
    # print('\n\n\n')
    print(x)
    # print('\n')
    # print('AI Content')
    # print(newcon)

save_list_to_json(AIcor, 'AIutocomplete_Continous.json')

#deepseek_api_key = 'sk-4a3b7fa7c48649f78ac338653b42494d'
#client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
