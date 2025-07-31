import os
import re
import json
from tqdm import tqdm



root_folder = "math_tex_27k"
context_grab_len = 1000


# strip the [stuff]
def a(s: str) -> str:
    if s[0] == '[':
        F = s.find(']')

        s = s[F + 1:]

        print(s)

# removes all char after first instance of a specified char
def remove_after_char(text, char):
  index = text.find(char)
  if index != -1:
    return text[:index]
  return text

#remove all chars before first instance of specifiied char
def remove_before_char(text, char):
  index = text.find(char)
  if index != -1:
    return text[index+1:]
  return text

#walk through all subdirectoies in math_tex and read all latex files in a subdirectory into a string
#then pass that string to get_conjectures_from_file
def parse_math_tex():
   theorums = []

   file_count_for_tqdm = 0
   for _, _, _filenames_for_count in os.walk(root_folder):
       file_count_for_tqdm += len(_filenames_for_count)

   pbar = tqdm(total=file_count_for_tqdm, unit="file")

   fails = 0
   for dirpath, dirnames, filenames in os.walk(root_folder):
      dir_text = ""
      for filename in filenames:
            try:
                filepath = os.path.join(dirpath, filename)
                dir_text += open(filepath, 'r').read()
            except:
                fails += 1
                print(f"--------failure---------{fails}")
            pbar.update(1)

      try:
        t = get_conjectures_from_file(dir_text,dirpath.removeprefix("math_tex_27k\\math_tex\\"))
        for x in t:
            theorums.append(x)
      except:
          print("--------failure---------")
   pbar.close()
   return theorums


#parse a file for latex theorums
def get_conjectures_from_file(latex,id):
    #find \newtheroum definition and get a lkist of all env names
   indexs = re.finditer(r"\\newtheorem", latex)
   conj_names = []
   conj_labels = {}
   for match in indexs:
      index = match.start()
      newthe = remove_after_char(latex[index:index+300],"\n")
      label = newthe
      label = remove_before_char(label,'}')
      label = remove_before_char(label, '{')
      label = remove_after_char(label,'}')
      newthe = newthe.replace(r"\newtheorem","")
      newthe = remove_after_char(newthe, "}")
      newthe = remove_before_char(newthe, "{")
      #print(newthe)
      #print(label)
      #print()
      conj_labels[newthe] = label
      conj_names.append(newthe)
   conj_names = set(conj_names)
   conj_names = list(conj_names)

   #loop for every env name defined
   contents = []
   conind = 0
   for env in conj_names:
       if not env:
           continue
        #find begining and end tags
       begin_tag = "\\begin{" + env + "}"
       end_tag = "\\end{" + env + "}"
       current_pos = 0
       while current_pos < len(latex):
           start_index = latex.find(begin_tag, current_pos)
           if start_index == -1:
               break

           content_start = start_index + len(begin_tag)



           end_index = latex.find(end_tag, content_start)
           if end_index == -1:

               current_pos = content_start
               continue
            #get content from theorum
           content = latex[content_start:end_index].strip()
           #get prev context
           if start_index > context_grab_len:
                before_context = latex[start_index-context_grab_len:start_index]
           else:
               before_context = latex[:start_index]
            #get post context

           if  end_index + len(end_tag)+ context_grab_len > len(latex):
               after_context = latex[end_index+len(end_tag):]
           else:
               after_context = latex[end_index+len(end_tag):end_index+len(end_tag)+context_grab_len]
            #if there is enough content add an entry
           if len(content) > 5:
               contents.append({
                   'id':id,
                   'env':env,
                   'type': conj_labels[env],
                   'text':content,
                   'previous context': before_context,
                   'following context': after_context


               })
           current_pos = end_index + len(end_tag)
   return contents






#sparcert = open("math_tex/0704_0002/sparsity-certifying.tex", 'r').read()
#get_conjectures_from_file(sparcert,"0704_0002")

#parse text, remove first blank entry
thes = parse_math_tex()
#print info
print(len(thes))
for x in range(1300,1600):
    #print(thes[x])
    print(x)




# --- Function to save the list to a JSON file ---
def save_list_to_json(data_list, filename="output_data_27k.json"):
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





save_list_to_json(thes,"data_27k_long_context.json")