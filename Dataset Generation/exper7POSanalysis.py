import json
import nltk
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexMathNode, LatexEnvironmentNode, \
    LatexGroupNode, LatexSpecialsNode, LatexCommentNode
import re
import random
from tqdm import tqdm

# --- NLTK Resource Download ---
# Run this once, or include at the start of your script for setup.
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' resource found.")
except LookupError:
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt')

try:
    # The error specifically mentioned 'punkt_tab' under 'tokenizers/punkt_tab/english/'
    # but downloading 'punkt_tab' should get the necessary files.
    nltk.data.find('tokenizers/punkt_tab') # Check for the base 'punkt_tab' directory
    print("NLTK 'punkt_tab' resource found.")
except LookupError:
    print("NLTK 'punkt_tab' resource not found. Downloading...")
    nltk.download('punkt_tab') # ADD THIS LINE
# --- End NLTK Resource Download ---

# ... rest of your code from the previous step ...

# --- Function to load the list from a JSON file ---



def swap_words_by_index(text_string, word1, index1, word2, index2):
    """
    Swaps two words in a string given the words and their starting indices.

    Args:
        text_string (str): The original string.
        word1 (str): The first word to swap.
        index1 (int): The starting index of the first word.
        word2 (str): The second word to swap.
        index2 (int): The starting index of the second word.

    Returns:
        str: The new string with the words swapped.

    Raises:
        ValueError: If the provided words/indices don't match the string content.
    """

    # --- Input Validation (Optional but Recommended) ---
    # Check if the words actually exist at the given indices
    if not (text_string[index1 : index1 + len(word1)] == word1 and \
            text_string[index2 : index2 + len(word2)] == word2):
        # For more specific error messages, you could check each word individually
        error_msg = "Mismatch: "
        if text_string[index1 : index1 + len(word1)] != word1:
            error_msg += f"Word '{word1}' not found at index {index1}. Found '{text_string[index1 : index1 + len(word1)]}'. "
        if text_string[index2 : index2 + len(word2)] != word2:
            error_msg += f"Word '{word2}' not found at index {index2}. Found '{text_string[index2 : index2 + len(word2)]}'. "
        raise ValueError(error_msg.strip())

    # --- Determine which word comes first in the original string ---
    # This simplifies the slicing logic.
    # 'first_word_in_string' is the word that appears earlier in the original string.
    # 'second_word_in_string' is the word that appears later.
    if index1 < index2:
        first_word_in_string = word1
        first_word_index_in_string = index1
        second_word_in_string = word2
        second_word_index_in_string = index2
    else: # index2 < index1 (or they are the same, though swapping same word is trivial)
        first_word_in_string = word2
        first_word_index_in_string = index2
        second_word_in_string = word1
        second_word_index_in_string = index1

    # --- Extract the parts of the string ---
    # 1. Part before the first word
    part1 = text_string[0 : first_word_index_in_string]

    # 2. The word that will now go into the first word's original slot
    #    This is the word that was originally *later* in the string.
    word_to_place_first = second_word_in_string if index1 < index2 else word1

    # 3. Part between the two words
    #    Starts after the first word ends, and ends before the second word begins.
    start_of_between = first_word_index_in_string + len(first_word_in_string)
    end_of_between = second_word_index_in_string
    part_between = text_string[start_of_between : end_of_between]

    # 4. The word that will now go into the second word's original slot
    #    This is the word that was originally *earlier* in the string.
    word_to_place_second = first_word_in_string if index1 < index2 else word2

    # 5. Part after the second word
    start_of_after = second_word_index_in_string + len(second_word_in_string)
    part_after = text_string[start_of_after : ]

    # --- Construct the new string ---
    new_string = part1 + word_to_place_first + part_between + word_to_place_second + part_after
    return new_string




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


class WORD:
    def __init__(self):
        # Three attributes
        self.token = ""
        self.POS = ''
        self.attributes = []
        self.type = "text"
        self.countNum = 0


    def describe(self):
        print(f" Token: {self.token} \n POS: {self.POS} \n'Atributes: {self.attributes} \n NUM: {self.countNum}'")


latex_sample_for_testing = r"""
\label{lem_56est} and some text.
\textbf{Bold text with \textit{italic content} and also \another{arg}.}
An equation $E = mc^2 + \sum_{i=0}^\infty \alpha_i$.
Optional arg \includegraphics[width=5cm]{image.png} here.
\simplecommand
"""
#latex_string = r"This is some text~with a non-breaking space."
# OR
latex_string = r"\begin{array}{cc} A & B \\ C & D \end{array}"
# OR
# latex_string = r"Variable $X_i$ and also $Y^j$."


conjects = load_list_from_json('data_27k.json')

def swap_contexts(cons,ind1,ind2):
    swapped = {}
    swapped[0] = cons[ind1]
    swapped[1] = cons[ind2]
    hold = swapped[0]['text']
    swapped[0]['text'] = swapped[1]['text']
    swapped[1]['text'] = hold
    return swapped

swap_contexts(conjects, 2, 4)








def recur_parse_latex(nodelist):
    words = []
    for node in nodelist:
        if node == None:
            continue
        if node.isNodeType(LatexCharsNode):
            # This is plain text, let's tokenize it
            text_chunk = node.chars
            tokens = nltk.word_tokenize(text_chunk)
            tagged = nltk.pos_tag(tokens)# ADDED: Tokenize the text chunk
            for token in tagged:
                #print(f"  TOKEN: '{token[0]}       part o speech : {token[1]}'")  # MODIFIED: Print individual tokens'
                w = WORD()
                w.token = token[0]
                w.POS = token[1]
                words.append(w)
        elif node.isNodeType(LatexMathNode):
            # This is an inline math mode, e.g., $...$ or \(...\)
           # print(f"  DELIM: '{node.delimiters[0]}' (LATEX_MATH_OPEN)")
            w = WORD()
            w.token = node.delimiters[0]
            w.POS = 'MDLIMB'
            words.append(w)

            man = recur_parse_latex(node.nodelist)
            for t in man:
                words.append(t)

           # print(f"  DELIM: '{node.delimiters[1]}' (LATEX_MATH_CLOSE)")
            w = WORD()
            w.token = node.delimiters[0]
            w.POS = 'MDLIME'
            words.append(w)
            # (Inside your recur_parse_latex function)
        elif node.isNodeType(LatexMacroNode):

            current_macro_node = node  # Rename for clarity

           # print(f"  MACRO_NAME: '\\{current_macro_node.macroname}'")
            w = WORD()
            w.token = current_macro_node.macroname
            w.POS = 'LMACRO'
            words.append(w)
            if node.nodeargd != None:
                if node.nodeargd.argnlist != None:
                    tim = recur_parse_latex(node.nodeargd.argnlist)
                    for e in tim:
                        words.append(e)
            #print(f"  MACRO_END: '\\{current_macro_node.macroname}'")
        elif node.isNodeType(LatexGroupNode):
            #print(f"  GROUP_OPEN: '{node.delimiters[0]}'")
            w = WORD()
            w.token = node.delimiters[0]
            w.POS = 'GDLIMB'
            words.append(w)
            # node_item.nodelist() for LatexGroupNode returns the Python list of nodes inside it
            xin = recur_parse_latex(node.nodelist)
            for m in xin:
                words.append(m)
            #print(f"  GROUP_CLOSE: '{node.delimiters[1]}'")
            w = WORD()
            w.token = node.delimiters[0]
            w.POS = 'GDLIME'
            words.append(w)
        elif node.isNodeType(LatexEnvironmentNode):
            env_name = node.environmentname
            #print(f"  ENV_BEGIN: '\\begin{{{env_name}}}'")  # Or however you want to represent it
            w = WORD()
            w.token = f'\\begin{{{env_name}}}'
            w.POS = 'ENVB'
            words.append(w)
            if node.nodelist:
                xi = recur_parse_latex(node.nodelist)  # Recurse on its .nodes
                for m in xi:
                    words.append(m)

            #print(f"  ENV_END: '\\end{{{env_name}}}'")
            w = WORD()
            w.token = f'\\end{{{env_name}}}'
            w.POS = 'ENVE'
            words.append(w)
        elif node.isNodeType(LatexSpecialsNode):
            # specials_chars attribute holds the character(s) like '&', '~', etc.
            special_char_token = node.specials_chars
            #print(f"  SPECIAL_CHAR_TOKEN: '{special_char_token}'")
            w = WORD()
            w.token = special_char_token
            w.POS = 'LSCT'
            words.append(w)
        elif node.isNodeType(LatexCommentNode):
            if node.comment != None:
                text_chunk = node.comment
                tokens = nltk.word_tokenize(text_chunk)
                tagged = nltk.pos_tag(tokens)  # ADDED: Tokenize the text chunk
                for token in tagged:
                    # print(f"  TOKEN: '{token[0]}       part o speech : {token[1]}'")  # MODIFIED: Print individual tokens'
                    w = WORD()
                    w.token = token[0]
                    w.POS = token[1]
                    words.append(w)


        else:
            # This is a LaTeX element (command, math, environment, etc.)
            #print(f"LATEX_STUFF: '{node.latex_verbatim()}'")
            w = WORD()
            w.token = node.latex_verbatim()
            w.POS = 'LOTHER'
            words.append(w)

    return words

def get_ats(words):
    ats = []
    count = {}
    for w in words:
        if w.POS == 'MDLIMB':
            ats.append('MDLIM')
        elif w.POS == 'MDLIME':
            ats.remove('MDLIM')
        elif w.POS == 'GDLIMB':
            ats.append('GDLIM')
        elif w.POS == 'GDLIME':
            ats.remove('GDLIM')
        elif w.POS == 'ENVB':
            ats.append('ENV')
        elif w.POS == 'ENVE':
            ats.remove('ENV')
        for a in ats:
            w.attributes.append(a)


        if w.token in count:
            count[w.token] += 1
        else:
            count[w.token] = 1

        w.countNum = count[w.token]
    return words




def swap_NNP(w,string):
    NNPs = []
    for n in w:
        if n.POS == 'NNP':
            NNPs.append(n)
    if len(NNPs) < 2:
        print("--------NOT ENOUGH NNPS--------")
    else:
        word1 = NNPs[random.randint(0, len(NNPs)-1)]
        NNPs.remove(word1)
        word2 = NNPs[random.randint(0, len(NNPs)-1)]

        indexs1 = list(re.finditer(word1.token, string))
        indexs2 = list(re.finditer(word2.token, string))

        corupt = swap_words_by_index(string,word1.token,indexs1[word1.countNum-1].start(),word2.token,indexs2[word2.countNum-1].start())
        print('original')
        print(string)
        print('\n\n\n')
        print(corupt)


def swap_POS(w,string,POS):
    NNPs = []
    for n in w:
        if n.POS == POS:
            NNPs.append(n)
    if len(NNPs) < 2:
        return -1
    else:
        word1 = NNPs[random.randint(0, len(NNPs)-1)]
        NNPs.remove(word1)
        word2 = NNPs[random.randint(0, len(NNPs)-1)]

        indexs1 = list(re.finditer(word1.token, string))
        indexs2 = list(re.finditer(word2.token, string))

        corupt = swap_words_by_index(string,word1.token,indexs1[word1.countNum-1].start(),word2.token,indexs2[word2.countNum-1].start())
        return corupt



def swap_POS_ATT(w,string,POS,ATT):
    NNPs = []
    for n in w:
        if n.POS == POS and ATT in n.attributes:
            NNPs.append(n)
    if len(NNPs) < 2:
        return -1
    else:
        word1 = NNPs[random.randint(0, len(NNPs)-1)]
        NNPs.remove(word1)
        word2 = NNPs[random.randint(0, len(NNPs)-1)]

        indexs1 = list(re.finditer(word1.token, string))
        indexs2 = list(re.finditer(word2.token, string))

        corupt = swap_words_by_index(string,word1.token,indexs1[word1.countNum-1].start(),word2.token,indexs2[word2.countNum-1].start())

pos_count = {}
newcons = []
fails= 0
for con in tqdm(conjects, desc="Processing Conjectures"):
    try:
        latex_string = con['text']
        lw = LatexWalker(latex_string)
        nodelist, _, _ = lw.get_latex_nodes()
        w = recur_parse_latex(nodelist)
        w = get_ats(w)
        for wor in w:
            if wor.POS in pos_count:
                pos_count[wor.POS] += 1
            else:
                pos_count[wor.POS] = 1




    except:
        fails += 1
print('fails')
print(fails)
print('successes')
print(pos_count)

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


liz = [pos_count]
save_list_to_json(liz, 'POS_Count.json')

#swap_POS_ATT(w,latex_string,'NNP','MDLIM')



