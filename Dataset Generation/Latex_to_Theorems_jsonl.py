import os
import re
import json
from tqdm import tqdm
import sys


ROOT_FOLDER = "/home/harris/PycharmProjects/PythonProject/math_src_EXTRACTED_start-1900"
OUTPUT_FILE = "theorems_start-1900.jsonl"
CONTEXT_GRAB_LEN = 1000


def get_theorems_from_latex(latex_content, paper_id):
    """
    Parses a string of LaTeX content to find theorem-like environments.
    This version uses more robust regular expressions.
    """

    # Regex to find \newtheorem definitions. It captures:
    # 1. The environment name (e.g., "thm")
    # 2. The label that will be displayed (e.g., "Theorem")
    # This handles optional arguments like [section] and whitespace more robustly.
    newtheorem_pattern = re.compile(r"\\newtheorem\*?\s*\{([^}]+)\}\s*(?:\[[^\]]+\])?\s*\{([^}]+)\}")

    theorem_labels = {}
    for match in re.finditer(newtheorem_pattern, latex_content):
        env_name, label = match.groups()
        theorem_labels[env_name.strip()] = label.strip()

    # Remove any empty keys that might have been captured
    theorem_labels.pop("", None)
    if not theorem_labels:
        return []  # No theorems defined in this file

    # --- Find all instances of the defined environments ---
    found_theorems = []
    for env, label in theorem_labels.items():
        # A robust regex to find the content between \begin{env} and \end{env}
        # re.escape(env) handles cases where an env name might have special regex characters
        # re.DOTALL (or flags=re.S) allows '.' to match newlines, which is crucial for multi-line theorems
        try:
            content_pattern = re.compile(r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{" + re.escape(env) + r"\}",
                                         re.DOTALL)
        except re.error:
            # This can happen if an 'env' name is badly formed, we can just skip it
            print(f"  [DEBUG] Skipped invalid environment name '{env}' in paper {paper_id}", file=sys.stderr)
            continue

        for content_match in re.finditer(content_pattern, latex_content):
            content = content_match.group(1).strip()

            # If the content is too short, it's likely not a real theorem
            if len(content) <= 10:
                continue

            # --- Context Grabbing ---
            start_index = content_match.start()
            end_index = content_match.end()

            before_context = latex_content[max(0, start_index - CONTEXT_GRAB_LEN):start_index]
            after_context = latex_content[end_index:end_index + CONTEXT_GRAB_LEN]

            found_theorems.append({
                'id': paper_id,
                'env': env,
                'type': label,
                'text': content,
                'previous_context': before_context,
                'following_context': after_context
            })

    return found_theorems


def process_tex_directories():
    """
    Walks through all subdirectories, reads all .tex files into a single string for each,
    and processes them to find theorems, saving results to a .jsonl file.
    """
    print(f"Starting processing of root folder: {ROOT_FOLDER}")

    # Get a list of all directories (papers) to process for tqdm
    # This ensures os.walk is only consumed once for the list.
    paper_dirs = []
    for dirpath, dirnames, _ in os.walk(ROOT_FOLDER):
        # We only want the subdirectories, not the root folder itself.
        if dirpath != ROOT_FOLDER:
            paper_dirs.append(dirpath)

    if not paper_dirs:
        print(f"[ERROR] No subdirectories found in {ROOT_FOLDER}. Please check the path.", file=sys.stderr)
        return

    # Open the output file once in write mode
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # Counters for the final summary
        papers_processed = 0
        papers_failed = 0
        theorems_found = 0

        pbar = tqdm(paper_dirs, unit="paper")
        for dirpath in pbar:
            dir_text = ""
            paper_id = os.path.relpath(dirpath, ROOT_FOLDER)
            pbar.set_description(f"Processing {paper_id}")

            # --- Robust File Reading ---
            try:
                filenames = [f for f in os.listdir(dirpath) if f.lower().endswith(('.tex', '.ltx'))]
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        # First, try to read with utf-8
                        with open(filepath, 'r', encoding='utf-8') as f:
                            dir_text += f.read()
                    except UnicodeDecodeError:
                        try:
                            # If utf-8 fails, fall back to latin-1
                            with open(filepath, 'r', encoding='latin-1') as f:
                                dir_text += f.read()
                        except Exception as e:
                            # If both fail, log it and move on
                            print(f"\n[ERROR] Could not read file: {filepath}. Reason: {e}", file=sys.stderr)
                    except IOError as e:
                        print(f"\n[ERROR] IOError reading file: {filepath}. Reason: {e}", file=sys.stderr)

                if not dir_text:
                    # This can happen if a folder contains no .tex files or they were unreadable
                    continue

                # --- Robust Parsing ---
                try:
                    theorems = get_theorems_from_latex(dir_text, paper_id)
                    if theorems:
                        for theorem_data in theorems:
                            # Write each theorem as a new line in the jsonl file
                            outfile.write(json.dumps(theorem_data, ensure_ascii=False) + '\n')
                        theorems_found += len(theorems)
                    papers_processed += 1
                except Exception as e:
                    print(f"\n[ERROR] Failed to parse content for paper: {paper_id}. Reason: {e}", file=sys.stderr)
                    papers_failed += 1

            except Exception as e:
                # Catch-all for any other unexpected error in the directory processing loop
                print(f"\n[CRITICAL ERROR] An unexpected error occurred in directory: {dirpath}. Reason: {e}",
                      file=sys.stderr)
                papers_failed += 1

    print("\n\n--- Processing Complete ---")
    print(f"Successfully processed {papers_processed} papers.")
    print(f"Failed to process {papers_failed} papers.")
    print(f"Found a total of {theorems_found} theorems.")
    print(f"Results saved to {OUTPUT_FILE}")


# --- Main Execution ---
if __name__ == "__main__":
    process_tex_directories()