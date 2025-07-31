import json
from tqdm import tqdm

# --- Configuration ---
# The input file with potential duplicates
INPUT_FILENAME = "negative_sample_pairs.json"
# The final, truly unique positive pairs dataset
OUTPUT_FILENAME = "negative_pairs_deduplicated.json"


def deduplicate_positive_pairs():
    """
    Reads a list of pairs, removes all duplicates, and saves the unique list.
    """

    # --- Step 1: Load the positive pairs data ---
    print(f"--- Step 1: Loading potentially duplicate pairs from '{INPUT_FILENAME}' ---")
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            all_pairs = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{INPUT_FILENAME}' was not found.")
        return

    # --- Step 2: Identify and store only the unique pairs ---
    print("\n--- Step 2: Processing and de-duplicating pairs ---")

    unique_pairs = []
    # Use a set to keep track of pairs we've already added
    seen_pairs_set = set()

    for pair in tqdm(all_pairs, desc="De-duplicating"):
        # Create a canonical (sorted, tuple) version of the pair
        canonical_pair = tuple(sorted(pair))

        # If we haven't seen this pair before...
        if canonical_pair not in seen_pairs_set:
            # ...add it to our set of seen pairs...
            seen_pairs_set.add(canonical_pair)
            # ...and add the original pair (as a list) to our final unique list.
            unique_pairs.append(pair)

    # --- Step 3: Final Report and Save ---
    print("\n--- De-duplication Complete ---")
    print(f"Original number of pairs: {len(all_pairs):,}")
    print(f"Number of unique pairs:   {len(unique_pairs):,}")
    print(f"Number of duplicates removed: {len(all_pairs) - len(unique_pairs):,}")

    print(f"\nSaving unique pairs to '{OUTPUT_FILENAME}'...")
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(unique_pairs, f, indent=2)
        print("Save complete.")
    except Exception as e:
        print(f"FATAL ERROR: Could not save the file. Reason: {e}")

    print("\nIMPORTANT: Your positive dataset has changed size.")
    print("You should now re-run the script to generate new negative samples")
    print("to ensure your positive and negative datasets are balanced.")


if __name__ == "__main__":
    deduplicate_positive_pairs()