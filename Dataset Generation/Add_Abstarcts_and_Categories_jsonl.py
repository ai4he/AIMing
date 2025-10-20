import json
from tqdm import tqdm
import os


METADATA_FILE = "arxiv-metadata-oai-snapshot.json"


THEOREMS_FILE = "theorems_start-1900.jsonl"


OUTPUT_FILE = "theorems_start-1900_absandcate.jsonl"


def load_arxiv_metadata(filepath):
    """
    Loads ArXiv metadata into a dictionary for quick lookups.
    Assumes the metadata file is in JSONL format (one JSON object per line).
    """
    if not os.path.exists(filepath):
        print(f"FATAL ERROR: Metadata file not found at '{filepath}'")
        print("Please update the METADATA_FILE variable in the script.")
        return None

    print(f"Loading metadata from {filepath} into memory...")
    metadata_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        # Use tqdm for a progress bar, as this can be a large file
        for line in tqdm(f, desc="Loading metadata"):
            try:
                # Each line is a JSON string
                meta_obj = json.loads(line)
                # Store only what we need: the abstract and categories
                metadata_map[meta_obj['id']] = {
                    'abstract': meta_obj.get('abstract'),
                    'categories': meta_obj.get('categories')
                }
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed lines or lines missing an 'id'
                print(f"\n[WARNING] Skipping malformed metadata line. Error: {e}")
                continue

    print(f"Successfully loaded metadata for {len(metadata_map)} articles.")
    return metadata_map


def enrich_theorems_file(metadata):
    """
    Reads the theorems JSONL file, adds metadata, and writes to a new file.
    """
    if not os.path.exists(THEOREMS_FILE):
        print(f"FATAL ERROR: Theorems file not found at '{THEOREMS_FILE}'")
        return

    print(f"Starting enrichment process. Reading from '{THEOREMS_FILE}'...")

    # Counters for the final summary
    theorems_processed = 0
    matches_found = 0

    with open(THEOREMS_FILE, 'r', encoding='utf-8') as infile, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:

        for line in tqdm(infile, desc="Enriching theorems"):
            try:
                theorem_data = json.loads(line)
                theorems_processed += 1

                # The paper ID from our extraction is like '1301_0001'
                # The metadata ID is like '1301.0001'. We must convert it.
                paper_id = theorem_data.get('id')
                if paper_id:
                    lookup_id = paper_id.replace('_', '.', 1)  # Replace only the first underscore

                    # Look up the metadata in our map
                    article_meta = metadata.get(lookup_id)

                    if article_meta:
                        # If found, add the new fields
                        theorem_data['abstract'] = article_meta.get('abstract', None)
                        theorem_data['categories'] = article_meta.get('categories', None)
                        matches_found += 1
                    else:
                        # If not found, add fields with None to maintain structure
                        theorem_data['abstract'] = None
                        theorem_data['categories'] = None

                # Write the (potentially) enriched data to the new file
                outfile.write(json.dumps(theorem_data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print(f"\n[WARNING] Skipping malformed JSON line in theorems file.")
                continue

    print("\n\n--- Enrichment Complete ---")
    print(f"Processed a total of {theorems_processed} theorems.")
    print(f"Successfully found and added metadata for {matches_found} entries.")
    # We count matches per entry, not per paper, so if a paper wasn't found, none of its theorems were matched.
    print(f"Results saved to the new file: '{OUTPUT_FILE}'")


# --- Main Execution ---
if __name__ == "__main__":
    arxiv_metadata = load_arxiv_metadata(METADATA_FILE)
    if arxiv_metadata:
        enrich_theorems_file(arxiv_metadata)