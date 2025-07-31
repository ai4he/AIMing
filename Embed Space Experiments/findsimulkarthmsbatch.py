# FILE: find_similar_theorems_batch.py
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import faiss
import os  # We need the 'os' module to handle file paths

# ===================================================================
# --- USER CONFIGURATION ---
# ===================================================================

# Provide a list of query strings to search for.
# Each item in this list will generate its own results.txt file.
QUERIES_TO_RUN = [
    "\\label{thm:mainelliptic} Suppose $\\chi(\\wp)=1$. %and $\\chi(\\wp^*)\\neq 1$\nThen\n$$\\widetilde{\\frak L}_p^\\prime(1,\\chi)=\\langle \\frak{e}_{1}^\\chi,\\frak{col}_0^\\chi\\rangle_{\\textup{Nek}}.$$",
    "\\label{th2}\\cite{berge2} An independent set $S$ belongs to $\\Omega(G)$ if and\nonly if every independent set $A$ of $G$, disjoint from $S$, can be matched\ninto $S$.",
    """\label{main-path-theorem} Let $n\geq 2$ and $1\leq k\leq n$, and let $\Cballot(n,k)$ be the number of arithmetical structures $(\dd,\rr)$ on $\P_n$ such that $\rr(1)=k$.  Then \begin{equation} \label{main-path-thm-eqn} \Cballot(n,k) = B(n-2,n-k) = \frac{k-1}{n-1} \binom{2n-2-k}{n-2}. \end{equation}""",
    """Let G be a finite, connected graph. An arithmetical structure on G is a pair of positive
integer vectors d, r such that (diag(d) − A)r = 0, where A is the adjacency matrix of G. We
investigate the combinatorics of arithmetical structures on path and cycle graphs, as well as the
associated critical groups (the torsion part of the cokernels of the matrices (diag(d)−A)). For paths,
we prove that arithmetical structures are enumerated by the Catalan numbers, and we obtain refined
enumeration results related to ballot sequences. For cycles, we prove that arithmetical structures are
enumerated by the binomial coefficients
2n−1
n−1

, and we obtain refined enumeration results related
to multisets. In addition, we determine the critical groups for all arithmetical structures on paths
and cycles."""  # Goldbach's Conjecture
]

# The number of similar theorems to find for EACH query.
TOP_K = 10

# The path to your final, complete Parquet file.
INDEX_FILE = "abstract_with_DEEPSEEK_MATH_7B_FULL.parquet"

# A folder to save the output .txt files to.
OUTPUT_DIRECTORY = "search_results_deepseek_abstract"

# --- Model Configuration ---
MODEL_NAME = 'deepseek-ai/deepseek-math-7b-instruct'
GPU_DEVICE_ID = 0


# ===================================================================

# The SemanticSearch class is identical to the previous version.
# (No changes needed here)
class SemanticSearch:
    def __init__(self, model_name, device_id=1):
        print("--- Initializing Semantic Search Engine ---")
        self.model_name = model_name
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        print(f"Loading tokenizer for '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Loading model '{self.model_name}' onto device {device_id}...")
        self.model = AutoModel.from_pretrained(
            self.model_name, torch_dtype=torch.float16, load_in_4bit=True, device_map={'': device_id}
        )
        self.model.eval()
        print(f"Model loaded successfully on device: {self.model.device}")
        self.faiss_index = None
        self.dataframe = None

    def embed_text(self, texts):
        if isinstance(texts, str): texts = [texts]
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(
            self.model.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean pooling logic
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()

    def build_index(self, parquet_file_path):
        print(f"\nLoading dataset from '{parquet_file_path}'...")
        self.dataframe = pd.read_parquet(parquet_file_path)
        embeddings = np.stack(self.dataframe['embedding'].values).astype('float32')
        dimension = embeddings.shape[1]
        print(f"Building FAISS index for {len(embeddings)} vectors...")
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        print("--- Index is built and ready for search. ---")

    def search(self, query_text, top_k=5):
        if self.faiss_index is None: raise RuntimeError("Index not built.")

        query_vector = self.embed_text(query_text)
        distances, indices = self.faiss_index.search(query_vector, top_k)

        # Instead of printing, we'll now return the results as a formatted string.
        report = []
        report.append("=" * 80)
        report.append(f"--- Search Results for Top {top_k} Theorems ---")
        report.append(f'Query: "{query_text[:150]}..."')
        report.append("=" * 80)

        for i, idx in enumerate(indices[0]):
            result = self.dataframe.iloc[idx]
            dist = distances[0][i]
            report.append(f"\n--- Result {i + 1} | Distance: {dist:.4f} ---")
            report.append(f"  ID:         {result['id']}")
            report.append(f"  Categories: {result['categories']}")
            report.append(f"  Text:       {result['abstract']}")  # Full text

        return "\n".join(report)


# ===================================================================
# --- Main Execution Block (Batch Processing Logic) ---
# ===================================================================

if __name__ == "__main__":
    # --- 1. Initialize the Search Engine (once) ---
    search_engine = SemanticSearch(model_name=MODEL_NAME, device_id=GPU_DEVICE_ID)

    # --- 2. Build the Index (once) ---
    search_engine.build_index(INDEX_FILE)

    # --- 3. Create the output directory if it doesn't exist ---
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"\nCreated output directory: '{OUTPUT_DIRECTORY}'")

    # --- 4. Loop through each query and process it ---
    print(f"\nStarting batch processing for {len(QUERIES_TO_RUN)} queries...")
    for i, query in enumerate(QUERIES_TO_RUN):
        print(f"  Processing query {i + 1}/{len(QUERIES_TO_RUN)}...")

        # Perform the search
        results_report = search_engine.search(query_text=query, top_k=TOP_K)

        # Create a safe filename from the query
        safe_filename = "".join(x for x in query[:50] if x.isalnum() or x in " _-").rstrip()
        output_filepath = os.path.join(OUTPUT_DIRECTORY, f"query_{i + 1}_{safe_filename}.txt")

        # Save the results to a text file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(results_report)

        print(f"    -> Results saved to '{output_filepath}'")

    print("\n--- BATCH PROCESSING COMPLETE ---")