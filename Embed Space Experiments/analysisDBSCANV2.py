# FILE: final_report.py (Corrected and Refined)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from umap import UMAP

# ===================================================================
# --- CONFIGURATION ---
# ===================================================================
INPUT_FILE = 'thethedeep15.parquet'
OUTPUT_PLOT_FILE = 'final_cluster_visualization.png'

# --- Analysis Parameters ---
SAMPLES_TO_SHOW = 7
TOP_N_CATEGORIES = 10
SILHOUETTE_SAMPLE_SIZE = 50000

# ===================================================================
# --- SCRIPT START ---
# ===================================================================

print("--- Final Cluster Analysis Report (Refined) ---")
print(f"Loading data from '{INPUT_FILE}'...")

try:
    df = pd.read_parquet(INPUT_FILE)
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{INPUT_FILE}' was not found.")
    exit()

X = np.stack(df['embedding'].values).astype('float32')
n_clusters = df['cluster'].max() + 1
print(f"Successfully loaded {len(df)} records with {n_clusters} clusters.")


# --- 1. High-Level Summary & Quality Score ---
print("\n" + "=" * 80)
print("--- 1. OVERALL CLUSTER SUMMARY & QUALITY ---")
print("=" * 80)
cluster_counts = df['cluster'].value_counts().sort_index()

print("Cluster Sizes:")
print(cluster_counts.to_string())

print("\nCalculating Silhouette Score (on a sample)...")
sample_indices = np.random.choice(df.index, size=min(SILHOUETTE_SAMPLE_SIZE, len(df)), replace=False)
sample_X = X[sample_indices]
sample_labels = df['cluster'].iloc[sample_indices]

if len(np.unique(sample_labels)) > 1:
    score = silhouette_score(sample_X, sample_labels)
    print(f"Silhouette Score: {score:.4f}")
else:
    print("Not enough clusters in the sample to calculate Silhouette Score.")


# --- 2. Deep Dive into Each Cluster ---
print("\n" + "=" * 80)
print("--- 2. DETAILED CLUSTER REPORTS ---")
print("=" * 80)

print("Calculating cluster centroids for representative analysis...")
centroids = np.array([X[df['cluster'] == i].mean(axis=0) for i in range(n_clusters)])
print("Centroids calculated.")

for cluster_id in sorted(df['cluster'].unique()):
    cluster_df = df[df['cluster'] == cluster_id].copy()

    print("\n\n" + "-" * 70)
    print(f"--- Cluster {cluster_id} | Size: {len(cluster_df)} ---")
    print("-" * 70)

    # A) Category Distribution
    print("\n[Analysis] Top Categories:")
    category_dist = cluster_df['categories'].value_counts(normalize=True)
    print(category_dist.head(TOP_N_CATEGORIES).to_string())
    if not category_dist.empty:
        top_cat = category_dist.index[0]
        top_pct = category_dist.iloc[0]
        print(f"--> Interpretation: This cluster is heavily focused on '{top_cat}' ({top_pct:.1%}).")

    # B) Most Representative Theorems
    print("\n[Analysis] Most Representative Members (Closest to Centroid):")
    cluster_center = centroids[cluster_id]
    cluster_embeddings = np.stack(cluster_df['embedding'].values)
    distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
    central_indices = np.argsort(distances)[:SAMPLES_TO_SHOW]
    representative_samples = cluster_df.iloc[central_indices]
    for _, row in representative_samples.iterrows():
        print(f"  - (ID: {row['id']}) {row['text'].replace(chr(10), ' ').strip()[:100]}...")

    # C) Duplicate ArXiv ID Analysis (interpretation removed as requested)
    print("\n[Analysis] ArXiv IDs appearing multiple times in this cluster:")
    id_counts = cluster_df['id'].value_counts()
    duplicates = id_counts[id_counts > 1]
    if duplicates.empty:
        print("  - None found.")
    else:
        print(duplicates.head().to_string())

# --- 3. Generate Overall Visualization ---
print("\n\n" + "=" * 80)
print("--- 3. GENERATING 2D VISUALIZATION ---")
print("=" * 80)
print("Running UMAP to reduce dimensions for plotting (this can take a moment)...")

sample_size = min(100000, len(X))
if sample_size < len(X):
    print(f"Using a sample of {sample_size} points for UMAP visualization for speed.")
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    plot_X = X[sample_indices]
    plot_labels = df['cluster'].iloc[sample_indices].values
else:
    plot_X = X
    plot_labels = df['cluster'].values

# --- THE FIX IS HERE: We call UMAP(...) directly ---
reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
embedding_2d = reducer.fit_transform(plot_X)

df_plot = pd.DataFrame(embedding_2d, columns=('x', 'y'))
df_plot['cluster'] = plot_labels

print("Creating scatter plot...")
plt.figure(figsize=(16, 12))
palette = sns.color_palette('hsv', n_colors=n_clusters)
sns.scatterplot(
    data=df_plot,
    x='x', y='y', hue='cluster', palette=palette, s=5, alpha=0.7, legend='full'
)
plt.title(f'UMAP Projection of Theorem Clusters (K={n_clusters})', fontsize=16)
plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE)
print(f"Plot saved to '{OUTPUT_PLOT_FILE}'")
print("\n--- MISSION COMPLETE ---")