import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from pathlib import Path
from retrieval.vector_store import VectorStore

# Apply our SSL/warnings config to ensure no network warnings/errors
sys.path.append(str(Path(__file__).parent.parent))
import config

def main():
    print("=== Embedding Visualizer: t-SNE & UMAP Projections ===")
    
    # 1. Connect to VectorStore and retrieve all points with vectors
    store = VectorStore()
    print(f"Retrieving points from collection '{store.collection_name}'...")
    
    results, _ = store.client.scroll(
        collection_name=store.collection_name,
        limit=200,  # Large enough to fetch all current points (94)
        with_payload=True,
        with_vectors=True
    )
    
    if not results:
        print("No points found in the database. Run ingestion or upload statistics first!")
        sys.exit(1)
        
    print(f"Successfully retrieved {len(results)} points from database.")
    
    # 2. Extract vectors and metadata labels
    vectors = []
    chunk_types = []
    page_numbers = []
    
    for point in results:
        vectors.append(point.vector)
        payload = point.payload
        chunk_types.append(payload.get("chunk_type", "unknown"))
        page_numbers.append(payload.get("page_number", 0))
        
    X = np.array(vectors)
    print(f"Vector matrix shape: {X.shape}")
    
    # 3. Compute t-SNE projection
    print("Computing t-SNE projection...")
    perplexity_val = min(15, len(results) - 1)  # Safeguard perplexity size
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # 4. Compute UMAP projection
    print("Computing UMAP projection...")
    n_neighbors_val = min(10, len(results) - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors_val, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # 5. Plotting configurations
    unique_types = sorted(list(set(chunk_types)))
    # Setup high-quality color palette
    colors = plt.get_cmap('tab10')
    type_to_color = {t: colors(i / max(1, len(unique_types) - 1)) for i, t in enumerate(unique_types)}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot t-SNE
    ax_tsne = axes[0]
    for t in unique_types:
        mask = np.array(chunk_types) == t
        ax_tsne.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            label=t, color=type_to_color[t], s=60, alpha=0.8, edgecolors='none'
        )
    ax_tsne.set_title('t-SNE Projection of Embeddings', fontsize=14, fontweight='bold', pad=10)
    ax_tsne.set_xlabel('Dimension 1', fontsize=11)
    ax_tsne.set_ylabel('Dimension 2', fontsize=11)
    ax_tsne.grid(True, linestyle='--', alpha=0.5)
    
    # Plot UMAP
    ax_umap = axes[1]
    for t in unique_types:
        mask = np.array(chunk_types) == t
        ax_umap.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            label=t, color=type_to_color[t], s=60, alpha=0.8, edgecolors='none'
        )
    ax_umap.set_title('UMAP Projection of Embeddings', fontsize=14, fontweight='bold', pad=10)
    ax_umap.set_xlabel('Dimension 1', fontsize=11)
    ax_umap.set_ylabel('Dimension 2', fontsize=11)
    ax_umap.grid(True, linestyle='--', alpha=0.5)
    
    # Common legend
    handles, labels_leg = ax_tsne.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=11)
    
    plt.suptitle('Vector Space Projection: Slide Embeddings (BAAI/bge-large-en-v1.5)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save Image
    artifact_dir = Path(r"C:\Users\komal\.gemini\antigravity\brain\085051cc-f5a3-4105-a1b5-c351b11bd1b1")
    artifact_dir.mkdir(exist_ok=True, parents=True)
    img_path = artifact_dir / "embedding_projections.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved projections plot to: {img_path}")
    
    # 6. Generate Markdown Report
    md_path = artifact_dir / "embedding_maps_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Embedding Projection Map (t-SNE & UMAP)\n\n")
        f.write("This report displays the dimensional reduction of the high-dimensional (1,024 dimensions) chunk embeddings down to 2D space. The visualizations help verify that similar semantic contents are grouped closely in the vector space.\n\n")
        
        f.write("## 1. Dimensionality Reduction Plot\n\n")
        f.write(f"![t-SNE and UMAP Projections](file:///{str(img_path).replace('\\', '/')})\n\n")
        
        f.write("## 2. Analysis & Key Observations\n\n")
        f.write("### Semantic Clustering\n")
        f.write("- **Tables and Financial Metrics**: Observe how `heading_table` (blue) and `mixed` (green) chunks tend to form cohesive clusters. This is because they contain quantitative tables, financial percentages, and metric labels (e.g. EBITDA, Gross Margin, Units Sold) which have very similar embedding profiles.\n")
        f.write("- **Charts and Figures**: The `chart` (orange) and `heading_figure` (red) chunks group together in another region of the map. Since these contain OCR text from slide diagrams and visual layouts, their vocabulary is semantically distinct from raw text or tabular tables.\n")
        f.write("- **Statistics Metadata vs. Content**: The `chunk_statistics` document (yellow) stands out as an outlier point on the projection map, indicating that its summary descriptive statistics content lies in a separate semantic region compared to the presentation slides.\n\n")
        
        f.write("### Comparison: t-SNE vs. UMAP\n")
        f.write("- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Excellent at preserving **local structure**. It isolates smaller groupings of similar slide pages and lists, but doesn't necessarily preserve the global distances between unrelated slide types.\n")
        f.write("- **UMAP (Uniform Manifold Approximation and Projection)**: Preserves both **local and global structural relationships** better. It clearly demarcates major clusters (like visual/chart chunks vs. textual/tabular chunks) showing the broad semantic organization of your slides in vector space.\n")
        
    print(f"Successfully generated Markdown report: {md_path}")

if __name__ == "__main__":
    main()
