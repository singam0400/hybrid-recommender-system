
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def build_item_item_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Builds an item-item similarity matrix using weighted implicit feedback.

    Args:
        interactions (pd.DataFrame): DataFrame with columns [user_id, product_id, weight]

    Returns:
        pd.DataFrame: item-item similarity matrix (square DataFrame)
    """
    # Pivot table: users as rows, products as columns, weights as values
    user_product_matrix = interactions.pivot_table(
        index='user_id',
        columns='product_id',
        values='weight',
        aggfunc='sum',
        fill_value=0
    )

    # Convert to sparse matrix for efficiency
    sparse_matrix = csr_matrix(user_product_matrix.values)

    # Compute cosine similarity between products (items = columns)
    item_sim = cosine_similarity(sparse_matrix.T)

    # Map back to product IDs
    product_ids = user_product_matrix.columns
    similarity_df = pd.DataFrame(item_sim, index=product_ids, columns=product_ids)

    return similarity_df


def get_similar_products(product_id, similarity_df, top_k=5):
    """
    Retrieves top-k similar products based on item-item similarity matrix.

    Args:
        product_id (int or str): The target product ID
        similarity_df (pd.DataFrame): Item-item similarity matrix
        top_k (int): Number of top similar products to return

    Returns:
        dict: product_id → similarity score
    """
    if product_id not in similarity_df.index:
        print(f"[Warning] Product ID {product_id} not found in similarity matrix.")
        return {}

    # Sort similarity scores in descending order, drop self
    sim_scores = similarity_df[product_id].sort_values(ascending=False).drop(product_id)

    # Return top_k as dictionary
    return sim_scores.head(top_k).to_dict()


#  Optional Test Script
if __name__ == "__main__":
    from data_loader import load_and_preprocess_interactions

    print(" Loading sample interactions...")
    df = load_and_preprocess_interactions(sample_size=5000)

    print(" Building item-item similarity matrix...")
    sim_matrix = build_item_item_matrix(df)

    sample_pid = df['product_id'].iloc[0]
    print(f"\n Top similar products to {sample_pid}:")
    top_similar = get_similar_products(sample_pid, sim_matrix, top_k=5)
    for pid, score in top_similar.items():
        print(f"  → {pid} (score: {score:.4f})")
