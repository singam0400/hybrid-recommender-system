
def get_hybrid_recommendations(product_id, 
                                collab_matrix, 
                                content_matrix, 
                                alpha=0.5, 
                                top_k=5):
    """
    Returns top-k recommended products based on weighted hybrid of
    collaborative and content-based scores.

    Args:
        product_id (int or str): target product
        collab_matrix (pd.DataFrame): item-item collaborative sim
        content_matrix (pd.DataFrame): item-item content sim
        alpha (float): weight for collaborative (between 0 and 1)
        top_k (int): number of recommendations

    Returns:
        dict: product_id → hybrid score
    """
    if product_id not in collab_matrix.index or product_id not in content_matrix.index:
        print(f"[Warning] Product {product_id} missing in one or both matrices.")
        return {}

    # Align scores: fill NaN with 0 to be safe
    collab_scores = collab_matrix.loc[product_id].fillna(0)
    content_scores = content_matrix.loc[product_id].fillna(0)

    # Drop self similarity
    collab_scores = collab_scores.drop(product_id, errors='ignore')
    content_scores = content_scores.drop(product_id, errors='ignore')

    # Union of indices (some products may exist in only one)
    all_products = set(collab_scores.index) | set(content_scores.index)

    # Final hybrid score computation
    final_scores = {}
    for pid in all_products:
        collab_val = collab_scores.get(pid, 0)
        content_val = content_scores.get(pid, 0)
        hybrid_val = alpha * collab_val + (1 - alpha) * content_val
        final_scores[pid] = hybrid_val

    # Sort and return top_k
    sorted_scores = dict(sorted(final_scores.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:top_k])
    return sorted_scores

if __name__ == "__main__":
    from data_loader import load_and_preprocess_interactions
    from collaborative import build_item_item_matrix
    from content_based import prepare_product_metadata, build_content_similarity_matrix

    print(" Loading and processing data...")
    interactions = load_and_preprocess_interactions(sample_size=5000)
    meta = prepare_product_metadata()

    print(" Building collaborative matrix...")
    collab_sim = build_item_item_matrix(interactions)

    print(" Building content-based matrix...")
    content_sim = build_content_similarity_matrix(meta)

    valid_ids = set(collab_sim.index) & set(content_sim.index)
    sample_pid = list(valid_ids)[0]

    print(f"\n Hybrid recommendations for {sample_pid}:")
    hybrid_recs = get_hybrid_recommendations(sample_pid, collab_sim, content_sim, alpha=0.7)
    for pid, score in hybrid_recs.items():
        print(f"  → {pid} (score: {score:.4f})")
