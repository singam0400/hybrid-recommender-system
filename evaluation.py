
def precision_at_k(recommended: list, actual: list, k: int = 5) -> float:
    """
    Precision@K: How many of the top-K recommended items are relevant?
    
    Args:
        recommended (list): List of recommended product IDs.
        actual (list): List of actual product IDs interacted with (ground truth).
        k (int): Top-K cutoff.

    Returns:
        float: Precision score.
    """
    if not recommended or not actual:
        return 0.0

    recommended_k = recommended[:k]
    relevant = set(recommended_k) & set(actual)
    return len(relevant) / k


def recall_at_k(recommended: list, actual: list, k: int = 5) -> float:
    """
    Recall@K: What fraction of the relevant items were recommended in top-K?

    Args:
        recommended (list): List of recommended product IDs.
        actual (list): List of actual relevant product IDs (e.g., purchased).
        k (int): Top-K cutoff.

    Returns:
        float: Recall score.
    """
    if not recommended or not actual:
        return 0.0

    recommended_k = recommended[:k]
    relevant = set(recommended_k) & set(actual)
    return len(relevant) / len(actual)

if __name__ == "__main__":
    recs = [101, 202, 303, 404, 505]
    actual = [202, 303, 777]

    print("Precision@5:", precision_at_k(recs, actual, k=5))   # → 2/5 = 0.40
    print("Recall@5:", recall_at_k(recs, actual, k=5))         # → 2/3 ≈ 0.67
