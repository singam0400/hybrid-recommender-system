
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_product_metadata(filepath="data/products.csv", max_rows=10000):
    df = pd.read_csv(filepath, nrows=max_rows)
    
    # Filter to just category and brand (or only category if limited)
    filtered = df[df['property'].isin(['categoryid'])]

    # Create one row per product with combined metadata as a string
    grouped = filtered.groupby('itemid')['value'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    grouped = grouped.rename(columns={'itemid': 'product_id', 'value': 'metadata'})
    
    return grouped  # Columns: product_id, metadata


def build_content_similarity_matrix(metadata_df):
    """
    Compute cosine similarity matrix using TF-IDF of product metadata
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(metadata_df['metadata'])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(similarity_matrix, 
                          index=metadata_df['product_id'], 
                          columns=metadata_df['product_id'])

    return sim_df


def get_similar_products_cb(product_id, similarity_df, top_k=5):
    if product_id not in similarity_df.index:
        return {}

    sim_scores = similarity_df[product_id].sort_values(ascending=False).drop(product_id)
    return sim_scores.head(top_k).to_dict()


if __name__ == "__main__":
    meta = prepare_product_metadata()
    sim_matrix = build_content_similarity_matrix(meta)

    sample_id = meta['product_id'].iloc[0]
    print(f"\n Top content-similar products to {sample_id}:")
    for pid, score in get_similar_products_cb(sample_id, sim_matrix).items():
        print(f"  → {pid} (score: {score:.4f})")
