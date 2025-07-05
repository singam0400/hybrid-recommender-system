
import pandas as pd

ACTION_WEIGHTS = {
    'view': 1,
    'addtocart': 2,
    'transaction': 3
}

def load_and_preprocess_interactions(filepath="data/interactions.csv", sample_size=5000):
    df = pd.read_csv(filepath)
    
    # Filter to only relevant events
    df = df[df['event'].isin(ACTION_WEIGHTS.keys())]
    
    # Sample small subset for quick training
    df = df.sample(n=sample_size, random_state=42)
    
    # Map event types to weights
    df['weight'] = df['event'].map(ACTION_WEIGHTS)
    
    # Rename for clarity
    df = df.rename(columns={
        'visitorid': 'user_id',
        'itemid': 'product_id'
    })

    # Drop unused columns
    df = df[['user_id', 'product_id', 'event', 'weight', 'timestamp']]
    
    return df
