"""
Create sample data for testing the pipeline
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import load_config, setup_logging, save_checkpoint

logger = setup_logging()

def create_sample_reddit_data(n_posts: int = 200) -> pd.DataFrame:
    """
    Create sample Reddit posts for testing
    """
    logger.info(f"Creating {n_posts} sample Reddit posts...")
    
    # Sample product-related titles and texts
    titles = [
        "Best laptop to buy under 50k?",
        "Review of new iPhone - totally worth it!",
        "Disappointed with Samsung quality",
        "Looking for gaming headphones recommendations",
        "Just bought Nike shoes, amazing quality",
        "OnePlus phone battery drains too fast",
        "Recommend good smartwatch for fitness",
        "Awesome deal on Amazon today!",
        "Product arrived damaged, poor packaging",
        "Great customer service from Flipkart"
    ]
    
    subreddits = ["india", "IndianGaming", "IndianFashion", "bangalore", "mumbai"]
    
    # Generate data
    data = []
    start_date = datetime.now() - timedelta(days=60)
    
    for i in range(n_posts):
        post_date = start_date + timedelta(days=i % 60, hours=np.random.randint(0, 24))
        
        data.append({
            'post_id': f'post_{i}',
            'created_utc': post_date,
            'subreddit': np.random.choice(subreddits),
            'title': np.random.choice(titles),
            'text': "Sample post text with product discussion and opinions.",
            'score': np.random.randint(1, 500),
            'num_comments': np.random.randint(0, 100),
            'upvote_ratio': np.random.uniform(0.6, 0.95),
            'url': f"https://reddit.com/r/india/comments/{i}",
            'author': f"user_{i}",
            'source': 'hot'
        })
    
    df = pd.DataFrame(data)
    logger.info(f"✓ Created {len(df)} sample posts")
    return df

def main():
    """
    Generate sample data for testing
    """
    logger.info("="*60)
    logger.info("Creating Sample Data for Testing")
    logger.info("="*60)
    
    config = load_config()
    
    # Create sample Reddit data
    reddit_df = create_sample_reddit_data(200)
    output_path = f"{config['paths']['raw_data']}/reddit_posts.csv"
    save_checkpoint(reddit_df, output_path)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Sample data created successfully")
    logger.info("Run pipeline: python src/data/clean_data.py")
    logger.info("="*60)

if __name__ == "__main__":
    main()
