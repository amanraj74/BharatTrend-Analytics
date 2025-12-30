"""
REAL Reddit Data Collection
Collects actual product discussions from Indian communities
"""
import pandas as pd
import praw
from datetime import datetime, timedelta
from typing import List, Dict
import time
from tqdm import tqdm
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import load_config, setup_logging, save_checkpoint

logger = setup_logging()
load_dotenv()

class RealRedditCollector:
    """
    Collect REAL product discussions from Indian subreddits
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.reddit_config = config['social_media']['reddit']
        self.keywords = self.reddit_config['keywords']
        
        # Initialize with real credentials
        logger.info("Connecting to Reddit API...")
        
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'BharatTrend/1.0')
        
        if not client_id or not client_secret:
            logger.error("Reddit credentials not found in .env file!")
            logger.info("Please create .env file with:")
            logger.info("  REDDIT_CLIENT_ID=your_id")
            logger.info("  REDDIT_CLIENT_SECRET=your_secret")
            raise ValueError("Missing Reddit API credentials")
        
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Test connection
        try:
            self.reddit.user.me()
            logger.info("✓ Connected to Reddit API successfully (READ-ONLY mode)")
        except Exception as e:
            logger.info("✓ API connected (anonymous access)")
    
    def is_product_related(self, text: str) -> bool:
        """Check if text is about products"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Product keywords
        product_keywords = [
            'buy', 'purchase', 'product', 'review', 'recommend',
            'laptop', 'phone', 'mobile', 'gaming', 'headphone',
            'shoes', 'clothes', 'fashion', 'electronics',
            'amazon', 'flipkart', 'myntra', 'quality',
            'worth', 'price', 'deal', 'discount', 'sale'
        ]
        
        return any(keyword in text_lower for keyword in product_keywords)
    
    def collect_from_subreddit(self, subreddit_name: str, limit: int = 500) -> pd.DataFrame:
        """
        Collect REAL posts from subreddit
        """
        logger.info(f"Collecting REAL data from r/{subreddit_name}...")
        
        posts_data = []
        subreddit = self.reddit.subreddit(subreddit_name)
        
        try:
            # Collect from multiple sources
            sources = [
                ('hot', subreddit.hot(limit=limit // 3)),
                ('new', subreddit.new(limit=limit // 3)),
                ('top', subreddit.top(time_filter='month', limit=limit // 3))
            ]
            
            for source_name, posts in sources:
                logger.info(f"  Fetching {source_name} posts...")
                
                for post in tqdm(posts, desc=f"r/{subreddit_name} ({source_name})"):
                    full_text = f"{post.title} {post.selftext}"
                    
                    # Only collect product-related posts
                    if self.is_product_related(full_text):
                        posts_data.append({
                            'post_id': post.id,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'text': post.selftext[:500],  # Limit text length
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'upvote_ratio': post.upvote_ratio,
                            'url': post.url,
                            'author': str(post.author) if post.author else '[deleted]',
                            'source': source_name,
                            'is_self': post.is_self
                        })
                    
                    time.sleep(0.1)  # Rate limiting
            
            df = pd.DataFrame(posts_data)
            logger.info(f"✓ Collected {len(df)} REAL product posts from r/{subreddit_name}")
            return df
            
        except Exception as e:
            logger.error(f"✗ Error collecting from r/{subreddit_name}: {e}")
            return pd.DataFrame()
    
    def collect_all(self, max_per_subreddit: int = None) -> pd.DataFrame:
        """
        Collect from all configured subreddits
        """
        all_posts = []
        subreddits = self.reddit_config['subreddits']
        posts_per_sub = max_per_subreddit or self.reddit_config['posts_per_subreddit']
        
        logger.info("="*60)
        logger.info(f"Starting REAL data collection from {len(subreddits)} subreddits")
        logger.info(f"Target: {posts_per_sub} posts per subreddit")
        logger.info("="*60)
        
        for subreddit in subreddits:
            try:
                df = self.collect_from_subreddit(subreddit, limit=posts_per_sub)
                if not df.empty:
                    all_posts.append(df)
                time.sleep(2)  # Pause between subreddits
            except Exception as e:
                logger.error(f"Failed to collect from r/{subreddit}: {e}")
                continue
        
        if all_posts:
            combined = pd.concat(all_posts, ignore_index=True)
            
            # Remove duplicates
            original_count = len(combined)
            combined = combined.drop_duplicates(subset=['post_id'])
            logger.info(f"Removed {original_count - len(combined)} duplicate posts")
            
            # Sort by date
            combined = combined.sort_values('created_utc', ascending=False)
            
            logger.info("="*60)
            logger.info(f"✓ Total REAL posts collected: {len(combined)}")
            logger.info(f"✓ Date range: {combined['created_utc'].min()} to {combined['created_utc'].max()}")
            logger.info("="*60)
            
            # Save
            output_path = f"{self.config['paths']['raw_data']}/reddit_posts_real.csv"
            save_checkpoint(combined, output_path)
            
            # Print sample
            logger.info("\nSample posts:")
            for _, row in combined.head(3).iterrows():
                logger.info(f"  • [{row['subreddit']}] {row['title'][:60]}...")
            
            return combined
        else:
            logger.error("No data collected from any subreddit!")
            return pd.DataFrame()

def main():
    """
    Run REAL data collection
    """
    logger.info("="*60)
    logger.info("BharatTrend REAL Data Collection")
    logger.info("="*60)
    
    config = load_config()
    
    # Check credentials
    if not os.getenv('REDDIT_CLIENT_ID'):
        logger.error("\n⚠️  SETUP REQUIRED!")
        logger.info("\n1. Go to: https://www.reddit.com/prefs/apps")
        logger.info("2. Create app (script type)")
        logger.info("3. Add to .env file:")
        logger.info("   REDDIT_CLIENT_ID=your_id")
        logger.info("   REDDIT_CLIENT_SECRET=your_secret")
        logger.info("   REDDIT_USER_AGENT=BharatTrend/1.0")
        return
    
    # Collect real data
    collector = RealRedditCollector(config)
    
    # Ask user for quantity
    logger.info("\nHow many posts per subreddit? (100-1000)")
    logger.info("Recommended: 500 for good analysis")
    
    try:
        limit = int(input("Enter number (or press Enter for 500): ") or 500)
    except:
        limit = 500
    
    real_data = collector.collect_all(max_per_subreddit=limit)
    
    if not real_data.empty:
        logger.info("\n✅ SUCCESS! REAL data collected and saved.")
        logger.info(f"Location: data/raw/reddit_posts_real.csv")
        logger.info(f"\nNext step: python src/data/clean_data.py")
    else:
        logger.error("\n❌ Data collection failed. Check your API credentials.")

if __name__ == "__main__":
    main()
