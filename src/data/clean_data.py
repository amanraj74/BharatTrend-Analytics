"""
Data Cleaning and Preprocessing Module
Cleans raw social media and sales data for analysis
"""
import pandas as pd
import re
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import load_config, setup_logging, load_data, save_checkpoint

logger = setup_logging()

class DataCleaner:
    """
    Clean and preprocess data for sentiment analysis and forecasting
    """
    
    def __init__(self, config: dict):
        """
        Initialize data cleaner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = config['paths']
    
    def clean_text(self, text: str) -> str:
        """
        Clean text data - remove URLs, special chars, extra spaces
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove Reddit markdown
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove [text](url)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def clean_reddit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Reddit posts data
        
        Args:
            df: Raw Reddit data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning Reddit data: {len(df)} posts")
        
        df = df.copy()
        
        # Clean text fields
        df['title_clean'] = df['title'].apply(self.clean_text)
        df['text_clean'] = df['text'].apply(self.clean_text)
        
        # Combine title and text
        df['full_text'] = df['title_clean'] + ' ' + df['text_clean']
        
        # Remove empty posts
        df = df[df['full_text'].str.len() > 10]
        
        # Remove deleted/removed posts
        df = df[~df['author'].isin(['[deleted]', '[removed]'])]
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['created_utc']):
            df['created_utc'] = pd.to_datetime(df['created_utc'])
        
        # Extract date features
        df['date'] = df['created_utc'].dt.date
        df['year'] = df['created_utc'].dt.year
        df['month'] = df['created_utc'].dt.month
        df['day_of_week'] = df['created_utc'].dt.dayofweek
        df['hour'] = df['created_utc'].dt.hour
        
        # Sort by date
        df = df.sort_values('created_utc')
        
        logger.info(f"✓ Cleaned: {len(df)} posts remaining")
        return df
    
    def clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean e-commerce sales data from Kaggle
        
        Args:
            df: Raw sales data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning sales data: {len(df)} records")
        
        df = df.copy()
        
        # Common column name standardization
        column_mapping = {
            'Date': 'date',
            'date': 'date',
            'Product': 'product',
            'product_name': 'product',
            'Category': 'category',
            'Sales': 'sales',
            'Quantity': 'quantity',
            'Price': 'price'
        }
        
        # Rename columns if they exist
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        
        # Remove negative values
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df = df[df[col] >= 0]
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        logger.info(f"✓ Cleaned: {len(df)} records remaining")
        return df
    
    def process_raw_data(self):
        """
        Process all raw data files
        """
        logger.info("="*60)
        logger.info("Starting Data Cleaning Pipeline")
        logger.info("="*60)
        
        # Process Reddit data
        reddit_path = f"{self.paths['raw_data']}/reddit_posts.csv"
        if Path(reddit_path).exists():
            logger.info("\n1. Processing Reddit data...")
            reddit_df = load_data(reddit_path)
            if reddit_df is not None:
                reddit_clean = self.clean_reddit_data(reddit_df)
                output_path = f"{self.paths['processed_data']}/reddit_cleaned.csv"
                save_checkpoint(reddit_clean, output_path)
        else:
            logger.warning(f"Reddit data not found: {reddit_path}")
        
        # Process sales data
        sales_files = list(Path(self.paths['external_data']).glob('*.csv'))
        if sales_files:
            logger.info(f"\n2. Processing {len(sales_files)} sales data files...")
            for file in sales_files:
                logger.info(f"Processing: {file.name}")
                df = load_data(str(file))
                if df is not None:
                    df_clean = self.clean_sales_data(df)
                    output_path = f"{self.paths['processed_data']}/{file.stem}_cleaned.csv"
                    save_checkpoint(df_clean, output_path)
        else:
            logger.warning(f"No sales data found in {self.paths['external_data']}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ Data cleaning pipeline completed")
        logger.info("="*60)

def main():
    """
    Main execution
    """
    config = load_config()
    cleaner = DataCleaner(config)
    cleaner.process_raw_data()

if __name__ == "__main__":
    main()
