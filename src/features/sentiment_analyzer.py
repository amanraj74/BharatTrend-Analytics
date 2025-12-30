"""
Multilingual Sentiment Analysis Module
Uses transformer models for Hindi and English sentiment detection
"""
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import load_config, setup_logging, load_data, save_checkpoint

logger = setup_logging()

class SentimentAnalyzer:
    """
    Analyze sentiment of social media posts
    Handles both English and Indian languages
    """
    
    def __init__(self, config: Dict):
        """
        Initialize sentiment analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sentiment_config = config['sentiment']
        self.model_name = self.sentiment_config['model_name']
        self.device = 0 if torch.cuda.is_available() else -1
        
        logger.info(f"Loading sentiment model: {self.model_name}")
        logger.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        self.sentiment_pipeline = None
        self.load_model()
    
    def load_model(self):
        """
        Load pretrained sentiment analysis model
        """
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                max_length=self.sentiment_config['max_length'],
                truncation=True
            )
            logger.info("✓ Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            logger.info(f"Trying backup model: {self.sentiment_config['backup_model']}")
            
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.sentiment_config['backup_model'],
                    device=self.device
                )
                logger.info("✓ Backup model loaded successfully")
            except Exception as e2:
                logger.error(f"✗ Backup model also failed: {e2}")
                raise
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of single text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment label and score
        """
        if not text or len(text.strip()) < 3:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            result = self.sentiment_pipeline(text[:512])[0]  # Limit text length
            
            # Normalize label to positive/negative/neutral
            label = result['label'].upper()
            score = result['score']
            
            # Map different label formats
            if 'POS' in label or label == 'LABEL_2':
                sentiment = 'POSITIVE'
            elif 'NEG' in label or label == 'LABEL_0':
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            return {
                'sentiment': sentiment,
                'confidence': score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5}
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
        """
        Analyze sentiment for entire dataframe
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            
        Returns:
            DataFrame with sentiment columns added
        """
        logger.info(f"Analyzing sentiment for {len(df)} texts...")
        
        df = df.copy()
        
        # Batch processing for efficiency
        texts = df[text_column].fillna("").tolist()
        results = []
        
        batch_size = self.sentiment_config['batch_size']
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch = texts[i:i + batch_size]
            batch_results = [self.analyze_text(text) for text in batch]
            results.extend(batch_results)
        
        # Add results to dataframe
        df['sentiment'] = [r['sentiment'] for r in results]
        df['sentiment_confidence'] = [r['confidence'] for r in results]
        
        # Calculate sentiment score (-1 to 1)
        df['sentiment_score'] = df.apply(
            lambda row: row['sentiment_confidence'] if row['sentiment'] == 'POSITIVE'
                       else -row['sentiment_confidence'] if row['sentiment'] == 'NEGATIVE'
                       else 0.0,
            axis=1
        )
        
        logger.info(f"✓ Sentiment analysis completed")
        
        # Print summary stats
        sentiment_counts = df['sentiment'].value_counts()
        logger.info(f"Sentiment distribution:\n{sentiment_counts}")
        
        return df
    
    def get_sentiment_trends(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Calculate sentiment trends over time
        
        Args:
            df: DataFrame with sentiment data
            date_column: Name of date column
            
        Returns:
            DataFrame with daily sentiment aggregates
        """
        trends = df.groupby(date_column).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment': lambda x: (x == 'POSITIVE').sum() / len(x)  # Positive ratio
        }).reset_index()
        
        trends.columns = [date_column, 'avg_sentiment', 'sentiment_std', 'post_count', 'positive_ratio']
        
        return trends

def main():
    """
    Main execution - analyze sentiment in processed data
    """
    logger.info("="*60)
    logger.info("BharatTrend Sentiment Analysis Pipeline")
    logger.info("="*60)
    
    config = load_config()
    analyzer = SentimentAnalyzer(config)
    
    # Look for processed Reddit data
    processed_path = f"{config['paths']['processed_data']}/reddit_cleaned.csv"
    
    if not Path(processed_path).exists():
        logger.warning(f"No processed data found at {processed_path}")
        logger.info("Run data collection and cleaning first")
        return
    
    # Load and analyze
    logger.info(f"\nLoading data from: {processed_path}")
    df = load_data(processed_path)
    
    if df is not None and not df.empty:
        # Analyze sentiment
        df_with_sentiment = analyzer.analyze_dataframe(df)
        
        # Save results
        output_path = f"{config['paths']['processed_data']}/reddit_with_sentiment.csv"
        save_checkpoint(df_with_sentiment, output_path)
        
        # Calculate trends
        logger.info("\nCalculating sentiment trends...")
        trends = analyzer.get_sentiment_trends(df_with_sentiment)
        trends_path = f"{config['paths']['processed_data']}/sentiment_trends.csv"
        save_checkpoint(trends, trends_path)
        
        logger.info("\n" + "="*60)
        logger.info("✓ Sentiment analysis pipeline completed")
        logger.info("="*60)
    else:
        logger.error("Failed to load data")

if __name__ == "__main__":
    main()
