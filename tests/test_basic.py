"""
Basic unit tests for BharatTrend
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import DataProcessor

def test_data_loading():
    """Test if data loads correctly"""
    processor = DataProcessor()
    processor.load_data()
    
    assert processor.flipkart_df is not None
    assert len(processor.flipkart_df) > 0

def test_unified_products_exists():
    """Test if unified products file exists"""
    path = Path('data/processed/unified_products.csv')
    assert path.exists(), "Unified products file not found"
    
    df = pd.read_csv(path)
    assert len(df) > 0
    assert 'selling_price' in df.columns
    assert 'category' in df.columns

def test_enhanced_products_exists():
    """Test if enhanced products file exists"""
    path = Path('data/processed/enhanced_products.csv')
    assert path.exists(), "Enhanced products file not found"
    
    df = pd.read_csv(path)
    assert len(df) > 0
    assert 'main_category' in df.columns
    assert 'discount' in df.columns

def test_data_quality():
    """Test data quality"""
    df = pd.read_csv('data/processed/enhanced_products.csv')
    
    # Check for null values in critical columns
    assert df['selling_price'].notna().all()
    assert df['main_category'].notna().all()
    
    # Check price ranges
    assert (df['selling_price'] > 0).all()
    assert (df['discount'] >= 0).all()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
