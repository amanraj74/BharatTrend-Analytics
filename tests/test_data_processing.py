"""
Tests for data processing module
"""
import pytest
import pandas as pd
from pathlib import Path

def test_data_exists():
    """Test if data files exist"""
    data_path = Path("data/processed/enhanced_products.csv")
    assert data_path.exists(), "Enhanced products file not found"

def test_data_quality():
    """Test data quality"""
    df = pd.read_csv("data/processed/enhanced_products.csv")
    
    # Check columns exist
    assert 'selling_price' in df.columns
    assert 'main_category' in df.columns
    assert 'discount' in df.columns
    
    # Check no nulls in critical columns
    assert df['selling_price'].notna().all()
    assert df['main_category'].notna().all()
    
    # Check valid ranges
    assert (df['selling_price'] > 0).all()
    assert (df['discount'] >= 0).all()

def test_categories():
    """Test category processing"""
    df = pd.read_csv("data/processed/enhanced_products.csv")
    assert df['main_category'].nunique() > 0
    assert len(df) > 1000  # At least 1000 products
