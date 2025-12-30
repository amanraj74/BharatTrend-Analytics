"""
Utility functions for BharatTrend
Clean, maintainable helper functions following industry best practices
"""
import yaml
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
from typing import Dict, List, Optional

def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        raise

def setup_logging(log_dir: str = "logs") -> logger:
    """
    Setup logging system with console and file output
    Follows production logging patterns
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    
    # Remove default logger
    logger.remove()
    
    # Console output - INFO level with colors
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True
    )
    
    # File output - DEBUG level for troubleshooting
    logger.add(
        f"{log_dir}/bharattrend.log",
        rotation="50 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        encoding="utf-8"
    )
    
    logger.info("Logging system initialized")
    return logger

def ensure_directories(config: Dict) -> None:
    """
    Create all required directories from config
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        Path(path).mkdir(exist_ok=True, parents=True)
    logger.info("All required directories created")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate dataframe has required columns
    
    Args:
        df: Pandas dataframe to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"✓ DataFrame validated: {len(df)} rows, {len(df.columns)} columns")
    return True

def save_checkpoint(data: pd.DataFrame, 
                   filepath: str, 
                   format: str = 'csv',
                   index: bool = False) -> bool:
    """
    Save data checkpoint with proper error handling
    Follows immutability principle - never overwrite raw data
    
    Args:
        data: DataFrame to save
        filepath: Output file path
        format: File format ('csv' or 'parquet')
        index: Whether to save index
        
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'csv':
            data.to_csv(filepath, index=index, encoding='utf-8')
        elif format == 'parquet':
            data.to_parquet(filepath, index=index)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'")
        
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✓ Saved: {filepath} ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to save {filepath}: {e}")
        return False

def load_data(filepath: str, format: str = 'csv') -> Optional[pd.DataFrame]:
    """
    Load data from file with error handling
    
    Args:
        filepath: Path to data file
        format: File format ('csv' or 'parquet')
        
    Returns:
        DataFrame if successful, None otherwise
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None
    
    try:
        if format == 'csv':
            df = pd.read_csv(filepath, encoding='utf-8')
        elif format == 'parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✓ Loaded: {filepath} ({len(df)} rows)")
        return df
        
    except Exception as e:
        logger.error(f"✗ Failed to load {filepath}: {e}")
        return None

def get_project_root() -> Path:
    """
    Get project root directory
    
    Returns:
        Path object pointing to project root
    """
    return Path(__file__).parent.parent

if __name__ == "__main__":
    # Test utilities
    print("Testing BharatTrend utilities...")
    
    config = load_config()
    print(f"✓ Config loaded: {config['project']['name']}")
    
    setup_logging()
    print("✓ Logging initialized")
    
    ensure_directories(config)
    print("✓ Directories created")
    
    print("\n✅ All utilities working correctly!")
