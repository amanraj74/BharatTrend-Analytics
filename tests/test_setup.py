"""
Verify project setup is correct
Run this after initial setup to ensure everything works
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_structure():
    """Check all required directories exist"""
    required_dirs = [
        'src', 'data', 'configs', 'tests', 
        'models', 'logs', 'notebooks', 'docs'
    ]
    
    missing = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing.append(dir_name)
    
    if missing:
        print(f"✗ Missing directories: {missing}")
        return False
    
    print("✓ Project structure validated")
    return True

def test_imports():
    """Test critical package imports"""
    packages = {
        'pandas': 'Data processing',
        'transformers': 'NLP models',
        'prophet': 'Forecasting',
        'streamlit': 'Dashboard',
        'praw': 'Reddit API',
        'loguru': 'Logging'
    }
    
    failed = []
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✓ {package} ({description})")
        except ImportError as e:
            failed.append(package)
            print(f"✗ {package} - {e}")
    
    if failed:
        print(f"\n✗ Failed imports: {failed}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    try:
        from src.utils import load_config, setup_logging
        
        config = load_config()
        assert config['project']['name'] == 'BharatTrend', "Config name mismatch"
        print("✓ Configuration loaded correctly")
        
        logger = setup_logging()
        logger.info("Test log message")
        print("✓ Logging system working")
        
        return True
        
    except Exception as e:
        print(f"✗ Config/logging test failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    try:
        from src.utils import ensure_directories, get_project_root
        
        config = {'paths': {'test_dir': 'logs/test'}}
        ensure_directories(config)
        print("✓ Directory creation working")
        
        root = get_project_root()
        assert root.name == 'BharatTrend', "Project root incorrect"
        print("✓ Project root detection working")
        
        return True
        
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🔍 BharatTrend Setup Verification")
    print("="*60 + "\n")
    
    results = []
    
    print("1. Testing Project Structure...")
    results.append(test_structure())
    print()
    
    print("2. Testing Package Imports...")
    results.append(test_imports())
    print()
    
    print("3. Testing Configuration...")
    results.append(test_config())
    print()
    
    print("4. Testing Utilities...")
    results.append(test_utils())
    print()
    
    print("="*60)
    if all(results):
        print("🎉 ALL TESTS PASSED! Setup is complete.")
        print("✅ Ready to start building BharatTrend")
    else:
        print("⚠️  Some tests failed. Check errors above.")
    print("="*60)
