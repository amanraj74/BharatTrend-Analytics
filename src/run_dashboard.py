#!/usr/bin/env python
"""
Helper script to launch BharatTrend dashboard
"""
import os
import sys
import subprocess

def main():
    """Launch the Streamlit dashboard"""
    app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
    
    print("🚀 Launching BharatTrend Dashboard...")
    print(f"📍 Running: streamlit run {app_path}")
    
    try:
        subprocess.run(['streamlit', 'run', app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
