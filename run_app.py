#!/usr/bin/env python3
"""
PDR Decision System Launcher

This script launches the main PDR Decision System application.
"""

import subprocess
import sys
import os

def main():
    """Launch the PDR Decision System."""
    print("🚀 Starting PDR Decision System...")
    
    # Check if streamlit is available
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Get the path to the main app
    app_path = os.path.join("app", "main_app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ Main app not found at {app_path}")
        sys.exit(1)
    
    # Launch the app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 PDR Decision System stopped.")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 