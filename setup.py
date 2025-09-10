#!/usr/bin/env python3
"""Setup script for the time series prediction project."""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['results', 'data', 'logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main setup function."""
    print("Time Series Prediction Project Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if install_requirements():
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run a quick demo: python demo_improvements.py")
        print("2. Run specific prediction: python src/main.py --dataset stock --model both")
        print("3. Run all experiments: python run_experiments.py")
        print("4. Open Jupyter notebook: jupyter notebook notebooks/time_series_analysis.ipynb")
    else:
        print("\n✗ Setup failed. Please install requirements manually.")

if __name__ == "__main__":
    main()