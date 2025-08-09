"""
GPU Setup and Acceleration Script for Indi-ML

This script helps set up GPU acceleration for the ML models.
Run this to enable GPU support for TensorFlow, XGBoost, and LightGBM.
"""

import subprocess
import sys
import os

def install_gpu_packages():
    """Install GPU-enabled packages"""
    
    print("🚀 Setting up GPU acceleration for Indi-ML...")
    
    # Install TensorFlow with GPU support
    print("\n📦 Installing TensorFlow GPU...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]"], check=True)
    
    # Install XGBoost with GPU support
    print("\n📦 Installing XGBoost GPU...")
    subprocess.run([sys.executable, "-m", "pip", "install", "xgboost[gpu]"], check=True)
    
    # Install CuPy for GPU-accelerated NumPy operations
    print("\n📦 Installing CuPy...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "cupy-cuda12x"], check=True)
    except:
        print("⚠️  CuPy installation failed - continuing without it")
    
    print("\n✅ GPU packages installed successfully!")
    print("\n🔄 Please restart your Python environment to use GPU acceleration.")

def check_gpu_status():
    """Check current GPU status"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} devices")
        print(f"CUDA Built: {tf.test.is_built_with_cuda()}")
        
        if tf.config.list_physical_devices('GPU'):
            for gpu in tf.config.list_physical_devices('GPU'):
                print(f"GPU Device: {gpu}")
        else:
            print("❌ No GPU devices found")
            
    except ImportError:
        print("❌ TensorFlow not installed")

if __name__ == "__main__":
    print("Current GPU Status:")
    check_gpu_status()
    
    response = input("\nDo you want to install GPU packages? (y/n): ")
    if response.lower() == 'y':
        install_gpu_packages()
    else:
        print("Skipping GPU package installation")
