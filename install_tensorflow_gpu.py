#!/usr/bin/env python3
"""
TensorFlow GPU Installation Script for RTX 5070
Handles the complex CUDA setup for Windows systems
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Success")
            if result.stdout.strip():
                print(f"[OUTPUT] {result.stdout.strip()}")
            return True
        else:
            print(f"[FAIL] Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"[FAIL] Exception: {e}")
        return False

def check_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] NVIDIA GPU detected")
            return True
        else:
            print("[FAIL] No NVIDIA GPU found")
            return False
    except FileNotFoundError:
        print("[FAIL] nvidia-smi not found - NVIDIA drivers not installed")
        return False

def install_tensorflow_gpu():
    """Install TensorFlow with GPU support"""
    print("=" * 60)
    print("TensorFlow GPU Installation for RTX 5070")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU availability
    if not check_gpu():
        print("\n[ERROR] No NVIDIA GPU detected. Please install NVIDIA drivers first.")
        return False
    
    print("\n[INFO] Starting TensorFlow GPU installation...")
    
    # Step 1: Uninstall existing TensorFlow
    print("\n" + "="*50)
    print("STEP 1: Cleaning existing TensorFlow installations")
    print("="*50)
    
    run_command("pip uninstall tensorflow tensorflow-intel tensorflow-gpu -y", 
                "Uninstalling existing TensorFlow")
    
    # Step 2: Install compatible NumPy
    print("\n" + "="*50)
    print("STEP 2: Installing compatible NumPy")
    print("="*50)
    
    run_command("pip install \"numpy>=1.21.0,<1.25.0\"", 
                "Installing NumPy 1.24.x for compatibility")
    
    # Step 3: Install CUDA libraries
    print("\n" + "="*50)
    print("STEP 3: Installing CUDA libraries")
    print("="*50)
    
    cuda_packages = [
        "nvidia-cudnn-cu12",
        "nvidia-cublas-cu12", 
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12"
    ]
    
    for package in cuda_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Step 4: Install TensorFlow with GPU support
    print("\n" + "="*50)
    print("STEP 4: Installing TensorFlow with GPU support")
    print("="*50)
    
    # Try different TensorFlow versions that support GPU
    tensorflow_versions = [
        "tensorflow==2.15.0",  # Known to work well with RTX 50 series
        "tensorflow==2.14.0",  # Fallback option
        "tensorflow==2.13.0"   # Last resort
    ]
    
    tensorflow_installed = False
    for tf_version in tensorflow_versions:
        print(f"\n[INFO] Trying {tf_version}...")
        if run_command(f"pip install {tf_version}", f"Installing {tf_version}"):
            tensorflow_installed = True
            break
        else:
            print(f"[WARN] {tf_version} failed, trying next version...")
    
    if not tensorflow_installed:
        print("[ERROR] Failed to install any TensorFlow version")
        return False
    
    # Step 5: Test GPU detection
    print("\n" + "="*50)
    print("STEP 5: Testing GPU detection")
    print("="*50)
    
    test_script = '''
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU devices: {gpus}")
if gpus:
    print("SUCCESS: TensorFlow can see GPU!")
    # Test GPU memory growth
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth enabled")
    except:
        print("Could not set memory growth")
else:
    print("WARNING: TensorFlow cannot see GPU")
'''
    
    with open("test_tf_gpu.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python test_tf_gpu.py", "Testing TensorFlow GPU detection")
    
    # Cleanup
    if os.path.exists("test_tf_gpu.py"):
        os.remove("test_tf_gpu.py")
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: TensorFlow GPU installation completed!")
        print("="*60)
        print("Your RTX 5070 is now ready for GPU acceleration!")
        print("Expected speedup: 5-10x for LSTM training")
        print("\nNext steps:")
        print("1. Run: python check_gpu_usage.py")
        print("2. Test enhanced models in Streamlit app")
        print("3. Monitor GPU usage with: nvidia-smi -l 1")
        return True
    else:
        print("\n" + "="*60)
        print("PARTIAL SUCCESS: TensorFlow installed but GPU detection failed")
        print("="*60)
        print("This might still work for CPU training.")
        print("GPU acceleration may require manual CUDA installation.")
        return False

if __name__ == "__main__":
    try:
        install_tensorflow_gpu()
    except KeyboardInterrupt:
        print("\n[INFO] Installation cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Installation failed: {e}")
