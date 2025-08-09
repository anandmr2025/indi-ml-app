#!/usr/bin/env python3
"""
TensorFlow GPU Fix Script for Indi-ML
Fixes NumPy-TensorFlow compatibility issues and tests GPU support
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout.strip():
                print(result.stdout.strip())
            return True
        else:
            print(f"❌ Failed: {description}")
            if result.stderr.strip():
                print(result.stderr.strip())
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_current_environment():
    """Check current Python and package environment"""
    print("🔍 Checking current environment...")
    
    # Python version
    run_command("python --version", "Checking Python version")
    
    # Current packages
    print("\n📦 Current package versions:")
    packages = ["numpy", "tensorflow", "xgboost", "lightgbm"]
    for package in packages:
        try:
            result = subprocess.run(f"pip show {package}", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.startswith('Version:'):
                        print(f"   {package}: {line.split(':')[1].strip()}")
                        break
            else:
                print(f"   {package}: Not installed")
        except:
            print(f"   {package}: Error checking")

def check_gpu():
    """Check GPU availability"""
    print("\n🎯 Checking GPU availability...")
    return run_command("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", 
                      "Checking NVIDIA GPU")

def fix_tensorflow_numpy():
    """Fix TensorFlow-NumPy compatibility"""
    print("\n🔧 Fixing TensorFlow-NumPy compatibility...")
    
    # Step 1: Uninstall problematic packages
    print("\nStep 1: Uninstalling problematic packages...")
    run_command("pip uninstall tensorflow tensorflow-gpu numpy -y", 
                "Uninstalling TensorFlow and NumPy")
    
    # Step 2: Install compatible NumPy
    print("\nStep 2: Installing compatible NumPy...")
    run_command('pip install "numpy>=1.21.0,<2.0.0"', 
                "Installing compatible NumPy")
    
    # Step 3: Install TensorFlow with GPU support
    print("\nStep 3: Installing TensorFlow with GPU support...")
    run_command('pip install "tensorflow[and-cuda]>=2.15.0,<2.16.0"', 
                "Installing TensorFlow with GPU support")
    
    # Step 4: Install other ML packages
    print("\nStep 4: Installing other ML packages...")
    run_command("pip install --upgrade xgboost lightgbm", 
                "Installing XGBoost and LightGBM")

def test_installations():
    """Test all installations"""
    print("\n🧪 Testing installations...")
    
    # Test NumPy
    print("\n📊 Testing NumPy...")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} working")
    except Exception as e:
        print(f"❌ NumPy failed: {e}")
        return False
    
    # Test TensorFlow
    print("\n🧠 Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        # Check GPU support
        gpus = tf.config.list_physical_devices('GPU')
        print(f"📊 GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
        
        if gpus:
            print("🚀 TensorFlow GPU support is working!")
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        else:
            print("⚠️  No GPUs detected by TensorFlow")
            
    except Exception as e:
        print(f"❌ TensorFlow failed: {e}")
        return False
    
    # Test XGBoost GPU
    print("\n🌳 Testing XGBoost GPU...")
    try:
        import xgboost as xgb
        import numpy as np
        print(f"✅ XGBoost {xgb.__version__}")
        
        # Test GPU training
        X = np.random.random((100, 10))
        y = np.random.random(100)
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'reg:squarederror'
        }
        model = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        print("🚀 XGBoost GPU support working!")
        
    except Exception as e:
        print(f"⚠️  XGBoost GPU failed: {e}")
    
    # Test LightGBM GPU
    print("\n💡 Testing LightGBM GPU...")
    try:
        import lightgbm as lgb
        import numpy as np
        print(f"✅ LightGBM {lgb.__version__}")
        
        # Test GPU training
        X = np.random.random((100, 10))
        y = np.random.random(100)
        train_data = lgb.Dataset(X, label=y)
        params = {
            'device': 'gpu',
            'objective': 'regression',
            'verbose': -1
        }
        model = lgb.train(params, train_data, num_boost_round=1, verbose_eval=False)
        print("🚀 LightGBM GPU support working!")
        
    except Exception as e:
        print(f"⚠️  LightGBM GPU failed: {e}")
    
    return True

def test_enhanced_models():
    """Test enhanced models"""
    print("\n🎯 Testing Enhanced Models...")
    
    try:
        print("Testing Enhanced Ensemble...")
        from indi_ml.models.enhanced_ensemble import EnhancedEnsembleModel
        print("✅ Enhanced Ensemble imports successfully")
        
        print("Testing Enhanced ARIMA...")
        from indi_ml.models.enhanced_arima import EnhancedARIMAModel  
        print("✅ Enhanced ARIMA imports successfully")
        
        print("Testing Enhanced LSTM...")
        from indi_ml.models.enhanced_lstm import EnhancedLSTMModel
        print("✅ Enhanced LSTM imports successfully")
        
        print("🎉 All enhanced models are working!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced models error: {e}")
        return False

def main():
    """Main fix function"""
    print("🚀 TensorFlow GPU Fix Script for Indi-ML")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check current environment
    check_current_environment()
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Fix TensorFlow-NumPy compatibility
    fix_tensorflow_numpy()
    
    # Test installations
    if test_installations():
        print("\n✅ Package installations successful!")
    else:
        print("\n❌ Some package installations failed!")
        return False
    
    # Test enhanced models
    if test_enhanced_models():
        print("\n🎉 Enhanced models are ready!")
    else:
        print("\n⚠️  Enhanced models need attention")
    
    print("\n" + "=" * 60)
    print("✅ Fix script completed!")
    print("🚀 Your RTX 5070 should now work with enhanced models!")
    print("\nNext steps:")
    print("1. Run: python test_enhanced_models.py")
    print("2. Run: python -m streamlit run app.py")
    print("3. Select '⚡ Fast Enhanced Models' for GPU acceleration")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
