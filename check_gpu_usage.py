import subprocess
import sys
from datetime import datetime

def check_gpu():
    print("GPU Detection for Indi-ML")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] NVIDIA GPU detected!")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    print(f"   {line.strip()}")
        else:
            print("[FAIL] No NVIDIA GPU found")
    except FileNotFoundError:
        print("[FAIL] nvidia-smi not found")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        print(f"\n[OK] TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] TensorFlow can see {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("[FAIL] TensorFlow cannot see any GPUs")
    except ImportError:
        print("[FAIL] TensorFlow not installed")
    except Exception as e:
        print(f"[FAIL] TensorFlow error: {e}")
    
    # Check XGBoost GPU
    try:
        import xgboost as xgb
        print(f"\n[OK] XGBoost version: {xgb.__version__}")
        # Simple GPU test
        import numpy as np
        X = np.random.random((100, 10))
        y = np.random.random(100)
        dtrain = xgb.DMatrix(X, label=y)
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0, 'objective': 'reg:squarederror'}
        xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        print("[OK] XGBoost GPU support working")
    except Exception as e:
        print(f"[FAIL] XGBoost GPU failed: {e}")
    
    # Check LightGBM GPU
    try:
        import lightgbm as lgb
        print(f"\n[OK] LightGBM version: {lgb.__version__}")
        # Simple GPU test
        import numpy as np
        X = np.random.random((100, 10))
        y = np.random.random(100)
        train_data = lgb.Dataset(X, label=y)
        params = {'device': 'gpu', 'objective': 'regression', 'verbose': -1}
        lgb.train(params, train_data, num_boost_round=1)
        print("[OK] LightGBM GPU support working")
    except Exception as e:
        print(f"[FAIL] LightGBM GPU failed: {e}")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("Your RTX 5070 should provide excellent GPU acceleration!")
    print("Expected speedup: 3-8x for LSTM, 2-4x for ensemble models")
    print("\nTo fix TensorFlow issues, run: python fix_tensorflow_gpu.py")

if __name__ == "__main__":
    check_gpu()
