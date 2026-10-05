import sys
import tensorflow as tf
import sklearn
import pandas
import numpy
import platform

print("="*40)
print("Environment Verification")
print("="*40)
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"NumPy: {numpy.__version__}")
print("-" * 40)

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"✅ GPU Detected: {len(gpu_devices)} device(s)")
    for device in gpu_devices:
        print(f"   - {device}")
else:
    print("⚠️ No GPU Detected (Running on CPU)")
    print("   Note: This is fine for most introductory chapters.")

print("="*40)
print("✅ Setup looks good!")
