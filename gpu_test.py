# gpu_test.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import tensorflow as tf

print("⮞ PyTorch CUDA available:", torch.cuda.is_available())
print("⮞ PyTorch CUDA version:  ", torch.version.cuda)
try:
    cuda_version = tf.sysconfig.get_build_info().get("cuda_version")
except Exception:
    cuda_version = "N/A"
print("⮞ TF build CUDA version:", cuda_version)
print("⮞ TF sees GPUs:       ", tf.config.list_physical_devices("GPU"))
