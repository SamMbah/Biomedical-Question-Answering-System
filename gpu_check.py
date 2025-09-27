import torch
print("CUDA available? ", torch.cuda.is_available())
print("Torch CUDA ver:", torch.version.cuda)
print("GPU name:     ", torch.cuda.get_device_name(0))
