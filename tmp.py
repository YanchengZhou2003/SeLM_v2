import torch
print("torch version:", torch.__version__)
print("compiled with CUDA:", torch.version.cuda)
print("is CUDA available:", torch.cuda.is_available())
print("current device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current CUDA device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))