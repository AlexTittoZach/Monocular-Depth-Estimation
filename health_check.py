import sys
import torch
import os

print("[Health-Check] Python:", sys.version.split()[0])
print("[Health-Check] Torch:", torch.__version__)
print("[Health-Check] CUDA:", torch.version.cuda)
print("[Health-Check] CUDA-available:", torch.cuda.is_available())
print("[Health-Check] GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("[Health-Check] Accelerate config:", os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml"))
