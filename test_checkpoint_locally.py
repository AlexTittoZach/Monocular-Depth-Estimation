import torch
import cv2
import numpy as np

from src.models.depth_anything import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE = "path_to_000385.png"  #enter image path

CHECKPOINT = "finetuned/depth_anything_v2_vitl_lora.pth"

MODE = "lora"


model, _ = load_model(
    mode=MODE,
    encoder_type="vitl",
    checkpoint_path=CHECKPOINT,
)

for name, param in model.named_parameters():
    if "depth_head" in name:
        print(name)
        print(param.mean().item())
        print(param.std().item())
        break

model.to(DEVICE)
model.eval()

img = cv2.imread(IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (504, 378))

img = torch.from_numpy(img).float().permute(2,0,1)/255.

mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

img = ((img-mean)/std).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(img)

pred = pred.squeeze().cpu().numpy()
depth = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
depth = (depth * 255).astype(np.uint8)

cv2.imwrite("lora.png", depth)

print("min :", pred.min())
print("max :", pred.max())
print("mean:", pred.mean())
print("std :", pred.std())