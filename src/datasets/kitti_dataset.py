import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional

class KITTIDepthDataset(Dataset):
    """
    KITTI Monocular Depth Estimation Dataset.
    
    Supports:
    1. Real KITTI dataset structure:
       - image: RGB PNG image
       - depth: 16-bit PNG depth map (depth_in_meters = png_val / 256.0)
    2. Automatic synthetic sample generation if dataset directory is empty,
       allowing immediate baseline benchmarking and code testing.
    """
    def __init__(self, data_dir: str = "datasets/kitti", split: str = "val", target_size: Tuple[int, int] = (378, 504), transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.target_size = target_size
        self.transform = transform
        
        self.image_paths = []
        self.depth_paths = []
        
        if os.path.exists(self.data_dir):
            self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, "image_02", "*.png"))) + \
                               sorted(glob.glob(os.path.join(self.data_dir, "images", "*.jpg"))) + \
                               sorted(glob.glob(os.path.join(self.data_dir, "images", "*.png")))
            self.depth_paths = sorted(glob.glob(os.path.join(self.data_dir, "groundtruth_depth", "*.png"))) + \
                               sorted(glob.glob(os.path.join(self.data_dir, "depths", "*.png")))
                               
        if len(self.image_paths) == 0:
            print(f"[Dataset] No images found in {self.data_dir}. Generating 20 synthetic KITTI evaluation samples...")
            self._create_synthetic_samples()

    def _create_synthetic_samples(self):
        img_dir = os.path.join(self.data_dir, "images")
        depth_dir = os.path.join(self.data_dir, "depths")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        np.random.seed(42)
        for i in range(20):
            # Generate synthetic road scene
            h, w = 375, 1242
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            # Sky
            rgb[:h//2, :] = [220, 180, 130] # BGR
            # Road
            rgb[h//2:, :] = [80, 80, 80]
            # Add synthetic objects (cars / obstacles)
            cv2.rectangle(rgb, (500, 180), (700, 300), (0, 0, 200), -1)
            
            # Synthetic depth map (linear depth gradient from top 80m to bottom 2m)
            depth_map = np.zeros((h, w), dtype=np.float32)
            for r in range(h):
                depth_map[r, :] = 2.0 + (h - r) / h * 78.0
            depth_map[180:300, 500:700] = 15.0 # Car depth
            
            # Save synthetic ground truth in 16-bit uint PNG (scale by 256)
            depth_png = (depth_map * 256.0).astype(np.uint16)
            
            img_path = os.path.join(img_dir, f"sample_{i:04d}.png")
            depth_path = os.path.join(depth_dir, f"sample_{i:04d}.png")
            
            cv2.imwrite(img_path, rgb)
            cv2.imwrite(depth_path, depth_png)
            
            self.image_paths.append(img_path)
            self.depth_paths.append(depth_path)
            
        print(f"[Dataset] Created {len(self.image_paths)} synthetic samples at {self.data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[idx]
        
        # Load RGB Image
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # Load Ground Truth Depth Map (16-bit uint PNG in KITTI format)
        if idx < len(self.depth_paths):
            depth_path = self.depth_paths[idx]
            depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_png is not None:
                depth = depth_png.astype(np.float32) / 256.0
            else:
                depth = np.zeros((raw_img.shape[0], raw_img.shape[1]), dtype=np.float32)
        else:
            depth = np.zeros((raw_img.shape[0], raw_img.shape[1]), dtype=np.float32)

        # Resize for model forward pass
        img_resized = cv2.resize(raw_img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_CUBIC)
        depth_resized = cv2.resize(depth, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        # Normalize RGB to [0, 1] tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        # ImageNet Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        depth_tensor = torch.from_numpy(depth_resized).float()

        return img_tensor, depth_tensor, os.path.basename(img_path)
