import os
import sys
import glob
import shutil
import zipfile
import subprocess
import urllib.request

DATA_DIR = "datasets/kitti"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
EXTRACT_TARGET = os.path.join(DATA_DIR, "extracted_annotated")

print("=======================================================")
print("  PAIRING 92,750 FULL KITTI ANNOTATED RGB & LIDAR SET  ")
print("=======================================================\n")

os.makedirs(os.path.join(TRAIN_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, "depths"), exist_ok=True)
os.makedirs(os.path.join(VAL_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(VAL_DIR, "depths"), exist_ok=True)

# 1. Find all 92,750 annotated LiDAR ground truth depth maps
depth_files = sorted(glob.glob(os.path.join(EXTRACT_TARGET, "**", "groundtruth", "**", "*.png"), recursive=True))
print(f"[1/3] Found {len(depth_files):,} annotated LiDAR ground truth depth maps!")

# 2. Extract unique drive dates & sequence names (e.g. 2011_09_26_drive_0001_sync)
drives = set()
for df in depth_files:
    parts = df.split(os.sep)
    for p in parts:
        if "drive_" in p and "_sync" in p:
            drives.add(p)

drives = sorted(list(drives))
print(f"[2/3] Identified {len(drives)} unique driving sequences across all {len(depth_files):,} frames.")

RAW_S3_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
raw_zips_dir = os.path.join(DATA_DIR, "raw_drives")
os.makedirs(raw_zips_dir, exist_ok=True)

print(f"[3/3] Downloading raw RGB camera drives and pairing frames...\n")

paired_count = 0
for drive_idx, drive_name in enumerate(drives, 1):
    date_str = drive_name[:10] # 2011_09_26
    date_drive_folder = drive_name.replace("_sync", "") # 2011_09_26_drive_0001
    
    zip_url = f"{RAW_S3_BASE}/{date_drive_folder}/{drive_name}.zip"
    local_zip = os.path.join(raw_zips_dir, f"{drive_name}.zip")
    extracted_drive = os.path.join(raw_zips_dir, date_str, drive_name)

    if not os.path.exists(extracted_drive):
        if not os.path.exists(local_zip):
            print(f"[{drive_idx}/{len(drives)}] Downloading RGB drive: {drive_name}...")
            try:
                subprocess.run(["wget", "-q", "-c", "-O", local_zip, zip_url], check=True)
            except Exception as e:
                print(f"   [Warning] Error downloading {zip_url}: {e}")
                continue

        if os.path.exists(local_zip):
            try:
                with zipfile.ZipFile(local_zip, 'r') as zr:
                    zr.extractall(raw_zips_dir)
                os.remove(local_zip)
            except Exception as e:
                print(f"   [Warning] Error extracting {local_zip}: {e}")

    # Now pair all depth maps belonging to this drive
    drive_depths = [d for d in depth_files if drive_name in d]
    drive_paired = 0

    for depth_path in drive_depths:
        parts = depth_path.split(os.sep)
        filename = parts[-1]
        cam = parts[-2] # image_02 or image_03
        
        rgb_path = os.path.join(raw_zips_dir, date_str, drive_name, cam, "data", filename)
        if os.path.exists(rgb_path):
            unique_name = f"{drive_name}_{cam}_{filename}"
            
            # Split: 90% train, 10% val
            if paired_count % 10 < 9:
                shutil.copy(rgb_path, os.path.join(TRAIN_DIR, "images", unique_name))
                shutil.copy(depth_path, os.path.join(TRAIN_DIR, "depths", unique_name))
            else:
                shutil.copy(rgb_path, os.path.join(VAL_DIR, "images", unique_name))
                shutil.copy(depth_path, os.path.join(VAL_DIR, "depths", unique_name))

            paired_count += 1
            drive_paired += 1

    print(f"   ✓ Paired {drive_paired:,} frames for {drive_name} (Total: {paired_count:,})")

print(f"\n=======================================================")
print(f"  SUCCESSFULLY PAIRED {paired_count:,} FULL DATASET FRAMES!")
print(f"=======================================================")
