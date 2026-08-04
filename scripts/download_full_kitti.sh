#!/bin/bash
set -e

DATA_DIR="datasets/kitti"
TRAIN_DIR="${DATA_DIR}/train"
VAL_DIR="${DATA_DIR}/val"

echo "======================================================="
echo "  FULL KITTI DATASET SETUP (~14 GB Annotated Set)      "
echo "======================================================="

# 1. Clean existing training directory to prevent any mix-up
echo "[1/4] Clearing existing train directory..."
rm -rf "${TRAIN_DIR}"
mkdir -p "${TRAIN_DIR}/images" "${TRAIN_DIR}/depths"

# 2. Download official KITTI annotated depth benchmark zip (~14 GB)
ZIP_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
ZIP_FILE="${DATA_DIR}/data_depth_annotated.zip"

if [ ! -f "${ZIP_FILE}" ]; then
    echo "[2/4] Downloading official KITTI annotated dataset (~14 GB)..."
    wget -c --show-progress -O "${ZIP_FILE}" "${ZIP_URL}"
else
    echo "[2/4] Found existing archive at ${ZIP_FILE}. Skipping redownload."
fi

# 3. Extract and structure pairs cleanly
echo "[3/4] Extracting archive..."
TEMP_EXTRACT="${DATA_DIR}/temp_annotated"
mkdir -p "${TEMP_EXTRACT}"
unzip -q -o "${ZIP_FILE}" -d "${TEMP_EXTRACT}"

echo "[3/4] Organizing training image and depth pairs..."
python3 -c "
import os, glob, shutil

temp_dir = '${TEMP_EXTRACT}'
train_img_dir = '${TRAIN_DIR}/images'
train_depth_dir = '${TRAIN_DIR}/depths'

# Find all extracted groundtruth depth files
depth_files = glob.glob(os.path.join(temp_dir, 'train', '**/proj_depth/groundtruth/image_02/*.png'), recursive=True) + \
              glob.glob(os.path.join(temp_dir, 'train', '**/proj_depth/groundtruth/image_03/*.png'), recursive=True)

print(f'Found {len(depth_files)} annotated LiDAR ground truth depth maps.')

count = 0
for depth_path in depth_files:
    # Match corresponding RGB image path in train structure
    # KITTI format: train/date_drive/proj_depth/groundtruth/image_02/frame.png -> RGB image at date_drive/image_02/data/frame.png
    parts = depth_path.split(os.sep)
    filename = parts[-1]
    cam_folder = parts[-2] # image_02 or image_03
    drive_folder = parts[-5] # 2011_09_26_drive_0001_sync

    rgb_candidate = os.path.join(temp_dir, 'train', drive_folder, cam_folder, 'data', filename)
    
    if os.path.exists(rgb_candidate):
        unique_name = f'{drive_folder}_{cam_folder}_{filename}'
        shutil.copy(rgb_candidate, os.path.join(train_img_dir, unique_name))
        shutil.copy(depth_path, os.path.join(train_depth_dir, unique_name))
        count += 1

print(f'Successfully paired and organized {count} training pairs into {train_img_dir}')
"

# Clean temporary extraction folder
rm -rf "${TEMP_EXTRACT}"

# 4. Verify Zero Overlap between train and val
echo "[4/4] Verifying zero data leakage between train and val..."
python3 -c "
import glob, os
train_files = set(os.path.basename(f) for f in glob.glob('${TRAIN_DIR}/images/*.png'))
val_files = set(os.path.basename(f) for f in glob.glob('${VAL_DIR}/images/*.png'))

overlap = train_files & val_files
print(f'Total Train Images: {len(train_files)}')
print(f'Total Val Images  : {len(val_files)}')
print(f'Overlap Count     : {len(overlap)}')
assert len(overlap) == 0, 'Data leakage detected!'
print('✅ Zero-overlap assertion passed!')
"

echo "======================================================="
echo "  FULL KITTI DATASET SETUP COMPLETED SUCCESSFULLY!     "
echo "======================================================="
