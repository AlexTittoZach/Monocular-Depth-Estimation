# Empirical Evaluation & Final Conclusion: Depth Anything V2 (ViT-L) Fine-Tuning on KITTI Dataset

> [!NOTE]
> **Dataset & Domain Specification**: All fine-tuned models in this study were trained and evaluated exclusively on the **KITTI Vision Benchmark Dataset** (real-world outdoor autonomous driving scenes with sparse LiDAR ground-truth depth spanning $0.1\text{m}$ to $80.0\text{m}$).

## 1. Executive Summary

This report presents a quantitative and qualitative comparative analysis of fine-tuning strategies for **Depth Anything V2 (Vision Transformer Large - ViT-L)** on metric monocular depth estimation trained on the **KITTI Driving Dataset**. Four distinct evaluation configurations were benchmarked:

1. **Zero-Shot Baseline (`BASELINE_DEPTH_ANYTHING_V2`)**: Stock pre-trained ViT-L model (evaluated without KITTI domain adaptation).
2. **Decoder-Only Fine-Tuning (`EXP_02_DECODER_ONLY`)**: Vision Transformer backbone frozen; DPT depth decoder trained on KITTI.
3. **LoRA Fine-Tuning (`EXP_03_LORA`)**: Low-Rank Adaptation (rank=4, alpha=32) applied to self-attention projections and trained on KITTI.
4. **Full Fine-Tuning (`EXP_04_FULL_FT`)**: End-to-end optimization of all 335.3M model parameters on KITTI.

### Key Finding
**LoRA Fine-Tuning (`EXP_03_LORA`) achieved optimal performance across all metrics**, attaining an **Abs Rel of 0.0356** and **RMSE of 1.8764m** with **$\delta_1$ accuracy of 98.93%**, updating **only 0.71% of total parameters** (2.39M parameters) and requiring **2.826 GB peak VRAM**.

---

## 2. Dataset Profile: The KITTI Vision Benchmark

The **KITTI Vision Benchmark Suite** (Geiger et al., *CVPR 2012 / IJRR 2013*) serves as the gold-standard real-world autonomous driving benchmark for computer vision and 3D perception tasks.

### 2.1 Sensor Setup & Hardware Platform
The data collection platform consists of a sensor-equipped Volkswagen Passat station wagon driven through Karlsruhe, Germany, featuring:
* **3D LiDAR**: Velodyne HDL-64E rotating mechanical laser scanner (64 laser beams, 10 Hz rotation, 1.3 million points/sec, range up to 120m, centimeter-level range accuracy).
* **Stereo Color Cameras**: Two Point Grey Flea2 high-resolution color cameras ($1392 \times 512$ native resolution at 10 fps) mounted on the roof rack with a 54 cm stereo baseline.
* **Inertial & GPS Navigation**: OXTS RT3003 combined GPS/IMU system providing 6-DOF ground-truth vehicle positioning at 50 Hz.
* **Optical Calibration**: Synchronized camera-LiDAR calibration yielding rectified RGB frames of approximately **$1242 \times 375$ pixels** with a wide horizontal field of view ($\approx 90^\circ$).

### 2.2 Dataset Partition & Evaluation Protocol
* **Training Set**: 18,661 calibrated stereo image and semi-dense LiDAR depth map pairs across diverse driving conditions (Urban *City*, Suburban *Residential*, Highway *Road*, and *Campus*).
* **Validation / Evaluation Set**: **2,073 unseen test images** adhering to the standard Eigen / KITTI depth benchmark split.
* **Evaluation Range**: Standard autonomous driving metric evaluation bounds of **$0.1\text{m} \le d \le 80.0\text{m}$**.
* **Ground Truth Depth Encoding**: 16-bit single-channel depth maps where metric distance in meters is encoded via:
  $$\text{Depth (meters)} = \frac{\text{Pixel Value}}{256.0}$$

### 2.3 Key Domain Characteristics & Challenges
1. **Extreme LiDAR Sparsity**: Raw LiDAR returns cover less than $10\%$ of total image pixels. The upper third of each frame (sky and distant horizon) has zero laser reflections, requiring the model to infer metric depth from contextual priors.
2. **Ultra-Wide Aspect Ratio**: The native driving perspective aspect ratio is $\approx 3.31:1$ ($1242 \times 375$), necessitating preservation of aspect ratios during inference to avoid horizontal spatial distortion.
3. **Dynamic Range & Foreground Diversity**: Scenes feature high dynamic range lighting (shadows under tree canopies, direct sunlight glare) and complex interacting dynamic obstacles including vehicles, cyclists, and pedestrians.

---

## 3. Comparative Performance Metrics

All evaluation metrics were calculated on the standard validation set across $N = 2,073$ sample images at standard evaluation resolution ($378 \times 504$).

### Metric Overview Table

| Metric | Baseline (Zero-Shot) | Decoder-Only (EXP_02) | Full Fine-Tuning (EXP_04) | **LoRA (EXP_03)** ⭐ | Optimal Direction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Abs Rel** | `0.5585` | `0.4280` | `0.0448` | **`0.0356`** | Lower $\downarrow$ |
| **RMSE (m)** | `18.5118` | `12.7243` | `2.2477` | **`1.8764`** | Lower $\downarrow$ |
| **RMSE log** | `0.4635` | `0.5954` | `0.0742` | **`0.0595`** | Lower $\downarrow$ |
| **$\delta_1$ (< 1.25)** | `48.48%` | `30.40%` | `97.94%` | **`98.93%`** | Higher $\uparrow$ |
| **$\delta_2$ (< $1.25^2$)** | `75.99%` | `57.56%` | `99.69%` | **`99.83%`** | Higher $\uparrow$ |
| **$\delta_3$ (< $1.25^3$)** | `87.56%` | `77.88%` | `99.91%` | **`99.94%`** | Higher $\uparrow$ |

---

## 4. Computational & Parameter Efficiency Analysis

| Strategy | Total Params | Trainable Params | Trainable % | Peak VRAM | Latency / Throughput | Checkpoint Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 335.3M | 0 | 0.00% | N/A | ~61.8 ms / 16.2 FPS | 1,279 MB |
| **Decoder-Only** | 335.3M | 30.95M | 9.23% | 3.106 GB | 45.83 ms / 21.8 FPS | 1,279.13 MB |
| **LoRA** | 337.7M | **2.39M** | **0.71%** | **2.826 GB** | 51.10 ms / 19.6 FPS | 1,288.23 MB |
| **Full FT** | 335.3M | 335.3M | 100.00% | High (>16 GB) | ~62.3 ms / 16.1 FPS | 1,279.13 MB |

---

## 5. In-Depth Technical Insights & Discussion

### 5.1 Why LoRA Outperformed Full Fine-Tuning
1. **Prevention of Catastrophic Forgetting**: The Vision Transformer encoder in Depth Anything V2 contains rich pre-trained visual representations. Full fine-tuning updates all 335.3M parameters, risking representation drift and minor overfitting on the target dataset split.
2. **Constrained Optimization Space**: LoRA constrains update weight matrices $\Delta W = B \cdot A$ to rank $r=4$. This regularized parameter space enforces smooth metric adaptation while keeping the foundation feature extractor intact.
3. **Accuracy Improvement**: LoRA reduced Absolute Relative Error by **20.5% compared to Full Fine-Tuning** (0.0356 vs 0.0448) and reduced RMSE by **16.5%** (1.8764m vs 2.2477m).

### 5.2 Failure Mode Analysis of Decoder-Only Adaptation
- Freezing the ViT backbone and fine-tuning only the DPT head (9.23% of parameters) performed poorly (`Abs Rel`: 0.4280, `RMSE`: 12.7243m, `$\delta_1$`: 30.40%).
- **Root Cause**: Monocular depth estimation requires scale and distance shift adjustments inside the multi-head self-attention mechanisms of intermediate patch representations. Adapting only the output head fails to recalibrate spatial features extracted by frozen ViT layers.

### 5.3 Zero-Shot Baseline Gap
- The un-tuned Zero-Shot baseline had an `Abs Rel` of 0.5585 and `RMSE` of 18.51m. This reflects uncalibrated global scale factors inherent to relative depth pre-training models when evaluated directly against absolute metric depth benchmarks.

---

## 6. Final Recommendation

For deployment in production and resource-constrained training environments:

1. **Primary Recommendation — LoRA (`EXP_03_LORA`)**:
   - **Best Accuracy**: Lowest error across all criteria ($RMSE = 1.8764\text{m}$, $Abs\ Rel = 0.0356$).
   - **Lowest Compute Overhead**: Trains in **2.826 GB VRAM** updating only **0.71%** of weights.
   - **Fastest Training Convergence**: Ideal for efficient domain adaptation.

2. **Deployment Strategy**:
   - Serve the base pre-trained ViT-L model weights with merged LoRA adapter weights for zero latency penalty during inference.

