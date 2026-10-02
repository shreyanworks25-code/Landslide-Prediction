# Landslide Prediction Using Multi-Temporal Sentinel-1 and Sentinel-2 Data

## Overview

This project investigates **deep-learning-based landslide segmentation using multi-temporal and multimodal satellite imagery**.

The primary objective is to combine information from **Sentinel-1 Synthetic Aperture Radar (SAR)** and **Sentinel-2 multispectral imagery** and investigate whether temporal information can improve landslide detection compared with a conventional spatial segmentation approach.

The current implementation focuses on the **Chimanimani, Zimbabwe** dataset and establishes an end-to-end pipeline covering:

* Sentinel-1 and Sentinel-2 data inspection
* Spatial matching of Sentinel-1 and Sentinel-2 tiles
* Multimodal feature construction
* Temporal data fusion
* Dataset validation
* Baseline U-Net experimentation
* U-TAE-based temporal segmentation
* Training, validation, and test evaluation

The project is currently an **experimental research implementation**. The first U-TAE experiment demonstrates that the complete pipeline works, but the obtained test Dice score indicates that further experimentation and stabilization are required before drawing conclusions about the effectiveness of temporal fusion.

---

## Research Objective

The main research question investigated by this project is:

> **Can temporally-aware fusion of Sentinel-1 SAR and Sentinel-2 multispectral information improve automated landslide segmentation?**

The project combines complementary information:

* **Sentinel-2** provides optical/multispectral information.
* **Sentinel-1** provides SAR information that can be useful independently of optical illumination and cloud conditions.
* **Temporal observations** provide information about changes occurring before and after a landslide event.
* **U-TAE (U-TAE)** is used to model temporal information while performing spatial segmentation.

The goal is not simply to classify an entire tile as landslide/non-landslide, but to produce a **pixel-level landslide segmentation mask**.

---

# Dataset

## Study Region

The current experiment uses data from:

**Chimanimani, Zimbabwe**

The provided dataset contains harmonized Sentinel-1 and Sentinel-2 NetCDF (`.nc`) tiles.

### Sentinel-1

The Sentinel-1 dataset contains:

```text
1000 S1 tiles
```

Each tile contains:

```text
15 temporal observations
128 × 128 spatial resolution
```

Relevant variables include:

```text
VH
VV
MASK
DEM
```

The Sentinel-1 data used in the fused model are represented by:

```text
VV
VH
```

### Sentinel-2

The Sentinel-2 dataset contains:

```text
500 S2 tiles
```

Each tile contains:

```text
15 temporal observations
128 × 128 spatial resolution
```

The Sentinel-2 dataset contains several spectral bands and additional variables, including:

```text
B02
B03
B04
B05
B06
B07
B08
B8A
B11
B12
SCL
MASK
DEM
```

For the current multimodal experiment, three optical channels were selected:

```text
R = B04
G = B03
B = B02
```

These were combined with Sentinel-1:

```text
VV
VH
```

resulting in a five-channel input:

```text
[R, G, B, VV, VH]
```

---

# Sentinel-1 / Sentinel-2 Spatial Matching

The Sentinel-1 and Sentinel-2 tiles were matched using their spatial coordinates.

Initial datasets:

```text
S1 tiles: 1000
S2 tiles: 500
```

Spatial matching produced:

```text
Exact spatial matches: 500
```

Therefore, the current fused experiment uses:

```text
500 matched S1-S2 tile pairs
```

This ensures that Sentinel-1 and Sentinel-2 observations correspond to the same spatial locations before multimodal fusion.

Example matched pair:

```text
S1: chimanimani_s1asc_108.nc
S2: chimanimani_s2_108.nc

Coordinates:
(460160.0, 7834240.0)
```

---

# Temporal Fusion

After spatial matching, the corresponding Sentinel-1 and Sentinel-2 observations were combined temporally.

The resulting input tensor is:

```text
X shape:

(500, 15, 128, 128, 5)
```

where:

```text
500  = number of matched samples
15   = temporal observations
128  = image height
128  = image width
5    = input channels
```

The five channels are:

```text
Channel 1 → R
Channel 2 → G
Channel 3 → B
Channel 4 → VV
Channel 5 → VH
```

The segmentation target is:

```text
y shape:

(500, 128, 128, 1)
```

The target represents the pixel-level landslide mask.

---

# Dataset Statistics

The final temporal fused dataset contains:

```text
X:
(500, 15, 128, 128, 5)

y:
(500, 128, 128, 1)
```

Data types:

```text
X dtype: float32
y dtype: float32
```

Input value range:

```text
X min:  0.0
X max:  1.0
X mean: 0.32460114
```

Channel statistics:

```text
R:
min  = 0.0003
max  = 1.0000
mean = 0.2075

G:
min  = 0.0003
max  = 1.0000
mean = 0.2108

B:
min  = 0.0003
max  = 1.0000
mean = 0.1390

VV:
min  = 0.0000
max  = 1.0000
mean = 0.5221

VH:
min  = 0.0000
max  = 1.0000
mean = 0.5436
```

The segmentation masks contain relatively few positive pixels:

```text
Mask mean: 0.014275025
```

Samples containing at least one landslide pixel:

```text
260 / 500
```

Percentage:

```text
52.0%
```

This indicates that the dataset contains a substantial number of samples without landslide pixels, while landslide pixels themselves occupy only a small fraction of the image area.

---

# Train/Test Split

The 500 matched samples were divided into:

```text
Training samples: 400
Testing samples:  100
```

Therefore:

```text
80% training
20% testing
```

Training indices:

```text
(400,)
```

Testing indices:

```text
(100,)
```

The data pipeline was also tested using batches.

Training batches:

```text
100
```

Testing batches:

```text
25
```

Batch size:

```text
4
```

Example batch:

```text
X batch:
(4, 15, 128, 128, 5)

y batch:
(4, 128, 128, 1)
```

---

# Models

## 1. Baseline U-Net

A conventional convolutional U-Net was implemented as a spatial segmentation baseline.

The purpose of the baseline is to provide a reference point for evaluating whether adding temporal modelling provides an advantage.

The baseline follows an encoder-decoder segmentation architecture with skip connections.

---

# 2. U-TAE

The project then experiments with a **U-TAE-style temporal segmentation architecture**.

The model receives:

```text
Input:
(15, 128, 128, 5)
```

and produces:

```text
Output:
(128, 128, 1)
```

The implemented model contains convolutional encoder and decoder components together with skip connections.

The model used in the experiment contained approximately:

```text
118,561 trainable parameters
```

Model size:

```text
~463 KB
```

The model was trained for a maximum of:

```text
50 epochs
```

with early stopping and learning-rate reduction based on validation performance.

---

# Loss Function and Metric

The primary evaluation metric is the **Dice coefficient**, which measures overlap between the predicted landslide mask and the ground-truth mask.

A higher Dice coefficient indicates greater overlap between prediction and ground truth.

The training process monitored:

```text
dice_coef
val_dice_coef
loss
val_loss
```

The best model checkpoint was saved as:

```text
best_utae.keras
```

---

# First U-TAE Training Experiment

The first training experiment showed that the model was learning from the training data, but validation performance was unstable.

Important observations included:

```text
Epoch 1:
Training Dice ≈ 0.4374
Validation Dice ≈ 0.3233

Epoch 5:
Training Dice ≈ 0.4383
Validation Dice ≈ 0.3216

Epoch 9:
Training Dice ≈ 0.4157
Validation Dice ≈ 0.3168
```

The learning rate was reduced during training:

```text
2.5e-05
↓
1.25e-05
↓
6.25e-06
```

Early stopping eventually terminated the experiment.

The best validation model was from:

```text
Epoch 1
```

with:

```text
Best validation Dice ≈ 0.32325
```

This indicates that the initial U-TAE experiment was not yet sufficiently stable or optimized.

---

# Final U-TAE Test Result

The best saved U-TAE model was loaded and evaluated on the held-out test set.

Final result:

```text
Test Loss:
0.6926839351654053

Test Dice:
0.3588499426841736
```

Therefore:

```text
Final Test Dice ≈ 0.3589
```

This result should be considered a **baseline experimental result**, rather than a final optimized model performance.

---

# Prediction Analysis

A prediction sanity check was also performed.

Prediction tensor:

```text
(4, 128, 128, 1)
```

Prediction range:

```text
Minimum: 1.7085706e-09
Maximum: 0.99776554
Mean:    0.0050579645
```

Ground-truth mean:

```text
0.005004883
```

Predicted positive pixel percentage:

```text
0.43640137%
```

Actual positive pixel percentage:

```text
0.5004883%
```

The predicted and actual positive-pixel proportions are relatively close for the checked batch, but the overall Dice score remains moderate.

---

# Interpretation of the Current Result

The current test Dice score is:

```text
0.35885
```

This shows that the current pipeline is producing meaningful segmentation predictions, but the model is not yet sufficiently accurate for the project to claim that temporal multimodal fusion improves landslide segmentation.

Several factors may contribute to the current result.

## 1. Small Dataset

The current experiment contains only:

```text
500 matched samples
```

with:

```text
400 training samples
100 testing samples
```

For a temporal deep-learning segmentation model, this is relatively limited.

The model therefore has relatively little training data from which to learn robust spatial and temporal patterns.

---

## 2. Class Imbalance

The average mask value is:

```text
0.014275025
```

which indicates that landslide pixels occupy only a small portion of the images.

This creates a strong background-versus-landslide imbalance.

A segmentation model can therefore achieve reasonable pixel-level behaviour while still producing imperfect landslide boundaries.

---

## 3. Temporal Alignment

Each sample contains:

```text
15 temporal observations
```

However, the observations across different tiles are not necessarily identical in calendar timing.

For example, the metadata shows different acquisition dates and different pre/post-event configurations.

Therefore, the temporal sequence may not represent exactly the same physical time intervals across all samples.

This is an important issue to investigate before drawing strong conclusions about the temporal model.

---

## 4. Hyperparameter Optimization

The current U-TAE experiment represents an initial training configuration.

The training history shows that validation Dice remained around the low-to-mid 0.3 range and did not consistently improve after the first epoch.

This suggests that the training configuration requires further investigation.

Potential experiments include:

```text
Learning-rate tuning
Optimizer tuning
Batch-size experiments
Loss-function experiments
Class-imbalance handling
Data augmentation
Regularization
Temporal sampling
Model architecture tuning
```

---

# Current Research Status

The current implementation has successfully completed the following stages:

```text
[✓] Sentinel-1 data inspection

[✓] Sentinel-2 data inspection

[✓] Spatial coordinate extraction

[✓] Sentinel-1/Sentinel-2 matching

[✓] 500 exact spatial matches

[✓] Multimodal feature construction

[✓] Temporal sequence construction

[✓] Input normalization

[✓] Ground-truth mask preparation

[✓] Train/test split

[✓] Batch pipeline validation

[✓] Baseline U-Net experiment

[✓] U-TAE implementation

[✓] U-TAE training

[✓] Model checkpointing

[✓] Test evaluation

[✓] Prediction sanity check
```

The following stages remain:

```text
[ ] Training-stability investigation

[ ] Temporal alignment analysis

[ ] Hyperparameter optimization

[ ] Improved class-imbalance handling

[ ] Additional augmentation experiments

[ ] More systematic baseline comparison

[ ] Experiments on additional geographic regions

[ ] Final comparison of spatial vs temporal approaches

[ ] Final ablation studies

[ ] Final research conclusions
```

---

# Planned Improvements

## 1. Stabilize Training

The first priority is to investigate training stability.

Experiments will include different learning rates and training configurations.

For example:

```text
1e-4
5e-5
2.5e-5
1e-5
```

The objective is to determine whether the validation Dice can improve consistently rather than peaking at the beginning of training.

---

## 2. Investigate Temporal Alignment

The 15 temporal observations should be analyzed to determine:

* acquisition dates
* temporal spacing
* pre-event observations
* post-event observations
* event-date consistency
* seasonal differences

This is important because a temporal model assumes that the temporal sequence contains meaningful information.

---

## 3. Improve Class-Imbalance Handling

Because landslide pixels represent a small fraction of each image, alternative loss functions can be investigated.

Possible experiments include:

```text
Dice Loss
Binary Cross-Entropy
BCE + Dice Loss
Focal Loss
Tversky Loss
Focal Tversky Loss
```

These experiments should be evaluated systematically rather than changing multiple components simultaneously.

---

## 4. Data Augmentation

Spatial augmentation can potentially increase the effective size of the training dataset.

Possible transformations include:

```text
Horizontal flip
Vertical flip
90° rotation
Random rotation
Small spatial transformations
```

Any augmentation applied to the input must also be applied consistently to the segmentation mask.

---

## 5. Additional Regions

The current experiment focuses on Chimanimani.

The next stage is to investigate whether the same pipeline generalizes to other regions available in the dataset, such as:

```text
Italy
Dominica
Other available regions
```

This is important because performance on a single geographic region does not establish generalization.

---

## 6. Ablation Studies

A useful research extension is to compare different input configurations.

For example:

```text
Experiment A:
Sentinel-2 only

Experiment B:
Sentinel-1 only

Experiment C:
Sentinel-1 + Sentinel-2

Experiment D:
Sentinel-1 + Sentinel-2 + temporal information
```

These experiments can help determine which information contributes to the segmentation performance.

---

# Reproducibility

The repository contains the main notebooks used during development.

```text
notebooks/
│
├── 01_data_inspection.ipynb
├── 02_s1_s2_matching.ipynb
├── 03_temporal_fusion.ipynb
├── 04_baseline_unet.ipynb
└── 05_utae_training.ipynb
```

The notebooks represent the main experimental workflow:

```text
01
↓
Inspect Sentinel-1/Sentinel-2 datasets

02
↓
Find spatially corresponding S1/S2 tiles

03
↓
Construct the temporal multimodal dataset

04
↓
Train/evaluate a baseline U-Net

05
↓
Train/evaluate U-TAE
```

---

# Data and Model Files

Large generated datasets and model files are intentionally **not stored in this GitHub repository**.

The generated temporal dataset includes approximately:

```text
X size ≈ 2.29 GB
y size ≈ 0.031 GB
```

The project therefore uses `.gitignore` rules to prevent large data and model artifacts from being accidentally committed.

Ignored file types include:

```text
*.npy
*.npz
*.nc
*.keras
*.h5
*.pt
*.pth
```

The dataset generation process is documented through the notebooks.

---

# Project Structure

```text
Landslide-Prediction/
│
├── README.md
├── .gitignore
│
└── notebooks/
    ├── 01_data_inspection.ipynb
    ├── 02_s1_s2_matching.ipynb
    ├── 03_temporal_fusion.ipynb
    ├── 04_baseline_unet.ipynb
    └── 05_utae_training.ipynb
```

---

# Current Results Summary

| Component                               | Current Result |
| --------------------------------------- | -------------: |
| Sentinel-1 tiles                        |           1000 |
| Sentinel-2 tiles                        |            500 |
| Exact S1/S2 matches                     |            500 |
| Temporal observations/sample            |             15 |
| Image size                              |      128 × 128 |
| Input channels                          |              5 |
| Total samples                           |            500 |
| Training samples                        |            400 |
| Testing samples                         |            100 |
| Samples containing landslide pixels     |      260 / 500 |
| U-TAE parameters                        |        118,561 |
| Best validation Dice in first U-TAE run |       ≈ 0.3233 |
| Final U-TAE test Dice                   |      ≈ 0.35885 |
| Final U-TAE test loss                   |      ≈ 0.69268 |

---

# Conclusion

The project has established a complete experimental pipeline for **multimodal, multi-temporal landslide segmentation using Sentinel-1 and Sentinel-2 data**.

The current implementation successfully performs:

```text
Satellite data inspection
        ↓
Spatial S1/S2 matching
        ↓
Multimodal feature extraction
        ↓
Temporal fusion
        ↓
Segmentation dataset creation
        ↓
Baseline modelling
        ↓
U-TAE training
        ↓
Test evaluation
```

The initial U-TAE experiment achieved a test Dice coefficient of approximately:

```text
0.35885
```

However, this result should not yet be interpreted as evidence that temporal fusion improves performance. The current experiment is best viewed as a **working baseline for further research**.

The next phase will focus on improving training stability, validating temporal alignment, addressing class imbalance, tuning the model, and conducting controlled comparisons and ablation experiments.

The ultimate objective is to determine, through reproducible experiments, whether combining **spatial, spectral, SAR, and temporal information** can provide measurable improvements for automated landslide segmentation.

---

## Status

**Project stage:** Experimental / Under Development

**Current focus:** U-TAE training stabilization and multimodal-temporal evaluation

**Study region:** Chimanimani, Zimbabwe

**Primary task:** Pixel-level landslide segmentation

**Modalities:** Sentinel-1 SAR + Sentinel-2 multispectral

**Temporal observations:** 15 per sample
