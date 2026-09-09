# Landslide Prediction — Phase 1: Tabular Machine Learning

A machine learning project for predicting landslide occurrence using environmental and geological features.

This is **Phase 1** of a larger disaster-management project and focuses on binary landslide classification using tabular data.

---

## Project Overview

Landslides can be influenced by several environmental and geological conditions such as rainfall, slope, soil saturation, vegetation cover, earthquake activity, and proximity to water.

The objective of this project is to build and evaluate machine learning classification models that predict whether a landslide occurs based on these features.

The project includes:

- Data quality analysis
- Exploratory analysis
- Correlation analysis
- Train-test splitting
- Multiple machine learning models
- Cross-validation
- Model evaluation
- Feature importance analysis
- Model stability testing
- Feature ablation analysis

---

## Dataset

The dataset contains:

- **2,000 samples**
- **9 input features**
- **1 binary target variable**

### Input Features

| Feature | Description |
|---|---|
| `Rainfall_mm` | Rainfall measurement in millimeters |
| `Slope_Angle` | Slope angle of the terrain |
| `Soil_Saturation` | Soil saturation level |
| `Vegetation_Cover` | Vegetation coverage |
| `Earthquake_Activity` | Earthquake activity level |
| `Proximity_to_Water` | Proximity to a water source |
| `Soil_Type_Gravel` | Gravel soil indicator |
| `Soil_Type_Sand` | Sand soil indicator |
| `Soil_Type_Silt` | Silt soil indicator |

### Target

`Landslide`

- `0` — No Landslide
- `1` — Landslide

The target dataset is balanced:

- 1,000 No Landslide samples
- 1,000 Landslide samples

---

## Data Quality

The dataset was checked for common data-quality issues.

- Missing values: **0**
- Duplicate rows: **0**
- Duplicate feature rows: **0**
- Target distribution: **50% / 50%**
- No outliers were detected in the continuous features examined.

---

## Train-Test Split

The dataset was divided using an **80:20 stratified train-test split**.

- Training samples: **1,600**
- Testing samples: **400**
- Training features: **9**
- Testing features: **9**

Stratification was used to maintain the same class distribution in both training and testing sets.

---

## Machine Learning Models

The following classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. Gaussian Naive Bayes

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- 5-Fold Stratified Cross-Validation

---

## Model Performance

| Model | Test Accuracy | Mean CV Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.0000 | 1.0000 |
| KNN | 0.9925 | 0.9925 | 0.995 | 0.990 | 0.9925 | 0.9999 |
| SVM | 0.9750 | 0.9660 | 0.975 | 0.975 | 0.9750 | 0.9984 |

Several models achieved perfect performance on this dataset.

---

## Selected Model

**Random Forest** was selected as the primary model for Phase 1.

Although several models achieved the same perfect predictive performance, Random Forest was selected because it:

- Achieved 100% test accuracy
- Achieved 100% mean cross-validation accuracy
- Provides feature-importance analysis
- Can capture nonlinear relationships between features
- Is suitable for further experimentation

Logistic Regression and Decision Tree were also retained as useful baseline models.

---

## Feature Importance

Random Forest feature-importance analysis indicated that several features were particularly influential, including:

- `Slope_Angle`
- `Soil_Saturation`
- `Vegetation_Cover`
- `Earthquake_Activity`
- `Proximity_to_Water`

Other features, including rainfall and individual soil-type indicators, had comparatively lower importance.

---

## Model Stability

Additional experiments were performed to check whether the results depended heavily on a particular random split.

### Random Seed Testing

A Decision Tree was tested with random seeds from **0 to 4**.

All tested seeds achieved:

**Accuracy = 1.0000**

### Training Size Testing

Different training-set sizes were also tested:

- 10%
- 30%
- 50%

All tested configurations achieved:

**Accuracy = 1.0000**

These results indicate that the observed performance was stable on this dataset.

---

## Feature Ablation

A drop-one-feature experiment was performed using Random Forest with 5-fold Stratified Cross-Validation.

Each feature was removed individually and the model was retrained.

For every tested feature:

- Mean CV Accuracy = **1.0000**
- CV Standard Deviation = **0.0000**

This indicates that the dataset contains substantial redundancy and strong class separability.

---

## Important Observation

The models achieved unusually high performance, with several models reaching 100% accuracy.

This is largely because the dataset is **synthetically generated** and the two classes are highly separable.

Therefore, the results should **not** be interpreted as 100% real-world landslide prediction accuracy.

Real-world landslide data can contain:

- Measurement noise
- Missing observations
- Geographical variation
- Temporal variation
- Overlapping environmental conditions
- Sensor errors
- More complex relationships between environmental variables

The current project should therefore be considered a **Phase 1 proof of concept**.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure

```text
Landslide-Prediction/
│
├── landslide_prediction.ipynb
├── README.md
└── dataset/
   