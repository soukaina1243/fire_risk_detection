# Fire Risk Prediction Using Satellite Time Series

Deep learning project predicting wildfire risk in California by fusing three satellite data sources through a temporal LSTM model.

---

## Overview

This project frames wildfire risk as a binary classification problem: given the last 6 months of environmental conditions for a geographic region, will a fire occur next month?

The full pipeline runs on Google Earth Engine (no local data download) and is trained using TensorFlow/Keras on Google Colab.

---

## Results

| Metric | Value |
|---|---|
| AUC-ROC | 0.845 |
| Recall (fires) | 0.25 |
| Precision (fires) | 0.17 |
| F1-score | 0.20 |
| Gain vs best baseline | +0.177 AUC |

The model significantly outperforms all naive baselines (random, always-zero, seasonal rule), confirming that a real temporal signal was learned.

---

## Data Sources

All data is accessed directly through Google Earth Engine, no manual download required.

| Source | GEE Collection | Variable | Resolution | Frequency |
|---|---|---|---|---|
| NASA FIRMS | `FIRMS` | Active fires (T21 > 350K) | 375m | Daily |
| MODIS NDVI | `MODIS/061/MOD13A2` | Vegetation index | 1km | 16 days |
| ERA5 Land | `ECMWF/ERA5_LAND/DAILY_AGGR` | Temperature, precipitation, wind | 9km | Daily |

---

## Project Structure

```
fire-risk-prediction/
│
├── data/
│   ├── fire_risk_dataset_raw.csv        # Raw extracted dataset (69,900 rows)
│   ├── fire_risk_regional.csv           # Aggregated regional dataset (1x1 deg)
│   ├── fire_risk_2019.csv               # Annual extractions
│   ├── fire_risk_2020.csv
│   ├── fire_risk_2021.csv
│   ├── fire_risk_2022.csv
│   ├── fire_risk_2023.csv
│   ├── meta_train_reg.csv               # Train set metadata (lat, lon, year, month)
│   └── meta_test_reg.csv                # Test set metadata
│
├── numpy_arrays/
│   ├── X_train_reg.npy                  # Train sequences (N, 6, 11)
│   ├── y_train_reg.npy                  # Train labels
│   ├── X_val_reg.npy                    # Validation sequences
│   ├── y_val_reg.npy
│   ├── X_test_reg.npy                   # Test sequences
│   └── y_test_reg.npy
│
├── models/
│   ├── lstm_fire_risk_final.keras       # Trained LSTM model
│   ├── model_config_final.json          # Hyperparameters, threshold, feature list
│   ├── scaler_regional.pkl              # Fitted StandardScaler (train only)
│   └── training_log_regional.csv        # Loss and metrics per epoch
│
├── figures/
│   ├── fig1_performance.png             # ROC curve, Precision-Recall, confusion matrix
│   ├── fig2_temporal_analysis.png       # Seasonal error analysis
│   ├── fig3_spatial_analysis.png        # Spatial distribution of errors
│   ├── fig4_feature_importance.png      # Permutation importance
│   └── fig5_fn_zones_history.png        # NDVI and temperature history of missed fires
│
├── reports/
│   └── error_analysis_report_final.md   # Full error analysis report
│
├── notebook/
│   └── fire_risk_prediction.ipynb       # Main Colab notebook (all phases)
│
├── .gitignore
└── README.md
```

---

## Methodology

### Phase 1 - Data Collection

A regular 0.25 x 0.25 degree grid (1165 points) is created server-side in GEE covering California. Each source is extracted independently using `sampleRegions()` to avoid mask propagation, then merged in pandas using lat/lon as keys. FIRMS is initialized to 0 everywhere and updated only where fires are detected.

### Phase 2 - Preprocessing

Five engineered features are added to capture temporal trends:

- `ndvi_anomaly`: deviation from seasonal mean NDVI
- `temp_roll3`: 3-month rolling average temperature
- `precip_roll3`: 3-month cumulative precipitation
- `ndvi_roll3`: 3-month rolling NDVI trend
- `hydric_deficit`: max(0, potential evaporation - precipitation)

A strict chronological split is applied to prevent temporal leakage:

- Train: 2019, 2020, 2021
- Validation: 2022
- Test: 2023

Normalization (StandardScaler) is fitted on the train set only and applied to all splits.

Sequences of length 6 are created per geographic point using a sliding window, yielding shapes of (34950, 6, 11) for training.

### Phase 3 - LSTM Model

```
Input      (6, 11)
LSTM 64    return_sequences=True
Dropout    0.3
LSTM 32    return_sequences=False
Dropout    0.3
Dense 16   relu
Dense  1   sigmoid  ->  P(fire) in [0,1]
```

Three strategies address the extreme class imbalance (99.9% negative):

1. Focal Loss (gamma=2.0, alpha=0.75) penalizes missed fires heavily
2. Temporal augmentation x10: positive sequences duplicated with Gaussian noise (sigma=0.05)
3. Regional aggregation to 1x1 degree cells multiplies positives by approximately 15

### Phase 4 - Evaluation

The decision threshold is optimized on the validation set (not the test set) to prevent leakage. Permutation importance reveals that `ndvi_anomaly` is the most critical feature, which is physically coherent: vegetation that is drier than usual for the season is the strongest fire precursor.

---

## Key Design Decisions

**Why extract sources separately?**
FIRMS only has values at fire pixels. Using `addBands()` in GEE propagates the FIRMS mask to the combined image, returning only 3-4 points instead of 1165. Separate extraction with a default-zero FIRMS base solves this.

**Why chronological split?**
A random split would allow future data to contaminate training, making evaluation invalid. Fire risk prediction is a strictly causal problem.

**Why AUC and not accuracy?**
With 99.9% negative class, a model predicting always-zero reaches 99.9% accuracy but detects nothing. AUC measures the model's ability to rank fire regions above non-fire regions, regardless of threshold.

**Why Focal Loss?**
Standard binary cross-entropy does not penalize missed fires sufficiently when they represent 0.1% of data. Focal Loss down-weights easy negatives and forces the model to focus on hard positive examples.

---

## Reproducing the Project

### Requirements

```
earthengine-api
geemap
tensorflow>=2.12
scikit-learn
pandas
numpy
matplotlib
scipy
```

### Setup

1. Create a Google Earth Engine account at https://earthengine.google.com
2. Open the notebook in Google Colab
3. Run the restoration cell at the top to reload all files from GitHub or Drive

```python
import os, shutil

GITHUB_USERNAME = "your_username"
GITHUB_TOKEN    = "your_token"
GITHUB_REPO     = "fire-risk-prediction"

if not os.path.exists('/content/fire_risk_dataset_raw.csv'):
    os.system(f'git clone https://{GITHUB_USERNAME}:{GITHUB_TOKEN}'
              f'@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git')
    for subdir in ['data','models','figures','reports','numpy_arrays']:
        path = f'/content/{GITHUB_REPO}/{subdir}'
        if os.path.exists(path):
            for f in os.listdir(path):
                shutil.copy2(f'{path}/{f}', f'/content/{f}')
    print("Restored successfully")
```

### Saving Progress

Run at the end of every session to push all files to GitHub:

```python
push_to_github("description of what you did")
```

---

## Limitations

- Only 80 positive examples over 5 years forces heavy reliance on synthetic augmentation
- Monthly temporal resolution cannot capture sudden fires lasting only a few days
- 1x1 degree spatial resolution loses fine geographic precision
- `pot_evap_mm` and `ndvi_roll3` show negative permutation importance and should be removed in future iterations

---

## Deliverables

| Deliverable | File | Description |
|---|---|---|
| Data fusion pipeline | `data/` + notebook Phase 1 | Full GEE extraction and merge logic |
| Temporal modeling justification | Section 3 of report + notebook Phase 3 | LSTM choice, architecture, training |
| Error analysis report | `reports/error_analysis_report_final.md` | FN/FP analysis, feature importance, threshold sensitivity |

---

## Academic Context

Project 3 — Fire Risk Prediction Using Satellite Time Series
Deep Learning course — 2024/2025
