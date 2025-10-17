# EnhancedResConvLSTM-CA

## Overview

This repository contains a PyTorch implementation of an urban expansion simulation framework that combines residual ConvLSTM units with an Efficient Channel–Position Attention (ECPA) mechanism and classic residual convolutional blocks. The code reads multi-temporal land cover rasters and spatial growth factors (e.g., CBD distance, road proximity, population density, slope, restricted areas), trains a model to predict new urban conversions, evaluates results with standard and change-specific metrics, and supports multi-year forward simulation while preserving categorical land-use classes.

The main script (`main.py` or the project Python file) includes data preparation, model building, training, evaluation, advanced analyses (variable importance, attention evolution, neighborhood effects, transition pattern analysis), and export of GeoTIFF prediction maps.

## Key Features

* Residual ConvLSTM architecture (stacked ResConvLSTM units) for spatio-temporal modeling.
* ECPA attention blocks with extraction and analysis of channel and spatial attention weights.
* Training, validation and patch-based prediction workflow (non-overlapping patches by default).
* Land-use–aware prediction that preserves categorical classes (urban, vegetation, water, paddy) and applies conversion constraints.
* Evaluation metrics: accuracy, F1-score, IoU, Figure of Merit (FoM), Allocation Disagreement (AD), Quantity Disagreement (QD), counts of hits/misses/false alarms.
* Experiment utilities: variable importance (correlation + ablation), attention evolution visualization, spatial autocorrelation (simplified Moran's I), cluster and edge analyses.
* GeoTIFF export of predicted maps with categorical integer datatype.

## Repository Structure (recommended)

```
README.md
requirements.txt
main.py                       # Your main script (the code you provided)
DataColombo/                  # Example data folder (GIS rasters)
  2015_cleaned.tif
  2020_cleaned.tif
  2025_cleaned.tif
  CBD_cleaned (1).tif
  road_cleaned.tif
  pop_cleaned.tif
  slope_cleaned.tif
  restricted_cleaned.tif
outputs/                       # Saved models, PNGs and GeoTIFF outputs
final_model.pth
variable_importance.png
attention_distributions.png
training_history.png
predicted_2025_accuracy_check.tif
predicted_2030.tif
predicted_2035.tif
```

Adjust folder and file names to match your environment.

## Requirements and Installation

**Recommended environment:** Miniconda/Anaconda (easiest for GDAL and GPU-enabled PyTorch).

1. Create a conda environment (example):

```bash
conda create -n urban-exp python=3.10 -y
conda activate urban-exp
```

2. Install GDAL (use conda-forge for binary compatibility):

```bash
conda install -c conda-forge gdal -y
```

3. Install PyTorch (choose the right CUDA version from [https://pytorch.org](https://pytorch.org)):

```bash
# CPU-only (example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# OR GPU (example for CUDA 12.1) -- replace with appropriate command from PyTorch site
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install the remaining Python packages:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas rasterio scikit-image
# If you prefer gdal python bindings instead of rasterio keep the conda-installed gdal
pip install tqdm
```

5. Optional: create a `requirements.txt` with pinned versions for reproducibility:

```
numpy
scipy
scikit-learn
matplotlib
seaborn
pandas
rasterio
torch
torchvision
tqdm
```

> **Note:** Installing `gdal` via `pip` is often problematic; prefer `conda install -c conda-forge gdal`.

## Data preparation

1. Place your multi-temporal land cover rasters and factor rasters inside a data folder (e.g., `DataColombo/`).
2. Filenames in the script are set to:

   * `2015_cleaned.tif`, `2020_cleaned.tif`, `2025_cleaned.tif` (optional 2025 used for validation)
   * Factor rasters: `CBD_cleaned (1).tif`, `road_cleaned.tif`, `pop_cleaned.tif`, `slope_cleaned.tif`, `restricted_cleaned.tif`
3. Land cover class convention used in the code (change if needed):

   * `1` = Urban
   * `2` = Vegetation
   * `3` = Water
   * `4` = Paddy lands

If your class codes differ, update the mapping logic in `predict_next_year()` and any evaluation masks accordingly.

## How to run (example)

1. Edit file paths and parameters at the bottom of `main.py` (or the script file). Confirm `patch_size` setting matches your raster dimensions (non-overlapping grid is used: `rows // patch_size` must be >= 1).

2. Run training + experiments:

```bash
python main.py
```

This will:

* Load data, build the model and train for the number of epochs set in the script.
* Save `final_model.pth` and PNG visualizations into the working directory.
* Export prediction GeoTIFFs (predicted 2025, 2030, 2035) in the working folder.

### Quick tips

* If memory/GPU is tight, reduce `batch_size` or `patch_size` in the script.
* If no patches are created, increase the number of negative sample ratio or reduce `patch_size`.

## Configuration options (what to edit in code)

* `patch_size` (default `64`) — spatial patch dimension for training and prediction.
* `epochs` and `batch_size` in `train(epochs=..., batch_size=...)`.
* File paths for land cover and factor rasters at the bottom of `main.py`.
* Class mapping inside `predict_next_year()` if your dataset uses different labels.
* Threshold for urban expansion `> 0.5` — consider tuning to maximize IoU/FoM.

## Model details

The provided model (`EnhancedResConvLSTMAttentionModel`) stacks several `ResConvLSTMUnit` blocks with channel reductions (128 → 64 → 32 → 1), interleaves ECPA attention blocks and residual convolutional blocks, and finishes with a `1x1` convolution + sigmoid to return per-pixel urban probability maps. Attention weights are stored in the ECPA blocks for later analysis.

## Evaluation and metrics

Evaluation is performed using:

* Pixel-wise metrics: Accuracy, F1-score, IoU (Jaccard).
* Change-specific metrics: Figure of Merit (FoM), Quantity Disagreement (QD), Allocation Disagreement (AD), hits/misses/false alarms.

The evaluation excludes water (`class == 3`) from change-area calculations by default. Adjust the `eval_mask` logic in `evaluate()` if a different exclusion rule is needed.

## Advanced experiments included

* **Variable importance:** correlation with model predictions and ablation (zeroing factors) with plots saved as `variable_importance.png`.
* **Attention analysis:** saves attention distribution histograms and computes entropy/stability over time.
* **Neighborhood effects:** simple Moran's I proxy, cluster analysis (connected components), edge vs interior density analysis with plots saved as `neighborhood_analysis.png`.
* **Transition pattern analysis:** transition probabilities grouped by original land cover type and spatial consistency checks.

## Outputs and where to find them

* Model weights: `final_model.pth`
* Visualizations: `training_history.png`, `variable_importance.png`, `attention_distributions.png`, `neighborhood_analysis.png`
* GeoTIFFs: `predicted_2025_accuracy_check.tif`, `predicted_2030.tif`, `predicted_2035.tif`

All outputs are saved to the working directory unless you customize the paths in the script.

## Troubleshooting

* **GDAL errors while reading rasters:** make sure GDAL is installed (use conda-forge) and that the raster files are not corrupted. Use `gdalinfo <file>` to inspect.
* **No patches created:** check `patch_size` and raster size alignment (`rows % patch_size` and `cols % patch_size`). Make sure target transitions exist (change pixels present); otherwise relax the negative sampling condition.
* **Out of memory (OOM):** reduce `batch_size`, `patch_size`, or run on CPU for debugging by forcing `device = torch.device('cpu')`.
* **Low IoU / poor predictions:** tune threshold, inspect class imbalance, add class weighting or focal loss, augment training samples, or add more informative growth factors.

## How to upload this project to GitHub (basic steps)

1. Initialize a git repo (if not already):

```bash
git init
git add .
git commit -m "Initial commit: EnhancedResConvLSTM-CA implementation"
```

2. Create a new repository on GitHub (via web UI) and then push:

```bash
git remote add origin git@github.com:<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

Or use HTTPS URL if you prefer:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Citation and credit

If you adapt or use this code in research, please cite this repository and any related publications you produce. Consider including an appropriate citation (BibTeX) section here once you have a paper or DOI.

## License

This repository is provided under the MIT License by default. Add `LICENSE` file with the MIT text if you want permissive reuse. Replace with another license if you prefer.

## Contact

For questions, bug reports, or improvements, open an issue in the repository or contact the author (add your email or ORCID here).

---

*Created from the user's DeepLearning CA script. Edit the README to include project-specific details (author name, repository URL, precise dependencies and versions).*
