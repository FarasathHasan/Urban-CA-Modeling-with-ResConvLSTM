# Deep Learning Cellular Automata for Urban Expansion Prediction

A PyTorch-based implementation of a Deep Learning Cellular Automata (DLCA) model that combines ResConvLSTM units with Efficient Channel Position Attention (ECPA) mechanisms to predict urban expansion patterns. This project is designed for spatio-temporal land use/land cover (LULC) change modeling and urban growth simulation.

## Overview

Urban expansion is a critical phenomenon affecting environmental sustainability, resource management, and urban planning. This project implements an advanced deep learning approach to:

- Predict urban expansion patterns based on spatial growth factors
- Learn transition rules from temporal LULC data
- Simulate future urban growth scenarios
- Analyze attention mechanisms and variable importance
- Evaluate predictions using land-change-specific metrics

The model architecture integrates:
- **ResConvLSTM**: Residual Convolutional LSTM units for spatio-temporal feature learning
- **ECPA Blocks**: Efficient Channel Position Attention for capturing important spatial and channel features
- **Residual Blocks**: Standard convolutional residual connections for improved gradient flow

## Features

### Core Functionality
- Multi-temporal land cover classification and change detection
- Patch-based deep learning approach for efficient processing of large raster data
- Support for 2-3 time periods of land cover data
- Flexible growth factor integration (CBD distance, road networks, population, slope, restricted areas, etc.)
- GPU acceleration support via PyTorch

### Advanced Analysis
- **Variable Importance Analysis**: Correlation-based and ablation-based assessment of growth factors
- **Attention Mechanism Analysis**: Distribution statistics and evolution tracking of attention weights
- **Neighborhood Effects Analysis**: Spatial autocorrelation, urban cluster patterns, and edge effects
- **Transition Pattern Analysis**: Land cover-specific transition probabilities and prediction confidence

### Evaluation Metrics
- Traditional metrics: Accuracy, F1 Score, IoU (Jaccard Index)
- Land-change-specific metrics:
  - Figure of Merit (FoM)
  - Allocation Disagreement (AD)
  - Quantity Disagreement (QD)
  - Hits, Misses, False Alarms

### Prediction & Simulation
- Year-by-year urban expansion prediction
- Multi-year future simulation with land use category preservation
- GeoTIFF export with proper spatial reference information

## Requirements

### System Requirements
- Python 3.7+
- CUDA 11.0+ (for GPU acceleration, optional)
- 8GB+ RAM (16GB+ recommended for large rasters)

### Python Dependencies

```
torch>=1.9.0
numpy>=1.19.0
gdal>=3.0.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.1.0
scipy>=1.5.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dlca-urban-expansion.git
cd dlca-urban-expansion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: Installing GDAL can be challenging. For easier installation, use conda:
```bash
conda create -n dlca python=3.9
conda activate dlca
conda install gdal torch scikit-learn matplotlib seaborn pandas scipy
```

## Data Format

### Input Data Requirements

The model requires:

1. **Land Cover Maps** (GeoTIFF format):
   - Multi-band or single-band raster files
   - Integer values representing land use categories
   - Same spatial extent and resolution
   - Recommended categories:
     - 1 = Urban/Built-up
     - 2 = Vegetation/Forest
     - 3 = Water
     - 4 = Paddy/Agricultural lands

2. **Growth Factors** (GeoTIFF format):
   - CBD Distance: Distance to Central Business District (continuous)
   - Road Network: Distance to nearest road (continuous)
   - Population: Population density or count (continuous)
   - Slope: Topographic slope in degrees (continuous)
   - Restricted Areas: Binary mask (0=unrestricted, 1=restricted)

### Example Data Structure
```
project/
├── DataColombo/
│   ├── 2015_cleaned.tif
│   ├── 2020_cleaned.tif
│   ├── 2025_cleaned.tif
│   ├── CBD_cleaned.tif
│   ├── road_cleaned.tif
│   ├── population_cleaned.tif
│   ├── slope_cleaned.tif
│   └── restricted_cleaned.tif
├── script.py
└── README.md
```

## Usage

### Basic Workflow

```python
import numpy as np
from script import (
    LandCoverData, GrowthFactors, DeepLearningCA,
    exportPredicted
)

# 1. Load land cover data for multiple time periods
landcover = LandCoverData(
    "DataColombo/2015_cleaned.tif",
    "DataColombo/2020_cleaned.tif",
    "DataColombo/2025_cleaned.tif"
)

# 2. Load growth factors
factors = GrowthFactors(
    "DataColombo/CBD_cleaned.tif",
    "DataColombo/road_cleaned.tif",
    "DataColombo/population_cleaned.tif",
    "DataColombo/slope_cleaned.tif",
    "DataColombo/restricted_cleaned.tif"
)

# 3. Initialize the model
model = DeepLearningCA(landcover, factors, patch_size=64)
model.build_model()

# 4. Train the model
history = model.train(epochs=100, batch_size=16)

# 5. Evaluate on validation year
accuracy, f1, iou, predicted_2025 = model.evaluate(landcover.arr_lc3)

# 6. Run advanced experiments
results = model.run_advanced_experiments(X_val, y_val)

# 7. Simulate future expansion
future_preds = model.simulate_future(landcover.arr_lc3, years=5)
exportPredicted(future_preds[5], 'predicted_2030.tif', landcover.ds_lc1)
```

### Run Complete Pipeline

Execute the main script:
```bash
python script.py
```

The script will:
- Load and validate data
- Prepare patches for training
- Train the model for 100 epochs
- Evaluate on 2025 data
- Run comprehensive experiments
- Generate future simulations (2025→2030, 2030→2035)
- Export results as GeoTIFF files
- Generate visualizations (PNG)

## Model Architecture

### EnhancedResConvLSTMAttentionModel

```
Input (channels, H, W)
    ↓
ResConvLSTM Block 1 (128 filters, 3 units)
    ↓
ResConvLSTM Block 2 (64 filters, 3 units)
    ↓
ECPA Block 1 (Channel + Spatial Attention)
    ↓
Residual Block (64 filters)
    ↓
ResConvLSTM Block 3 (32 filters, 1 unit)
    ↓
ECPA Block 2 (Channel + Spatial Attention)
    ↓
Residual Block (32 filters)
    ↓
ResConvLSTM Block 4 (1 filter)
    ↓
Final Conv + Sigmoid
    ↓
Output (0-1 probability of urban)
```

### Key Components

**ConvLSTM**: Convolutional LSTM cell combining convolutional operations with LSTM gating:
- Input convolution: projects input and applies gates
- Hidden convolution: processes hidden state
- Cell state update: I⊙g + f⊙c (input gate, candidate, forget gate)

**ECPA Block**: Dual attention mechanism:
- Channel Attention: Learns channel importance via adaptive pooling and FC layers
- Spatial Attention: Captures spatial patterns via mean/max pooling

**ResConvLSTMUnit**: Combines ConvLSTM with residual connections and batch normalization for improved training stability

## Output Files

The model generates the following outputs:

### Predictions
- `predicted_2025_accuracy_check.tif`: Predicted 2025 land cover for validation
- `predicted_2030.tif`: Simulated 2030 land cover
- `predicted_2035.tif`: Simulated 2035 land cover
- `final_model.pth`: Trained model weights

### Visualizations
- `training_history.png`: Training/validation loss and accuracy curves
- `variable_importance.png`: Growth factor importance analysis
- `attention_distributions.png`: Distribution of attention weights
- `neighborhood_analysis.png`: Spatial clustering and edge effects

### Console Output
- Land use distribution statistics
- Training epoch metrics
- Evaluation results with land-change-specific metrics
- Advanced experiment summaries

## Advanced Features

### Variable Importance Analysis
Identifies which growth factors most influence urban expansion predictions through:
- Correlation analysis between factors and predictions
- Ablation study (removing each factor and measuring performance drop)

### Attention Mechanism Analysis
Extracts and analyzes learned attention patterns:
- Channel attention statistics (mean, std, range, entropy)
- Spatial attention distributions
- Attention evolution during training

### Neighborhood Effect Analysis
Quantifies spatial patterns in predictions:
- Moran's I spatial autocorrelation
- Urban cluster size and density
- Edge vs. interior urban concentration

### Transition Pattern Analysis
Examines learned transition rules:
- Transition probabilities by original land cover type
- Spatial consistency of predictions
- Prediction confidence distribution

## Performance Considerations

### Memory Usage
- Large rasters (>3000x3000 pixels) with multiple factors may require chunking
- Adjust batch_size and patch_size based on available GPU memory
- Use GPU for training (10-50x faster than CPU)

### Training Time
- 100 epochs on typical datasets: 30-120 minutes (GPU)
- Factors affecting speed:
  - Number of patches (depends on patch_size and raster dimensions)
  - Image resolution
  - Hardware (GPU >> CPU)

### Optimization Tips
- Use patch_size=64 for balance between local context and memory efficiency
- Reduce batch_size if GPU runs out of memory
- Use ReduceLROnPlateau scheduler for adaptive learning rate
- Monitor validation loss to prevent overfitting

## Customization

### Modify Model Architecture
Edit `EnhancedResConvLSTMAttentionModel.__init__()`:
```python
# Add/remove layers
self.resconvlstm1 = ResConvLSTMUnit(input_channels, 256)  # More filters
self.dropout = nn.Dropout(0.5)  # Add dropout
```

### Change Training Parameters
```python
model.train(
    epochs=200,              # Increase training iterations
    batch_size=8             # Reduce batch size for small GPUs
)
```

### Adjust Patch Size
```python
model = DeepLearningCA(landcover, factors, patch_size=128)
# Larger patches: more context, less samples, slower training
# Smaller patches: less context, more samples, faster training
```

### Add Custom Growth Factors
Include additional factors in the GrowthFactors tuple and ensure they have the same spatial extent as land cover data.

## Troubleshooting

### GDAL Import Error
```bash
# Try conda installation
conda install gdal
# Or use pre-built wheels from Unofficial Windows Binaries
```

### GPU Memory Error
- Reduce batch_size (16 → 8 or 4)
- Reduce patch_size (64 → 32)
- Use CPU instead (slower but works)

### File Not Found Error
- Check file paths match exactly
- Ensure GeoTIFF files are properly formatted
- Verify all raster extents and resolutions match

### Poor Prediction Quality
- Ensure growth factors are meaningful for urban expansion
- Check data normalization (should be near zero mean, unit variance)
- Increase training epochs
- Try adjusting learning rate or using different optimizer
- Verify land cover categories are consistent across time periods

## Methodological Notes

### Land Use Preservation
The model predicts binary urban expansion but preserves original land use categories:
- Class 1 (Urban): Never reverts to other classes
- Class 2 (Vegetation) & 4 (Paddy): Can convert to Urban
- Class 3 (Water): Never changes (excluded from evaluation)

### Evaluation Strategy
- 80% training, 20% validation split
- Water pixels (class 3) excluded from evaluation metrics
- Land-change metrics specifically designed for transition models

### Validation Approach
If 2025 actual data available:
- Train on 2015→2020 and 2020→2025 transitions
- Evaluate predicted 2025 against actual 2025
- If no 2025 data, use cross-validation on 2015→2020

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: hasan.mohomadasath@connect.polyu.hk
- Include:
  - Detailed error message and traceback
  - Sample data or code to reproduce issue
  - System information (OS, Python version, GPU info)

## Acknowledgments

- PyTorch development team
- GDAL/OGR libraries
- ResearchGate community for methodological discussions

---

**Last Updated**: 2025
**Version**: 1.0.0
**Status**: Actively Maintained
