# P300 ERP-BCI: Brain-Computer Interface Speller

A complete Brain-Computer Interface (BCI) system using P300 Event-Related Potentials (ERP) for spelling applications. This project implements deep learning models (EEGNet, EEGInception) to detect P300 responses in EEG signals, enabling users to spell words through brain activity.

## 🧠 Overview

This BCI system detects P300 waves - positive voltage deflections in EEG signals occurring approximately 300ms after a stimulus. By analyzing these responses, the system determines which character a user intends to select, enabling hands-free communication.

## 📁 Project Structure

```
├── backend/
│   ├── DatasetAnalyser.py          # EEG dataset exploration and statistics
│   ├── EEGInceptiontrain.py        # EEG-Inception model training
│   ├── EEGmodelvisualisation.py    # Model performance visualization
│   ├── bigp3_processor.py          # P300 data preprocessing pipeline
│   ├── eegnet_p300.py              # EEGNet model implementation
│   └── p300_backend.py             # Backend API server
│
└── frontend/
    └── p300-speller/               # React-based web interface
```

## 🚀 Features

### Backend
- **Multiple Model Architectures**
  - EEGNet: Compact CNN for EEG classification
  - EEG-Inception: Advanced architecture with inception modules
  
- **Comprehensive Data Processing**
  - Automated EEG data cleaning and artifact removal
  - Feature extraction and normalization
  - Train/validation/test split management

- **Advanced Visualization**
  - Model architecture diagrams
  - Confusion matrices
  - ROC and Precision-Recall curves
  - Performance metrics dashboards
  - Error analysis plots

- **Dataset Analysis Tools**
  - ERP waveform visualization
  - Channel-wise signal analysis
  - Class distribution statistics
  - Data quality metrics

### Frontend
- Interactive P300 speller interface
- Real-time EEG signal visualization
- Character selection feedback
- User-friendly controls

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 14+ (for frontend)
- GPU recommended for training (CUDA-compatible)

### Backend Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib seaborn
pip install tensorflow keras scikit-learn
pip install h5py mne scipy
pip install flask flask-cors  # For backend API
```

### Frontend Setup

```bash
cd frontend/p300-speller
npm install
```

## 📊 Dataset

This project uses the **GIB-UVA ERP-BCI** dataset (or similar P300 EEG datasets).

### Expected Data Format (HDF5)
- `features`: EEG signal data (samples × time × channels)
- `target`: Binary labels (0: non-target, 1: P300/target)
- `subjects`: Subject IDs
- `trials`: Trial indices
- Additional metadata fields

### Data Specifications
- **Channels**: 8 EEG electrodes
- **Sampling Rate**: Variable (typically 128-256 Hz)
- **Epoch Length**: ~800ms (including baseline)
- **Classes**: Binary (P300 vs Non-P300)

## 🎯 Usage

### 1. Dataset Analysis

```bash
python backend/DatasetAnalyser.py --data path/to/dataset.hdf5
```

Generates:
- Signal quality reports
- Channel statistics
- ERP waveform plots
- Class distribution analysis

### 2. Model Training

#### EEGNet Training
```bash
python backend/eegnet_p300.py --data path/to/dataset.hdf5 --epochs 100 --batch-size 32
```

#### EEG-Inception Training
```bash
python backend/EEGInceptiontrain.py --data path/to/dataset.hdf5 --epochs 100
```

Training outputs:
- Trained model (.h5 file)
- Training history (loss/accuracy plots)
- Model checkpoints

### 3. Model Evaluation & Visualization

```bash
python backend/EEGmodelvisualisation.py --model model.h5 --data dataset.hdf5
```

Generates 12 visualization files:
- `00_model_summary_report.txt` - Detailed performance metrics
- `01_model_architecture.png` - Model structure
- `02_confusion_matrix.png` - Classification results
- `03_classification_metrics.png` - Accuracy, precision, recall, F1
- `04_roc_curve.png` - ROC-AUC analysis
- `05_precision_recall_curve.png` - PR curve
- `06_prediction_distribution.png` - Probability distributions
- `07_threshold_analysis.png` - Optimal threshold selection
- `08_class_wise_performance.png` - Per-class metrics
- `09_error_analysis.png` - Error breakdown
- `10_model_parameters.png` - Parameter count analysis
- `11_comprehensive_dashboard.png` - All-in-one overview

### 4. Run Backend API

```bash
python backend/p300_backend.py
```

Starts Flask server on `http://localhost:5000`

### 5. Run Frontend

```bash
cd frontend/p300-speller
npm start
```

Opens web interface on `http://localhost:3000`

## 🏗️ Model Architectures

### EEGNet
- **Input**: (8 channels, 128 timepoints, 1)
- **Architecture**:
  - Temporal convolution (filters frequency bands)
  - Depthwise convolution (spatial filtering per channel)
  - Separable convolution (feature extraction)
  - Dense layers for classification
- **Parameters**: ~2,500
- **Best for**: Fast inference, limited data

### EEG-Inception
- **Input**: (8 channels, 128 timepoints, 1)
- **Architecture**:
  - Multi-scale inception modules
  - Parallel convolutions (1x1, 3x3, 5x5)
  - Batch normalization
  - Global average pooling
- **Parameters**: ~15,000
- **Best for**: Higher accuracy, more data available

## 📈 Performance Metrics

Typical performance on P300 detection:

| Metric | EEGNet | EEG-Inception |
|--------|--------|---------------|
| Accuracy | 70-72% | 75-80% |
| Precision | 83-88% | 86-91% |
| Recall | 80-85% | 84-89% |
| F1-Score | 82-87% | 85-90% |
| ROC-AUC | 0.90-0.95 | 0.92-0.96 |

*Note: Results vary based on dataset quality and preprocessing*

## 🔧 Configuration

### Model Hyperparameters

Edit in respective training scripts:

```python
# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5

# Data parameters
CHANNELS = 8
TIMEPOINTS = 128
TEST_SPLIT = 0.2
```

### Data Preprocessing

Modify in `bigp3_processor.py`:

```python
# Filtering
LOW_FREQ = 0.1  # Hz
HIGH_FREQ = 30  # Hz

# Epoching
EPOCH_START = -0.2  # seconds
EPOCH_END = 0.8     # seconds

# Baseline correction
BASELINE_WINDOW = (-0.2, 0)  # seconds
```

## 🐛 Troubleshooting

### Common Issues

**1. ImportError: cannot import name '_ccallback_c'**
```bash
pip uninstall scipy -y
pip install scipy --no-cache-dir
```

**2. Shape mismatch errors**
- Check data dimensions match model input
- Verify channel/timepoint configuration
- Use `DatasetAnalyser.py` to inspect data shape

**3. Low accuracy**
- Increase training epochs
- Adjust learning rate
- Try data augmentation
- Check data quality (use `DatasetAnalyser.py`)

**4. Out of memory**
- Reduce batch size
- Use data generators
- Clear GPU cache: `tf.keras.backend.clear_session()`

## 📚 References

- [EEGNet Paper](https://arxiv.org/abs/1611.08024) - Lawhern et al., 2018
- [P300 BCI Overview](https://doi.org/10.1088/1741-2560/3/2/R01) - Farwell & Donchin
- [GIB-UVA Dataset](https://github.com/gib-uva/erp-bci-dataset)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Joel Sahay D
- Harshitha
- Anagha
- Divagar

## 🙏 Acknowledgments

- GIB-UVA for the ERP-BCI dataset
- EEGNet authors for the model architecture
- MNE-Python community for EEG processing tools

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [joel.sahayjes.a@gmail.com]

---
