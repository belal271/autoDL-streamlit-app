# AutoML Model Builder

A streamlined machine learning application built with Streamlit that allows users to build, train, and download deep learning models without writing code. This application supports both classification (binary and multi-class) and regression problems.

## Features

### Data Handling
- Easy data upload through web interface
- Automatic data type detection
- Comprehensive data profiling with pandas-profiling
- Automatic handling of missing values
- Support for both numerical and categorical data

### Preprocessing
- Automatic handling of NaN values
  - Numerical columns: Mean imputation
  - Categorical columns: Most frequent value imputation
- Automatic feature scaling (StandardScaler)
- Automatic categorical encoding
- Boolean to integer conversion

### Model Building
- Automatic problem type detection (Classification/Regression)
- Customizable neural network architecture
  - Variable number of layers
  - Configurable neurons per layer
  - Multiple activation function options
- Dropout layers for regularization
- Early stopping to prevent overfitting

### Training
- Interactive training progress visualization
- Real-time metrics tracking
- Train/Validation split
- Configurable:
  - Number of epochs (up to 1000)
  - Batch size
  - Learning rate
  - Early stopping patience

### Visualization
- Interactive training metrics plots
- Model architecture visualization
- Data profiling reports
- Real-time training progress

### Model Export
- Download trained models in H5 format
- Model summary and configuration export
- Final performance metrics included

## Installation

1. Clone the repository:
```bash
git clone https://github.com/belal271/autoDL-streamlit-app.git
cd autoDL-streamlit-app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run model_builder.py
```

2. Upload your data:
   - Click "Upload" in the sidebar
   - Select your CSV file
   - View the data preview

3. Profile your data (optional):
   - Click "Profiling" in the sidebar
   - Explore the comprehensive data report

4. Build and train your model:
   - Click "Modelling" in the sidebar
   - Select your target column
   - Configure model architecture
   - Click "Build Model" and then "Train Model"

5. Download your model:
   - Click "Download" in the sidebar
   - Download the trained model in H5 format

## Requirements
```
streamlit
pandas
numpy
tensorflow
scikit-learn
plotly
ydata-profiling
streamlit-pandas-profiling
```

## File Structure
- `model_builder.py`: Main application file with Streamlit interface
- `preprocessing.py`: Data preprocessing utilities
- `requirements.txt`: Package dependencies
- `README.md`: Documentation

## Model Loading Example
```python
from tensorflow.keras.models import load_model

# Load the downloaded model
model = load_model('trained_model.h5')

# Make predictions
predictions = model.predict(your_data)
```

## Supported Problems
- Binary Classification
- Multi-class Classification
- Regression

## Contributing
Feel free to open issues or submit pull requests for any improvements.

## Author
Belal Mohamed
