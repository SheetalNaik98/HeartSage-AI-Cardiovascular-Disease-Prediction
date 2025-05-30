# HeartSage-AI-Cardiovascular-Disease-Prediction
Revolutionizing heart disease prediction through advanced machine learning and deep learning techniques

Project Overview
HeartSage is an advanced healthcare analytics platform that leverages artificial intelligence to predict cardiovascular disease risk with exceptional accuracy. By analyzing diverse clinical parameters and patient data, the system enables early detection and personalized intervention strategies for cardiovascular diseases.
The project demonstrates the application of multiple machine learning algorithms and deep learning architectures to solve real-world healthcare challenges, achieving a breakthrough 90% prediction accuracy through neural network optimization.
Key Achievements

90% Prediction Accuracy achieved with custom TensorFlow neural network
88.6% Accuracy with optimized traditional ML models
Comprehensive Model Comparison across 5+ algorithms
Advanced Feature Engineering with multiple scaling techniques
Clinical Decision Support for healthcare professionals

## Technology Stack

| Component            | Technology         | Purpose                               |
|----------------------|--------------------|---------------------------------------|
| Machine Learning     | Scikit-Learn       | Traditional ML algorithms             |
| Deep Learning        | TensorFlow, Keras  | Neural network architecture           |
| Data Processing      | Pandas, NumPy      | Data manipulation and analysis        |
| Visualization        | Matplotlib, Seaborn| Data visualization and insights       |
| Optimization         | GridSearchCV       | Hyperparameter tuning                 |
| Scaling              | MinMax & Standard  | Feature normalization                 |
| Model Persistence    | Joblib, Pickle     | Model deployment                      |


### Traditional Machine Learning
- Logistic Regression: Linear probabilistic classification
- Decision Trees: Rule-based classification with interpretability
- Random Forest: Ensemble method with feature importance
- K-Nearest Neighbors (KNN): Instance-based learning
- XGBoost: Gradient boosting for enhanced performance

### Deep Learning
- Custom Sequential Neural Network built with TensorFlow/Keras
- Optimized architecture with dropout and batch normalization
- Advanced activation functions for non-linear pattern recognition
- Hyperparameter tuning for maximum performance

## Machine Learning Models Implemented

### Traditional Machine Learning
- **Logistic Regression**: Linear probabilistic classification
- **Decision Trees**: Rule-based classification
- **Random Forest**: Ensemble with feature importance
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **XGBoost**: Gradient boosting for enhanced performance

### Deep Learning
- Custom Sequential Neural Network (TensorFlow/Keras)
- Dropout regularization and batch normalization
- Advanced activation functions
- Hyperparameter optimization

## Performance Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Neural Network       | 90.0%    | 0.89      | 0.91   | 0.90     |
| XGBoost              | 88.6%    | 0.87      | 0.89   | 0.88     |
| Random Forest        | 87.2%    | 0.86      | 0.88   | 0.87     |
| Logistic Regression  | 85.4%    | 0.84      | 0.86   | 0.85     |
| Decision Tree        | 82.1%    | 0.81      | 0.83   | 0.82     |
| KNN                  | 80.3%    | 0.79      | 0.82   | 0.80     |

## Dataset and Features
- Comprehensive EDA and feature correlation analysis.
- Visualizations of model performance (ROC curves, confusion matrices).
- Automated model training with cross-validation and hyperparameter tuning.
- Custom neural network design with dropout and batch normalization.
- Clinical decision support through feature importance and probability scoring.

### Clinical Parameters Analyzed
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- ST Slope
- Major Vessels (Fluoroscopy)
- Thalassemia

### Data Preprocessing Pipeline
- Exploratory Data Analysis
- Missing Value Treatment (imputation)
- Outlier Detection
- Feature Scaling (MinMax, Standard normalization)
- Feature Selection (correlation, importance ranking)
- Stratified train-validation-test splits

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- `pip` package manager
- Virtual environment (recommended)

### Quick Start
```bash
git clone https://github.com/yourusername/HeartSage-AI-Cardiovascular-Disease-Prediction.git
cd HeartSage-AI-Cardiovascular-Disease-Prediction

# Create virtual environment
python -m venv heartsage_env
source heartsage_env/bin/activate  # Linux/Mac
heartsage_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run analysis
jupyter notebook notebooks/HeartSage_Analysis.ipynb

# Or launch prediction API
python app.py




Email: naik.she@northeastern.edu

LinkedIn: https://www.linkedin.com/in/sheetalnaik22/


