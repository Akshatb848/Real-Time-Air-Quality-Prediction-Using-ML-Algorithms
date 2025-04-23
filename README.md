# Real-time Air Quality Prediction using Machine Learning

This project aims to predict air quality metrics using real-time data and machine learning models. The dataset used is the [Air Quality Data Set](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set) from Kaggle.

## 📊 Dataset

The dataset contains hourly averaged responses from an array of chemical sensors embedded in an Air Quality Chemical Multisensor Device.

- Source: [Kaggle - Air Quality Data Set](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set)
- Format: CSV
- Features: CO(GT), PT08.S1(CO), NMHC(GT), C6H6(GT), T, RH, etc.

## ⚙️ Project Workflow

1. **Data Preprocessing**: Handling missing values and encoding.
2. **Feature Selection**: Using feature importance from Random Forest.
3. **Model Training**:
   - Linear Regression
   - Random Forest Regressor with RandomizedSearchCV for hyperparameter tuning
4. **Evaluation Metrics**:
   - Mean Squared Error (MSE)
   - R² Score

## ✅ Results

- **Best Model**: Random Forest Regressor
- **Best Hyperparameters**:  
  ```python
  {
      'n_estimators': 250,
      'min_samples_split': 5,
      'min_samples_leaf': 1,
      'max_features': 'sqrt',
      'max_depth': 30,
      'bootstrap': False
  }
