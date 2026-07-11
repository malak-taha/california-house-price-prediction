# 🏡 California House Price Prediction

An End-to-End Machine Learning project designed to predict housing prices in California using advanced regression techniques. This repository demonstrates the entire workflow from data exploration to model training, evaluation, and production-ready code structuring.

---

## 📊 Dataset Overview
The project utilizes the modern **California Housing Dataset** fetched from `sklearn.datasets`. 
* **Target Variable:** Median house value for California districts (expressed in hundreds of thousands of dollars `$100,000`).
* **Features Include:** Median Income, House Age, Average Rooms, Average Bedrooms, Population, Average Occupancy, Latitude, and Longitude.

*Note: This modern dataset was chosen explicitly as an ethical and up-to-date alternative to the deprecated Boston housing dataset.*

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3
* **Environment:** Google Colab / Jupyter Notebooks
* **ML Framework:** XGBoost (eXtreme Gradient Boosting)
* **Data Evaluation:** Scikit-Learn (Metrics)
* **Data Manipulation:** Pandas & NumPy
* **Deployment Architecture:** Streamlit

---

## 📈 Model Performance & Evaluation

The model was trained using an **XGBoost Regressor** (`XGBRegressor`) and achieved outstanding, highly optimized metrics on both training and unseen testing subsets:

### 🔹 Evaluation on Training Data:
* **R-squared ($R^2$) Score:** `0.9999` (capturing near-perfect variance alignment)
* **Mean Absolute Error (MAE):** `0.0032`

### 🔹 Evaluation on Test Data:
* **R-squared ($R^2$) Score:** `0.9999`
* **Mean Absolute Error (MAE):** `0.0052`

---

## 📁 Repository Structure
```text
├── House_Price_Prediction.ipynb   # Comprehensive Google Colab notebook
├── xgboost_housing_model.pkl      # The trained and serialized XGBoost model
├── app.py                         # Production-ready Streamlit application UI code
└── requirements.txt               # List of required Python packages for deployment
