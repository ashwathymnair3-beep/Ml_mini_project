# 🧠 Personality Prediction System (KNN + Streamlit)

A machine learning web app that predicts whether a person is an **Extrovert**, **Introvert**, or **Ambivert** based on personality-related inputs.

## 🎯 Purpose
Build an interactive ML application to demonstrate an end-to-end workflow:
data preprocessing → model training → evaluation → deployment UI.

## ⚙️ Features
- Slider-based user input form in Streamlit
- Trains a **KNN Classifier** on a personality dataset
- Applies **StandardScaler** for feature scaling
- Shows prediction result instantly
- Evaluates performance using **accuracy score** and **confusion matrix**

## 🧑‍💻 My Contribution
- Cleaned dataset and selected relevant features
- Built the ML training pipeline using Scikit-learn
- Implemented scaling + train/test split
- Developed Streamlit UI for real-time predictions
- Added evaluation metrics for model validation

## 🛠 Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn (KNN, StandardScaler, train_test_split)
- Streamlit

## 📂 Project Structure
- `personality.py` → Streamlit app + model training
- `personality_synthetic_dataset.csv` → dataset used for training

## ▶️ How to Run Locally
```bash
pip install pandas numpy scikit-learn streamlit
streamlit run personality.py
