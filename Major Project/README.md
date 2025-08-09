
# Loan Default Prediction – Major Project

## 📌 Overview
This project predicts the probability of a customer defaulting on a loan using machine learning techniques.  
It includes data preprocessing, model training (Logistic Regression, Random Forest, XGBoost), and a Flask-based web application for user-friendly predictions.

---

## 📂 Folder Structure
Major Project/
│
├── requirements.txt # Python dependencies
│
├── app/ # Flask web application
│ ├── app.py # Main application script
│ ├── static/
│ │ └── css/style.css # CSS styles
│ └── templates/
│ ├── index.html # Home page
│ └── result.html # Prediction results page
│
├── data/
│ └── Loan_default.csv # Dataset
│
├── models/
│ ├── model.pkl # Trained model
│ └── preprocessor.pkl # Preprocessing pipeline
│
├── plots/ # Model performance visualizations
│ ├── LogisticRegression_cm.png
│ ├── LogisticRegression_roc.png
│ ├── RandomForest_cm.png
│ ├── RandomForest_roc.png
│ ├── XGBoost_cm.png
│ └── XGBoost_roc.png
│
├── src/ # Source code for ML pipeline
│ ├── data_prep.py # Data cleaning & preprocessing
│ ├── predict.py # Prediction logic
│ └── train_model.py # Model training & evaluation
│
└── notebooks/ (optional) # Jupyter notebooks for exploration
├── 01_EDA_and_Preprocessing.ipynb
└── 02_Model_Training_Evaluation.ipynb


---

## 🚀 Features
- **Data Preprocessing** – Handles missing values, encodes categorical variables, and scales features.
- **Multiple Models** – Logistic Regression, Random Forest, and XGBoost.
- **Performance Metrics** – Confusion Matrix, ROC Curve, and Accuracy.
- **Flask Web App** – Interactive web interface for predictions.
- **Pre-trained Model** – Ready-to-use model with saved preprocessing pipeline.

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows

# Install dependencies
pip install -r requirements.txt


MIT License

Copyright (c) 2025 [Goldi Kumari]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND.
