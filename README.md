# Customer Churn Prediction Project

This project predicts customer churn using machine learning models.  
It follows a clean, modular, production-ready pipeline including data ingestion, exploratory data analysis (EDA), feature engineering, model training, evaluation, and model saving.

---

## 📂 Project Structure

```
Predict-Customer-Churn/
├── churn_notebook.ipynb                # Development notebook
├── churn_library.py                    # Refactored helper functions
├── churn_script_logging_and_tests.py   # Training and testing script with logging
├── README.md                           # Project overview
├── requirements.txt                    # Required Python packages
├── .gitignore                          # Files/folders to ignore
├── data/
│   └── bank_data.csv                   # Input dataset
├── images/
│   ├── eda/                            # EDA plots
│   └── results/                        # Model evaluation plots
├── logs/
│   └── churn_library.log               # Log file (optional)
└── models/
    ├── logistic_model.pkl              # Saved Logistic Regression model
    └── rfc_model.pkl                   # Saved Random Forest model
```

---

## 🚀 How to Run the Project

1. **Clone the repository:**

```bash
git clone https://github.com/EsraaKamel11/Predict-Customer-Churn.git
cd Predict-Customer-Churn
```

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
# Activate environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the training and evaluation script:**

```bash
python churn_script_logging_and_tests.py
```

This will:
- Import the dataset
- Perform EDA and save plots
- Engineer features
- Train Logistic Regression and Random Forest models
- Save models and evaluation plots
- Generate logs

---

## 🧠 Key Features

- 📊 **Exploratory Data Analysis (EDA)** with automated plot generation
- 🏗️ **Modularized Python code** with `churn_library.py`
- 🔎 **Logging** of all operations to `logs/`
- 🧪 **Unit testing** using Pytest in `churn_script_logging_and_tests.py`
- 📈 **Model evaluation** with ROC Curves, Confusion Matrices, Feature Importances
- 💾 **Model persistence** with Joblib
- 🗂️ **Production-ready folder structure**

---

## ⚙️ Requirements

The project uses the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- statsmodels
- joblib
- pytest

All dependencies can be installed with:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

**Esraa Kamel**  
Date: September 26, 2024

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
