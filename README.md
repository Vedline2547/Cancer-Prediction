# Cancer Risk Prediction System

## Overview

The **Cancer Risk Prediction System** is a machine learning project that uses patient risk-factor data to classify individuals into three cancer risk categories: **Low, Medium, and High**.

The project explores machine learning techniques for multiclass classification, including data preprocessing, feature scaling, class imbalance handling, hyperparameter optimization, and model evaluation.

The best-performing model was an **Optuna-tuned, class-weighted XGBoost classifier**, integrated into a Flask web application to allow users to submit input data and receive a predicted risk category.

> **Disclaimer:** This is an educational machine learning project. Its predictions are not medical diagnoses and should not be used for clinical decision-making.

## Objectives

- Build a multiclass classification model for cancer risk categorization.
- Explore and preprocess patient risk-factor data.
- Address class imbalance using sample weighting.
- Optimize model performance using Optuna.
- Evaluate performance using classification metrics.
- Deploy the trained model through a Flask web application.

## Technologies and Libraries

- **Python** – Main programming language
- **Pandas & NumPy** – Data manipulation and numerical computation
- **Scikit-learn** – Preprocessing, model evaluation, and data splitting
- **XGBoost** – Gradient boosting classification
- **Optuna** – Hyperparameter optimization
- **Matplotlib & Seaborn** – Data visualization
- **Joblib** – Model and preprocessing artifact serialization
- **Flask** – Web application and model deployment
- **HTML & CSS** – Frontend interface

## Machine Learning Workflow

### 1. Data Preprocessing
- Loaded and explored the dataset.
- Separated input features from the target variable (`Risk_Level`).
- Excluded identifier and non-predictive columns such as `Patient_ID` and `Cancer_Type`.
- Removed `Overall_Risk_Score` to reduce the possibility of target leakage.
- Encoded target labels using `LabelEncoder`.
- Split the dataset into training and testing sets using stratified sampling.
- Applied `StandardScaler` to standardize numerical features.

### 2. Model Development

Several machine learning approaches were explored and compared, including:

- Random Forest
- Random Forest with SMOTE
- Optuna-tuned Random Forest
- XGBoost with SMOTE
- Class-weighted XGBoost
- Optuna-tuned class-weighted XGBoost

### 3. Handling Class Imbalance

Class weighting was implemented in XGBoost using sample weights calculated from the distribution of classes in the training data.

This approach assigns greater importance to underrepresented classes during model training.

### 4. Hyperparameter Optimization

Optuna was used to search for suitable XGBoost hyperparameters.

The optimization process explored parameters such as:

- Number of estimators
- Maximum tree depth
- Learning rate
- Subsample ratio
- Column sampling ratio
- Gamma
- L1 regularization (`reg_alpha`)
- L2 regularization (`reg_lambda`)

Stratified K-Fold cross-validation and macro F1-score were used to evaluate candidate configurations.

### 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Macro F1-score

The Optuna-tuned, class-weighted XGBoost model was selected as the best-performing approach among the models explored.

## Flask Web Application

The trained model is integrated into a Flask application.

Users can:
1. Open the web application.
2. Enter the required patient risk-factor values.
3. Submit the form.
4. Receive a predicted risk category: Low, Medium, or High.

The application loads the trained model together with its preprocessing components to ensure that input data is transformed consistently with the training pipeline.

## Project Structure

```text
Cancer-prediction/
│
├── app.py
├── train.py
├── model_xgb_new.pkl
│
├── templates/
│   └── home.html
│
├── cancer-risk-factors.csv
│
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2647/Cancer-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Cancer-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost optuna flask joblib matplotlib seaborn
```

### 6. Run the Flask application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Future Improvements

- Experiment with additional machine learning algorithms.
- Improve model generalization through further validation.
- Add more informative visualizations and prediction explanations.
- Improve the user interface and input validation.
- Deploy the application to a cloud hosting platform.
- Evaluate the model on independent, clinically validated datasets.

## Key Learning Outcomes

Through this project, I gained practical experience in:

- Multiclass classification.
- Feature engineering and preprocessing.
- Handling imbalanced datasets.
- XGBoost model development.
- Hyperparameter tuning with Optuna.
- Cross-validation and model evaluation.
- Saving and loading trained machine learning pipelines.
- Building Flask applications for machine learning deployment.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

[GitHub](https://github.com/Vedline2547) · [LinkedIn](https://www.linkedin.com/in/vedline-ochieng-025146290/)

---

*This project was developed for educational and portfolio purposes. It is not a clinically validated cancer screening or diagnostic system.*
