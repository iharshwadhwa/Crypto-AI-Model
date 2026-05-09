import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings('ignore')

print("============================================")
print("AI Fraud Detection in Cryptocurrency")
print("============================================")

# -------------------------------
# LOAD DATASET
# -------------------------------

# Replace with your dataset file name
file_path = 'kaggle_crypto_dataset.csv'

try:
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully")
except:
    print("Dataset file not found")
    exit()

print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

# -------------------------------
# DATA PREPROCESSING
# -------------------------------

print("\nChecking Missing Values...")
print(data.isnull().sum())

# Assume last column is target
# Remove Date column
data = data.drop(columns=['Date'])

# Create artificial fraud labels
# If closing price increased sharply -> fraud = 1 else 0

data['target'] = np.where(
    (data['Close'] - data['Open']) > 1000,
    1,
    0
)

# Features and target
X = data.drop('target', axis=1)
y = data['target']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -------------------------------
# MACHINE LEARNING MODELS
# -------------------------------

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),

    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='logloss',
        random_state=42
    ),

    'Support Vector Machine': SVC(
        probability=True,
        kernel='rbf'
    )
}

# -------------------------------
# TRAINING & EVALUATION
# -------------------------------

results = []

print("\n============================================")
print("MODEL TRAINING STARTED")
print("============================================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Probability predictions for AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        auc
    ])

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("AUC ROC  :", round(auc, 4))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

# -------------------------------
# RESULTS DATAFRAME
# -------------------------------

results_df = pd.DataFrame(results, columns=[
    'Model',
    'Accuracy',
    'Precision',
    'Recall',
    'F1 Score',
    'AUC ROC'
])

print("\n============================================")
print("FINAL RESULTS")
print("============================================")
print(results_df)

# Save results
results_df.to_csv('model_results.csv', index=False)

print("\nResults saved as model_results.csv")

# -------------------------------
# VISUALIZATION
# -------------------------------

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC']

for metric in metrics:

    plt.figure(figsize=(8, 5))

    plt.bar(results_df['Model'], results_df[metric])

    plt.title(f'{metric} Comparison')
    plt.xlabel('Models')
    plt.ylabel(metric)

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(f'{metric}.png')

    plt.close()

print("\nIndividual graphs generated successfully")

# -------------------------------
# COMBINED GRAPH
# -------------------------------

plt.figure(figsize=(10, 6))

for metric in metrics:
    plt.plot(results_df['Model'], results_df[metric], marker='o', label=metric)

plt.title('Overall Model Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Score')

plt.legend()
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig('combined_graph.png')

plt.show()

print("\nCombined graph generated successfully")

print("\n============================================")
print("PROJECT EXECUTED SUCCESSFULLY")
print("============================================")

