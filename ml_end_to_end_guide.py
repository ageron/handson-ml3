#!/usr/bin/env python3
"""
End-to-End Machine Learning Project Guide
==========================================

Complete workflow: Data Loading → EDA → Preparation → Training → Evaluation → Tuning

This script demonstrates all phases using the Iris dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("END-TO-END MACHINE LEARNING PROJECT GUIDE")
print("=" * 70)

# ============================================================================
# PHASE 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 1: DATA LOADING AND EXPLORATION")
print("=" * 70)

iris = load_iris()
X = iris.data
y = iris.target

print(f"\n📊 Dataset Overview:")
print(f"   Shape: {X.shape}")
print(f"   Features: {X.shape[1]} (sepal length/width, petal length/width)")
print(f"   Samples: {X.shape[0]}")
print(f"   Classes: {len(np.unique(y))} (setosa, versicolor, virginica)")

# Convert to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]

print(f"\n📋 First 5 samples:")
print(df.head())

print(f"\n📈 Statistical Summary:")
print(df.describe().round(2))

print(f"\n🎯 Class Distribution:")
print(df['species'].value_counts())

# Check for missing values
print(f"\n❓ Missing Values: {df.isnull().sum().sum()}")

# Correlations
print(f"\n🔗 Feature Correlations:")
corr_matrix = df.drop('species', axis=1).corr()
print(corr_matrix.round(2))

# ============================================================================
# PHASE 2: DATA PREPARATION
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 2: DATA PREPARATION")
print("=" * 70)

print("\n📍 Step 1: Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples (80%)")
print(f"   Test set: {X_test.shape[0]} samples (20%)")
print(f"   ✅ Stratified split ensures balanced classes")

print("\n📍 Step 2: Feature Scaling (StandardScaler)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Before scaling - Mean: {X_train.mean(axis=0).round(2)}")
print(f"   After scaling  - Mean: {X_train_scaled.mean(axis=0).round(2)}")
print(f"   ✅ Features now standardized (mean=0, std=1)")

# ============================================================================
# PHASE 3: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 3: MODEL TRAINING WITH CROSS-VALIDATION")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}

print(f"\n🔄 Training models with 5-fold cross-validation...\n")
print(f"{'Model':<25} {'CV Mean':<15} {'CV Std':<15} {'Train Score':<12}")
print("-" * 67)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_score': train_score
    }
    
    print(f"{name:<25} {cv_scores.mean():.4f}±{cv_scores.std():.4f}    {train_score:.4f}")

best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   CV Accuracy: {results[best_model_name]['cv_mean']:.4f}")

# ============================================================================
# PHASE 4: HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 4: HYPERPARAMETER TUNING")
print("=" * 70)

print(f"\n🎯 Tuning Random Forest hyperparameters...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ Best Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n   Best CV Score: {grid_search.best_score_:.4f}")

# ============================================================================
# PHASE 5: FINAL EVALUATION ON TEST SET
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 5: FINAL EVALUATION ON TEST SET")
print("=" * 70)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("   Predicted:")
print("         Setosa  Versicolor  Virginica")
for i, row in enumerate(cm):
    print(f"   {iris.target_names[i]:<10} {row}")

print(f"\n📋 Classification Report (per class):")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ============================================================================
# PHASE 6: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 6: FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

feature_importance = best_model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]

print(f"\n🎯 Feature Importance (Random Forest):")
for idx in sorted_indices:
    print(f"   {iris.feature_names[idx]:<28} {feature_importance[idx]:.4f}")

# ============================================================================
# PHASE 7: PREDICTIONS ON NEW DATA
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 7: MAKING PREDICTIONS ON NEW DATA")
print("=" * 70)

sample_indices = [0, 10, 20]

print(f"\n🔮 Predictions on test samples:")
for idx in sample_indices:
    sample = X_test_scaled[idx].reshape(1, -1)
    prediction = best_model.predict(sample)[0]
    probabilities = best_model.predict_proba(sample)[0]
    actual = y_test[idx]
    
    print(f"\n   Sample {idx}:")
    print(f"   Predicted: {iris.target_names[prediction]}")
    print(f"   Actual:    {iris.target_names[actual]}")
    print(f"   Confidence:")
    for i, prob in enumerate(probabilities):
        print(f"      {iris.target_names[i]:<15} {prob:.2%}")

# ============================================================================
# SUMMARY AND KEY LEARNINGS
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: COMPLETE ML WORKFLOW")
print("=" * 70)

summary = """
✅ PHASES COMPLETED:
   1. Data Loading & Exploration (EDA)
   2. Data Preparation (Split, Scale)
   3. Model Training (4 models, cross-validation)
   4. Hyperparameter Tuning (GridSearchCV)
   5. Evaluation on Test Set
   6. Feature Importance Analysis
   7. Predictions on New Data

🔑 KEY PRINCIPLES:
   1. Always split data FIRST (before any preprocessing)
   2. Scale features (especially for distance/gradient-based algorithms)
   3. Try multiple models (different algorithms work differently)
   4. Use cross-validation (more reliable than single split)
   5. Tune hyperparameters systematically (GridSearchCV, RandomSearchCV)
   6. Evaluate ONLY on test set (don't touch it until final evaluation)
   7. Analyze errors (confusion matrix, feature importance)
   8. Interpret results (why does the model work?)

📈 WORKFLOW DIAGRAM:
   Raw Data
      ↓
   Exploratory Analysis
      ↓
   Train-Test Split
      ↓
   Feature Scaling
      ↓
   Model Training (multiple algorithms)
      ↓
   Cross-Validation
      ↓
   Hyperparameter Tuning
      ↓
   Final Evaluation on Test Set
      ↓
   Deployment/Production

⚠️ COMMON PITFALLS TO AVOID:
   • Data leakage (scaling before split)
   • Overfitting (tuning on test set)
   • Ignoring class imbalance (use stratified split)
   • Not using cross-validation (single split is unreliable)
   • Using test set for model selection (use validation set instead)
"""

print(summary)

print("\n" + "=" * 70)
print("✨ PROJECT COMPLETE!")
print("=" * 70)
