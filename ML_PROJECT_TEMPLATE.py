"""
ML Project Template: Copy and Adapt to Your Problem
====================================================

This is a complete template with all the essential steps.
Follow the comments to adapt to your specific problem.
"""

# ============================================================================
# STEP 1: IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

# TODO: Replace with your data loading
# df = pd.read_csv('your_data.csv')
# For this template, we'll use a sample dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 70)
print("DATA LOADING")
print("=" * 70)
print(f"Shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.DataFrame(X, columns=feature_names)
df['species'] = target_names[y]

print("\n📊 Basic Statistics:")
print(df.describe().round(2))

print("\n📌 Class Distribution:")
print(df['species'].value_counts())

print("\n❓ Missing Values:")
print(df.isnull().sum().sum())

print("\n🔗 Feature Correlations:")
print(df.drop('species', axis=1).corr().round(2))

# ============================================================================
# STEP 4: DATA PREPARATION
# ============================================================================

print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

# 4.1: Train-Test Split (BEFORE any preprocessing)
print("\n1️⃣  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Keep class distribution
)
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 4.2: Handle Missing Values (if any)
print("\n2️⃣  Handling missing values...")
# TODO: Implement if needed
# X_train = pd.DataFrame(X_train).fillna(X_train.mean())
# X_test = pd.DataFrame(X_test).fillna(X_train.mean())

# 4.3: Feature Scaling
print("\n3️⃣  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   Mean: {X_train_scaled.mean(axis=0).round(2)}")
print(f"   Std:  {X_train_scaled.std(axis=0).round(2)}")

# 4.4: Feature Engineering (optional)
print("\n4️⃣  Feature engineering...")
# TODO: Add custom features if needed
# Example:
# X_train_scaled = np.column_stack([X_train_scaled, X_train_scaled[:, 0] ** 2])
# X_test_scaled = np.column_stack([X_test_scaled, X_test_scaled[:, 0] ** 2])

# ============================================================================
# STEP 5: TRAIN MULTIPLE MODELS
# ============================================================================

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

results = {}

print("\n🔄 Training with 5-fold cross-validation...\n")
print(f"{'Model':<25} {'CV Mean':<12} {'CV Std':<12} {'Train Score':<12}")
print("-" * 61)

for name, model in models.items():
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_score': train_score,
        'cv_scores': cv_scores
    }
    
    print(f"{name:<25} {cv_scores.mean():.4f}±{cv_scores.std():.4f}    {train_score:.4f}")

# Find best model
best_name = max(results, key=lambda x: results[x]['cv_mean'])
print(f"\n🏆 Best model: {best_name}")

# ============================================================================
# STEP 6: HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

print("\n🎯 Tuning hyperparameters for Random Forest...\n")

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
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

print(f"✅ Tuning complete!")
print(f"Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# ============================================================================
# STEP 7: FINAL EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION ON TEST SET")
print("=" * 70)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Confusion Matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification Report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ============================================================================
# STEP 8: FEATURE IMPORTANCE (if applicable)
# ============================================================================

if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    importances = best_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print(f"\n🎯 Feature Importance:")
    for idx in sorted_idx:
        print(f"   {feature_names[idx]:<30} {importances[idx]:.4f}")

# ============================================================================
# STEP 9: PREDICTIONS ON NEW DATA
# ============================================================================

print("\n" + "=" * 70)
print("PREDICTIONS ON NEW DATA")
print("=" * 70)

# Make predictions
sample_idx = 0
sample = X_test_scaled[sample_idx:sample_idx+1]
pred_class = best_model.predict(sample)[0]
pred_proba = best_model.predict_proba(sample)[0]

print(f"\n🔮 Sample prediction:")
print(f"   Predicted class: {target_names[pred_class]}")
print(f"   Actual class: {target_names[y_test[sample_idx]]}")
print(f"   Probabilities:")
for i, prob in enumerate(pred_proba):
    print(f"      {target_names[i]:<15} {prob:.2%}")

# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("MODEL SAVING")
print("=" * 70)

import pickle
import joblib

# Option 1: pickle
pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Option 2: joblib (better for sklearn)
joblib.dump(best_model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("\n✅ Model saved!")
print("   - model.pkl / model.joblib")
print("   - scaler.pkl / scaler.joblib")

# ============================================================================
# NEXT STEPS
# ============================================================================

print("\n" + "=" * 70)
print("✨ PROJECT COMPLETE!")
print("=" * 70)

print("""
Next steps:
1. ✅ Verify the model meets business requirements
2. ✅ Create an API for predictions (Flask, FastAPI)
3. ✅ Set up monitoring for model performance
4. ✅ Plan retraining schedule
5. ✅ Deploy to production

Good luck! 🚀
""")
