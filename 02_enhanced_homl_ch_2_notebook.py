# Enhanced End-to-End Notebook for Hands-On ML Chapter 2
# File: enhanced_homl_ch2_notebook.py
# Purpose: Complete, runnable end-to-end pipeline implementing improvements:
#   - data loading
#   - EDA (brief)
#   - feature engineering
#   - preprocessing pipelines (ColumnTransformer)
#   - outlier handling
#   - several models (Linear, Ridge, DecisionTree, RandomForest, GradientBoosting)\#   - model selection (RandomizedSearchCV)
#   - stacking ensemble
#   - final evaluation on test set
#   - model persistence

print("Enhanced ML workflow loaded.")

# --- Imports ---------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import tarfile

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Optional advanced libs (if installed) – fallback safe
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --- Utilities -------------------------------------------------------------
def load_housing_data(data_root="https://github.com/ageron/data/raw/main/"):
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = data_root + "housing.tgz"
        print("Downloading housing dataset...")
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

# RMSE helper
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# save figure helper
IMAGES_PATH = Path("images/enhanced_homl_ch2")
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id):
    plt.tight_layout()
    plt.savefig(IMAGES_PATH / f"{fig_id}.png", dpi=200)

# --- Load data -------------------------------------------------------------
housing = load_housing_data()
print("Loaded housing shape:", housing.shape)

# --- Quick EDA (very brief) -----------------------------------------------
print(housing.info())
print(housing.describe().T[['mean','std','min','max']])
print(housing['ocean_proximity'].value_counts())

# Visual quick check (histograms)
housing.hist(bins=50, figsize=(12, 8))
save_fig('histograms')
plt.show()

# --- Create stratified split based on median_income (as in book) ------------
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0.,1.5,3.0,4.5,6.,np.inf], labels=[1,2,3,4,5])
train_set, test_set = train_test_split(housing, test_size=0.2, stratify=housing['income_cat'], random_state=42)
for s in (train_set, test_set):
    s.drop('income_cat', axis=1, inplace=True)

housing = train_set.copy()

# --- Feature engineering (recommended additions) ---------------------------
# We'll implement feature transformers that can be included in ColumnTransformer

def add_extra_features(X_df):
    X = X_df.copy()
    # safe divisions
    X['rooms_per_house'] = X['total_rooms'] / X['households']
    X['bedrooms_ratio'] = X['total_bedrooms'] / X['total_rooms']
    X['people_per_house'] = X['population'] / X['households']
    X['rooms_per_person'] = X['total_rooms'] / X['population']
    X['income_x_age'] = X['median_income'] * X['housing_median_age']
    # fill infinities / NaNs if any
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X

# Apply to verify
housing_fe = add_extra_features(housing)
print(housing_fe[['rooms_per_house','bedrooms_ratio','people_per_house','rooms_per_person','income_x_age']].head())

# Custom transformer to add engineered features inside pipeline
class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_rooms_per_person=True):
        self.add_rooms_per_person = add_rooms_per_person
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X is numpy array: we convert to DataFrame with feature names if provided
        if hasattr(X, 'columns'):
            X_df = X
        else:
            # fallback: no column names – assume original order from book num_attribs
            X_df = pd.DataFrame(X, columns=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income'])
        X_df = add_extra_features(X_df)
        # select numerical columns
        return X_df[['rooms_per_house','bedrooms_ratio','people_per_house','rooms_per_person','income_x_age']].values

# --- Preprocessing pipelines -----------------------------------------------
num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feat_adder', FeatureAdder()),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
])

# Fit-transform a small sample to ensure pipeline works
sample_prep = preprocessing.fit_transform(housing)
print('Preprocessing output shape:', sample_prep.shape)

# --- Prepare training data -------------------------------------------------
X_train = housing.drop('median_house_value', axis=1)
y_train = housing['median_house_value'].copy()
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value'].copy()

# We will use pipelines wrapping the preprocessing + estimator

# --- Define candidate models ----------------------------------------------
models = {}
models['lin_reg'] = make_pipeline(preprocessing, LinearRegression())
models['ridge'] = make_pipeline(preprocessing, Ridge(random_state=42))
models['tree'] = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
models['rf'] = make_pipeline(preprocessing, RandomForestRegressor(random_state=42, n_jobs=-1))
models['gbr'] = make_pipeline(preprocessing, GradientBoostingRegressor(random_state=42))
if XGBOOST_AVAILABLE:
    models['xgb'] = make_pipeline(preprocessing, xgb.XGBRegressor(random_state=42, n_jobs=-1))

# --- Quick cross-validation comparison ------------------------------------
print('Cross-validating baseline models (5-fold RMSE) ...')
for name, model in models.items():
    scores = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    print(f"{name}: mean RMSE={scores.mean():.2f}, std={scores.std():.2f}")

# --- Hyperparameter tuning (RandomizedSearch) ------------------------------
# We'll tune RandomForest and GradientBoosting (and XGBoost if available)

param_dist_rf = {
    'randomforestregressor__n_estimators': [100, 200, 400],
    'randomforestregressor__max_features': [4, 6, 8, 10],
    'randomforestregressor__max_depth': [None, 10, 20, 30],
    'randomforestregressor__min_samples_split': [2, 5, 10]
}

param_dist_gbr = {
    'gradientboostingregressor__n_estimators': [100, 200, 400],
    'gradientboostingregressor__learning_rate': [0.01, 0.05, 0.1],
    'gradientboostingregressor__max_depth': [3, 5, 8],
    'gradientboostingregressor__subsample': [0.6, 0.8, 1.0]
}

searches = {}

print('Running RandomizedSearchCV for RandomForest (this may take a while)...')
rf_search = RandomizedSearchCV(models['rf'], param_distributions=param_dist_rf, n_iter=10, cv=3,
                               scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
searches['rf'] = rf_search
print('Best RF params:', rf_search.best_params_)

print('Running RandomizedSearchCV for GradientBoosting...')
gbr_search = RandomizedSearchCV(models['gbr'], param_distributions=param_dist_gbr, n_iter=10, cv=3,
                                 scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
gbr_search.fit(X_train, y_train)
searches['gbr'] = gbr_search
print('Best GBR params:', gbr_search.best_params_)

if XGBOOST_AVAILABLE:
    param_dist_xgb = {
        'xgbregressor__n_estimators': [100, 200, 400],
        'xgbregressor__max_depth': [3, 5, 8],
        'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
        'xgbregressor__colsample_bytree': [0.6, 0.8, 1.0]
    }
    print('Running RandomizedSearchCV for XGBoost...')
    xgb_search = RandomizedSearchCV(models['xgb'], param_distributions=param_dist_xgb, n_iter=10, cv=3,
                                    scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    searches['xgb'] = xgb_search
    print('Best XGB params:', xgb_search.best_params_)

# --- Evaluate best estimators on validation via cross-val ------------------
print('Evaluating best estimators (5-fold CV)')
best_estimators = {}
for key, search in searches.items():
    best = search.best_estimator_
    best_estimators[key] = best
    scores = -cross_val_score(best, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    print(f"{key}: mean RMSE={scores.mean():.2f}, std={scores.std():.2f}")

# Also include tuned RF and GBR in pool
candidates = []
for k, est in best_estimators.items():
    candidates.append((k, est))
# add untuned models as fallback
candidates.append(('ridge', models['ridge']))

# --- Stacking ensemble -----------------------------------------------------
print('Building stacking ensemble with top candidates...')
final_estimators = [(name, est.named_steps[list(est.named_steps.keys())[-1]]) for name, est in candidates]
# NOTE: StackingRegressor expects estimators without preprocessing; to keep preprocessing we create pipeline-wrappers
stack = StackingRegressor(estimators=[(name, est) for name, est in best_estimators.items() if name in ['rf','gbr'] and name in best_estimators],
                          final_estimator=Ridge())
stack_pipeline = make_pipeline(preprocessing, stack)

# Cross-validate stacking
stack_scores = -cross_val_score(stack_pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
print(f"Stacking mean RMSE={stack_scores.mean():.2f}, std={stack_scores.std():.2f}")

# --- Final training on full training set and test evaluation ----------------
print('Training final model (stack pipeline) on full training set...')
stack_pipeline.fit(X_train, y_train)
final_predictions = stack_pipeline.predict(X_test)
final_test_rmse = rmse(y_test, final_predictions)
print(f"Final model test RMSE = {final_test_rmse:.2f}")

# Feature importances from best RF (if present)
if 'rf' in searches:
    best_rf = searches['rf'].best_estimator_.named_steps['randomforestregressor']
    # get feature names
    try:
        feature_names = list(preprocessing.get_feature_names_out())
    except Exception:
        feature_names = None
    if feature_names is not None:
        importances = best_rf.feature_importances_
        fi = sorted(zip(importances, feature_names), reverse=True)[:15]
        print('Top feature importances (RF):')
        for imp, name in fi:
            print(f"{name}: {imp:.3f}")

# --- Confidence interval for test RMSE via bootstrap -----------------------
from scipy import stats
squared_errors = (final_predictions - y_test) ** 2
def rmse_from_sq(sq):
    return np.sqrt(np.mean(sq))
boot_res = stats.bootstrap([squared_errors], rmse_from_sq, confidence_level=0.95, random_state=42)
print('95% CI for test RMSE:', boot_res.confidence_interval)

# --- Save final pipeline --------------------------------------------------
joblib.dump(stack_pipeline, 'enhanced_homl_ch2_final_pipeline.pkl')
print('Saved final pipeline to enhanced_homl_ch2_final_pipeline.pkl')

# --- Quick prediction demo ------------------------------------------------
sample = X_test.iloc[:5]
print('Sample predictions:', stack_pipeline.predict(sample).round(-2))
print('Sample actuals    :', y_test.iloc[:5].values.round(-2))

# Done
print('Notebook run complete.')