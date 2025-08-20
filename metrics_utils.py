# metrics_utils.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    recall_score, f1_score, confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna

def calculate_metrics(y_test, y_pred_proba):
    """Calculate binary classification metrics given true labels and predicted probabilities."""
    threshold = np.mean(y_pred_proba)  # Dynamic threshold based on mean
    y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred_proba]

    accuracy = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    g_mean = np.sqrt(recall * specificity)

    metrics_dict = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'fp_rate': fp_rate,
        'g_mean': g_mean
    }

    return metrics_dict

def callf1(X_train, y_train, X_test, y_test):
    """Train a LightGBM DART GPU model with Optuna tuning and return metrics (original logic preserved)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("optuna")
    # Optuna objective for hyperparameter optimization
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "dart",
            "device": "gpu",
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        
        dtrain = lgb.Dataset(X_train_scaled, label=y_train)
        model = lgb.train(params, dtrain, num_boost_round=200)
        y_pred_proba = model.predict(X_test_scaled)
        
        # Maximize F1-score (aligned with your original logic)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        return np.max(f1_scores)

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

    # Train final model with best params
    best_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "dart",
        "device": "gpu",
        **study.best_params  # Inject Optuna's best hyperparameters
    }
    
    dtrain = lgb.Dataset(X_train_scaled, label=y_train)
    model = lgb.train(best_params, dtrain, num_boost_round=200)
    y_pred_proba = model.predict(X_test_scaled)

    # Your original metric calculation logic (unchanged)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-6)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    metrics = calculate_metrics(y_test, y_pred_proba)  # Ensure this function is defined
    return metrics  # Return format identical to your original function
"""def callf1(X_train, y_train, X_test, y_test):
   

    scaler = StandardScaler()

    dtrain = lgb.Dataset(X_train, label=y_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "dart",
        "device": "gpu"
    }

    model = lgb.train(params, dtrain, num_boost_round=200)

    y_pred_proba = model.predict(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-6)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    metrics = calculate_metrics(y_test, y_pred_proba)
    return metrics
"""

def callf1_opt(X_test, y_test, model):
    """Use a trained model to predict probabilities and return metrics."""

    y_pred_proba = model.predict(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-6)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    metrics = calculate_metrics(y_test, y_pred_proba)
    return metrics
