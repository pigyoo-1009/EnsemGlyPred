import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, matthews_corrcoef, roc_auc_score
)
import os
import random

RANDOM_STATE = 58
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

MODEL_WEIGHTS =(0.4, 0.33, 0.27) # PAAC+XGBoost, AAC+XGBoost, ProGen+BiLSTM

# 预测阈值
THRESHOLDS = {
    'paac': 0.45,  # PAAC+XGBoost 模型阈值
    'aac': 0.46,  # AAC+XGBoost 模型阈值
    'progen': 0.46,  # ProGen+BiLSTM 模型阈值
    'ensemble': 0.46 # 集成模型阈值
}


def load_test_data():
    print("Loading test data features...")
    try:
        X_test_paac = np.load('X_test_paac.npy')
        X_test_aac = np.load('X_test_aac.npy')
        X_test_progen = np.load('X_test_progen_pca.npy')
        y_test = np.load('y_test_paac.npy')

        print(f"Test data loaded successfully: {len(y_test)} samples")
        print(f"  - Positive samples: {np.sum(y_test)}")
        print(f"  - Negative samples: {len(y_test) - np.sum(y_test)}")

        return (X_test_paac, X_test_aac, X_test_progen), y_test

    except FileNotFoundError as e:
        print(f"Error: Could not find feature files: {e}")
        return None, None


def load_models():
    print("Loading pre-trained models...")
    try:
        # Load XGBoost models
        model1 = joblib.load('paac_xgb.pkl')  # PAAC+XGBoost
        model2 = joblib.load('aac_xgb.pkl')  # AAC+XGBoost
        # Load BiLSTM model
        model3 = load_model('progen2_bilstm.h5')  # ProGen+BiLSTM
        print("All models loaded successfully")
        return model1, model2, model3
    except FileNotFoundError as e:
        print(f"Error: Could not load models: {e}")
        return None, None, None


def evaluate_ensemble(features, y_test, models):
    """Evaluate the ensemble model performance"""
    if features is None or models is None:
        print("Cannot evaluate without features or models")
        return None

    X_test_paac, X_test_aac, X_test_progen = features
    model1, model2, model3 = models
    weights = MODEL_WEIGHTS
    thresholds = THRESHOLDS


    X_test_progen_3d = X_test_progen.reshape((X_test_progen.shape[0], 1, X_test_progen.shape[1]))

    prob1 = model1.predict_proba(X_test_paac)[:, 1]  # PAAC+XGBoost
    prob2 = model2.predict_proba(X_test_aac)[:, 1]  # AAC+XGBoost
    prob3 = model3.predict(X_test_progen_3d)
    if prob3.ndim > 1 and prob3.shape[1] == 1:
        prob3 = prob3.reshape(-1)
    # ProGen+BiLSTM

    ensemble_prob = (
            weights[0] * prob1 +
            weights[1] * prob2 +
            weights[2] * prob3
    )

    y_pred1 = (prob1 >= thresholds['paac']).astype(int)
    y_pred2 = (prob2 >= thresholds['aac']).astype(int)
    y_pred3 = (prob3 >= thresholds['progen']).astype(int)
    y_pred_ensemble = (ensemble_prob >= thresholds['ensemble']).astype(int)

    metrics = {
        'PAAC+XGBoost': {
            'Accuracy': accuracy_score(y_test, y_pred1),
            'Recall': recall_score(y_test, y_pred1),
            'Precision': precision_score(y_test, y_pred1, zero_division=1),
            'F1': f1_score(y_test, y_pred1),
            'MCC': matthews_corrcoef(y_test, y_pred1),
            'AUC': roc_auc_score(y_test, prob1)
        },
        'AAC+XGBoost': {
            'Accuracy': accuracy_score(y_test, y_pred2),
            'Recall': recall_score(y_test, y_pred2),
            'Precision': precision_score(y_test, y_pred2, zero_division=1),
            'F1': f1_score(y_test, y_pred2),
            'MCC': matthews_corrcoef(y_test, y_pred2),
            'AUC': roc_auc_score(y_test, prob2)
        },
        'ProGen+BiLSTM': {
            'Accuracy': accuracy_score(y_test, y_pred3),
            'Recall': recall_score(y_test, y_pred3),
            'Precision': precision_score(y_test, y_pred3, zero_division=1),
            'F1': f1_score(y_test, y_pred3),
            'MCC': matthews_corrcoef(y_test, y_pred3),
            'AUC': roc_auc_score(y_test, prob3)
        },
        'Ensemble': {
            'Accuracy': accuracy_score(y_test, y_pred_ensemble),
            'Recall': recall_score(y_test, y_pred_ensemble),
            'Precision': precision_score(y_test, y_pred_ensemble, zero_division=1),
            'F1': f1_score(y_test, y_pred_ensemble),
            'MCC': matthews_corrcoef(y_test, y_pred_ensemble),
            'AUC': roc_auc_score(y_test, ensemble_prob)
        }
    }

    save_results(y_test, prob1, prob2, prob3, ensemble_prob, y_pred_ensemble, weights, thresholds)

    return metrics


def save_results(y_test, prob1, prob2, prob3, ensemble_prob, y_pred_ensemble, weights, thresholds):
    """Save detailed prediction results to CSV file"""
    results = []
    for i in range(len(y_test)):
        result_item = {
            'Sample_ID': f"Seq_{i + 1}",
            'PAAC+XGB_Score': prob1[i],
            'AAC+XGB_Score': prob2[i],
            'ProGen+BiLSTM_Score': prob3[i],
            'Ensemble_Score': ensemble_prob[i],
            'Prediction': "Glycation Site" if y_pred_ensemble[i] == 1 else "Non-Glycation Site",
            'True_Label': "Glycation Site" if y_test[i] == 1 else "Non-Glycation Site",
            'Correct': y_pred_ensemble[i] == y_test[i]
        }
        results.append(result_item)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("glycation_ensemble_results.csv", index=False)
    print(f"Detailed prediction results saved to: glycation_ensemble_results.csv")

    # Save summary information
    summary = {
        'Total_Samples': len(y_test),
        'Positive_Samples': int(sum(y_test)),
        'Negative_Samples': int(len(y_test) - sum(y_test)),
        'Correct_Predictions': int(sum(y_pred_ensemble == y_test)),
        'PAAC_Weight': weights[0],
        'AAC_Weight': weights[1],
        'ProGen_Weight': weights[2],
        'PAAC_Threshold': thresholds['paac'],
        'AAC_Threshold': thresholds['aac'],
        'ProGen_Threshold': thresholds['progen'],
        'Ensemble_Threshold': thresholds['ensemble']
    }
    pd.DataFrame([summary]).to_csv("glycation_ensemble_summary.csv", index=False)
    print(f"Summary information saved to: glycation_ensemble_summary.csv")


def print_metrics(metrics):
    """Print evaluation metrics """
    if metrics is None:
        return

    print("\n" + "=" * 80)
    print(" " * 30 + "MODEL EVALUATION RESULTS")
    print("=" * 80)

    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name} Model:")
        print("-" * 50)
        for metric_name, metric_value in model_metrics.items():
            print(f"{metric_name:10}: {metric_value:.4f}")

    # Print performance improvement of ensemble over best single model
    ensemble_metrics = metrics['Ensemble']
    best_single_model = max(['PAAC+XGBoost', 'AAC+XGBoost', 'ProGen+BiLSTM'],
                            key=lambda x: metrics[x]['MCC'])
    best_single_metrics = metrics[best_single_model]

    print("\nEnsemble Model Performance Improvement:")
    print(f"Best Single Model: {best_single_model}, MCC: {best_single_metrics['MCC']:.4f}")
    print(f"Ensemble Model: MCC: {ensemble_metrics['MCC']:.4f}, " +
          f"Improvement: {ensemble_metrics['MCC'] - best_single_metrics['MCC']:.4f}")


def main():
    print("======== Lysine Glycation Site Prediction Ensemble Model Evaluation ========")

    # Load test data
    features, y_test = load_test_data()
    if features is None:
        print("Error: Failed to load test data. Exiting.")
        return

    # Load models
    models = load_models()
    if None in models:
        print("Error: Failed to load models. Exiting.")
        return

    print(f"Using configured weights: {MODEL_WEIGHTS}")
    print(f"Using configured thresholds: {THRESHOLDS}")

    # Evaluate ensemble model
    metrics = evaluate_ensemble(features, y_test, models)

    # Print evaluation results
    print_metrics(metrics)

    print("\n======== Evaluation Complete ========")


if __name__ == "__main__":
    main()