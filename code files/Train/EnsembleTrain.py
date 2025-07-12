import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import random
import tensorflow as tf
import os
from xgboost import XGBClassifier
from collections import OrderedDict

RANDOM_STATE = 58
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.experimental.enable_op_determinism()

# PAAC+XGBoost model
MODEL1_PARAMS = OrderedDict({
    'colsample_bytree': 0.64,
    'gamma': 0.98,
    'learning_rate': 0.031,
    'max_depth': 7,
    'min_child_weight': 9,
    'n_estimators': 189,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.6,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
})

# AAC+XGBoost model parameters
MODEL2_PARAMS = OrderedDict({
    'colsample_bytree': 0.68,
    'gamma': 0.5,
    'learning_rate': 0.02,
    'max_depth': 6,
    'min_child_weight': 5,
    'n_estimators': 325,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.64,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
})

# Thresholds
PAAC_THRESHOLD = 0.45
AAC_THRESHOLD = 0.46
PROGEN_THRESHOLD = 0.46
ENSEMBLE_THRESHOLD = 0.46
MANUAL_WEIGHTS = (0.4, 0.33, 0.27)


def build_bilstm_model(input_shape, lstm_units=64, dropout_rate=0.5, learning_rate=0.001):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(lstm_units))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_and_process_data():
    X_paac = np.load('X_paac.npy')
    X_aac = np.load('X_aac.npy')
    X_progen = np.load('X_progen_pca.npy')
    y_paac = np.load('y_paac.npy')
    y_aac = np.load('y_aac.npy')

    if np.array_equal(y_paac, y_aac):
        y = y_paac
    else:
        raise ValueError("Labels in PAAC and AAC datasets don't match")

    return X_paac, X_aac, X_progen, y


def weighted_ensemble_cross_validate(X_paac, X_aac, X_progen, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    base_metrics = {
        'paac': {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'mcc': [], 'auc': []},
        'aac': {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'mcc': [], 'auc': []},
        'progen': {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'mcc': [], 'auc': []},
        'ensemble': {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'mcc': [], 'auc': []}
    }

    best_epochs_list = []

    progen_feature_dim = X_progen.shape[1]
    fold_indices = list(kf.split(X_paac))

    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        print(f"\n{'=' * 20} Fold {fold + 1}/10 {'=' * 20}")
        X_train_paac, X_test_paac = X_paac[train_idx], X_paac[test_idx]
        X_train_aac, X_test_aac = X_aac[train_idx], X_aac[test_idx]
        X_train_progen = X_progen[train_idx].reshape(len(train_idx), 1, progen_feature_dim)
        X_test_progen = X_progen[test_idx].reshape(len(test_idx), 1, progen_feature_dim)

        y_train, y_test = y[train_idx], y[test_idx]

        # Train PAAC+XGBoost model (Model 1)
        print("Training PAAC+XGBoost model...")
        model1 = XGBClassifier(**MODEL1_PARAMS)
        model1.fit(X_train_paac, y_train)

        # Train AAC+XGBoost model (Model 2)
        print("Training AAC+XGBoost model...")
        model2 = XGBClassifier(**MODEL2_PARAMS)
        model2.fit(X_train_aac, y_train)

        # Train ProGen2+BiLSTM model (Model 3)
        print("Training ProGen2+BiLSTM model...")
        model3 = build_bilstm_model(
            input_shape=(1, progen_feature_dim),
            lstm_units=64,
            dropout_rate=0.5,
            learning_rate=0.001
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model3.fit(
            X_train_progen,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_progen, y_test),
            callbacks=[early_stop],
            verbose=0
        )

        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_epochs_list.append(best_epoch)

        prob1 = model1.predict_proba(X_test_paac)[:, 1]
        prob2 = model2.predict_proba(X_test_aac)[:, 1]
        prob3 = model3.predict(X_test_progen).flatten()

        pred1 = (prob1 >= PAAC_THRESHOLD).astype(int)
        pred2 = (prob2 >= AAC_THRESHOLD).astype(int)
        pred3 = (prob3 >= PROGEN_THRESHOLD).astype(int)

        ensemble_prob = (
                MANUAL_WEIGHTS[0] * prob1 +
                MANUAL_WEIGHTS[1] * prob2 +
                MANUAL_WEIGHTS[2] * prob3
        )
        ensemble_pred = (ensemble_prob >= ENSEMBLE_THRESHOLD).astype(int)

        # Calculate metrics for PAAC+XGBoost model
        base_metrics['paac']['accuracy'].append(accuracy_score(y_test, pred1))
        base_metrics['paac']['recall'].append(recall_score(y_test, pred1, zero_division=0))
        base_metrics['paac']['precision'].append(precision_score(y_test, pred1, zero_division=1))
        base_metrics['paac']['f1'].append(f1_score(y_test, pred1))
        base_metrics['paac']['mcc'].append(matthews_corrcoef(y_test, pred1))
        base_metrics['paac']['auc'].append(roc_auc_score(y_test, prob1))

        # Calculate metrics for AAC+XGBoost model
        base_metrics['aac']['accuracy'].append(accuracy_score(y_test, pred2))
        base_metrics['aac']['recall'].append(recall_score(y_test, pred2, zero_division=0))
        base_metrics['aac']['precision'].append(precision_score(y_test, pred2, zero_division=1))
        base_metrics['aac']['f1'].append(f1_score(y_test, pred2))
        base_metrics['aac']['mcc'].append(matthews_corrcoef(y_test, pred2))
        base_metrics['aac']['auc'].append(roc_auc_score(y_test, prob2))

        # Calculate metrics for ProGen2+BiLSTM model
        base_metrics['progen']['accuracy'].append(accuracy_score(y_test, pred3))
        base_metrics['progen']['recall'].append(recall_score(y_test, pred3, zero_division=0))
        base_metrics['progen']['precision'].append(precision_score(y_test, pred3, zero_division=1))
        base_metrics['progen']['f1'].append(f1_score(y_test, pred3))
        base_metrics['progen']['mcc'].append(matthews_corrcoef(y_test, pred3))
        base_metrics['progen']['auc'].append(roc_auc_score(y_test, prob3))

        # Calculate metrics for ensemble model
        base_metrics['ensemble']['accuracy'].append(accuracy_score(y_test, ensemble_pred))
        base_metrics['ensemble']['recall'].append(recall_score(y_test, ensemble_pred, zero_division=0))
        base_metrics['ensemble']['precision'].append(precision_score(y_test, ensemble_pred, zero_division=1))
        base_metrics['ensemble']['f1'].append(f1_score(y_test, ensemble_pred))
        base_metrics['ensemble']['mcc'].append(matthews_corrcoef(y_test, ensemble_pred))
        base_metrics['ensemble']['auc'].append(roc_auc_score(y_test, ensemble_prob))

        # Print current fold results
        print(f"\n{'*' * 10} Fold {fold + 1} Results {'*' * 10}")

        print("\n1. PAAC+XGBoost model:")
        print(f"Accuracy: {base_metrics['paac']['accuracy'][-1]:.4f}")
        print(f"Recall:    {base_metrics['paac']['recall'][-1]:.4f}")
        print(f"Precision: {base_metrics['paac']['precision'][-1]:.4f}")
        print(f"F1-score:  {base_metrics['paac']['f1'][-1]:.4f}")
        print(f"MCC:       {base_metrics['paac']['mcc'][-1]:.4f}")
        print(f"AUC:       {base_metrics['paac']['auc'][-1]:.4f}")

        print("\n2. AAC+XGBoost model:")
        print(f"Accuracy: {base_metrics['aac']['accuracy'][-1]:.4f}")
        print(f"Recall:    {base_metrics['aac']['recall'][-1]:.4f}")
        print(f"Precision: {base_metrics['aac']['precision'][-1]:.4f}")
        print(f"F1-score:  {base_metrics['aac']['f1'][-1]:.4f}")
        print(f"MCC:       {base_metrics['aac']['mcc'][-1]:.4f}")
        print(f"AUC:       {base_metrics['aac']['auc'][-1]:.4f}")

        print("\n3. ProGen2+BiLSTM model:")
        print(f"Accuracy: {base_metrics['progen']['accuracy'][-1]:.4f}")
        print(f"Recall:    {base_metrics['progen']['recall'][-1]:.4f}")
        print(f"Precision: {base_metrics['progen']['precision'][-1]:.4f}")
        print(f"F1-score:  {base_metrics['progen']['f1'][-1]:.4f}")
        print(f"MCC:       {base_metrics['progen']['mcc'][-1]:.4f}")
        print(f"AUC:       {base_metrics['progen']['auc'][-1]:.4f}")

        print("\n4. Weighted Ensemble model:")
        print(f"Accuracy: {base_metrics['ensemble']['accuracy'][-1]:.4f}")
        print(f"Recall:    {base_metrics['ensemble']['recall'][-1]:.4f}")
        print(f"Precision: {base_metrics['ensemble']['precision'][-1]:.4f}")
        print(f"F1-score:  {base_metrics['ensemble']['f1'][-1]:.4f}")
        print(f"MCC:       {base_metrics['ensemble']['mcc'][-1]:.4f}")
        print(f"AUC:       {base_metrics['ensemble']['auc'][-1]:.4f}")

    # 计算平均最佳轮次
    avg_best_epochs = int(np.mean(best_epochs_list))
    print(f"\n交叉验证平均最佳轮次: {avg_best_epochs}")

    print("\n" + "=" * 50)
    print("10-fold Cross-Validation Average Results:")
    print("=" * 50)

    models = ['PAAC+XGBoost', 'AAC+XGBoost', 'ProGen2+BiLSTM', 'Weighted Ensemble']
    model_keys = ['paac', 'aac', 'progen', 'ensemble']

    for i, (model_name, model_key) in enumerate(zip(models, model_keys)):
        avg_metrics = {k: np.mean(v) for k, v in base_metrics[model_key].items()}
        print(f"\n{i + 1}. {model_name} Model Average Metrics:")
        print(f"Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"Recall:    {avg_metrics['recall']:.4f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"F1-score:  {avg_metrics['f1']:.4f}")
        print(f"MCC:       {avg_metrics['mcc']:.4f}")
        print(f"AUC:       {avg_metrics['auc']:.4f}")

    result_metrics = {
        'paac': {k: np.mean(v) for k, v in base_metrics['paac'].items()},
        'aac': {k: np.mean(v) for k, v in base_metrics['aac'].items()},
        'progen': {k: np.mean(v) for k, v in base_metrics['progen'].items()},
        'ensemble': {k: np.mean(v) for k, v in base_metrics['ensemble'].items()},
        'best_epochs': avg_best_epochs
    }

    return result_metrics


def train_final_models(X_paac, X_aac, X_progen, y, avg_metrics):
    """Train final models using all data"""
    # 获取ProGen特征维度
    progen_feature_dim = X_progen.shape[1]
    X_progen_3d = X_progen.reshape(-1, 1, progen_feature_dim)

    # 训练PAAC模型
    print("Training final PAAC+XGBoost model...")
    model1 = XGBClassifier(**MODEL1_PARAMS)
    model1.fit(X_paac, y)
    print("PAAC+XGBoost model training complete")

    # 训练AAC模型
    print("Training final AAC+XGBoost model...")
    model2 = XGBClassifier(**MODEL2_PARAMS)
    model2.fit(X_aac, y)
    print("AAC+XGBoost model training complete")

    # 第一阶段：使用验证集和早停确定最佳轮次
    print("\n第一阶段：使用验证集和早停训练模型...")
    temp_model = build_bilstm_model(
        input_shape=(1, progen_feature_dim),
        lstm_units=64,
        dropout_rate=0.5,
        learning_rate=0.001
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = temp_model.fit(
        X_progen_3d,
        y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"第一阶段确定的最佳轮次: {best_epoch}")

    # 第二阶段：使用所有数据重新训练固定轮次
    print("\n第二阶段：使用所有数据训练最终模型...")
    model3 = build_bilstm_model(
        input_shape=(1, progen_feature_dim),
        lstm_units=64,
        dropout_rate=0.5,
        learning_rate=0.001
    )

    model3.fit(
        X_progen_3d,
        y,
        epochs=best_epoch,
        batch_size=32,
        verbose=1
    )
    print("ProGen2+BiLSTM model training complete")

    # 保存模型
    print("Saving models...")
    joblib.dump(model1, 'paac_xgb.pkl')
    joblib.dump(model2, 'aac_xgb.pkl')
    model3.save('progen2_bilstm.h5')

    # 保存集成配置
    ensemble_info = {
        'thresholds': {
            'paac': PAAC_THRESHOLD,
            'aac': AAC_THRESHOLD,
            'progen': PROGEN_THRESHOLD,
            'ensemble': ENSEMBLE_THRESHOLD
        },
        'weights': MANUAL_WEIGHTS
    }
    joblib.dump(ensemble_info, 'ensemble_config.pkl')
    print("All models and configurations saved")


if __name__ == "__main__":
    print("Loading and processing data...")
    X_paac, X_aac, X_progen, y = load_and_process_data()
    print("Data loaded successfully")

    print("Starting 10-fold cross-validation...")
    avg_metrics = weighted_ensemble_cross_validate(X_paac, X_aac, X_progen, y)

    print("\nTraining final models on all data...")
    train_final_models(X_paac, X_aac, X_progen, y, avg_metrics)