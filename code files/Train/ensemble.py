import random
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef)
from xgboost import XGBClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from read_file import *
from sklearn.metrics import roc_auc_score
SEED = 58
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)  # TensorFlow的随机种子
tf.config.experimental.enable_op_determinism()

# --------------------- 模型参数配置 ---------------------
# PAAC+XGBoost模型参数 (来自第一个模型的网格搜索最佳参数)
MODEL1_PARAMS = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.01,
    'max_depth': 7,
    'n_estimators': 200,
    'subsample': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss',
}

# AAC+XGBoost模型参数 (来自第二个模型的随机搜索最佳参数示例)
MODEL2_PARAMS = {
    'colsample_bytree': 0.9692763545078751,
    'learning_rate': 0.010778765841014329,
    'max_depth': 6,
    'n_estimators': 326,
    'subsample': 0.8087407548138583,
    'random_state': 42,
    'eval_metric': 'logloss',
}

# BiLSTM模型参数 (来自第三个模型的最佳参数)
def build_bilstm_model(input_shape):
    """构建BiLSTM模型"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --------------------- 数据加载 ---------------------
# 加载预处理好的特征矩阵
X_paac = np.load('X_paac.npy')  # PAAC特征 (n_samples, 25)
X_aac = np.load('X_aac.npy')  # AAC特征 (n_samples, 20)
X_progen = np.load('X_progen_pca.npy')  # ProGen2特征 (n_samples, 300)

# 创建标签 (假设正样本在前，负样本在后)
pos_seqs = read_fasta_file('trainPos.txt')
neg_seqs = read_fasta_file('trainNeg.txt')
y = np.concatenate([np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))])

# --------------------- 手动设置权重 ---------------------
# 手动设置权重值（权重和为1）
MANUAL_WEIGHTS = (0.58, 0.3, 0.12)# 分别对应 model1, model2, model3 的权重

# --------------------- 交叉验证流程 ---------------------
def weighted_ensemble_cross_validate():
    """执行加权集成模型的交叉验证"""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'mcc': [],
        'auc': []  # 新增AUC指标
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_paac, y)):
        print(f"\nFold {fold + 1}/10")

        # 划分训练集/验证集 (保持不变)
        X_train_paac, X_test_paac = X_paac[train_idx], X_paac[test_idx]
        X_train_aac, X_test_aac = X_aac[train_idx], X_aac[test_idx]
        X_train_progen = X_progen[train_idx].reshape(-1, 1, 300)
        X_test_progen = X_progen[test_idx].reshape(-1, 1, 300)
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练三个基模型 (保持不变)
        model1 = XGBClassifier(**MODEL1_PARAMS)
        model1.fit(X_train_paac, y_train)

        model2 = XGBClassifier(**MODEL2_PARAMS)
        model2.fit(X_train_aac, y_train)

        model3 = build_bilstm_model((1, 300))
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model3.fit(X_train_progen, y_train,
                   epochs=50,
                   batch_size=32,
                   validation_data=(X_test_progen, y_test),
                   callbacks=[early_stop],
                   verbose=0)

        # 加权集成预测 (保持不变)
        prob1 = model1.predict_proba(X_test_paac)[:, 1]
        prob2 = model2.predict_proba(X_test_aac)[:, 1]
        prob3 = model3.predict(X_test_progen).flatten()

        ensemble_prob = (
            MANUAL_WEIGHTS[0] * prob1 +
            MANUAL_WEIGHTS[1] * prob2 +
            MANUAL_WEIGHTS[2] * prob3
        )
        y_pred = (ensemble_prob >= 0.5).astype(int)

        # 计算指标 (增加AUC计算)
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=1))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['mcc'].append(matthews_corrcoef(y_test, y_pred))
        metrics['auc'].append(roc_auc_score(y_test, ensemble_prob))  # 新增AUC计算

        # 打印当前折结果 (增加AUC输出)
        print(f"Accuracy: {metrics['accuracy'][-1]:.4f}")
        print(f"Recall:    {metrics['recall'][-1]:.4f}")
        print(f"Precision: {metrics['precision'][-1]:.4f}")
        print(f"F1-score:  {metrics['f1'][-1]:.4f}")
        print(f"MCC:       {metrics['mcc'][-1]:.4f}")
        print(f"AUC:       {metrics['auc'][-1]:.4f}")  # 新增AUC输出

    # 计算平均指标 (增加AUC平均)
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print("\nAverage Cross-Validation Metrics:")
    print(f"Accuracy:  {avg_metrics['accuracy']:.4f}")
    print(f"Recall:    {avg_metrics['recall']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"F1-score:  {avg_metrics['f1']:.4f}")
    print(f"MCC:       {avg_metrics['mcc']:.4f}")
    print(f"AUC:       {avg_metrics['auc']:.4f}")  # 新增AUC平均输出

# --------------------- 最终模型训练 ---------------------
def train_final_models():
    """使用全部数据训练最终模型"""
    # 转换ProGen特征维度
    X_progen_3d = X_progen.reshape(-1, 1, 300)

    # 训练模型1
    model1 = XGBClassifier(**MODEL1_PARAMS)
    model1.fit(X_paac, y)

    # 训练模型2
    model2 = XGBClassifier(**MODEL2_PARAMS)
    model2.fit(X_aac, y)

    # 训练模型3
    model3 = build_bilstm_model((1, 300))
    model3.fit(X_progen_3d, y, epochs=50, batch_size=32, verbose=0)

    # 保存模型
    joblib.dump(model1, 'paac_xgb.pkl')
    joblib.dump(model2, 'aac_xgb.pkl')
    model3.save('progen2_bilstm.h5')
    print("Final models saved successfully.")

# --------------------- 主程序 ---------------------
if __name__ == "__main__":
    # 执行交叉验证
    print("Starting 10-fold cross-validation with manual weights...")
    weighted_ensemble_cross_validate()

    # 训练最终模型
    print("\nTraining final models on full dataset...")
    train_final_models()