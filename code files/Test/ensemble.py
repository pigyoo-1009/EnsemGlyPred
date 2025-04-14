import numpy as np
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef)
from xgboost import XGBClassifier
import joblib
from tensorflow.keras.models import load_model

# 设置手动权重（与训练时相同）
MANUAL_WEIGHTS =  (0.58, 0.3, 0.12)  # model1, model2, model3


def load_test_data():
    """加载测试数据和标签"""
    # 加载特征数据
    X_test_paac = np.load('X_test_paac.npy')  # PAAC特征 (n_samples, 25)
    X_test_aac = np.load('X_test_aac.npy')  # AAC特征 (n_samples, 20)
    X_test_progen = np.load('X_test_progen.npy')  # ProGen特征 (n_samples, 300)

    # 调整ProGen特征的维度用于BiLSTM
    X_test_progen = X_test_progen.reshape(-1, 1, 300)  # 调整为 (n_samples, 1, 300)

    # 创建测试标签
    # 假设正样本在前，负样本在后
    pos_seqs = read_fasta_file('testPos.txt')
    neg_seqs = read_fasta_file('testNeg.txt')
    y_test = np.concatenate([np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))])

    return (X_test_paac, X_test_aac, X_test_progen), y_test


def evaluate_ensemble():
    """评估集成模型"""
    # 加载测试数据
    (X_paac, X_aac, X_progen), y_test = load_test_data()

    # 加载保存的模型
    try:
        model1 = joblib.load('paac_xgb.pkl')  # PAAC+XGBoost
        model2 = joblib.load('aac_xgb.pkl')  # AAC+XGBoost
        model3 = load_model('progen2_bilstm.h5')  # ProGen+BiLSTM
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 获取各模型的预测概率
    try:
        prob1 = model1.predict_proba(X_paac)[:, 1]  # XGBoost模型输出概率
        prob2 = model2.predict_proba(X_aac)[:, 1]  # XGBoost模型输出概率
        prob3 = model3.predict(X_progen).flatten()  # BiLSTM模型输出概率
    except Exception as e:
        print(f"模型预测失败: {e}")
        return

    # 加权集成预测
    ensemble_prob = (
            MANUAL_WEIGHTS[0] * prob1 +
            MANUAL_WEIGHTS[1] * prob2 +
            MANUAL_WEIGHTS[2] * prob3
    )
    y_pred = (ensemble_prob >= 0.5).astype(int)  # 将概率转换为类别

    # 计算评估指标
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=1),
        'F1-score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

    # 打印结果
    print("Test Set Performance:")
    for name, value in metrics.items():
        print(f"{name:10}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    # 确保read_fasta_file函数可用
    from read_file import read_fasta_file

    # 执行评估
    test_metrics = evaluate_ensemble()