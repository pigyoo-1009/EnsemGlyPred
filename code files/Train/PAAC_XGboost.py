import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef,
                             roc_auc_score)
import joblib
from process_protein import *
from read_file import *

# 设置随机种子
np.random.seed(42)


def compute_paac(sequence, lambda_value=5, w=0.05):
    """
    计算单个序列的PAAC特征
    :param sequence: 蛋白质序列
    :param lambda_value: 控制序列顺序信息的参数
    :param w: 权重参数
    :return: PAAC特征向量
    """
    # 20种标准氨基酸
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}
    num_aa = len(amino_acids)

    # 初始化PAAC特征向量
    paac = np.zeros(num_aa + lambda_value)

    # 计算氨基酸组成
    for aa in sequence:
        if aa in aa_index:
            paac[aa_index[aa]] += 1

    # 归一化氨基酸组成
    paac[:num_aa] /= len(sequence)

    # 计算序列顺序信息
    for j in range(1, lambda_value + 1):
        for i in range(len(sequence) - j):
            if sequence[i] in aa_index and sequence[i + j] in aa_index:
                paac[num_aa + j - 1] += (hydrophobicity[sequence[i]] - hydrophobicity[sequence[i + j]]) ** 2

    # 归一化序列顺序信息
    paac[num_aa:] /= (len(sequence) - lambda_value)

    # 加权合并
    paac[num_aa:] *= w

    return paac


def compute_paac_features(sequences, lambda_value=5, w=0.05):
    """
    计算所有序列的PAAC特征
    :param sequences: 蛋白质序列列表
    :param lambda_value: 控制序列顺序信息的参数
    :param w: 权重参数
    :return: PAAC特征矩阵
    """
    return np.array([compute_paac(seq, lambda_value, w) for seq in sequences])


# 定义疏水性值（可根据需要调整）
hydrophobicity = {
    'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
    'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
    'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
    'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26
}


def cross_validate(X, y, model):
    """10折交叉验证，返回平均结果"""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "recall": [], "precision": [], "f1": [], "mcc": [], "auc": []}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 获取预测概率以计算AUC
        y_prob = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=1))
        metrics["f1"].append(f1_score(y_test, y_pred))
        metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))
        metrics["auc"].append(roc_auc_score(y_test, y_prob))

    return {k: np.mean(v) for k, v in metrics.items()}


# 读取数据
pos_seqs = read_fasta_file('trainPos.txt')
neg_seqs = read_fasta_file('trainNeg.txt')

# 处理序列
proc_pos = [protein(seq) for seq in pos_seqs]
proc_neg = [protein(seq) for seq in neg_seqs]

# 计算PAAC特征
X_pos = compute_paac_features(proc_pos)
X_neg = compute_paac_features(proc_neg)

# 合并数据
X = np.concatenate([X_pos, X_neg], axis=0)
y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

# 保存 PAAC 特征
np.save('X_paac.npy', X)  # 保存 PAAC 特征矩阵
print("PAAC features saved to X_paac.npy")

# 使用已知的最佳参数
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.01,
    'max_depth': 7,
    'n_estimators': 200,
    'subsample': 0.8
}

print("Using best parameters: ", best_params)

# 使用最佳参数初始化XGBoost模型
best_model = XGBClassifier(**best_params, random_state=42)

# 交叉验证
results = cross_validate(X, y, best_model)

# 输出平均结果，包括AUC
print("Average Cross-Validation Results:")
for metric, value in results.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# 使用所有数据重新训练模型
best_model.fit(X, y)

# 计算全数据集的AUC
y_prob = best_model.predict_proba(X)[:, 1]
full_auc = roc_auc_score(y, y_prob)
print(f"\nFull Dataset AUC: {full_auc:.4f}")

# 保存模型
joblib.dump(best_model, 'xgb_paac.pkl')
print("XGBoost model saved as xgb_paac.pkl")