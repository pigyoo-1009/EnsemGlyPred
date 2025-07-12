import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
import joblib
import random
from process_protein import *
from read_file import *

random.seed(58)
np.random.seed(58)
RANDOM_SEED = 58

def compute_paac(sequence, lambda_value=5, w=0.05):
    """计算单个序列的PAAC特征"""
    filtered_sequence = ''.join(aa for aa in sequence if aa != 'X')
    if len(filtered_sequence) == 0:
        raise ValueError("序列中没有有效的氨基酸残基")

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}
    num_aa = len(amino_acids)
    paac = np.zeros(num_aa + lambda_value)

    # 氨基酸组成部分
    for aa in filtered_sequence:
        if aa in aa_index:
            paac[aa_index[aa]] += 1
    paac[:num_aa] /= len(filtered_sequence)

    # 序列顺序部分
    valid_pairs_count = np.zeros(lambda_value)
    for j in range(1, lambda_value + 1):
        for i in range(len(filtered_sequence) - j):
            if filtered_sequence[i] in aa_index and filtered_sequence[i + j] in aa_index:
                paac[num_aa + j - 1] += (hydrophobicity[filtered_sequence[i]] - hydrophobicity[
                    filtered_sequence[i + j]]) ** 2
                valid_pairs_count[j - 1] += 1

    # 归一化处理
    for j in range(lambda_value):
        if valid_pairs_count[j] > 0:
            paac[num_aa + j] /= valid_pairs_count[j]

    # 加权合并
    paac[num_aa:] *= w
    return paac


def compute_paac_features(sequences, lambda_value=5, w=0.05):
    """计算所有序列的PAAC特征"""
    return np.array([compute_paac(seq, lambda_value, w) for seq in sequences])


hydrophobicity = {
    'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
    'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
    'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
    'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26
}


def cross_validate(X, y, model):
    """10折交叉验证，返回平均结果"""
    kf = KFold(n_splits=10, shuffle=True, random_state=58)
    metrics = {
        "accuracy": [],
        "recall": [],
        "precision": [],
        "f1": [],
        "mcc": []
    }

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.45).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=1))
        metrics["f1"].append(f1_score(y_test, y_pred))
        metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))

    return {k: np.mean(v) for k, v in metrics.items()}


if __name__ == "__main__":
    try:
        X = np.load('X_paac.npy')
        y = np.load('y_paac.npy')
    except FileNotFoundError:
        pos_seqs = read_fasta_file('trainPos.txt')
        neg_seqs = read_fasta_file('trainNeg.txt')
        X = np.concatenate([
            compute_paac_features([protein(seq) for seq in pos_seqs]),
            compute_paac_features([protein(seq) for seq in neg_seqs])
        ], axis=0)
        y = np.concatenate([np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))])
        np.save('X_paac.npy', X)
        np.save('y_paac.npy', y)

    params = {
        'colsample_bytree': 0.64,
        'gamma': 0.98,
        'learning_rate': 0.031,
        'max_depth': 7,
        'min_child_weight': 9,
        'n_estimators': 189,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'subsample': 0.6,
        'random_state': 58,
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**params)

    print("\n开始10折交叉验证...")
    results = cross_validate(X, y, model)
    print("\n评估结果：")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Recall:   {results['recall']:.4f}")
    print(f"Precision:{results['precision']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"MCC:     {results['mcc']:.4f}")

    model.fit(X, y)
    joblib.dump(model, 'xgboost.pkl')