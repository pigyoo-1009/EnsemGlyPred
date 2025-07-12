from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, recall_score,
                            precision_score, f1_score, matthews_corrcoef)
from xgboost import XGBClassifier
import joblib
from collections import OrderedDict
import numpy as np
import random
from process_protein import protein
from read_file import read_fasta_file

random.seed(58)
np.random.seed(58)
RANDOM_SEED = 58

def aac_encode(seqs):
    """计算氨基酸组成(AAC)"""
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    num_aa = len(amino_acids)
    num_seqs = len(seqs)
    encoded = np.zeros((num_seqs, num_aa))
    for i, seq in enumerate(seqs):
        filtered_seq = ''.join(aa for aa in seq if aa != 'X')
        seq_length = max(1, len(filtered_seq))
        for j, aa in enumerate(amino_acids):
            encoded[i, j] = filtered_seq.count(aa) / seq_length
    return encoded

def cross_validate(X, y, model, threshold=0.46):
    """10折交叉验证,返回平均结果"""
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    metrics = {
        "Accuracy": [],
        "Recall": [],
        "Precision": [],
        "F1": [],
        "Mcc": []
    }
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["Recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["Precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["F1"].append(f1_score(y_test, y_pred, zero_division=0))
        metrics["Mcc"].append(matthews_corrcoef(y_test, y_pred))
    return {k: np.mean(v) for k, v in metrics.items()}

pos_seqs = read_fasta_file('trainPos.txt')
neg_seqs = read_fasta_file('trainNeg.txt')

processed_pos_seqs = []
for seq in pos_seqs:
    processed_seq = protein(seq)
    processed_pos_seqs.append(processed_seq)
processed_neg_seqs = []
for seq in neg_seqs:
    processed_seq = protein(seq)
    processed_neg_seqs.append(processed_seq)

all_processed_seqs = processed_pos_seqs + processed_neg_seqs

# 创建标签（正样本为1，负样本为0）
y = np.array([1] * len(processed_pos_seqs) + [0] * len(processed_neg_seqs))
X = aac_encode(all_processed_seqs)

# 设置XGBoost参数
params = OrderedDict({
    'colsample_bytree': 0.68,
    'gamma': 0.5,
    'learning_rate': 0.02,
    'max_depth': 6,
    'min_child_weight': 5,
    'n_estimators': 325,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.64
})
model = XGBClassifier(
    **params,
    objective='binary:logistic',
    random_state=RANDOM_SEED,
    eval_metric='logloss',
)

threshold = 0.46
results = cross_validate(X, y, model, threshold=threshold)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

model.fit(X, y)
joblib.dump(model, 'xgboost.pkl')

np.save('X_aac.npy', X)
np.save('y_aac.npy', y)
