import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef,
                             roc_auc_score)
import joblib
from read_file import read_fasta_file
from process_protein import protein
from scipy.stats import randint, uniform

# 设置随机种子
np.random.seed(42)


def aac_encode(seqs):
    """计算氨基酸组成（AAC）"""
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    num_aa = len(amino_acids)
    num_seqs = len(seqs)
    encoded = np.zeros((num_seqs, num_aa))

    for i, seq in enumerate(seqs):
        seq_length = max(1, len(seq))  # 避免除零
        for j, aa in enumerate(amino_acids):
            encoded[i, j] = seq.count(aa) / seq_length
    return encoded


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

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.shape == (1, 1):
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        elif cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError("Unexpected shape of confusion matrix.")

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        metrics["auc"].append(roc_auc_score(y_test, y_prob))
        try:
            metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))
        except:
            metrics["mcc"].append(0)

    return {k: np.mean(v) for k, v in metrics.items()}


# 读取数据
pos_seqs = read_fasta_file('trainPos.txt')
neg_seqs = read_fasta_file('trainNeg.txt')

# 处理序列
proc_pos = [protein(seq) for seq in pos_seqs]
proc_neg = [protein(seq) for seq in neg_seqs]

# 计算AAC特征
X_pos = aac_encode(proc_pos)
X_neg = aac_encode(proc_neg)

# 合并数据
X = np.concatenate([X_pos, X_neg], axis=0)
y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

# 保存 AAC 特征
np.save('X_aac.npy', X)  # 保存 AAC 特征矩阵
print("AAC features saved to X_aac.npy")

# 使用给定的最佳参数初始化 XGBoost 模型
best_params = {
    'colsample_bytree': 0.9692763545078751,
    'learning_rate': 0.010778765841014329,
    'max_depth': 6,
    'n_estimators': 326,
    'subsample': 0.8087407548138583
}

# 使用最佳参数初始化 XGBoost 模型
best_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')

# 交叉验证
results = cross_validate(X, y, best_model)

# 输出平均结果，包括AUC
print("\nAverage Cross-Validation Results:")
for metric, value in results.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# 使用所有数据重新训练模型
best_model.fit(X, y)

# 计算全数据集的AUC
y_prob = best_model.predict_proba(X)[:, 1]
full_auc = roc_auc_score(y, y_prob)
print(f"\nFull Dataset AUC: {full_auc:.4f}")

# 保存模型
joblib.dump(best_model, 'xgboost_aac.pkl')
print("Model saved as xgboost_aac.pkl")