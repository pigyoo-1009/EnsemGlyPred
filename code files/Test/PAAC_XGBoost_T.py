from process_protein import *
from read_file import *
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef

model = joblib.load('xgboost.pkl')

def compute_paac(sequence, lambda_value=5, w=0.05):
    """
    计算单个序列的PAAC特征
    """
    filtered_sequence = ''.join(aa for aa in sequence if aa != 'X')
    if len(filtered_sequence) == 0:
        raise ValueError("序列中没有有效的氨基酸残基")

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_index = {aa: i for i, aa in enumerate(amino_acids)}
    num_aa = len(amino_acids)
    paac = np.zeros(num_aa + lambda_value)
    for aa in filtered_sequence:
        if aa in aa_index:
            paac[aa_index[aa]] += 1

    paac[:num_aa] /= len(filtered_sequence)

    valid_pairs_count = np.zeros(lambda_value)
    for j in range(1, lambda_value + 1):
        for i in range(len(filtered_sequence) - j):
            if filtered_sequence[i] in aa_index and filtered_sequence[i + j] in aa_index:
                paac[num_aa + j - 1] += (hydrophobicity[filtered_sequence[i]] - hydrophobicity[filtered_sequence[i + j]]) ** 2
                valid_pairs_count[j - 1] += 1

    for j in range(lambda_value):
        if valid_pairs_count[j] > 0:
            paac[num_aa + j] /= valid_pairs_count[j]

    # 加权合并
    paac[num_aa:] *= w
    return paac


def compute_paac_features(sequences, lambda_value=5, w=0.05):
    """
    计算所有序列的PAAC特征
    """
    return np.array([compute_paac(seq, lambda_value, w) for seq in sequences])

hydrophobicity = {
    'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
    'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
    'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
    'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26
}

test_pos_seqs = read_fasta_file('testPos.txt')
test_neg_seqs = read_fasta_file('testNeg.txt')

# 处理测试集序列
proc_test_pos = [protein(seq) for seq in test_pos_seqs]
proc_test_neg = [protein(seq) for seq in test_neg_seqs]

# 计算测试集的PAAC特征
X_test_pos = compute_paac_features(proc_test_pos)
X_test_neg = compute_paac_features(proc_test_neg)

# 合并测试集数据和标签
X_test = np.concatenate([X_test_pos, X_test_neg], axis=0)
y_test = np.concatenate([np.ones(len(X_test_pos)), np.zeros(len(X_test_neg))])

# 保存PAAC特征
np.save('X_test_paac.npy', X_test)
np.save('y_test_paac.npy', y_test)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.45).astype(int)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# 输出评估结果
print("Test Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")