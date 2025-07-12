import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import joblib
import random
from process_protein import protein
from read_file import read_fasta_file

random.seed(58)
np.random.seed(58)

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


test_pos_seqs = read_fasta_file('testPos.txt')
test_neg_seqs = read_fasta_file('testNeg.txt')

processed_test_pos_seqs = []
for seq in test_pos_seqs:
    processed_seq = protein(seq)
    processed_test_pos_seqs.append(processed_seq)
processed_test_neg_seqs = []
for seq in test_neg_seqs:
    processed_seq = protein(seq)
    processed_test_neg_seqs.append(processed_seq)

all_processed_test_seqs = processed_test_pos_seqs + processed_test_neg_seqs

# 创建测试标签
y_test = np.array([1] * len(processed_test_pos_seqs) + [0] * len(processed_test_neg_seqs))
X_test = aac_encode(all_processed_test_seqs)
model = joblib.load('xgboost.pkl')
threshold = 0.46
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

np.save('X_test_aac.npy', X_test)
np.save('y_test_aac.npy', y_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mcc: {mcc:.4f}")