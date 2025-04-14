import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import joblib
from read_file import read_fasta_file
from process_protein import protein
import os

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

# 读取测试数据
test_pos_seqs = read_fasta_file('testPos.txt')
test_neg_seqs = read_fasta_file('testNeg.txt')

# 处理测试序列
proc_test_pos = [protein(seq) for seq in test_pos_seqs]
proc_test_neg = [protein(seq) for seq in test_neg_seqs]

# 检查是否已经存在 AAC 特征文件
if os.path.exists('X_test_aac.npy'):
    # 如果存在，直接加载
    X_test_aac = np.load('X_test_aac.npy')
    print("Loaded AAC features from X_test_aac.npy")
else:
    # 如果不存在，计算 AAC 特征并保存
    print("X_test_aac.npy not found. Generating AAC features...")
    X_test_pos = aac_encode(proc_test_pos)
    X_test_neg = aac_encode(proc_test_neg)
    X_test_aac = np.concatenate([X_test_pos, X_test_neg], axis=0)
    np.save('X_test_aac.npy', X_test_aac)
    print("AAC features saved to X_test_aac.npy")

# 合并测试数据
y_test = np.concatenate([np.ones(len(test_pos_seqs)), np.zeros(len(test_neg_seqs))])

# 加载训练好的 XGBoost 模型
model = joblib.load('xgboost.pkl')

# 进行预测
y_pred = model.predict(X_test_aac)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# 输出评估结果
print("Test Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# 输出混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nConfusion Matrix:")
print(cm)