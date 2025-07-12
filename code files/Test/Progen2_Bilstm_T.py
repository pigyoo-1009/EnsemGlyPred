import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef
)
import joblib
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import random
import tensorflow as tf
from process_protein import *
from read_file import *

RANDOM_STATE = 58
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
tf.random.set_seed(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

PROGEN2_MODEL_PATH = "D:\\Python\\PPcharm\\progen2-model"
PROGEN2_TOKENIZER_PATH = "D:\\Python\\PPcharm\\progen2-tokenizer"
PCA_DIMS = 300
MODEL_PATH = "bilstm.h5"
PCA_PATH = "pca_model.pkl"
TEST_POS_PATH = "testPos.txt"
TEST_NEG_PATH = "testNeg.txt"

def extract_progen2_features(sequences, model, tokenizer, max_length=31, batch_size=32):
    features = []
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    processed_sequences = sequences

    for i in range(0, len(processed_sequences), batch_size):
        batch_sequences = processed_sequences[i:i + batch_size]
        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).to("cpu")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            masks = []
            for seq in batch_sequences:
                mask = [1 if (c != 'X' and c != tokenizer.pad_token) else 0 for c in seq]
                masks.append(mask)

            masks = torch.tensor(masks, dtype=torch.float32).to("cpu")
            masks = masks.unsqueeze(-1)

            masked_hidden = hidden_states * masks
            valid_counts = masks.sum(dim=1)
            valid_counts[valid_counts == 0] = 1
            batch_features = masked_hidden.sum(dim=1) / valid_counts
            features.append(batch_features.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"已处理 {min(i + batch_size, len(sequences))}/{len(sequences)} 个序列")

    return np.concatenate(features, axis=0)

test_pos = read_fasta_file(TEST_POS_PATH)
test_neg = read_fasta_file(TEST_NEG_PATH)

processed_pos = []
processed_neg = []

for seq in test_pos:
    processed_pos.append(protein(seq))
for seq in test_neg:
    processed_neg.append(protein(seq))

try:
    progen2_pos = np.load('test_progen2_pos.npy')
    progen2_neg = np.load('test_progen2_neg.npy')
except FileNotFoundError:
    progen2_tokenizer = AutoTokenizer.from_pretrained(PROGEN2_TOKENIZER_PATH, local_files_only=True,
                                                     trust_remote_code=True)
    progen2_model = AutoModelForCausalLM.from_pretrained(PROGEN2_MODEL_PATH, local_files_only=True,
                                                        trust_remote_code=True)
    progen2_model.eval()

    progen2_pos = extract_progen2_features(processed_pos, progen2_model, progen2_tokenizer)
    progen2_neg = extract_progen2_features(processed_neg, progen2_model, progen2_tokenizer)

    # 保存特征到缓存
    np.save('test_progen2_pos.npy', progen2_pos)
    np.save('test_progen2_neg.npy', progen2_neg)

# 数据准备
X_test = np.concatenate([progen2_pos, progen2_neg])
y_test = np.concatenate([np.ones(len(progen2_pos)), np.zeros(len(progen2_neg))])
pca = joblib.load(PCA_PATH)
X_test_pca = pca.transform(X_test)

np.save('X_test_progen_pca.npy', X_test_pca)
np.save('y_test_progen.npy', y_test)
X_test_reshaped = X_test_pca.reshape((X_test_pca.shape[0], 1, X_test_pca.shape[1]))

model = load_model(MODEL_PATH)

# 预测
print("\n进行预测...")
y_pred_proba = model.predict(X_test_reshaped)
y_pred = (y_pred_proba >= 0.46).astype(int)

# 评估
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=1),
    "F1-Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

for name, value in metrics.items():
    print(f"{name:10}: {value:.4f}")