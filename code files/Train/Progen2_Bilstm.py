import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# 配置参数
PROGEN2_MODEL_NAME = "hugohrban/progen2-base"
PCA_DIMS = 300
N_JOBS = -1  # 使用所有CPU核心

# 读取FASTA文件
def read_fasta_file(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()

    sequences = []
    sequence = ''
    for line in lines:
        if line.startswith('>'):
            if sequence:
                sequences.append(sequence)
                sequence = ''
        else:
            sequence += line.strip()
    if sequence:
        sequences.append(sequence)

    print(f"Loaded {len(sequences)} sequences from {fasta_file}")
    return sequences

# 处理蛋白质序列
def process_protein_sequence(sequence):
    len_seq = len(sequence)
    k_index = sequence.find('K')

    if k_index == -1 or len_seq <= 31:
        if len_seq < 31:
            sequence = 'X' * (31 - len_seq) + sequence
        return sequence[:31]

    start = max(0, k_index - 15)
    end = min(len_seq, k_index + 16)
    new_sequence = sequence[start:end]

    if len(new_sequence) < 31:
        new_sequence = 'X' * (31 - len(new_sequence)) + new_sequence

    return new_sequence[:31]

# 提取ProGen2特征（使用隐藏层）
def extract_progen2_features(sequences, model, tokenizer, max_length=31, batch_size=32):
    features = []
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).to("cpu")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏状态
            batch_features = hidden_states.mean(dim=1).cpu().numpy()
            features.append(batch_features)

        # 进度输出
        if (i // batch_size) % 10 == 0:
            print(f"Processed {min(i + batch_size, len(sequences))}/{len(sequences)} sequences")

    return np.concatenate(features, axis=0)

# 构建BiLSTM模型
def build_bilstm_model(input_shape, lstm_units=64, dropout_rate=0.5, learning_rate=0.001):
    """构建BiLSTM模型"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(lstm_units))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 交叉验证
def cross_validate(X, y, params, n_splits=10):
    """执行交叉验证"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = {"accuracy": [], "recall": [], "precision": [], "f1": [], "mcc": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 构建模型
        model = build_bilstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=params['lstm_units'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )

        # 训练模型
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        # 验证集评估
        y_val_pred = (model.predict(X_val) > 0.5).astype(int)
        metrics["accuracy"].append(accuracy_score(y_val, y_val_pred))
        metrics["recall"].append(recall_score(y_val, y_val_pred))
        metrics["precision"].append(precision_score(y_val, y_val_pred, zero_division=1))
        metrics["f1"].append(f1_score(y_val, y_val_pred))
        metrics["mcc"].append(matthews_corrcoef(y_val, y_val_pred))

        # 打印当前fold结果
        print(f"Fold {fold + 1} Metrics:")
        print(f"Accuracy: {metrics['accuracy'][-1]:.4f}")
        print(f"Recall: {metrics['recall'][-1]:.4f}")
        print(f"Precision: {metrics['precision'][-1]:.4f}")
        print(f"F1-Score: {metrics['f1'][-1]:.4f}")
        print(f"MCC: {metrics['mcc'][-1]:.4f}")

    return {k: np.mean(v) for k, v in metrics.items()}

# 加载ProGen2模型
print("Loading ProGen2 model...")
progen2_tokenizer = AutoTokenizer.from_pretrained(PROGEN2_MODEL_NAME, trust_remote_code=True)
progen2_model = AutoModelForCausalLM.from_pretrained(PROGEN2_MODEL_NAME, trust_remote_code=True)
progen2_model.eval()

# 数据预处理
print("\nProcessing training data...")
train_pos = read_fasta_file('trainPos.txt')
train_neg = read_fasta_file('trainNeg.txt')

processed_pos = [process_protein_sequence(seq) for seq in train_pos]
processed_neg = [process_protein_sequence(seq) for seq in train_neg]

# 示例输出
print("\nSample processed sequences:")
print(f"Positive example: {processed_pos[0][:15]}...{processed_pos[0][-15:]}")
print(f"Negative example: {processed_neg[0][:15]}...{processed_neg[0][-15:]}")

# 特征提取（带缓存）
try:
    progen2_pos = np.load('progen2_pos.npy')
    progen2_neg = np.load('progen2_neg.npy')
    print("\nLoaded cached ProGen2 features")
except FileNotFoundError:
    print("\nExtracting ProGen2 features...")
    progen2_pos = extract_progen2_features(processed_pos, progen2_model, progen2_tokenizer)
    progen2_neg = extract_progen2_features(processed_neg, progen2_model, progen2_tokenizer)
    np.save('progen2_pos.npy', progen2_pos)
    np.save('progen2_neg.npy', progen2_neg)
    print("ProGen2 features saved to disk")

# 数据准备
X = np.concatenate([progen2_pos, progen2_neg])
y = np.concatenate([np.ones(len(progen2_pos)), np.zeros(len(progen2_neg))])

# PCA降维
pca = PCA(n_components=PCA_DIMS)
X_pca = pca.fit_transform(X)

# 保存 PCA 降维后的特征
np.save('X_progen_pca.npy', X_pca)  # 保存 PCA 降维后的特征
print("PCA-transformed ProGen2 features saved to X_progen_pca.npy")

# 调整输入形状为3D (samples, timesteps, features)
X_reshaped = X_pca.reshape((X_pca.shape[0], 1, X_pca.shape[1]))

# 使用最佳参数
best_params = {
    'lstm_units': 64,
    'dropout_rate': 0.5,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32
}

# 执行交叉验证
print("\nPerforming 10-fold cross-validation with best parameters...")
avg_metrics = cross_validate(X_reshaped, y, best_params, n_splits=10)

# 输出平均结果
print("\nAverage cross-validation results:")
print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
print(f"Recall: {avg_metrics['recall']:.4f}")
print(f"Precision: {avg_metrics['precision']:.4f}")
print(f"F1-Score: {avg_metrics['f1']:.4f}")
print(f"MCC: {avg_metrics['mcc']:.4f}")

# 训练最终模型
print("\nTraining final model on full dataset...")
final_model = build_bilstm_model(
    input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]),
    lstm_units=best_params['lstm_units'],
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate']
)
final_model.fit(X_reshaped, y, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# 保存模型
final_model.save('final_bilstm_model.h5')
print("\nFinal model saved as final_bilstm_model.h5")

# 保存PCA模型
joblib.dump(pca, 'pca_model.pkl')
print("PCA model saved as pca_model.pkl")