from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import random
import numpy as np
import torch
import tensorflow as tf

RANDOM_STATE = 58
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

PROGEN2_MODEL_PATH = "D:\\Python\\PPcharm\\progen2-model"
PROGEN2_TOKENIZER_PATH = "D:\\Python\\PPcharm\\progen2-tokenizer"
PCA_DIMS = 300
N_JOBS = -1


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
    return sequences


def protein(sequence):
    len_seq = len(sequence)
    k_index = sequence.find('K')

    if k_index == -1:
        raise ValueError("输入序列必须包含'K'(赖氨酸)。")
    start = k_index - 15
    end = k_index + 16

    left_pad = max(0, -start)
    right_pad = max(0, end - len_seq)
    start = max(0, start)
    end = min(len_seq, end)
    central_part = sequence[start:end]
    padded_sequence = ('X' * left_pad) + central_part + ('X' * right_pad)
    if len(padded_sequence) != 31 or padded_sequence[15] != 'K':
        raise ValueError("序列处理失败 - K没有正确居中。")
    return padded_sequence


def extract_progen2_features(sequences, model, tokenizer, max_length=31, batch_size=32):
    """
    从ProGen2模型提取序列特征
    """
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


def build_bilstm_model(input_shape, lstm_units=64, dropout_rate=0.5, learning_rate=0.001):
    """
    BiLSTM模型
    """
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


def cross_validate(X, y, params, n_splits=10):
    """
    执行交叉验证
    """
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = {"accuracy": [], "recall": [], "precision": [], "f1": [], "mcc": []}

    best_epochs_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X)):
        print(f"\n第 {fold + 1}/{n_splits} 折")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 构建模型
        model = build_bilstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=params['lstm_units'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_epochs_list.append(best_epoch)

        y_val_pred = (model.predict(X_val) >= 0.46).astype(int)
        metrics["accuracy"].append(accuracy_score(y_val, y_val_pred))
        metrics["recall"].append(recall_score(y_val, y_val_pred))
        metrics["precision"].append(precision_score(y_val, y_val_pred, zero_division=1))
        metrics["f1"].append(f1_score(y_val, y_val_pred))
        metrics["mcc"].append(matthews_corrcoef(y_val, y_val_pred))

        print(f"第 {fold + 1} 折评估指标:")
        print(f"Acc: {metrics['accuracy'][-1]:.4f}")
        print(f"Rec: {metrics['recall'][-1]:.4f}")
        print(f"Pre: {metrics['precision'][-1]:.4f}")
        print(f"F1-Score: {metrics['f1'][-1]:.4f}")
        print(f"MCC: {metrics['mcc'][-1]:.4f}")

    # 计算平均最佳轮次
    avg_best_epochs = int(np.mean(best_epochs_list))

    metrics_result = {k: np.mean(v) for k, v in metrics.items()}
    metrics_result["best_epochs"] = avg_best_epochs

    return metrics_result


# 主函数
def main():
    print("加载ProGen2模型...")
    progen2_tokenizer = AutoTokenizer.from_pretrained(PROGEN2_TOKENIZER_PATH, local_files_only=True,
                                                      trust_remote_code=True)
    progen2_model = AutoModelForCausalLM.from_pretrained(PROGEN2_MODEL_PATH, local_files_only=True,
                                                         trust_remote_code=True)
    progen2_model.eval()

    # 读取训练数据
    print("读取训练数据...")
    train_pos = read_fasta_file('trainPos.txt')
    train_neg = read_fasta_file('trainNeg.txt')

    # 处理序列
    print("处理序列...")
    processed_pos = []
    processed_neg = []

    for seq in train_pos:
        processed_pos.append(protein(seq))
    for seq in train_neg:
        processed_neg.append(protein(seq))

    # 提取或加载特征
    print("提取或加载特征...")
    try:
        progen2_pos = np.load('progen2_pos.npy')
        progen2_neg = np.load('progen2_neg.npy')
        print("已从缓存加载特征")
    except FileNotFoundError:
        print("从ProGen2模型提取特征...")
        progen2_pos = extract_progen2_features(processed_pos, progen2_model, progen2_tokenizer)
        progen2_neg = extract_progen2_features(processed_neg, progen2_model, progen2_tokenizer)
        np.save('progen2_pos.npy', progen2_pos)
        np.save('progen2_neg.npy', progen2_neg)

    X = np.concatenate([progen2_pos, progen2_neg])
    y = np.concatenate([np.ones(len(progen2_pos)), np.zeros(len(progen2_neg))])

    print("应用PCA降维...")
    pca = PCA(n_components=PCA_DIMS)
    X_pca = pca.fit_transform(X)
    np.save('X_progen_pca.npy', X_pca)

    X_reshaped = X_pca.reshape((X_pca.shape[0], 1, X_pca.shape[1]))

    # 设置参数
    best_params = {
        'lstm_units': 64,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32
    }

    print("开始交叉验证...")
    avg_metrics = cross_validate(X_reshaped, y, best_params, n_splits=10)
    print("\n交叉验证平均指标:")
    print(f"Acc: {avg_metrics['accuracy']:.4f}")
    print(f"Rec: {avg_metrics['recall']:.4f}")
    print(f"Pre: {avg_metrics['precision']:.4f}")
    print(f"F1-Score: {avg_metrics['f1']:.4f}")
    print(f"MCC: {avg_metrics['mcc']:.4f}")

    # 第一阶段：使用验证集和早停确定最佳轮次
    print("\n第一阶段：使用验证集和早停训练模型...")
    temp_model = build_bilstm_model(
        input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]),
        lstm_units=best_params['lstm_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = temp_model.fit(
        X_reshaped, y,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"第一阶段确定的最佳轮次: {best_epoch}")

    # 第二阶段：使用所有数据重新训练固定轮次
    print("\n第二阶段：使用所有数据训练最终模型...")
    final_model = build_bilstm_model(
        input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]),
        lstm_units=best_params['lstm_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )

    final_model.fit(
        X_reshaped, y,
        epochs=best_epoch,
        batch_size=best_params['batch_size'],
        verbose=1
    )

    print("保存最终模型...")
    # 保存模型
    final_model.save('bilstm.h5')
    joblib.dump(pca, 'pca_model.pkl')
    print("完成！模型和PCA已保存。")

if __name__ == "__main__":
    main()