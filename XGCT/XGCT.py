import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Dense, Flatten, Conv1D
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from datetime import datetime


# Function to build CNN+Transformer model
def cnn1d_transformer_model(input_shape):
    input1 = keras.Input(shape=input_shape)
    x = CNN1D(input1)
    x = stack_block_transformer_spatial(1, x)
    x = Dense(128, activation='selu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='selu')(x)
    x = Dropout(0.1)(x)
    answer = Dense(1, activation='sigmoid')(x)
    model = keras.Model(input1, answer)
    opt = optimizers.Nadam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def CNN1D(input):
    input1 = input
    x = Conv1D(filters=8, kernel_size=3, activation='selu', padding='valid')(input1)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=1, kernel_size=3, activation='selu', padding='valid')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    return x

def stack_block_transformer_spatial(num_transformer_blocks, x):
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, 10 * 18, 4)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    return x

def transformer_encoder(inputs, key_dim, num_heads):
    dropout = 0.3
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads)(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Dense(key_dim, activation='softmax')(x)
    return x + res


# 加载数据集
file_path = 'pd_speech_features.xlsx'
data = pd.read_excel(file_path, skiprows=1)

# 分离特征和标签
X = data.iloc[:, 2:-1]
y = data.iloc[:, -1]

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用XGBoost训练模型并获取特征重要性
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
feature_importances = model.feature_importances_

# 根据特征重要性选择保留最重要的特征子集
num_features_to_keep = 96 # 设置要保留的特征数量
selected_features_indices = np.argsort(feature_importances)[::-1][:num_features_to_keep]
selected_features = X.columns[selected_features_indices]
print(selected_features) 