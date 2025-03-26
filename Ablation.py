import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Nadam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load dataset
file_path = 'pd_selectedx_features.xlsx'
data = pd.read_excel(file_path, skiprows=1)
X = data.iloc[:, 2:-1].values
y = data.iloc[:, -1].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define the input shape
input_shape = X.shape[1:]

# CNN model
def cnn_only_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv1D(filters=8, kernel_size=3, activation='selu', padding='same')(input_layer)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=32, kernel_size=3, activation='selu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='selu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='selu')(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Transformer model
def transformer_encoder(inputs, key_dim, num_heads):
    dropout = 0.1
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads)(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Dense(key_dim, activation='softmax')(x)
    return x + res

def transformer_only_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = transformer_encoder(input_layer, key_dim=32, num_heads=4)
    x = transformer_encoder(x, key_dim=32, num_heads=4)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = Dense(128, activation='selu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='selu')(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Select model to train
# model = cnn_only_model(input_shape)  
# Change to transformer_only_model(input_shape) to use the Transformer model
model = transformer_only_model(input_shape)

# Define K-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_accuracies = []
all_sensitivities = []
all_specificities = []
all_f1_scores = []

for train, test in kfold.split(X, y):
    # Fit the model
    model.fit(X[train], y[train], epochs=10, batch_size=10, verbose=1)
    # Predict on test data
    y_pred = (model.predict(X[test]) > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y[test], y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y[test], y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y[test], y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Store results
    all_accuracies.append(accuracy)
    all_sensitivities.append(sensitivity)
    all_specificities.append(specificity)
    all_f1_scores.append(f1_score)
    
    print(f"Fold Accuracy: {accuracy:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, F1-Score: {f1_score:.2f}")

# Compute average of the metrics
average_accuracy = np.mean(all_accuracies)
average_sensitivity = np.mean(all_sensitivities)
average_specificity = np.mean(all_specificities)
average_f1_score = np.mean(all_f1_scores)

print(f"\nAverage Accuracy: {average_accuracy:.4f}")
print(f"Average Sensitivity: {average_sensitivity:.4f}")
print(f"Average Specificity: {average_specificity:.4f}")
print(f"Average F1-Score: {average_f1_score:.4f}")
