import os
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dropout, Dense, Flatten, Conv1D
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from datetime import datetime
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

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

# Load dataset
file_path = 'pd_speech_features.xlsx'
xyz = pd.read_excel(file_path, skiprows=1)
X = xyz.iloc[:, 2:-1].values
y = xyz.iloc[:, -1].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Define a function to create the Keras model
def create_model():
    input_shape = X_scaled.shape[1:]
    model = cnn1d_transformer_model(input_shape)
    return model

# Wrap the Keras model with KerasClassifier
estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size=32, verbose=1)

# Ten-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize RFECV with the wrapped Keras model
rfe = RFECV(estimator=estimator, step=1, cv=kfold, scoring='accuracy')

# 将输入数据调整为二维数组
X_reshaped = X_scaled.reshape(X_scaled.shape[0], -1)

# Fit RFECV to the training data
rfe.fit(X_reshaped, y)

# Get the selected features
selected_features_indices = rfe.support_
X_selected = X_scaled[:, selected_features_indices]

# Define the current time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Create a directory with the current time
result_dir = os.path.join(os.getcwd(), current_time)
os.makedirs(result_dir)

# Lists to store results
all_accuracies = []

for fold, (train_index, val_index) in enumerate(kfold.split(X_selected, y), 1):
    print(f"\nTraining and evaluating Fold {fold}/{kfold.get_n_splits()}:")
    X_train_fold, X_val_fold = X_selected[train_index], X_selected[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Train your model
    estimator.fit(X_train_fold, y_train_fold)

    # Evaluate the model on the validation set
    y_pred_val = estimator.predict(X_val_fold)
    val_accuracy = accuracy_score(y_val_fold, y_pred_val)

    # Save individual fold results
    fold_result_filename = os.path.join(result_dir, f"Fold_{fold}_Result.txt")
    with open(fold_result_filename, 'w') as f:
        f.write('Validation Accuracy: {}\n'.format(val_accuracy))
    
    # Append metrics for averaging
    all_accuracies.append(val_accuracy)

# Save overall performance metrics to a file
overall_result_filename = os.path.join(result_dir, "Overall_Result.txt")
with open(overall_result_filename, 'w') as f:
    f.write('Average Accuracy: {}\n'.format(np.mean(all_accuracies)))

print("\nOverall Performance Metrics:")
print(f'Average Accuracy: {np.mean(all_accuracies)}')
