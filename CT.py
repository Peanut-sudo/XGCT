import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Dense, Flatten, Conv1D
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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
    x = Conv1D(filters=8, kernel_size=3, activation='selu', padding='same')(input)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=1, kernel_size=3, activation='selu', padding='same')(x)
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
file_path = 'feature2_selected_XGBT.xlsx'
xyz = pd.read_excel(file_path, skiprows=1)
X = xyz.iloc[:, 2:-1].values
y = xyz.iloc[:, -1].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define model
input_shape = X.shape[1:]
model = cnn1d_transformer_model(input_shape)

# Ten-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define the current time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Create a directory with the current time
result_dir = os.path.join(os.getcwd(), current_time)
os.makedirs(result_dir)

# Start total training time
total_start_time = datetime.now()

# Lists to store results and training times
all_accuracies = []
all_precisions = []
all_recalls = []
all_f1_scores = []
all_sensitivities = []
all_specificities = []
all_auc_scores = []
all_confusion_matrices = []
all_y_true = []
all_y_pred_prob = []

for fold, (train_index, val_index) in enumerate(kfold.split(X, y), 1):
    print(f"\nTraining and evaluating Fold {fold}/{kfold.get_n_splits()}:")

    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Train your model
    history = model.fit(X_train_fold, y_train_fold, epochs=30, batch_size=16, verbose=1, validation_data=(X_val_fold, y_val_fold))

    # Evaluate the model on the validation set
    y_pred_val = model.predict(X_val_fold)
    y_pred_val_binary = (y_pred_val > 0.5).astype(int)

    # Save true labels and predicted probabilities for ROC/AUC calculation
    all_y_true.extend(y_val_fold)
    all_y_pred_prob.extend(y_pred_val)

    # Compute performance metrics on validation set
    val_accuracy = accuracy_score(y_val_fold, y_pred_val_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val_fold, y_pred_val_binary, average='binary')
    sensitivity = recall
    tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred_val_binary).ravel()
    specificity = tn / (tn + fp)
    auc_score = roc_auc_score(y_val_fold, y_pred_val)

    # Append metrics for averaging
    all_accuracies.append(val_accuracy)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1)
    all_sensitivities.append(sensitivity)
    all_specificities.append(specificity)
    all_auc_scores.append(auc_score)
    all_confusion_matrices.append(confusion_matrix(y_val_fold, y_pred_val_binary))

# End total training time
total_end_time = datetime.now()
total_training_time = total_end_time - total_start_time

# Calculate average performance metrics
average_accuracy = np.mean(all_accuracies)
average_precision = np.mean(all_precisions)
average_recall = np.mean(all_recalls)
average_f1_score = np.mean(all_f1_scores)
average_sensitivity = np.mean(all_sensitivities)
average_specificity = np.mean(all_specificities)
average_auc_score = np.mean(all_auc_scores)

# Print overall performance metrics
print("\nOverall Performance Metrics:")
print(f'Average Accuracy: {average_accuracy}')
print(f'Average Precision: {average_precision}')
print(f'Average Recall (Sensitivity): {average_sensitivity}')
print(f'Average Specificity: {average_specificity}')
print(f'Average F1 Score: {average_f1_score}')
print(f'Average AUC Score: {average_auc_score}')
print('Summed Confusion Matrix:')
summed_confusion_matrix = np.sum(all_confusion_matrices, axis=0).astype(int)
print(summed_confusion_matrix)

# Print total training time
print("Total training time:", total_training_time)

# Plot summed confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(summed_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.title("Summed Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Convert lists to numpy arrays
all_y_true = np.array(all_y_true)
all_y_pred_prob = np.array(all_y_pred_prob)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred_prob)
roc_auc = roc_auc_score(all_y_true, all_y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='yellow', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print(f'Overall AUC: {roc_auc}')
print("Total training time:", total_training_time)
