import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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
num_features_to_keep = 100 # 设置要保留的特征数量
selected_features_indices = np.argsort(feature_importances)[::-1][:num_features_to_keep]
selected_features = X.columns[selected_features_indices]
print(selected_features)

# 根据选择的特征子集重新组织输入数据
X_train_selected = X_train[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]
print(X_train_selected)
print(np.shape(X_train_selected))
print(X_test_selected)
print(np.shape(X_test_selected))

# 在保留的特征子集上重新训练模型
model_selected_features = xgb.XGBClassifier()
model_selected_features.fit(X_train_selected, y_train)

# 在测试集上评估模型
accuracy = model_selected_features.score(X_test_selected, y_test)
print("Accuracy with selected features:", accuracy)

