import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

# 创建示例数据
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 LassoCV 模型
lasso = LassoCV(cv=5)

# 训练模型
lasso.fit(X_train, y_train)

# 输出特征的系数
print("Coefficients:", lasso.coef_)

# 使用非零系数的特征进行预测
selected_features_train = X_train[:, lasso.coef_ != 0]
selected_features_test = X_test[:, lasso.coef_ != 0]

# 打印选择的特征的形状
print("Selected Features (Train):", selected_features_train.shape)
print("Selected Features (Test):", selected_features_test.shape)
