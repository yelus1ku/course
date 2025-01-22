import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time  
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 读取训练和测试数据集
train_data = pd.read_csv("E:/robotstudy/mnist_01_train.csv")  # 读取训练集数据
test_data = pd.read_csv("E:/robotstudy/mnist_01_test.csv")  # 读取测试集数据

# 提取特征数据和标签
X_train = train_data.drop(columns=['label']).values  # 获取训练集特征数据（去除标签列）
print(X_train)
y_train = train_data['label'].values  # 获取训练集标签数据
X_test = test_data.drop(columns=['label']).values  # 获取测试集特征数据（去除标签列）
y_test = test_data['label'].values  # 获取测试集标签数据

# 特征标准化：使每个特征的均值为0，标准差为1
scaler = StandardScaler()  # 创建标准化对象
X_train = scaler.fit_transform(X_train)  # 对训练集特征进行标准化
X_test = scaler.transform(X_test)  # 对测试集特征进行标准化（使用训练集的标准化参数）

# --- 使用现成的SVM软件包训练模型 ---
# 使用线性核函数训练 SVM
start_time = time.time()  
svm_linear = SVC(C=1, kernel="linear", gamma="auto")  # 创建线性核的SVM模型
svm_linear.fit(X_train, y_train)  # 训练SVM模型
y_pred_linear = svm_linear.predict(X_test)  # 在测试集上进行预测
accuracy_linear = accuracy_score(y_test, y_pred_linear)  # 计算并输出模型的准确率
linear_train_time = time.time() - start_time  # 计算训练时长

#使用高斯核训练 SVM
start_time = time.time()  
svm_rbf = SVC(C=1, kernel="rbf", gamma="auto")  # 创建高斯核SVM模型
svm_rbf.fit(X_train, y_train)  # 上训练SVM模型
y_pred_rbf = svm_rbf.predict(X_test)  # 在测试集上进行预测
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)  # 计算并输出模型的准确率
rbf_train_time = time.time() - start_time  # 计算训练时长

# --- 手动实现线性分类模型（Hinge Loss 和 Cross-Entropy Loss） ---

# Hinge Loss 损失函数（SVM模型）
def hinge_loss(W, b, X, y, reg=0.1):
    # 计算 SVM 模型的损失（Hinge Loss）
    margin = y * (np.dot(X, W) + b)  # margin 是每个样本点的 margin，y 是标签，W 是权重，b 是偏置项
    # 计算 hinge loss 和正则化项，reg 是正则化强度
    loss = np.mean(np.maximum(0, 1 - margin)) + 0.5 * reg * np.sum(W ** 2)
    return loss

# 使用梯度下降法优化（Hinge Loss）
def svm_gradient_descent(X, y, X_test, y_test, learning_rate=0.01, epochs=200, reg=0.5):
    # 获取样本数量和特征数量
    n_samples, n_features = X.shape
    # 初始化权重和偏置
    W = np.zeros(n_features)
    b = 0
    loss_history = []  # 存储每个epoch的损失值
    accuracy_history = []  # 存储每个epoch的准确率
    
    # 进行梯度下降
    for epoch in range(epochs):
        margin = y * (np.dot(X, W) + b)  # 计算每个样本的 margin
        dw = np.zeros_like(W)  
        db = 0  
        # 对每个样本进行处理，计算梯度
        for i in range(n_samples):
            if margin[i] < 1:  # 如果 margin 小于1，表示该样本被误分类
                dw -= y[i] * X[i]  # 权重梯度
                db -= y[i]  # 偏置项梯度
        
        dw /= n_samples  # 平均梯度
        db /= n_samples  # 平均偏置项梯度
        dw += reg * W  # 添加正则化项的梯度
        W -= learning_rate * dw  # 更新权重
        b -= learning_rate * db  # 更新偏置项
        
        # 计算当前epoch的损失和准确率，并记录下来
        loss_history.append(hinge_loss(W, b, X, y, reg))
        
        # 在测试集上进行预测并计算准确率
        y_pred = np.sign(np.dot(X_test, W) + b)  # 使用 sign 函数获取类别预测
        accuracy = accuracy_score(y_test, np.where(y_pred == -1, 0, 1))  # 计算准确率，注意需要将 -1 转换为 0
        accuracy_history.append(accuracy)
        
    return W, b, loss_history, accuracy_history

# 训练集标签转为 -1 和 1
y_train_hinge = np.where(y_train == 0, -1, 1)

start_time = time.time()  
# 使用梯度下降法训练 SVM 模型
W_hinge, b_hinge, hinge_loss_history, hinge_accuracy_history = svm_gradient_descent(X_train, y_train_hinge, X_test, y_test)
hinge_train_time = time.time() - start_time  # 计算训练时长

# Cross-Entropy Loss 损失函数（逻辑回归）
def cross_entropy_loss(W, b, X, y, reg=0.1):
    # 计算逻辑回归模型的损失（Cross-Entropy Loss）
    n_samples = X.shape[0]
    linear_output = np.dot(X, W) + b  # 计算每个样本的线性输出
    predictions = 1 / (1 + np.exp(-linear_output))  # 使用 Sigmoid 函数计算预测的概率
    # 计算 cross-entropy 损失和正则化项，reg 是正则化强度
    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) + 0.5 * reg * np.sum(W ** 2)
    return loss

# 使用梯度下降法优化（Cross-Entropy Loss）
def cross_entropy_gradient_descent(X, y, X_test, y_test, learning_rate=0.01, epochs=200, reg=0.5):
    # 获取样本数量和特征数量
    n_samples, n_features = X.shape
    # 初始化权重和偏置
    W = np.zeros(n_features)
    b = 0
    loss_history = []  # 存储每个epoch的损失值
    accuracy_history = []  # 存储每个epoch的准确率
    
    # 进行梯度下降
    for epoch in range(epochs):
        linear_output = np.dot(X, W) + b  # 计算每个样本的线性输出
        predictions = 1 / (1 + np.exp(-linear_output))  # 通过 Sigmoid 函数得到预测的概率
        dw = np.dot(X.T, (predictions - y)) / n_samples + reg * W  # 权重梯度
        db = np.mean(predictions - y)  # 偏置项梯度
        
        W -= learning_rate * dw  # 更新权重
        b -= learning_rate * db  # 更新偏置项
        
        # 计算当前epoch的损失和准确率，并记录下来
        loss_history.append(cross_entropy_loss(W, b, X, y, reg))
        
        # 在测试集上进行预测并计算准确率
        y_pred = np.round(1 / (1 + np.exp(-(np.dot(X_test, W) + b))))  # 使用 Sigmoid 函数得到预测概率，并将概率大于0.5的样本预测为1
        accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
        accuracy_history.append(accuracy)
        
    return W, b, loss_history, accuracy_history

# 训练集标签转为 0 和 1
y_train_ce = np.where(y_train == 0, 0, 1)

start_time = time.time()
# 使用梯度下降法训练逻辑回归模型
W_ce, b_ce, ce_loss_history, ce_accuracy_history = cross_entropy_gradient_descent(X_train, y_train_ce, X_test, y_test)
ce_train_time = time.time() - start_time  

# 输出
print(f'线性核函数准确率: {accuracy_linear:.4f} (Training time: {linear_train_time:.4f} seconds)')
print(f'高斯核函数准确率: {accuracy_rbf:.4f} (Training time: {rbf_train_time:.4f} seconds)')
print(f'Hinge Loss Model Accuracy: {hinge_accuracy_history[-1]:.4f} (Training time: {hinge_train_time:.4f} seconds)')
print(f'Cross-Entropy Loss Model Accuracy: {ce_accuracy_history[-1]:.4f} (Training time: {ce_train_time:.4f} seconds)')

# --- 可视化损失和准确率 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Hinge Loss 的损失曲线
axes[0, 0].plot(hinge_loss_history, label='Hinge Loss', color='blue')
axes[0, 0].set_title('Hinge Loss')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Loss')
axes[0, 1].set_ylim(0, 1.0)  # 设置 y 轴范围为 0 到 1.0
axes[0, 1].set_yticks(np.arange(0, 1.1, 0.2))  # 设置 y 轴的间隔为 0.2
axes[0, 0].legend()

# Cross-Entropy Loss 的损失曲线
axes[0, 1].plot(ce_loss_history, label='Cross-Entropy Loss', color='green')
axes[0, 1].set_title('Cross-Entropy Loss')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_ylim(0, 1.0)  # 设置 y 轴范围为 0 到 1.0
axes[0, 1].set_yticks(np.arange(0, 1.1, 0.2))  # 设置 y 轴的间隔为 0.2
axes[0, 1].legend()

# Hinge Loss 的准确率曲线
axes[1, 0].plot(hinge_accuracy_history, label='Hinge Accuracy', color='blue')
axes[1, 0].set_title('Hinge Accuracy')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()

#  Cross-Entropy Loss 的准确率曲线
axes[1, 1].plot(ce_accuracy_history, label='Cross-Entropy Accuracy', color='green')
axes[1, 1].set_title('Cross-Entropy Accuracy')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
