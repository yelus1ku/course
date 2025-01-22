import pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 数据加载函数
def load_data(dir):
    import pickle
    import numpy as np
    X_train = []
    Y_train = []
    for i in range(1, 6):
        with open(dir + r'/data_batch_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        X_train.append(dict[b'data'])
        Y_train += dict[b'labels']
    X_train = np.concatenate(X_train, axis=0)
    with open(dir + r'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X_test = dict[b'data']
    Y_test = dict[b'labels']

    return X_train, Y_train, X_test, Y_test

# 加载数据
data_dir = r"E:/robotstudy/Assignment2/material\data"
X_train, Y_train, X_test, Y_test = load_data(data_dir)

# 数据归一化和重塑
X_train = X_train.reshape(-1, 3, 32, 32) / 255.0  # 归一化到 [0, 1]
X_test = X_test.reshape(-1, 3, 32, 32) / 255.0

# 转换为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型定义
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i-1], hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # Conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                # Pool1
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),# Conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                # Pool2
            #新增一层
            # #------------------
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Conv3
            #-----
            nn.ReLU()
            #------------------------
        )
        self.fc_input_dim = None
        self.fc_layers = None
        self.num_classes = num_classes

    def _initialize_fc_layers(self, x_shape):
        self.fc_input_dim = x_shape[1] * x_shape[2] * x_shape[3]
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        if self.fc_layers is None:
            self._initialize_fc_layers(x.shape)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# 训练和测试函数
def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs):
    train_losses, train_accuracies = [], []
    test_accuracies = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()               # 梯度清零
            outputs = model(X_batch)            # 前向传播
            loss = criterion(outputs, Y_batch)  # 计算损失
            loss.backward()                     # 反向传播
            optimizer.step()                    # 更新参数

            total_loss += loss.item()           # 累计损失
            total_correct += (outputs.argmax(dim=1) == Y_batch).sum().item()     # 计算正确样本数

        train_losses.append(total_loss / len(train_loader))                      # 记录平均训练损失
        train_accuracies.append(total_correct / len(train_loader.dataset))       # 记录训练集准确率

        # 测试阶段
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                outputs = model(X_batch)
                total_correct += (outputs.argmax(dim=1) == Y_batch).sum().item()
        test_accuracies.append(total_correct / len(test_loader.dataset))

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, "
              f"Train Accuracy: {train_accuracies[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    return train_losses, train_accuracies, test_accuracies

# 配置实验
models = {
    "Softmax": SoftmaxClassifier(3 * 32 * 32, 10),
    "MLP": MLP(3 * 32 * 32, [256,128], 10),
    "CNN": CNN(10)
}
optimizers = {
    "SGD": lambda model: optim.SGD(model.parameters(), lr=0.01),
    "SGD_Momentum": lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    "Adam": lambda model: optim.Adam(model.parameters(), lr=0.01)
}
criterion = nn.CrossEntropyLoss()
num_epochs = 80

# 运行实验并绘图
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    optimizer = optimizers["Adam"](model)
    losses, train_acc, test_acc = train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs)
    results[model_name] = (losses, train_acc, test_acc)

# 绘制损失曲线和准确率
for model_name, (losses, train_acc, test_acc) in results.items():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label=f"{model_name} Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label=f"{model_name} Train Accuracy")
    plt.plot(test_acc, label=f"{model_name} Test Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
