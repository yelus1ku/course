import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from munkres import Munkres
import time

EPOCHS = 100
NEW_DIMENSION = 60

class myKmeans:
    def __init__(self, init_data, K=2, init_method='kmeans++'):
        '''
        初始化 K-Means 模型
        init_data: 初始数据，二维数组
        K: 聚类数量
        init_method: 聚类中心的初始化方法，默认为随机
        '''
        self.K = K  # 聚类数量
        self.dimension = init_data.shape[1]  # 数据的特征维度
        self.centroids = None  # 用于存储聚类中心
        self.initCentroids(init_data, init_method)  # 初始化聚类中心


    def initCentroids(self, data, init_method):
        '''
        初始化聚类中心
        init_method: 初始化方法，可选 'random' 或 'kmeans++'
        '''
        if init_method == 'random':
            # 随机初始化
            indexes = np.arange(data.shape[0])  # 获取所有样本点的索引
            np.random.shuffle(indexes)  # 随机打乱索引
            self.centroids = np.zeros((self.K, data.shape[1]))  # 初始化聚类中心矩阵
            for i in range(self.K):
                self.centroids[i] = data[indexes[i]]  # 从随机顺序中选择前 K 个样本作为初始中心
        elif init_method == 'kmeans++':
            # K-Means++ 初始化
            self.centroids = []  # 初始化为空列表
            # 随机选择第一个聚类中心
            self.centroids.append(data[np.random.randint(data.shape[0])])
            # 选择剩余的聚类中心
            for _ in range(1, self.K):
                # 计算每个点到最近中心的距离
                distances = np.min(self.getDistanceToCenters(data, self.centroids), axis=1)
                # 概率分布：距离的平方
                probabilities = distances**2 / np.sum(distances**2)
                # 根据概率选择新的中心
                new_center = data[np.random.choice(data.shape[0], p=probabilities)]
                self.centroids.append(new_center)
            self.centroids = np.array(self.centroids)  # 转为 numpy 数组

    def getDistanceToCenters(self, data, centers):
        """
        计算样本到已选择中心的距离
        centers: 已选择的聚类中心
        return样本到中心的距离矩阵
        """
        distances = np.zeros((data.shape[0], len(centers)))  # 初始化距离矩阵
        for i, center in enumerate(centers):
            distances[:, i] = np.sum((data - center)**2, axis=1)**0.5  # 欧式距离计算
        return distances
    def getDistance(self, data):
        """
        计算每个样本点到所有聚类中心的距离
        数据集
        :return样本到每个聚类中心的距离矩阵
        """
        distances = np.zeros((data.shape[0], self.centroids.shape[0]))  # 初始化距离矩阵
        for i in range(data.shape[0]):
            distances[i] = np.sum((self.centroids - data[i])**2, axis=1)**0.5  # 欧式距离计算
        return distances


    def getClusters(self, data):
        """
        根据距离分配样本点到最近的聚类中心
        return分配的簇标签和平均距离
        """
        distances = self.getDistance(data)  # 计算所有样本到聚类中心的距离
        clusters = np.argmin(distances, axis=1)  # 找到每个样本最近的聚类中心
        avgDistances = np.sum(np.min(distances, axis=1)) / data.shape[0]  # 计算平均距离
        return clusters, avgDistances


    def getCentroids(self, data, clusters):
        """
        根据样本分配更新聚类中心
        clusters: 样本分配的簇标签
        return更新后的聚类中心
        """
        oneHotClusters = np.zeros((data.shape[0], self.K))  # 初始化 one-hot 矩阵
        oneHotClusters[np.arange(data.shape[0]), clusters] = 1  # 将样本标签转化为 one-hot 编码
        return np.dot(oneHotClusters.T, data) / np.sum(oneHotClusters, axis=0).reshape((-1, 1))  # 计算每个簇的均值

    def getAccuracy(self, predLabels, trueLabels):
        """
        计算聚类准确率
        predLabels: 预测的簇标签
        rueLabels: 实际的标签
        return聚类准确率
        """
        predLabelType = np.unique(predLabels)  # 预测标签的种类
        trueLabelType = np.unique(trueLabels)  # 真实标签的种类
        labelNum = np.maximum(len(predLabelType), len(trueLabelType))  # 取两者的最大值
        costMatrix = np.zeros((labelNum, labelNum))  # 初始化代价矩阵
        for i in range(len(predLabelType)):
            chosenPredLabels = (predLabels == predLabelType[i]).astype(float)
            for j in range(len(trueLabelType)):
                chosenTrueLabels = (trueLabels == trueLabelType[j]).astype(float)
                costMatrix[i, j] = -np.sum(chosenPredLabels * chosenTrueLabels)  # 计算代价矩阵
        m = Munkres()
        indexes = m.compute(costMatrix)  # 使用匈牙利算法找到最优匹配
        mappedPredLabels = np.zeros_like(predLabels, dtype=int)
        for index1, index2 in indexes:
            if index1 < len(predLabelType) and index2 < len(trueLabelType):
                mappedPredLabels[predLabels == predLabelType[index1]] = trueLabelType[index2]  # 映射预测标签
        return np.sum((mappedPredLabels == trueLabels).astype(float)) / trueLabels.size

    def train(self, data):
        """
        进行一次训练迭代
        return聚类中心的变化幅度
        """
        clusters, _ = self.getClusters(data)
        newCentroids = self.getCentroids(data, clusters)  # 更新聚类中心
        diff = np.sum((newCentroids - self.centroids)**2)**0.5  # 计算变化幅度
        self.centroids = newCentroids
        return diff

    def test(self, data, labels):
        """
        测试模型性能
        labels: 测试数据的真实标签
        return准确率和平均距离
        """
        clusters, avgDistance = self.getClusters(data)
        return self.getAccuracy(clusters, labels), avgDistance


class myGMM:
    def __init__(self, n_components, init_data, init_type='random', cov_type='full', reg_covar=1e-6, isUsedPi=False):
        """
        初始化 GMM 模型
        n_components: 高斯分量数目
        init_data: 用于初始化的数据
        init_type: 初始化类型(随机选择 'random' 或其他)
        cov_type: 协方差矩阵类型('full', 'diag', 'tied')
        reg_covar: 正则化项，防止协方差矩阵不可逆
        isUsedPi: 是否使用高斯分布中的常数项
        """
        self.n_components = n_components
        self.dimension = init_data.shape[1]
        self.weights = None
        self.means = None
        self.cov = None
        self.cov_type = cov_type  # 协方差矩阵类型
        self.reg_covar = reg_covar
        self.isUsedPi = isUsedPi
        self._init_parameters(init_type=init_type, cov_type=cov_type, init_data=init_data)

    def _init_parameters(self, init_data, init_type='random', cov_type='full'):
        indexes = np.arange(init_data.shape[0])
        np.random.shuffle(indexes)
        self.means = init_data[indexes[:self.n_components]]  # 随机选择均值
        self.weights = np.ones(self.n_components) / self.n_components  # 初始化权重

        # 根据 cov_type 初始化协方差矩阵
        tempCov = np.cov(init_data, rowvar=False) + np.eye(self.dimension) * self.reg_covar
        if cov_type == 'full':
            self.cov = tempCov[np.newaxis, :].repeat(self.n_components, axis=0)  # 每个分量独立的完整协方差矩阵
        elif cov_type == 'diag':
            self.cov = np.diag(tempCov).repeat(self.n_components, axis=0).reshape(self.n_components, -1)  # 对角矩阵
        elif cov_type == 'tied':
            self.cov = tempCov  # 所有分量共享一个协方差矩阵

    def gaussianFunc(self, x, mean, cov):
        """
        高斯分布概率密度函数计算
        """
        dev = x - mean
        if self.cov_type == 'diag':
            cov_matrix = np.diag(cov)
        elif self.cov_type == 'tied':
            cov_matrix = self.cov  # 确保是二维矩阵
        else:
            cov_matrix = cov

        maha = np.sum(np.dot(dev, np.linalg.pinv(cov_matrix)) * dev, axis=1)  # 马氏距离
        y = np.exp(-0.5 * maha) / np.sqrt(np.linalg.det(cov_matrix))  # 高斯概率密度
        if self.isUsedPi:
            return y / (2 * np.pi)**(self.dimension / 2)
        else:
            return y

    def EStep(self, data):
        gamma = np.zeros((data.shape[0], self.n_components))
        for k in range(self.n_components):
            gamma[:, k] = self.weights[k] * self.gaussianFunc(data, self.means[k], self.cov[k])
        gamma /= np.sum(gamma, axis=1).reshape(-1, 1)
        return gamma

    def MStep(self, data, gamma):
        """
        M 步，更新 GMM 模型参数
        gamma: 样本属于各个高斯分量的责任值
        """
        self.means = np.dot(gamma.T, data) / np.sum(gamma, axis=0).reshape(-1, 1)  # 更新均值
        self.weights = np.sum(gamma, axis=0) / data.shape[0]  # 更新权重

        if self.cov_type == 'full':
            # 每个分量有独立的完整协方差矩阵
            for k in range(self.n_components):
                diff = data - self.means[k]
                self.cov[k] = np.dot(gamma[:, k] * diff.T, diff) / np.sum(gamma[:, k]) + np.eye(self.dimension) * self.reg_covar
        elif self.cov_type == 'diag':
            # 每个分量有独立的对角协方差矩阵
            for k in range(self.n_components):
                diff = data - self.means[k]
                self.cov[k] = np.sum(gamma[:, k][:, np.newaxis] * diff**2, axis=0) / np.sum(gamma[:, k]) + self.reg_covar
        elif self.cov_type == 'tied':
            # 所有分量共享一个完整协方差矩阵
            tied_cov = np.zeros((self.dimension, self.dimension))
            for k in range(self.n_components):
                diff = data - self.means[k]
                tied_cov += np.dot((gamma[:, k][:, np.newaxis] * diff).T, diff)
            self.cov = tied_cov / data.shape[0] + np.eye(self.dimension) * self.reg_covar

    def train(self, data):
        gamma = self.EStep(data)
        self.MStep(data, gamma)

    def test(self, data, labels):
        gamma = self.EStep(data)
        clusters = np.argmax(gamma, axis=1)
        return self.getAccuracy(clusters, labels), -np.sum(np.log(np.max(gamma, axis=1)))

    def getAccuracy(self, predLabels, trueLabels):
        # 获取预测标签和真实标签的种类
        predLabelType = np.unique(predLabels)
        trueLabelType = np.unique(trueLabels)
        
        # 确定代价矩阵的大小，取预测标签和真实标签种类数的较大值
        labelNum = np.maximum(len(predLabelType), len(trueLabelType))
        costMatrix = np.zeros((labelNum, labelNum))
        
        # 构造代价矩阵，表示预测标签和真实标签之间的匹配代价
        for i in range(len(predLabelType)):
            # 当前预测标签对应的样本掩码
            chosenPredLabels = (predLabels == predLabelType[i]).astype(float)
            for j in range(len(trueLabelType)):
                # 当前真实标签对应的样本掩码
                chosenTrueLabels = (trueLabels == trueLabelType[j]).astype(float)
                # 代价矩阵中存储负的交集样本数
                costMatrix[i, j] = -np.sum(chosenPredLabels * chosenTrueLabels)
        
        # 使用 Munkres（匈牙利算法）计算最优匹配
        m = Munkres()
        indexes = m.compute(costMatrix)
        
        # 根据最优匹配结果映射预测标签到真实标签
        mappedPredLabels = np.zeros_like(predLabels, dtype=int)
        for index1, index2 in indexes:
            if index1 < len(predLabelType) and index2 < len(trueLabelType):
                mappedPredLabels[predLabels == predLabelType[index1]] = trueLabelType[index2]
        
        # 计算准确率：映射后的预测标签与真实标签相等的样本比例
        return np.sum((mappedPredLabels == trueLabels).astype(float)) / trueLabels.size


def load_and_preprocess_data(train_path, test_path, new_dimension=NEW_DIMENSION):
    """
    加载并预处理数据，包括归一化和 PCA 降维
    train_path: 训练集路径
    test_path: 测试集路径
    new_dimension: 降维的目标维度
    return预处理后的训练集和测试集
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.iloc[:, 1:].values.astype(np.float32) / 255.0  # 像素归一化
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values.astype(np.float32) / 255.0
    y_test = test_data.iloc[:, 0].values
    pca = PCA(n_components=new_dimension)  # PCA 降维
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, y_train, X_test, y_test


def plot_metrics(epochs, diffs, train_accuracies, test_accuracies, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot diff
    plt.subplot(2, 1, 1)
    plt.plot(epochs, diffs, label='Diff')
    plt.xlabel('Epochs')
    plt.ylabel('Diff')
    plt.title(f"{title} - Diff")
    plt.legend()

    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f"{title} - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_metrics1(epochs, train_losses, train_accuracies, test_losses, test_accuracies, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    train_path = "E:/robotstudy/Assignment3/mnist_train.csv"
    test_path = "E:/robotstudy/Assignment3/mnist_test.csv"
    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_path, test_path)
    # K-Means
    np.random.seed(0)
    kmeans = myKmeans(init_data=X_train, K=10, init_method='random')
    print("Training K-Means...")

    diffs = []
    train_accuracies = []
    test_accuracies = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        diff = kmeans.train(X_train)
        train_acc, _ = kmeans.test(X_train, y_train)
        test_acc, _ = kmeans.test(X_test, y_test)

        diffs.append(diff)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        print(f"  Diff: {diff:.6f}")
        print(f"  Train - Acc: {train_acc:.4f}")
        print(f"  Test  - Acc: {test_acc:.4f}")

        if diff < 1e-6:
            break

    end_time = time.time()
    print(f"K-Means Training Completed in {end_time - start_time:.2f} seconds.")

    # 绘制 K-Means 的 diff 曲线
    plot_metrics(
        epochs=list(range(1, len(diffs) + 1)),
        diffs=diffs,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        title="K-Means Diff and Accuracies"
    )
    # GMM
    np.random.seed(0)
    gmm = myGMM(n_components=10, init_data=X_train, init_type='kmeans', cov_type='tied', isUsedPi=False)
    print("Training GMM...")
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    start_time = time.time()

    for epoch in range(EPOCHS):
        gmm.train(X_train)
        train_acc, train_loss = gmm.test(X_train, y_train)
        test_acc, test_loss = gmm.test(X_test, y_test)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    end_time = time.time()
    print(f"GMM Training Completed in {end_time - start_time:.2f} seconds.")

    # 绘制 GMM 的训练曲线
    plot_metrics1(
        epochs=list(range(1, len(train_losses) + 1)),
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        title="GMM"
    )
    

if __name__ == "__main__":
    main()
