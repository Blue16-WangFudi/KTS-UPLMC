import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import csv


# 基本参数设置
directory_path = './data_mat' # 数据集路径
num_eporch = 100 # 训练轮次（10的时候效果不佳，100的时候感觉就很准了，按需调整）
num_folds = 2 # 进行几折交叉验证

def load_data_from_file(file_path):
    '''
    导入.mat数据集
    Data_mat中数据集格式：
    1.每一行为一条数据
    2.最后一列为标签（1或-1，二分类问题）
    3.除了最后一列，每一列均为一个特征（为浮点数，可能为0）
    :param file_path: 单个.mat数据集路径
    :return: features（特征）, labels（0，1标签）
    '''
    if file_path.endswith(".mat"):
        mat = sio.loadmat(file_path)
        key = list(mat.keys())[-1]
        data = mat[key]
        labels = data[:, -1]
        features = data[:, :-1]
        labels = np.where(labels == -1, 0, 1)
        return features, labels
    else:
        raise ValueError("The provided file is not a .mat file.")

# 为表格数据定义一个完全连接的神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# 定义一个具有交叉验证（cross-validation）的训练与评估函数
def train_and_evaluate(file_path):
    features, labels = load_data_from_file(file_path)

    # 将数据集特征的数据0替换成一个较小的数字，防止训练报错
    features = np.where(features == 0, 1e-6, features)

    # 数据预处理：正则化数据、将数据转为张量
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # 交叉验证
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    auc_scores = []
    f1_scores = []
    gmean_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(features)):
        train_features, test_features = features[train_index], features[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        model = SimpleNN(input_dim=train_features.shape[1]) # 初始化CNN网络
        criterion = nn.BCELoss() # 损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器

        # 进行多轮训练
        for epoch in range(num_eporch):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_features)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            # 打印训练过程数据
            if (epoch + 1) % 2 == 0:
                print(f"数据集: {file_path}, Fold: {fold+1}, Epoch: {epoch+1}, Loss: {loss.item():.4f}")

        # 进行模型评估
        model.eval()
        with torch.no_grad():
            outputs = model(test_features).numpy()
            predicted = (outputs >= 0.5).astype(int)
            test_labels = test_labels.numpy()

        auc = roc_auc_score(test_labels, outputs)
        f1 = f1_score(test_labels, predicted)
        tn, fp, fn, tp = confusion_matrix(test_labels, predicted).ravel()
        gmean = np.sqrt(tp / (tp + fn) * tn / (tn + fp))

        auc_scores.append(auc)
        f1_scores.append(f1)
        gmean_scores.append(gmean)

    # 计算平均得分
    avg_auc = np.mean(auc_scores)
    avg_f1 = np.mean(f1_scores)
    avg_gmean = np.mean(gmean_scores)

    return [file_path, avg_auc, avg_f1, avg_gmean]

if __name__ == '__main__':
    results = [["Dataset", "AUC", "F1", "G-Mean"]]
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".mat"):
            file_path = os.path.join(directory_path, file_name)
            result = train_and_evaluate(file_path)
            results.append(result)
    # 把结果保存到CSV中
    with open('CNN_evaluation_results_'+str(num_eporch)+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
    print(f"结果已经保存到 CNN_evaluation_results.csv")
