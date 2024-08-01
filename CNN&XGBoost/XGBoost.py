import csv
import os
import scipy.io as sio
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# 数据集路径
data_dir = './Data_mat'
# 交叉验证进行次数
n_folds = 2

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


# 计算模型评价指标
def evaluate_model(y_true, y_pred, y_prob):
    # G-mean
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    gmean = np.sqrt(tp / (tp + fn) * tn / (tn + fp))
    # AUC
    auc = roc_auc_score(y_true, y_prob)
    # F1 Score
    f1 = f1_score(y_true, y_pred)
    return gmean, auc, f1

if __name__ == '__main__':
    results = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.mat'):
            file_path = os.path.join(data_dir, filename)
            X, y = load_data_from_file(file_path)

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_results = {'G-mean': [], 'AUC': [], 'F1 Score': []}

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # 训练模型
                model = XGBClassifier()
                model.fit(X_train, y_train)

                # 预测模型
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # 评估模型
                gmean, auc, f1 = evaluate_model(y_test, y_pred, y_prob)

                # 保存结果
                fold_results['G-mean'].append(gmean)
                fold_results['AUC'].append(auc)
                fold_results['F1 Score'].append(f1)

            # 计算平均得分
            avg_results = {metric: np.mean(fold_results[metric]) for metric in fold_results}
            avg_results['filename'] = filename
            results.append(avg_results)


    for result in results:
        print(f"数据集: {result['filename']}")
        print(f"G-mean: {result['G-mean']}")
        print(f"AUC: {result['AUC']}")
        print(f"F1 Score: {result['F1 Score']}")
        print('-----------------------------')

    headers = ['Dataset', 'G-mean', 'AUC', 'F1 Score']
    with open("XGBoost_evaluation_results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for result in results:
            row = [result['filename'], result['G-mean'], result['AUC'], result['F1 Score']]
            writer.writerow(row)

    print(f"结果已经保存到 XGBoost_evaluation_results.csv")
