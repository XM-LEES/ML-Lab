import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from KNNLib import KNN
from sklearn.neighbors import KNeighborsClassifier

import time

def LoadFrame(path):
    # 加载数据
    df = pd.read_csv(path)
    # 查看数据集行列数
    print("该数据集共有 {} 行 {} 列".format(df.shape[0], df.shape[1]))
    print("分类特征：", df.columns[0:12].tolist(), "\n分类标签：", df.columns[12])

    # 删除空值行
    df.dropna(inplace=True)
    print("删除空行后该数据集共有 {} 行 {} 列".format(df.shape[0], df.shape[1]))

    # 将酒种类特征字符串转为整数 酒质量等级设定分类标签
    df['type'] = df['type'].apply(lambda x: 1 if x == 'white' else 0)
    df['quality'] = df['quality'].apply(lambda x: 1 if x <= 4 else (2 if x <= 6 else 3))
    # df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

    X = np.array(df[df.columns[0:12]])
    Y = np.array(df.quality)
    return X, Y


def Normalisation(X, Y):
    # 数据处理
    # 数据打乱并划分训练集与测试集（3:1）
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    # 对属性进行归一化处理
    # 最小-最大归一化方法
    scalar = MinMaxScaler()
    X_train = scalar.fit_transform(X_train_raw)
    X_test = scalar.transform(X_test_raw)
    # Z-score标准化
    # scaler = StandardScaler().fit(X_train_raw)
    # X_train = scaler.transform(X_train_raw)
    # X_test = scaler.transform(X_test_raw)
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    X, Y = LoadFrame("../winequalityN.csv")
    X_train, X_test, Y_train, Y_test = Normalisation(X, Y)
    start1 = time.perf_counter() ##
    for k in range(1,9):
        knn = KNN(k=k)
        knn.LoadData(X_train, Y_train)
        predictlist = knn.Predict(X_test)
        end1 = time.perf_counter()
        print(metrics.classification_report(Y_test, predictlist))
        print(k, "自行实现测试准确率：", accuracy_score(Y_test, predictlist))
        print("耗时", end1 - start1)


        start2 = time.perf_counter() ##
        clf = KNeighborsClassifier(k)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        end2 = time.perf_counter()
        # accuracy = accuracy_score(Y_test, Y_pred)
        print(k, "调用库算法测试准确率：", accuracy_score(Y_test, Y_pred))
        print("耗时", end2 - start2)
        # print("测试集准确率：{}%".format(round(accuracy * 100, 2)))