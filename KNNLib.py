import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances  # 计算欧氏距离


class KNN:
    def __init__(self, k: int):
        self.k: int = k
        self.xtrain = None
        self.ytrain = None

    def LoadData(self, x, y):
        self.xtrain = x
        self.ytrain = y

    def ManhattanDistance(self, xtest):
        distances = []

        for row in range(len(self.xtrain)):
            distance = np.sum(np.abs(self.xtrain[row] - xtest))
            distances.append((self.xtrain[row], self.ytrain[row], distance))

        distances.sort(key=lambda x: x[2])
        # 截取k个邻居
        return distances[:self.k]

    def EuclideanDistance(self, xtest):
        distances = []

        for row in range(len(self.xtrain)):
            distance = np.sqrt(np.sum((self.xtrain[row] - xtest) ** 2))
            distances.append((self.xtrain[row], self.ytrain[row], distance))

        distances.sort(key=lambda x: x[2])
        # 截取k个邻居
        return distances[:self.k]


    # def EuclideanDistance(self, xtest):
    #     # 使用euclidean_distances计算欧氏距离
    #     distances = euclidean_distances(self.xtrain, [xtest])
    #
    #     # 将距离与对应的标签组成元组
    #     neighbors = [(self.xtrain[i], self.ytrain[i], distances[i][0]) for i in range(len(self.xtrain))]
    #
    #     # 根据距离排序
    #     neighbors.sort(key=lambda x: x[2])
    #
    #     # 截取k个邻居
    #     return neighbors[:self.k]


    def Predict(self, xtest):
        print(len(xtest))
        predictlist = []
        for i in range(len(xtest)):
            # neighbors = self.ManhattanDistance(xtest[i])
            neighbors = self.EuclideanDistance(xtest[i])

            counter = Counter(neighbor[1] for neighbor in neighbors)
            # print("共{}个邻居，值为{}的最多{}个".format(self.k, counter.most_common(1)[0][0], counter.most_common(1)[0][1]))
            prediction = counter.most_common(1)[0][0]
            predictlist.append(prediction)
        return predictlist