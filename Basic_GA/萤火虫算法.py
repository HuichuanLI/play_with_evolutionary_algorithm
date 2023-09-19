import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import time
class FA:
    def __init__(self, D, N, Beta0,gama, alpha, T, bound):
        self.D = D          #问题维数
        self.N = N          #群体大小
        self.Beta0 = Beta0  #最大吸引度
        self.gama = gama    #光吸收系数
        self.alpha = alpha  #步长因子
        self.T = T
        self.X = (bound[1] - bound[0]) * np.random.random([N, D]) + bound[0]
        self.X_origin = copy.deepcopy(self.X)
        self.FitnessValue = np.zeros(N)
        for n in range(N):
            self.FitnessValue[n] = self.FitnessFunction(n)
    def DistanceBetweenIJ(self,i, j):
        return np.linalg.norm(self.X[i,:] - self.X[j,:])
    def BetaIJ(self,i, j):  # AttractionBetweenIJ
        return self.Beta0 * \
        np.math.exp(-self.gama * (self.DistanceBetweenIJ(i,j) ** 2))
    def update(self,i,j):
        self.X[i,:] = self.X[i,:] + \
        self.BetaIJ(i,j) * (self.X[j,:] - self.X[i,:]) + \
        self.alpha * (np.random.rand(self.D) - 0.5)
    def FitnessFunction(self,i):
        x_ = self.X[i,:]
        return np.linalg.norm(x_) ** 2
    def iterate(self):
        t = 0
        while t <self.T:
            for i in range(self.N):
                FFi = self.FitnessValue[i]
                for j in range(self.N):
                    FFj = self.FitnessValue[j]
                    if FFj<FFi:
                        self.update(i,j)
            self.FitnessValue[i] = self.FitnessFunction(i)
            FFi = self.FitnessValue[i]
            t += 1
    def find_min(self):
        v = np.min(self.FitnessValue)
        n = np.argmin(self.FitnessValue)
        return v,self.X[n,:]
    def plot(X_origin,X):
        fig_origin = plt.figure(0)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(X_origin[:, 0],X_origin[:, 1], c='r')
        plt.scatter(X[:, 0], X[:, 1], c='g')
        plt.show()
if __name__ == '__main__':
    t = np.zeros(10)
    value = np.zeros(10)
    for i in range(10):
        fa = FA(2,20,1,0.000001,0.97,100,[-100,100])
        time_start = time.time()
        fa.iterate()
        time_end = time.time()
        t[i] = time_end - time_start
        value[i],n = fa.find_min()
        plt.plot(fa.X_origin,fa.X)
    print("平均值：",np.average(value))
    print("最优值：",np.min(value))
    print("最差值：",np.max(value))
    print("平均时间：",np.average(t))
