import numpy as np
import copy as copy


def initialization(pop, ub, lb, dim):
    ''' 黏菌种群初始化函数'''
    '''
    pop:为种群数量
    dim:每个个体的维度
    ub:每个维度的变量上边界，维度为[dim,1]
    lb:为每个维度的变量下边界，维度为[dim,1]
    X:为输出的种群，维度[pop,dim]
    '''
    X = np.zeros([pop, dim])  # 声明空间
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]  # 生成[lb,ub]之间的随机数

    return X


def BorderCheck(X, ub, lb, pop, dim):
    '''边界检查函数'''
    '''
    dim:为每个个体数据的维度大小
    X:为输入数据，维度为[pop,dim]
    ub:为个体数据上边界，维度为[dim,1]
    lb:为个体数据下边界，维度为[dim,1]
    pop:为种群数量
    '''
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


def CaculateFitness(X, fun):
    '''计算种群的所有个体的适应度值'''
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


def SortFitness(Fit):
    '''适应度排序'''
    '''
    输入为适应度值
    输出为排序后的适应度值，和索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    '''根据适应度对位置进行排序'''
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def SMA(pop, dim, lb, ub, MaxIter, fun):
    '''黏菌优化算法'''
    '''
    输入：
    pop:为种群数量
    dim:每个个体的维度
    ub:为个体上边界信息，维度为[1,dim]
    lb:为个体下边界信息，维度为[1,dim]
    fun:为适应度函数接口
    MaxIter:为最大迭代次数
    输出：
    GbestScore:最优解对应的适应度值
    GbestPositon:最优解
    Curve:迭代曲线
    '''
    z = 0.03  # 位置更新参数
    X = initialization(pop, ub, lb, dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    W = np.zeros([pop, dim])  # 权重W矩阵
    for t in range(MaxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        S = bestFitness - worstFitness + 10E-8  # 当前最优适应度于最差适应度的差值，10E-8为极小值，避免分母为0；
        for i in range(pop):
            if i < pop / 2:  # 适应度排前一半的W计算
                W[i, :] = 1 + np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
            else:  # 适应度排后一半的W计算
                W[i, :] = 1 - np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
        # 惯性因子a,b
        tt = -(t / MaxIter) + 1
        if tt != -1 and tt != 1:
            a = np.math.atanh(tt)
        else:
            a = 1
        b = 1 - t / MaxIter
        # 位置更新
        for i in range(pop):
            if np.random.random() < z:
                X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T  # 公式（1.4）第一个式子
            else:
                p = np.tanh(abs(fitness[i] - GbestScore))
                vb = 2 * a * np.random.random([1, dim]) - a
                vc = 2 * b * np.random.random([1, dim]) - b
                for j in range(dim):
                    r = np.random.random()
                    A = np.random.randint(pop)
                    B = np.random.randint(pop)
                    if r < p:
                        X[i, j] = GbestPositon[j] + vb[0, j] * (W[i, j] * X[A, j] - X[B, j])  # 公式（1.4）第二个式子
                    else:
                        X[i, j] = vc[0, j] * X[i, j]  # 公式(1.4)第三个式子

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
