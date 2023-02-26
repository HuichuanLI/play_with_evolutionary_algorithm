'''基于黏菌优化算法的三杆桁架设计'''
import numpy as np
from matplotlib import pyplot as plt
import SMA

'''适应度函数'''


def fun(X):
    x1 = X[0]
    x2 = X[1]
    l = 100
    P = 2
    sigma = 2
    # 约束条件判断
    g1 = (np.sqrt(2) * x1 + x2) * P / (np.sqrt(2) * x1 ** 2 + 2 * x1 * x2) - sigma
    g2 = x2 * P / (np.sqrt(2) * x1 ** 2 + 2 * x1 * x2) - sigma
    g3 = P / (np.sqrt(2) * x2 + x1) - sigma
    if g1 <= 0 and g2 <= 0 and g3 <= 0:
        # 如果满足约束条件则计算适应度值
        fitness = (2 * np.sqrt(2) * x1 + x2) * l
    else:
        # 如果不满足约束条件，则设置适应度值为很大的一个惩罚数
        fitness = 10E32

    return fitness


'''主函数 '''
# 设置参数
pop = 30  # 种群数量
MaxIter = 100  # 最大迭代次数
dim = 2  # 维度
lb = np.array([0.001, 0.001])  # 下边界
ub = np.array([1, 1])  # 上边界
# 适应度函数选择
fobj = fun
GbestScore, GbestPositon, Curve = SMA.SMA(pop, dim, lb, ub, MaxIter, fobj)
print('最优适应度值：', GbestScore)
print('最优解[x1,x2]：', GbestPositon)

# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('SMA', fontsize='large')
plt.show()
