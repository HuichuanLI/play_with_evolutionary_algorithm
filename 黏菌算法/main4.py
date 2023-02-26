'''基于黏菌优化算法的拉压弹簧设计'''
import numpy as np
from matplotlib import pyplot as plt
import SMA

'''适应度函数'''


def fun(X):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    # 约束条件判断
    g1 = 1 - (x2 ** 3 * x3) / (71785 * x1 ** 4)
    g2 = (4 * x2 ** 2 - x1 * x2) / (12566 * (x2 * x1 ** 3 - x1 ** 4)) + 1 / (5108 * x1 ** 2) - 1
    g3 = 1 - (140.45 * x1) / (x2 ** 2 * x3)
    g4 = (x1 + x2) / 1.5 - 1
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
        # 如果满足约束条件则计算适应度值
        fitness = (x3 + 2) * x2 * x1 ** 2
    else:
        # 如果不满足约束条件，则设置适应度值为很大的一个惩罚数
        fitness = 10E32

    return fitness


'''主函数 '''
# 设置参数
pop = 30  # 种群数量
MaxIter = 100  # 最大迭代次数
dim = 3  # 维度
lb = np.array([0.05, 0.25, 2])  # 下边界
ub = np.array([2, 1.3, 15])  # 上边界
# 适应度函数选择
fobj = fun
GbestScore, GbestPositon, Curve = SMA.SMA(pop, dim, lb, ub, MaxIter, fobj)
print('最优适应度值：', GbestScore)
print('最优解[x1,x2,x3]：', GbestPositon)

# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('SMA', fontsize='large')
plt.show()
