'''黏菌优化算法求解压力容器设计'''
import numpy as np
from matplotlib import pyplot as plt
import SMA

'''适应度函数'''


def fun(X):
    x1 = X[0]  # Ts
    x2 = X[1]  # Th
    x3 = X[2]  # R
    x4 = X[3]  # L

    # 约束条件判断
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.math.pi * x3 ** 2 - 4 * np.math.pi * x3 ** 3 / 3 + 1296000
    g4 = x4 - 240
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
        # 如果满足约束条件则计算适应度值
        fitness = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3
    else:
        # 如果不满足约束条件，则设置适应度值为很大的一个惩罚数
        fitness = 10E32

    return fitness


'''主函数 '''
# 设置参数
pop = 50  # 种群数量
MaxIter = 500  # 最大迭代次数
dim = 4  # 维度
lb = np.array([0, 0, 10, 10])  # 下边界
ub = np.array([100, 100, 100, 100])  # 上边界
# 适应度函数选择
fobj = fun
GbestScore, GbestPositon, Curve = SMA.SMA(pop, dim, lb, ub, MaxIter, fobj)
print('最优适应度值：', GbestScore)
print('最优解[Ts,Th,R,L]：', GbestPositon)

# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('SMA', fontsize='large')
plt.show()
