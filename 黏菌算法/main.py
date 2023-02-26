import numpy as np
from matplotlib import pyplot as plt
import SMA

'''适应度函数'''


def fun(X):
    O = X[0] ** 2 + X[1] ** 2
    return O


'''黏菌优化算法求解x1^2 + x2^2的最小值'''

'''主函数 '''
# 设置参数
pop = 50  # 种群数量
MaxIter = 100  # 最大迭代次数
dim = 2  # 维度
lb = -10 * np.ones(dim)  # 下边界
ub = 10 * np.ones(dim)  # 上边界
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
