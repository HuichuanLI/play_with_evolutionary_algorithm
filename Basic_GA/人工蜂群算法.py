#定义待优化函数：只能处理行向量形式的单个输入，若有矩阵形式的多个输入应当进行迭代
import os
import matplotlib.pyplot as plt
import numpy as np
def CostFunction(input):
    x = input[0]
    y = input[1]
    result=-20*np.exp(-0.2*np.sqrt((x*x+y*y)/2))-np.exp((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2)+20+np.exp(1)
    return result
#初始化各参数
#代价函数中参数数目和范围
nVar = 2
VarMin = -4
VarMax = 4
#蜂群算法基本参数
iter_max = 60
nPop = 100
nOnLooker = 100
L = np.around(0.6*nVar*nPop)
a = 1
#创建各记录矩阵
PopPosition = np.zeros([nPop,nVar])
PopCost = np.zeros([nPop,1])
Probability = np.zeros([nPop,1])
BestSol = np.zeros([iter_max+1,nVar])
BestCost = np.inf*np.ones([iter_max+1,1])
Mine = np.zeros([nPop,1])
#初始化蜜源位置
PopPosition = 8*np.random.rand(nPop,nVar) - 4
for i in range(nPop):
    PopCost[i][0] = CostFunction(PopPosition[i])
    if PopCost[i][0] <BestCost[0][0]:
        BestCost[0][0] = PopCost[i][0]
        BestSol[0] = PopPosition[i]
for iter in range(iter_max):
    #雇佣蜂阶段
    #寻找下一个蜜源
    for i in range(nPop):
        while True:
            k = np.random.randint(0,nPop)
            if k != i:
                break
        phi = a*(-1+2*np.random.rand(2))
        NewPosition = PopPosition[i] + phi*(PopPosition[i]-PopPosition[k])
        #进行贪婪选择
        NewCost = CostFunction(NewPosition)
        if NewCost < PopCost[i][0]:
            PopPosition[i] = NewPosition
            PopCost[i][0] = NewCost
        else:
            Mine[i][0] = Mine[i][0]+1
    #跟随蜂阶段
    #计算选择概率矩阵
    Mean = np.mean(PopCost)
    for i in range(nPop):
        Probability[i][0] = np.exp(-PopCost[i][0]/Mean)
    Probability = Probability/np.sum(Probability)
    CumProb = np.cumsum(Probability)
    for k in range(nOnLooker):
        #执行轮盘赌选择法
        m = 0
        for i in range(nPop):
            m = m + CumProb[i]
            if m >= np.random.rand(1):
                break
            #重复雇佣蜂操作
        while True:
            k = np.random.randint(0,nPop)
            if k != i:
                break
        phi = a*(-1+2*np.random.rand(2))
        NewPosition = PopPosition[i] + phi*(PopPosition[i]-PopPosition[k])
            #进行贪婪选择
        NewCost = CostFunction(NewPosition)
        if NewCost<PopCost[i][0]:
            PopPosition[i] = NewPosition
            PopCost[i][0] = NewCost
        else:
            Mine[i][0] = Mine[i][0]+1
        #侦查蜂阶段
        for i in range(nPop):
            if Mine[i][0] >= L:
                PopPosition[i] = 8*np.random.rand(1,nVar) - 4
                PopCost[i][0] = CostFunction(PopPosition[i])
                Mine[i][0] = 0
        #保存历史最优解
        for i in range(nPop):
            if PopCost[i][0] <BestCost[iter+1][0]:
                BestCost[iter+1][0] = PopCost[i][0]
                BestSol[iter+1] = PopPosition[i]
    #输出结果
y = np.zeros(iter_max+1)
print(BestSol[iter_max-1])
for i in range(iter_max):
    if i % 5 == 0:
        print(i,BestCost[i])
    y[i] = BestCost[i][0]
x = [i for i in range(iter_max+1)]
plt.plot(x,y)
plt.show()
