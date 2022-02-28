

import numpy as np



# 链接信息的关系矩阵
a = np.array([[0,1,1,0],
             [1,0,0,1],
             [1,0,0,1],
             [1,1,0,0]],dtype=float)
# print(a.shape[0]) #输出行数
# print(a.shape[1]) #输出列数
# print(a.shape) (4,4)
# c = np.zeros(a.shape)
# print(c)

# 构造转移矩阵
def transPre(data):
    b = np.transpose(data) #转置
    c = np.zeros(a.shape) #输出4*4全为0的矩阵
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            c[i][j] = data[i][j] / (b[j].sum())
    return c # 转移矩阵已经是转置过的矩阵

# print(transPre(a))

# PR初始化
def initiPre(c):
    pr = np.zeros((c.shape[0],1),dtype=float)
    for i in range(c.shape[0]):
        pr[i] = float(1) / c.shape[0]
    return pr

# print(initiPre(a))

def PageRank(q, A, pr):
    # q是阻尼系数0.85，A是转移矩阵，pr是初始化的pr
    # 迭代n次后，当影响因子不发生变化，迭代停止
    n = 1
    while (pr == q*np.dot(A, pr) + (1-q)*pr).all() == False:  # all()是元素里有空或0，则为false
        pr = q*np.dot(A, pr) + (1-q)*pr
        print('第{}次迭代，结果是{}'.format(n,(pr == q*np.dot(A, pr) + (1-q)*pr).all()))
        n += 1
    return pr


q = 0.85
AT = transPre(a)
pr = initiPre(AT)

print(PageRank(q,AT,pr))