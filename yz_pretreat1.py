from Pretreatment_wzj1 import pretreat
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

path1 = r'C:\Users\Think\Documents\32 对刘富强和姜小刚温度校正文章图表的进一步复现\刘富强贡梨NIRS温度校正数据\EPO_GLSW适用数据集'
xn = np.load(os.path.join(path1,'XN.npy'))
yn = np.load(os.path.join(path1,'YN.npy'))#原始建模集
w = np.load(os.path.join(path1,'wavelength.npy'))#波长
xp = np.load(os.path.join(path1,'XP.npy'))
yp = np.load(os.path.join(path1,'YP.npy'))#原始测试集
x4 = np.load(os.path.join(path1,'X4.npy'))#温差矩阵光谱

print('原始建模集：',xn.shape,yn.shape)
"""验证标准预处理
xn_a = pretreat(xn,method='autoscaling')
yn_a = pretreat(yn,method='autoscaling')
print('预处理后的建模集：',xn_a.shape,yn_a.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行标准化预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w,xn_a.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_a.shape[0]),yn_a)
plt.show()
"""
"""
#验证均值中心化：
xn_c = pretreat(xn,method='center')
yn_c = pretreat(yn,method='center')
print('预处理后的建模集：',xn_c.shape,yn_c.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w,xn_c.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_c.shape[0]),yn_c)
plt.show()
"""
"""
#验证pareto预处理：
xn_p = pretreat(xn,method='pareto')
yn_p = pretreat(yn,method='pareto')
print('预处理后的建模集：',xn_p.shape,yn_p.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w,xn_p.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_p.shape[0]),yn_p)
plt.show()
"""
"""
#验证最小最大预处理方法：
xn_p = pretreat(xn,method='minmax')
yn_p = pretreat(yn,method='minmax')
print('预处理后的建模集：',xn_p.shape,yn_p.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w,xn_p.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_p.shape[0]),yn_p)
plt.show()
"""
"""
#验证msc多元散射校正预处理方法：
xn_p = pretreat(xn,method='msc')
yn_p = pretreat(yn,method='msc')
print('预处理后的建模集：',xn_p.shape,yn_p.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w,xn_p.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_p.shape[0]),yn_p)
plt.show()
"""
"""
#验证d1 一阶微分预处理方法：
xn_p = pretreat(xn,method='d1')
yn_p = pretreat(yn,method='d1')
print('预处理后的建模集：',xn_p.shape,yn_p.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w[1:],xn_p.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_p.shape[0]),yn_p)
plt.show()
"""
"""
#验证d2 二阶微分预处理：
xn_p = pretreat(xn,method='d2')
yn_p = pretreat(yn,method='d2')
print('预处理后的建模集：',xn_p.shape,yn_p.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w[2:],xn_p.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_p.shape[0]),yn_p)
plt.show()
"""
"""
#验证sg平滑滤波预处理方法：
xn_p = pretreat(xn,method='sg')
yn_p = pretreat(yn,method='sg')
print('预处理后的建模集：',xn_p.shape,yn_p.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w,xn_p.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_p.shape[0]),yn_p)
plt.show()
"""

#验证DT 趋势校正预处理方法：
xn_p = pretreat(xn,method='dt')
yn_p = pretreat(yn,method='dt')
print('预处理后的建模集：',xn_p.shape,yn_p.shape)
#绘图认识数据集的变化：
# 设置Matplotlib使用SimHei字体显示中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei' 等其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.title('建模集进行预处理')
plt.subplot(4,1,1)
plt.plot(w,xn.T)
plt.subplot(4,1,2)
plt.plot(w,xn_p.T)
plt.subplot(4,1,3)
plt.scatter(range(yn.shape[0]),yn)
plt.subplot(4,1,4)
plt.scatter(range(yn_p.shape[0]),yn_p)
plt.show()

