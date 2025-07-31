#不是类，直接使用函数pretreat(x,method='autoscaling')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pywt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from copy import deepcopy
import pandas as pd

def pretreat(x,method='autoscaling'):
    #x:输入既可以是光谱（n*m)，也可以是标签（（n,） 或 （n,1））
    #n是样品数，m是波长点数
    #method: 预处理方法，默认为'autoscaling': 标准化预处理
    n,m = x.shape
    if x.ndim == 1 or m ==1: #x是(n,)或(n,1)标签，或为（n,1)光谱
        if method == 'autoscaling': #标准化预处理
            para1 = np.mean(x)
            para2 = np.std(x)
        elif method == 'center': #去中心化(均值中心化）
            para1 = np.mean(x)
            para2 = 1
        elif method == 'pareto':#帕累托预处理
            para1 = np.mean(x)
            para2 = np.sqrt(np.std(x))
        elif method == 'minmax': #最小最大化预处理
            para1 = np.min(x)
            maxv = np.max(x)
            para2 = maxv - para1
        elif method == 'msc':#多元散射校正预处理
            print('MSC method is only suitable to the spectrum with n*m dimension')
            return x
        elif method == 'd1': #一阶微分预处理
            x = deepcopy(x)
            if isinstance(x,pd.DataFrame):
                x = x.values
            x = x.reshape(1,-1)
            temp1 = pd.DataFrame(x)
            temp2 = temp1.diff(axis=1)
            temp3 = temp2.values
            temp4 = np.delete(temp3,0,axis=1)
            return temp4.squeeze()
        elif method == 'd2':#二阶微分预处理：
            x = deepcopy(x)
            if isinstance(x, pd.DataFrame):
                x = x.values
            x = x.reshape(1, -1)
            temp2 = (pd.DataFrame(x)).diff(axis=1)
            temp3 = np.delete(temp2.values,0,axis=1)
            temp4 = (pd.DataFrame(temp3)).diff(axis=1)
            spec_D2 = np.delete(temp4.values,0,axis=1)
            return spec_D2.squeeze()
        elif method == 'sg': #Savitzky-Golay平滑滤波
            print('SG method is only suitable to the spectrum with n*m dimension')
            return x
        elif method == 'dt': #趋势校正
            print('DT method is only suitable to the spectrum with n*m dimension')
            return x
        x_p = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_p[i] = (x[i] - para1) / para2  # 对标签每一个数值进行预处理计算
        return x_p
    else:#x:n*m
        if method == 'autoscaling': #标准化预处理
            para1 = np.mean(x,axis=0)
            para2 = np.std(x,axis=0)
        elif method == 'center': #去中心化(均值中心化）
            para1 = np.mean(x,axis=0)
            para2 = 1
        elif method == 'pareto':#帕累托预处理
            para1 = np.mean(x,axis=0)
            para2 = np.sqrt(np.std(x,axis=0))
        elif method == 'minmax': #最小最大化预处理
            para1 = np.min(x,axis=0)
            maxv = np.max(x,axis=0)
            para2 = maxv - para1
        elif method == 'msc':#多元散射校正预处理
            n = x.shape[0]  # 样本数量
            k = np.zeros(x.shape[0])
            b = np.zeros(x.shape[0])  # k,b用于存放线性模型参数y = k*x+b
            M = np.array(np.mean(x, axis=0))  # 沿波长点方向求平均光谱M
            for i in range(n):
                xi = x[i, :]
                xi = xi.reshape(-1, 1)
                M = M.reshape(-1, 1)
                model = LinearRegression()
                model.fit(M, xi)
                k[i] = model.coef_[0][0]
                b[i] = model.intercept_[0]
            spec_msc = np.zeros_like(x)  # 保存msc预处理后的光谱
            for i in range(n):
                bb = np.repeat(b[i], x.shape[1])
                kk = np.repeat(k[i], x.shape[1])
                temp = (x[i, :] - bb) / kk
                spec_msc[i, :] = temp
            return spec_msc
        elif method == 'd1': #一阶微分预处理
            x = deepcopy(x)
            if isinstance(x,pd.DataFrame):
                x = x.values
            temp1 = pd.DataFrame(x)
            temp2 = temp1.diff(axis=1)
            temp3 = temp2.values
            return np.delete(temp3,0,axis=1)
        elif method == 'd2':#二阶微分预处理：
            x = deepcopy(x)
            if isinstance(x, pd.DataFrame):
                x = x.values
            temp2 = (pd.DataFrame(x)).diff(axis=1)
            temp3 = np.delete(temp2.values,0,axis=1)
            temp4 = (pd.DataFrame(temp3)).diff(axis=1)
            spec_D2 = np.delete(temp4.values,0,axis=1)
            return spec_D2
        elif method == 'sg': #Savitzky-Golay平滑滤波
            x = deepcopy(x)
            x_p = savgol_filter(x,5,polyorder=3,deriv=0)
            return x_p
        elif method == 'dt': #趋势校正
            length = x.shape[1]
            xx = np.asarray(range(length),dtype=np.float32)
            ll = LinearRegression()
            out = np.zeros(x.shape)
            for i in range(x.shape[0]):
                ll.fit(xx.reshape(-1,1),x[i].reshape(-1,1))
                k = ll.coef_
                b = ll.intercept_
                for j in range(x.shape[1]):
                    out[i,j] = out[i,j]-(j*k[0][0]+b[0])
            return out

        x_p = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_p[i] = (x[i] - para1) / para2  # 对标签每一个数值进行预处理计算
        return x_p