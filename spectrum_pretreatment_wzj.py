###说明：将各种常用光谱预处理方法写成一个类
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from scipy.signal import savgol_filter
from copy import deepcopy
import pywt
from sklearn.preprocessing import MinMaxScaler,StandardScaler

class Pretreatment:
    def PlotSpectrum(self,spec,title='原始光谱',x=0,m=5):
        #输入参数 spec: shape(n_samples,n_features),读取的pandas光谱数据（原始）
        #返回值return: xx:光谱矩阵；y:糖度值；w：波长点数
        if isinstance(spec,pd.DataFrame):
            spec = spec.values #如果spec是pandas数据，则取其值，得到np array
        xx = spec[:,1:] #光谱
        y = spec[:,0] #糖度
        w = np.linspace(x,x+(spec.shape[1]-1)*m,spec.shape[1]-1)
        plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']= False #用来正常显示负号
        
        fonts = 6
        plt.figure(figsize=(5.2,3.1),dpi=200)
        plt.plot(w,xx.T)
        plt.xlabel('Wavelength/nm',fontsize=fonts)
        plt.ylabel('Reabsorbance',fontsize=fonts)
        plt.title(title,fontsize=fonts)
        plt.grid()
        plt.show()
        return xx,y,w
    
    ###纯绘图函数
    def PlotSpectrum1(self,spec_data,w,title='原始光谱'):#spec_data:光谱数据，w：波长数据
        if isinstance(spec_data,pd.DataFrame):
            data = spec_data.values #如果spec是pandas数据，则取其值，得到np array
        else:
            data = spec_data
        fonts = 6
        plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']= False #用来正常显示负号
        plt.figure(figsize=(5.2,3.1),dpi=200)
        plt.plot(w,data.T)
        plt.xlabel('Wavelength/nm',fontsize=fonts)
        plt.ylabel('Absorbance',fontsize=fonts)
        plt.title(title,fontsize=fonts)
        plt.grid()
        plt.show()
    
    def mean_centralization(self,sdata):
        #均值中心化:对50个样品光谱，在每一个波长点处减去其平均值。
        #输入参数 sdata：光谱数组数据
        #返回参数：均值中心化数据
        sdata = deepcopy(sdata)
        if isinstance(sdata,pd.DataFrame):
            sdata = sdata.values
        temp1 = np.mean(sdata,axis = 0) #对每一个波长点处求平均值
        temp2 = np.tile(temp1,sdata.shape[0]).reshape(   #np.tile()表示将数组a在行上重复x次
                (sdata.shape[0],sdata.shape[1]))
        return sdata-temp2
    
    def standardlize(self,sdata):
        #标准化预处理函数程序
        sdata = deepcopy(sdata)
        if isinstance(sdata,pd.DataFrame):
            sdata = sdata.values
        sdata = preprocessing.scale(sdata)###与后面的StandardScaler().fit_transform(sdata)
        return sdata                      ###相同.标准化数据（对每一个波长点）
    
    def msc(self,sdata):
        #多元散射校正MSC 预处理函数程序
        sdata = deepcopy(sdata)
        if isinstance(sdata,pd.DataFrame):
            sdata = sdata.values
        
        n = sdata.shape[0] #样本数量50
        k = np.zeros(sdata.shape[0])
        b = np.zeros(sdata.shape[0])
        
        M  = np.array(np.mean(sdata,axis=0))
        
        from sklearn.linear_model import LinearRegression
        for i in range(n): 
            y = sdata[i,:]
            y = y.reshape(-1,1)
            M = M.reshape(-1,1)
            model = LinearRegression()
            model.fit(M,y)
            k[i] = model.coef_[0][0]
            b[i] = model.intercept_[0]
        spec_msc = np.zeros_like(sdata)
        for i in range(n):
            bb = np.repeat(b[i],sdata.shape[1])
            kk = np.repeat(k[i],sdata.shape[1])
            temp = (sdata[i,:]-bb)/kk
            spec_msc[i,:] = temp
        return spec_msc
    
    def D1(self,sdata):
        #一阶差分
        sdata = deepcopy(sdata)
        if isinstance(sdata,pd.DataFrame):
            sdata = sdata.values
        temp1 = pd.DataFrame(sdata)#把np数组转换为pandas
        temp2 = temp1.diff(axis=1) #行方向求导
        temp3 = temp2.values #pandas 转换为np数组
        return np.delete(temp3,0,axis=1)
    
    def D2(self,sdata):
        #二阶差分
        sdata = deepcopy(sdata)
        if isinstance(sdata,pd.DataFrame):
            sdata = sdata.values
        temp2 = (pd.DataFrame(sdata)).diff(axis=1)#一阶导
        temp3 = np.delete(temp2.values,0,axis=1)
        temp4 = (pd.DataFrame(temp3)).diff(axis=1)#二阶导
        spec_D2 = np.delete(temp4.values,0,axis=1)
        return spec_D2
    
    # 标准正态变换
    def SNV(self,sdata):
        """
            :param data: raw spectrum data, shape (n_samples, n_features)
            :return: data after SNV :(n_samples, n_features)
        """
        m = sdata.shape[0]
        n = sdata.shape[1]
        print(m,n)  
        sdata_snv = np.zeros(sdata.shape)
        # 求标准差
        sdata_std = np.std(sdata,axis=1)  # 每条光谱的标准差,axis = 1,对行求标准差
        # 求平均值
        sdata_average = np.mean(sdata,axis=1)  # 每条光谱的平均值，axis=1，对行求平均值
        # SNV计算
        for i in range(m):
             sdata_snv[i,:] = (sdata[i,:]-sdata_average[i])/sdata_std[i]
        return  sdata_snv

    # Savitzky-Golay平滑滤波
    def SG(self,sdata, w=5, p=3, d=0):
        """
           :param data: raw spectrum data, shape (n_samples, n_features)
           :param w: int
           :param p: int
           :param d: int
           :return: data after SG :(n_samples, n_features)
        """
        if isinstance(sdata, pd.DataFrame):
            sdata = sdata.values
        else:
            sdata = deepcopy(sdata)
        data = savgol_filter(sdata,w,polyorder=p, deriv=d)
        return data

    #最小最大标准化：
    def MMN(self,sdata):
        if isinstance(sdata, pd.DataFrame):
            sdata = sdata.values
        else:
            sdata = deepcopy(sdata)
        #d_min = np.min(data, axis=0) #对列（每个波长点）取最小值，最大值
        #d_max = np.max(data, axis=0)
        #data_m = (data - d_min) / (d_max - d_min)
        data_m = MinMaxScaler().fit_transform(sdata)
        return data_m
    
    # 标准归一化（对每一个波长点）
    def SS(self,sdata):
        """
            :param data: raw spectrum data, shape (n_samples, n_features)
            :return: data after StandScaler :(n_samples, n_features)
        """
        if isinstance(sdata, pd.DataFrame):
            sdata = sdata.values
        else:
            sdata = deepcopy(sdata)
        return StandardScaler().fit_transform(sdata)
    
    # 均值中心化
    def CT(self,sdata):
        """
            :param data: raw spectrum data, shape (n_samples, n_features)
            :return: data after MeanScaler :(n_samples, n_features)
        """
        MEAN = np.zeros((sdata.shape[0],))
        data_ct = np.zeros(sdata.shape)

        for i in range(sdata.shape[0]):#循环样品数量次，对每一个样品，行
            MEAN[i] = np.mean(sdata[i])#对每一个样品求均值
            data_ct[i,:] = sdata[i,:]-MEAN[i]
        return data_ct
    
    # 移动平均平滑
    def MA(self,sdata, WSZ=11):
        """
           :param data: raw spectrum data, shape (n_samples, n_features)
           :param WSZ: int
           :return: data after MA :(n_samples, n_features)
        """
        data_ma = np.zeros(sdata.shape)
        for i in range(sdata.shape[0]):
            out0 = np.convolve(sdata[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
            r = np.arange(1, WSZ - 1, 2)
            start = np.cumsum(sdata[i, :WSZ - 1])[::2] / r
            stop = (np.cumsum(sdata[i, :-WSZ:-1])[::2] / r)[::-1]
            data_ma[i] = np.concatenate((start, out0, stop))
        return data_ma

    def move_avg(self,data_x, n=15, mode="valid"):
        # 滑动平均滤波
        data_x = deepcopy(data_x)
        if isinstance(data_x, pd.DataFrame):
            data_x = data_x.values
        tmp = None
        for i in range(data_x.shape[0]):
            if (i == 0):
                tmp = np.convolve(data_x[i, :], np.ones((n,)) / n, mode=mode)
            else:
                tmp = np.vstack((tmp, np.convolve(data_x[i, :], np.ones((n,)) / n, mode=mode)))
        return tmp
    
    def wave(self, data_x):  # 小波变换
        data_x = deepcopy(data_x)
        if isinstance(data_x, pd.DataFrame):
            data_x = data_x.values
        def wave_(data_x):
            w = pywt.Wavelet('db8')  # 选用Daubechies8小波
            maxlev = pywt.dwt_max_level(len(data_x), w.dec_len)
            coeffs = pywt.wavedec(data_x, 'db8', level=maxlev)
            threshold = 0.04
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            datarec = pywt.waverec(coeffs, 'db8')
            return datarec

        tmp = None
        for i in range(data_x.shape[0]):
            if (i == 0):
                tmp = wave_(data_x[i])
            else:
                tmp = np.vstack((tmp, wave_(data_x[i])))
        return tmp
    
    # 趋势校正(DT)
    def DT(self,data):
        """
           :param data: raw spectrum data, shape (n_samples, n_features)
           :return: data after DT :(n_samples, n_features)
        """
        from sklearn.linear_model import LinearRegression
        lenth = data.shape[1]
        x = np.asarray(range(lenth), dtype=np.float32)
        out = np.array(data)
        l = LinearRegression()
        for i in range(out.shape[0]):
            l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
            k = l.coef_
            b = l.intercept_
            for j in range(out.shape[1]):
                out[i][j] = out[i][j] - (j *k[0][0]+b[0])

        return out
