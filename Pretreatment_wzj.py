import numpy as np
from sklearn.linear_model import LinearRegression

class Pretreatment:
    def __init__(self,x,method='autoscaling'):
        self.x = x
        self.method = method
        # 创建一个数据集预处理函数pretreat():

    def pretreat(self):
        # x: 2维光谱或标签数组
        # method：表示预处理方法的字符串
        # 如：‘autoscaling’：标准化预处理
        # 'center':去均值；‘minmax’:最小最大预处理
        # 'pareto': pareto尺度预处理；'none':无预处理
        if self.x.ndim != 2:  # x为一维标签数组
            print("The input x should be a 1 dim label")
            if self.method == 'autoscaling':
                para1 = np.mean(self.x)
                para2 = np.std(self.x)
            elif self.method == 'center':
                para1 = np.mean(self.x)
                para2 = 1
            elif self.method == 'minmax':
                para1 = np.min(self.x)
                maxv = np.max(self.x)
                para2 = maxv - para1
            elif self.method == 'pareto':
                para1 = np.mean(self.x)
                para2 = np.sqrt(np.std(self.x))
            elif self.method == 'msc':  # 多元散射校正
                print('MSC method is only suitable to the spectrum,not the labels')
                return self.x
            elif self.method == 'none':
                para1 = 0
                para2 = 1
            else:
                print('Wrong data pretreat method')
                return self.x
            x_p = np.zeros(self.x.shape)
            for i in range(self.x.shape[0]):
                x_p[i] = (self.x[i] - para1) / para2  # 对标签每一个数值进行预处理计算
            return x_p

        Mx, Nx = self.x.shape
        if Nx == 1:
            print('The x data should be a label data')
            if self.method == 'autoscaling':
                para1 = np.mean(self.x)
                para2 = np.std(self.x)
            elif self.method == 'center':
                para1 = np.mean(self.x)
                para2 = 1
            elif self.method == 'minmax':
                para1 = np.min(self.x)
                maxv = np.max(self.x)
                para2 = maxv - para1
            elif self.method == 'pareto':
                para1 = np.mean(self.x)
                para2 = np.sqrt(np.std(self.x))
            elif self.method == 'msc':
                print('MSC method is only suitable to the spectrum,not the labels')
                return self.x
            elif self.method == 'none':
                para1 = 0
                para2 = 1
            else:
                print('Wrong data pretreat method')
                return
            x_p = np.zeros(self.x.shape)
            for i in range(self.x.shape[0]):
                x_p[i] = (self.x[i] - para1) / para2  # 对光谱每一别进行预处理计算
            return x_p

        if self.method == 'autoscaling':
            para1 = np.mean(self.x, axis=0)
            para2 = np.std(self.x, axis=0)  # axis=0,按列方法
        elif self.method == 'center':
            para1 = np.mean(self.x, axis=0)
            para2 = np.ones((Nx,))
        elif self.method == 'minmax':
            para1 = np.min(self.x, axis=0)
            maxv = np.max(self.x, axis=0)
            para2 = maxv - para1
        elif self.method == 'pareto':
            para1 = np.mean(self.x, axis=0)
            para2 = np.sqrt(np.std(self.x, axis=0))
        elif self.method == 'msc':  # 多元散射校正
            n = self.x.shape[0]  # 样本数量
            k = np.zeros(self.x.shape[0])
            b = np.zeros(self.x.shape[0])  # k,b用于存放线性模型参数y = k*x+b
            M = np.array(np.mean(self.x, axis=0))  # 沿波长点方向求平均光谱M
            for i in range(n):
                xi = self.x[i, :]
                xi = xi.reshape(-1, 1)
                M = M.reshape(-1, 1)
                model = LinearRegression()
                model.fit(M, xi)
                k[i] = model.coef_[0][0]
                b[i] = model.intercept_[0]
            spec_msc = np.zeros_like(self.x)  # 保存msc预处理后的光谱
            for i in range(n):
                bb = np.repeat(b[i], self.x.shape[1])
                kk = np.repeat(k[i], self.x.shape[1])
                temp = (self.x[i, :] - bb) / kk
                spec_msc[i, :] = temp
            para1 = k
            para2 = b
            return spec_msc

        elif self.method == 'none':
            para1 = np.zeros((Nx,))
            para2 = np.ones((Nx,))
        else:
            print('Wrong data pretreat method')
            return
        x_p = np.zeros((Mx, Nx))
        for i in range(Nx):
            x_p[:, i] = (self.x[:, i] - para1[i]) / para2[i]  # 对光谱每一别进行预处理计算
        print('The x data should be a 2 dim spectroscopy data')
        return x_p