import pandas as pd
import numpy as np


class dataLoader():
    def __init__(self, data_loc):
        df = pd.read_csv(data_loc)
        self.datas = np.array(df)
        self.encode_sex()
        
    def encode_sex(self):
        sex_encoder = np.zeros((len(self.datas),3))
        for i in range(0, len(self.datas)):
            if self.datas[i,0] == 'F':
                sex_encoder[i,0] = 1
            elif self.datas[i,0] == 'M':
                sex_encoder[i,1] = 1
            elif self.datas[i,0] == 'I':
                sex_encoder[i,2] = 1
        # print(sex_encoder.shape)
        # print(self.datas[:,1:].shape)
        self.datas = np.hstack((sex_encoder, self.datas[:,1:]))
        # print(self.datas)
        # print(self.datas.shape)

    def data_cut(self, segment_number:int = 10, segment_selected:int = 0):
    # 进行数据分割的函数，将training_data分为segment_number份，
    # 并且选中segment_selected当作测试集，其余数据当作训练集
        assert segment_selected>=0 and segment_selected < segment_number
        # 要求分段的合理性，选中数字应该在0到n-1中选择
        data_length = len(self.datas)
        segment_length = int(data_length/segment_number)
        segment_index_begain = segment_length*segment_selected
        segment_index_end = segment_length*(segment_selected+1)
        
        if segment_selected == segment_number-1:
            # 如果是最后一块，将最后末尾的当作测试集，即从尾开始数
            test_data = self.datas[-segment_length:]
            training_data = self.datas[0:data_length-segment_length]
        else:
            # 其他的都是从头开始算
            test_data = self.datas[segment_index_begain:segment_index_end]
            training_data = np.vstack((self.datas[0:segment_index_begain], self.datas[segment_index_end:]))

        return training_data, test_data





class Trainer():
# 传入参数包含有切分好的数据集与训练集
# algo：使用线性回归，岭回归，或者是Lasso回归，三选一，输入字符串，默认为线性回归

    def __init__(self, training_data:np.ndarray, test_data:np.ndarray, algo:str = "Linear"):
        self.training_data = training_data
        self.test_data = test_data
        self.algo = algo
        
        assert algo in ['Linear', 'Lasso', 'Ridge']
            

    
    def Y_hat(self, X:np.ndarray, theta:np.ndarray)->np.ndarray:
        return np.dot(X, theta)

    def gradient(self, X:np.ndarray, Y:np.ndarray, theta:np.ndarray, lambda_theta)->np.ndarray:
        grad = np.dot(X.T , (self.Y_hat(X, theta) - Y))
        grad /= X.shape[0]

        if self.algo == 'Lasso':
            # 这里务必使用copy，因为np的=传递的是引用，而不是拷贝。
            theta_of_x = theta.copy()
            theta_of_x[0,0] = 0
            for i in range(1,theta_of_x.shape[0]):
                if theta_of_x[i,0] > 0:
                    theta_of_x[i,0] = 1
                else:
                    theta_of_x[i,0] = -1
            grad += lambda_theta * theta_of_x

        elif self.algo == 'Ridge':
            theta_of_x = theta.copy()
            theta_of_x[0,0] = 0
            # print(grad)
            # print(theta_of_x)
            grad += lambda_theta * theta_of_x
            
            # print("*******************")
        return grad

    def MSE(self, X:np.ndarray, Y:np.ndarray, theta:np.ndarray)->np.ndarray:
        delta_y = self.Y_hat(X, theta) - Y

        return np.mean(np.square(delta_y) )

    def train(self, lr=1e-1, init_theta = np.zeros(9), lambda_theta = 0.001, l_steps = 10000):
        # 为了让偏置项与其他的参数写成一个矩阵，这边将一行1
        # 写在X的第一行，这样的话结果直接就是y=theta*x
        # 不需要在计算公式中另外加入偏置项。
        
        X_1 = np.ones((len(self.training_data),1))
    
        X = self.training_data[:, :-1]
        Y = self.training_data[:, -1].reshape(-1,1)
        X = np.hstack((X_1, X))

        theta = np.zeros((len(X[0]),1))
        # print(self.gradient(X, Y, theta))

        # print(self.MSE(X, Y, theta))

        
        for i in range(l_steps):
            if i % 1000 == 0:
                print("MSE@epoch", i, ":", self.MSE(X, Y, theta))
                # print(theta)
            theta = theta - lr * self.gradient(X, Y, theta, lambda_theta)

        print(theta)
            
        
        




if __name__ == "__main__":
    # data = dataLoader('./Data/abalone.data')
    
    data = dataLoader('C:\\Users\\Haoyu\\Desktop\\研一\\ML\\HomeWork1\\Data\\abalone.data')
    training_data, test_data = data.data_cut(10, 9)

    
    algo = "Linear"
    algo = "Lasso"
    algo = "Ridge"
    t = Trainer(training_data, test_data, algo)
    t.train()
    
    





