import numpy as np

class Trainer():
# 传入参数包含有切分好的数据集与训练集
# algo：使用线性回归，岭回归，或者是Lasso回归，三选一，输入字符串，默认为线性回归


    def __init__(self, training_data:np.ndarray, test_data:np.ndarray, algo:str = "Linear"):
        self.training_data = training_data
        self.test_data = test_data
        self.algo = algo
        
        assert algo in ['Linear', 'Lasso', 'Ridge', 'LWLR']
            

    
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

    def train(self, writer, lables,lr=1e-1, init_theta = np.zeros(9), lambda_theta = 0.001, l_steps = 5000):
        # 为了让偏置项与其他的参数写成一个矩阵，这边将一行1
        # 写在X的第一行，这样的话结果直接就是y=theta*x
        # 不需要在计算公式中另外加入偏置项。
        
        X_1 = np.ones((len(self.training_data),1))
    
        X = self.training_data[:, :-1]
        Y = self.training_data[:, -1].reshape(-1,1)
        X = np.hstack((X_1, X))

        theta = init_theta
        # print(self.gradient(X, Y, theta))

        # print(self.MSE(X, Y, theta))

        for i in range(l_steps):
            
            if i % 1000 == 0:
                print("MSE@epoch", i, ":", self.MSE(X, Y, theta))
                # print(theta)
            writer.add_scalars('loss', {lables :self.MSE(X, Y, theta)}, i)
            theta = theta - lr * self.gradient(X, Y, theta, lambda_theta)

        print(theta)
        return theta
            

    def test(self, theta):
        X_1 = np.ones((len(self.test_data),1))
    
        X = self.test_data[:, :-1]
        Y = self.test_data[:, -1].reshape(-1,1)
        X = np.hstack((X_1, X))
        # print(X)
        # print(Y)

        print("MSELoss@:", self.MSE(X, Y, theta))
        return self.MSE(X, Y, theta)
    


    def LWLR_Test(self, k):
        X_1 = np.ones((len(self.training_data),1))
        X = self.training_data[:, :-1]
        Y = self.training_data[:, -1].reshape(-1,1)
        # X = np.hstack((X_1, X))

        # 这边没有使用偏置项，很奇怪。没有偏置项预测精度好了一大截。
        
        
        theta = np.zeros((len(X[0]),1))
        
        X_1_test = np.ones((len(self.test_data),1))
        X_test = self.test_data[:, :-1]
        Y_test = self.test_data[:, -1].reshape(-1,1)


        MSE_LWLR = 0
        
        for x_i in range(X_test.shape[0]):
            W = np.zeros((X.shape[0],X.shape[0]))
            for i in range(X.shape[0]):
                W[i][i] =np.exp((-np.square(X_test[x_i] - X[i])).sum()/(2*k**2)) 
            
            # print(W.shape)
            
            XTWX = X.T @ W @ X
            # print(type(X), X.shape)
            # print(type(W), W.shape)


            r_XTWX = np.linalg.inv(XTWX)
            
            theta = r_XTWX @ X.T @ W @ Y 
            
           
            y_hat_single = np.dot(theta.T, X_test[x_i].reshape(-1,1))
           
            MSE_LWLR += (y_hat_single - Y_test[x_i])**2

        print(k,':',MSE_LWLR/X_test.shape[0])
        return MSE_LWLR/X_test.shape[0]

 
        
