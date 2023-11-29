from dataLoader import dataLoader
from Trainer import Trainer
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":


    lr=1e-2
    init_theta = np.zeros(9)
    lambda_theta = 1e-1
    l_steps = 5000

    data_path = './Data/abalone.data'
    writer = SummaryWriter(log_dir='./log/2_lasso')
    algo = "Linear"
    algo_Lasso = "Lasso"
    algo_Rigid = "Ridge"
    

    normization = None

    MSE_origin = np.zeros((10,))
    MSE_Lasso = np.zeros((10,))
    MSE_Rigid = np.zeros((10,))


    for i in range(10):

        data_origin = dataLoader(data_path)
        data_origin.normalization(normization)
   


        training_data_origin, test_data_origin = data_origin.data_cut(10, i)


        trainer_origin = Trainer(training_data_origin, test_data_origin, algo)
        trainer_Rigid = Trainer(training_data_origin, test_data_origin, algo_Rigid)
        trainer_Lasso = Trainer(training_data_origin, test_data_origin, algo_Lasso)

        

        theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='origin')
        theta_Rigid = trainer_Rigid.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='Rigid')
        thetaLasso = trainer_Lasso.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='Lasso')
        
        


        MSE_origin[i] = trainer_origin.test(theta_origin)
        MSE_Rigid[i] = trainer_Rigid.test(theta_Rigid)
        MSE_Lasso[i] = trainer_Lasso.test(thetaLasso)


        
    print(MSE_origin.mean())
    print(MSE_Rigid.mean())
    print(MSE_Lasso.mean())
    


        
        
    
    





