from dataLoader import dataLoader
from Trainer import Trainer
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":


    lr=1e-2
    init_theta1 = np.zeros((11,1))
    init_theta2 = np.ones((11,1))
    init_theta3 = np.random.random((11,1))
    init_theta4 = 100*np.ones((11,1))

    lambda_theta = 1e-3
    l_steps = 10000

    data_path = './Data/abalone.data'
    writer = SummaryWriter(log_dir='./log/1_standard_init')
    algo = "Linear"
    

    normization_type4 = None

    MSE_origin = np.zeros((4,10))
 
    data_origin = dataLoader(data_path)
    data_origin.normalization(normization_type4)

    

    for i in range(10):

        training_data_origin, test_data_origin = data_origin.data_cut(10, i)

        trainer_origin = Trainer(training_data_origin, test_data_origin, algo)
            
        
        theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta1 , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='init_0')
    
        MSE_origin[0][i] = trainer_origin.test(theta_origin)


        theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta2 , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='init_1')
    
        MSE_origin[1][i] = trainer_origin.test(theta_origin)



        theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta3 , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='init_rand')
    
        MSE_origin[2][i] = trainer_origin.test(theta_origin)



        theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta4 , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='init_100')
    
        MSE_origin[3][i] = trainer_origin.test(theta_origin)
       

    for i in range(4):
        
        print(MSE_origin[i].mean())


        
        
    
    





