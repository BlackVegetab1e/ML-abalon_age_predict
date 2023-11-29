from dataLoader import dataLoader
from Trainer import Trainer
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":


    lr=1e-2
    init_theta = np.zeros(9)
    lambda_theta = 1e-3
    l_steps = 5000

    data_path = './Data/abalone.data'
    writer = SummaryWriter(log_dir='./log/1_standard_lr')
    algo = "Linear"
    
    normization_type1 = 'MinMax'
    normization_type2 = 'Mean'
    normization_type3 = 'Standardization'
    normization_type4 = None

    MSE_origin = np.zeros((10,10))
 
    for j in range(1,10):
        now_lr = lr*j
        for i in range(10):

            data_origin = dataLoader(data_path)
            data_origin.normalization(normization_type4)

            training_data_origin, test_data_origin = data_origin.data_cut(10, i)
        
            trainer_origin = Trainer(training_data_origin, test_data_origin, algo)
            
            
            theta_origin = trainer_origin.train(lr=now_lr, init_theta=init_theta , \
                            lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='lr'+str(now_lr))
        
            MSE_origin[j][i] = trainer_origin.test(theta_origin)
       

        
    print(MSE_origin.mean())


        
        
    
    





