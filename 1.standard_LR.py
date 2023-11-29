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
    writer = SummaryWriter(log_dir='./log/1_standard')
    algo = "Linear"
    
    normization_type1 = 'MinMax'
    normization_type2 = 'Mean'
    normization_type3 = 'Standardization'
    normization_type4 = None

    MSE_origin = np.zeros((10,))
    MSE_MinMax = np.zeros((10,))
    MSE_Mean = np.zeros((10,))
    MSE_Standardization = np.zeros((10,))

    for i in range(10):

        data_origin = dataLoader(data_path)
        data_origin.normalization(normization_type4)
        data_MinMax = dataLoader(data_path)
        data_MinMax.normalization(normization_type1)


        data_Mean = dataLoader(data_path)
        data_Mean.normalization(normization_type2)
        data_Standardization = dataLoader(data_path)
        data_Standardization.normalization(normization_type3)


        training_data_origin, test_data_origin = data_origin.data_cut(10, i)
        training_data_MinMax, test_data_MinMax = data_MinMax.data_cut(10, i)
        training_data_Mean, test_data_Mean = data_Mean.data_cut(10, i)
        training_data_Standardization, test_data_Standardization = data_Standardization.data_cut(10, i)

        trainer_origin = Trainer(training_data_origin, test_data_origin, algo)
        trainer_MinMax = Trainer(training_data_MinMax, test_data_MinMax, algo)
        trainer_Mean = Trainer(training_data_Mean, test_data_Mean, algo)
        trainer_Standardization = Trainer(training_data_Standardization, test_data_Standardization, algo)
        

        theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='origin')
        theta_MinMax = trainer_MinMax.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='MinMax')
        thetaMean = trainer_Mean.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='Mean')
        theta_Standardization = trainer_Standardization.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='Standardization')
        


        MSE_origin[i] = trainer_origin.test(theta_origin)
        MSE_MinMax[i] = trainer_MinMax.test(theta_MinMax)
        MSE_Mean[i] = trainer_Mean.test(thetaMean)
        MSE_Standardization[i] = trainer_Standardization.test(theta_Standardization)

        
    print(MSE_origin.mean())
    print(MSE_MinMax.mean())
    print(MSE_Mean.mean())
    print(MSE_Standardization.mean())

        
        
    
    





