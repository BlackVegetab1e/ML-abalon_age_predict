from dataLoader import dataLoader
from Trainer import Trainer
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
if __name__ == "__main__":


    lr=1e-2
    init_theta = np.zeros(9)
    lambda_theta = 1e-3
    l_steps = 5000

    writer = SummaryWriter(log_dir='./log/3_LWLR')
    data_path = './Data/abalone.data'
    algo = "LWLR"
    LWLR_k = 0.3

    MSEs = np.zeros((10,))


    data = dataLoader(data_path)
    T1 = time.time()

    training_data, test_data = data.data_cut(10, 1)
    t = Trainer(training_data, test_data, algo)
    t.LWLR_Test(0.2)

    T2 = time.time()

    data_origin = dataLoader(data_path)
    training_data_origin, test_data_origin = data_origin.data_cut(10, 1)
    trainer_origin = Trainer(training_data_origin, test_data_origin, 'Linear')
    theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='origin')
    trainer_origin.test(theta_origin)
        
    T3 = time.time()
    
    data_origin = dataLoader(data_path)
    training_data_origin, test_data_origin = data_origin.data_cut(10, 1)
    trainer_origin = Trainer(training_data_origin, test_data_origin, 'Lasso')
    theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='origin')
    trainer_origin.test(theta_origin)

    T4 = time.time()

    data_origin = dataLoader(data_path)
    training_data_origin, test_data_origin = data_origin.data_cut(10, 1)
    trainer_origin = Trainer(training_data_origin, test_data_origin, 'Ridge')
    theta_origin = trainer_origin.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps , writer= writer, lables='origin')
    trainer_origin.test(theta_origin)
    
    T5 = time.time()

    print((T2-T1)*1000)
    print((T3-T2)*1000)
    print((T4-T3)*1000)
    print((T5-T4)*1000)
    

