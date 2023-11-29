from dataLoader import dataLoader
from Trainer import Trainer
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":


    lr=1e-2
    init_theta = np.zeros(9)

    writer = SummaryWriter(log_dir='./log/3_LWLR')
    data_path = './Data/abalone.data'
    algo = "LWLR"
    LWLR_k = 0.3

    MSEs = np.zeros((10,))

    for i in range(10):
        data = dataLoader(data_path)

        training_data, test_data = data.data_cut(10, i)

        t = Trainer(training_data, test_data, algo)

        init_k = 0.3

        MSEs[i] = t.LWLR_Test(init_k)
    print(MSEs.mean())

        
    
    
    

