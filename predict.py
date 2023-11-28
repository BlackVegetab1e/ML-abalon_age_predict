from dataLoader import dataLoader
from Trainer import Trainer
import numpy as np
if __name__ == "__main__":


    lr=1e-1
    init_theta = np.zeros(9)
    lambda_theta = 0.001
    l_steps = 5000

    data_path = './Data/abalone.data'

    algo = "Linear"
    algo = "Lasso"
    # algo = "Ridge"
    # algo = "LWLR"
    normization_type = 'MinMax'
    normization_type = 'Mean'
    normization_type = 'Standardization'
    normization_type = None
    LWLR_k = 0.16



    data = dataLoader(data_path)
    data.normalization(normization_type)
    training_data, test_data = data.data_cut(10, 9)


    t = Trainer(training_data, test_data, algo)
    if algo == 'LWLR':
        t.LWLR_Test(LWLR_k)
    else:
        theta = t.train(lr=lr, init_theta=init_theta , \
                        lambda_theta=lambda_theta , l_steps=l_steps )
        t.test(theta)
    
    





