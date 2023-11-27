import numpy as np


# n1 = np.random.random((11,3760))
# n2 = np.random.random((3760,3760))

# print(type(n1[0][0]))
# print(type(n2[0][0]))
# for i in range(100):
    
#     print(np.dot(np.dot(n1,n2),n1.T).shape)

# Y_HAT = np.ones(0)
# print(Y_HAT)
n1 = np.array([1,2,3,4])
print(np.square(n1).sum())
datas = np.array([[1.0,2.0,3.0,4],
                  [2.0,1.0,1.0,1],
                  [3.0,2.0,2.0,2],
                  [4.0,2.0,2.0,2],
                  [5.0,2.0,2.0,2]])

# min_col = np.min(datas, axis=0)
# max_col = np.max(datas, axis=0)
mean_col = np.mean(datas, axis=0)
var = np.var(datas, axis=0)
# for i in range(datas[0].shape[0]-1):
#     datas[:,i] = (datas[:,i] - min_col[i])/(max_col[i] - min_col[i])

print(var)