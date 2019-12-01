import numpy as np

a = np.reshape(np.arange(10),(-1,1))
print(a)

b = np.rot90(a,2)
print(b)
