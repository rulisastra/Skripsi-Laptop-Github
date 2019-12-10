import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.linalg import cholesky

data = pd.read_csv('Currency Converter - Copy.csv',
                    usecols=[1],
                    engine='python',
                    delimiter=',',
                    decimal=".",
                    thousands=',',
                    header=None,
                    names=['date','value'] )
data['value'] = data['value'].values
data['value'] = data['value'].astype('float32')



data = [1,2,3,4,4]
data2 = np.array([1,1,2,2,2])
sum_ = np.sum(data)
sum2_ = np.sum(data2)
mean = np.mean(data)
mean2 = np.mean(data2)
variance = np.var(data)
variance2 = np.var(data2)
covariance = np.cov(data)
covariance2 = np.cov(data2)
covariance_all = np.cov(data,data2)
std = np.std(data)
std2 = np.std(data2)
koef = np.corrcoef(data)
koef2 = np.corrcoef(data2)
koef_all = np.corrcoef(data,data2)
P = np.eye(6)
Pp = cholesky(P) * P

X = data2

X[0] = X 
n = X.size
for k in range (n):
    X[k+1] = np.subtract(X, -P[k])
    X[n+k+1] = np.subtract(X, P[k])

#plt.plot(data, norm.pdf(data))

# =============================================================================
# incomes = np.random.normal(50, 30, 1000)
# plt.hist(incomes,50)
# plt.show()
# =============================================================================

x = np.arange(-3, 3, 0.001)
plt.plot(x, norm.pdf(x,1, 0.2),label ='std=0,2 dan mean = 1',c='y')
plt.plot(x, norm.pdf(x,1, 0.5), label ='std=0,5 dan mean = 1',c='r')
plt.plot(x, norm.pdf(x,1, 1),label = 'std=1 dan mean = 1',c='g')
plt.legend()
plt.show()

plt.plot(x, norm.pdf(x,0.2, 1),label ='std=1 dan mean = 0.2',c='y')
plt.plot(x, norm.pdf(x,0.5, 1), label ='std=1 dan mean = 0.5',c='r')
plt.plot(x, norm.pdf(x,1, 1),label = 'std=1 dan mean = 1',c='g')
plt.legend()
plt.show()

plt.scatter(data,data)
plt.show()

x = np.linspace(data, 2*np.pi, 100)
plt.rcParams['lines.linewidth'] = 1
plt.figure()
plt.plot(x, np.sin(x), label='zorder=10', zorder=10)  # on top
plt.plot(x, np.sin(1.1*x), label='zorder=1', zorder=1)  # bottom
plt.plot(x, np.sin(1.2*x), label='zorder=3',  zorder=3)
plt.axhline(0, label='zorder=2', color='grey', zorder=2)
plt.title('Custom order of elements')
l = plt.legend(loc='upper right')
l.set_zorder(20)  # put the legend on top
plt.show()
