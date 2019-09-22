import pandas as pd
import numpy as np

data = pd.read_csv('data.csv',usecols=[1],
                    engine='python',
                    delimiter=',',
                    decimal=".",
                    thousands=',',
                    header=None,
                    names=['date','value'] )
data = data.values
data = data.astype('float32')
print(data)

#normalisasi dulualan
def normalize(data,scale):
    normalized = []
    for i in range(len(data)):
       a = (min(scale))+((data[i]-min(data))*
              max(scale)-min(scale))/(max(data)-min(data))
       normalized.append(a)
    return np.array(normalized)

scale = (-1,1)
normalized = normalize(data,scale)
print(normalized)

#pisah data 7:3 disimpan di variabel normalized!
def pisahData(data,a,b):
     if((a+b)!=1):
            print("pemisahan tidak valid")
     else:
        train = []
        test = []
        train_size = int(len(normalized)*a)
        train = normalized[0:train_size-1]
        test = normalized[train_size-1:len(normalized)]
     return np.array(train),np.array(test)

train_data, test_data = pisahData(normalized, 0.7, 0.3)

#createwindowSize =3 untuk input
def createDataset(data, windowSize):
    dataX, dataY = [],[]
    for i in range(len(data)-windowSize):
        a = []
        for j in range(i, i+windowSize):
            a.append(data[j,0])
        dataX.append(a)
        dataY.append(data[i+windowSize,0])
    return np.array(dataX), np.array(dataY)
windowSize = 3
trainX, trainY = createDataset(train_data,windowSize)
testX, testY = createDataset(test_data, windowSize)

#fungsi aktivasi dan turunannya
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def dtanh(x):
    return (1-tanh(x)**2)

#inisialisasi dimensi layer
input_dim = windowSize
hidden_dim = 1 
output_dim = 1

#inisialisasi random bobot awal JST
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 #inisialisasi
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 #bobotjaringan dg
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 #interval[-1,1]

#inisialisasi Q,R wajib buat UKF sbg kovarian proses dan measurement
#inisiallsasi P sbg error kovarian untuk saat ini (state)
context_layer = np.full((trainX.shape[0],hidden_dim),0)
layer_h_deltas = np.zeros(hidden_dim)
Q = (np.std(data)*np.std(data))*np.eye(windowSize) #kovarian Noise process
R = np.std(data)*np.std(data) #Kovarian Noise measurement(observasi)
P = np.eye(windowSize) #kovarian estimasi vektor state
V = np.zeros((windowSize), dtype=int)

#forward pass (hitug NILAI neuron pada layer)
layer_1 = tanh(np.dot(trainX,synapse_0) +
              np.dot(context_layer,synapse_h)) #di hidden layer (sbg zh di paper)
layer_2 = tanh(np.dot(layer_1,synapse_1)) #di output layer (sbg yo di paper)

#update bobot (GAPERLU)
layer_2_error = layer_2 - trainY #nilai error dari prediksi buat jacobian
layer_2_delta = layer_2_error*dtanh(layer_2)
layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) +
                 np.multiply(layer_2_delta,synapse_1.T))*dtanh(layer_1)
synapse_1_update = np.dot(np.atleast_2d(layer_1).T, (layer_2_delta))
synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
synapse_0_update = np.dot(trainX.T,(layer_1_delta))



#----------------------------------------------------------
#UNSCENTED Kalman Filter (ALL)

#INISIALISASI
from filterpy.kalman import MerweScaledSigmaPoints

#inisialisasi faktor skalar
#inisialisasi PARAMETER -> van der merwe suggest using beta =2 kappa = 3-n
points = MerweScaledSigmaPoints(n=2, alpha=.1, beta=2., kappa=1.)
points.sigma_points(x=[0.,0], P=[[1.,.1],[.1, 1]])


##WEIGHTS
lambda_ = alpha**2 * (n + kappa) - n
Wc = np.full(2*n + 1,  1. / (2*(n + lambda_)))
Wm = np.full(2*n + 1,  1. / (2*(n + lambda_)))
Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
Wm[0] = lambda_ / (n + lambda_)

#SIGMA POINTS

s = np.sqrt(data) #data atau x??? data kok
np.dot(s, s.T)

sigmas = np.zeros((2*n+1, n))
U = np.sqrt((n+lambda_)*P) # sqrt

def sigmaPoint(data,X,k):
    sigmas[0] = X
    for k in range(n):
        sigmas[k+1]   = X + U[k]
        sigmas[n+k+1] = X - U[k]
        x = np.dot(Wm, sigmas) #jumlah sigma mean atau means
        return np.dot(x,n)
    
x = np.dot(Wm, sigmas)
n = sigmas.shape


def PE(imax,n):
    P = np.zeros((n, n))
    for i in range(imax):
        y = sigmas[i] - x
        P += Wc[i] * np.outer(y, y) 
        P += Q
        y = (sigmas[i] - x).reshape(imax, 1) # convert into 2D array
        P += Wc[i] * np.dot(y, y.T) #P += Wc[K] * np.dot(y, y.T)
        return()
PE

#PREDIKSI (STEP)
def predict(self, sigma_points_fn):
    """ Performs the predict step of the UKF. On return, 
    self.xp and self.Pp contain the predicted state (xp) 
    and covariance (Pp). 'p' stands for prediction.
    """

    # calculate sigma points for given mean and covariance
    sigmas = sigma_points_fn(self.x, self.P)

    for i in range(self._num_sigmas):
        self.sigmas_f[i] = self.fx(sigmas[i], self._dt)

    self.xp, self.Pp = (self.sigmas_f, self.Wm, self.Wc, self.Q).T #transform
    return(predict)
    
predict

#UPDATE STEP
def update(self, z):
    # rename for readability
    sigmas_f = self.sigmas_f
    sigmas_h = self.sigmas_h

    # transform sigma points into measurement space
    for i in range(self._num_sigmas):
        sigmas_h[i] = self.hx(sigmas_f[i])

    # mean and covariance of prediction passed through UT
    zp, Pz = (sigmas_h, self.Wm, self.Wc, self.R).T #transform

    # compute cross variance of the state and the measurements
    Pxz = np.zeros((self._dim_x, self._dim_z))
    for i in range(self._num_sigmas):
        Pxz += self.Wc[i] * np.outer(sigmas_f[i] - self.xp,
                                    sigmas_h[i] - zp)

    K = np.dot(Pxz, np.linalg.inv(Pz)) # Kalman gain

    self.x = self.xp + np.dot(K, z-zp)
    self.P = self.Pp - np.dot(np.dot(K, Pz),np.dot(K.T))
    return np.array(self.x),np.array(self.P)

innovation = ((trainY-layer_2).sum()/len(trainY))
#xNew = np.dot((self.x)-windowSize)