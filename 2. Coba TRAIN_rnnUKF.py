import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv

#persiapan data
data = pd.read_csv('data.csv',usecols=[1],
                    engine='python',
                    delimiter=',',
                    decimal=".",
                    thousands=',',
                    header=None,
                    names=['date','value'] )
data = data.values
data = data.astype('float32')

#normalisasi duluan
def normalize(data,scale):
    normalized = []
    for i in range(len(data)):
       a = (min(scale))+((data[i]-min(data))*
            max(scale)-min(scale))/(max(data)-min(data))
       normalized.append(a)
    return np.array(normalized)

scale = (-1,1)
normalized = normalize(data,scale)

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
train_data = train_data.reshape(-1,1) #reshape data dengan range -1,1
test_data = test_data.reshape(-1,1)

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

#initialize neuron size
batch_dim = trainX.shape[0]

#inisiallisasi dimensi layer
input_dim = windowSize
hidden_dim = 6
output_dim = 1

np.random.seed(4)

#fungsi aktivasi dan turunannya
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

tanh_ = tanh()
print(tanh_)

def dtanh(x):
    return (1-tanh(x)**2)

dtanh_ = dtanh()
print(dtanh_)

#inisialisasi random BOBOOOTT awal JST
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 #inisialisasi
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 #bobotjaringan dg
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 #interval[-1,1]

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#log mse tiap epoch
mse_all = []

jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)+(hidden_dim*output_dim)
Q = 0.01*np.identity(jumlah_w)
R = 1*np.identity(output_dim)
P = 1*np.identity(jumlah_w)

#inisialisasi Q,R wajib buat UKF sbg kovarian proses dan measurement
#inisiallsasi P sbg error kovarian untuk saat ini (state)
context_layer = np.full((batch_dim,hidden_dim),0) # bedanya batch dengan trainX???
#context_layer = np.full((trainX.shape[0],hidden_dim),0)
# =============================================================================
layer_h_deltas = np.zeros(hidden_dim)
# Q = (np.std(data)*np.std(data))*np.eye(windowSize) #kovarian Noise process
# R = np.std(data)*np.std(data) #Kovarian Noise measurement(observasi)
# P = np.eye(windowSize) #kovarian estimasi vektor state
# V = np.zeros((windowSize), dtype=int)
# =============================================================================

# =============================================================================
# forward pass (hitug NILAI neuron pada layer)
layer_1 = tanh(np.dot(trainX,synapse_0) +
              np.dot(context_layer,synapse_h)) #di hidden layer (sbg zh di paper)
layer_2 = tanh(np.dot(layer_1,synapse_1)) #di output layer (sbg yo di paper)

#update bobot (GAPERLU)
layer_2_error = trainY - layer_2 #nilai error dari prediksi buat jacobian
layer_2_delta = layer_2_error*dtanh(layer_2)

layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) + 
                 np.dot(layer_2_delta,synapse_1.T)) * dtanh(layer_1)

# layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) +
#                  np.multiply(layer_2_delta,synapse_1.T))*dtanh(layer_1)
synapse_1_update = np.dot(np.atleast_2d(layer_1).T, (layer_2_delta))
# =============================================================================
synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
synapse_0_update = np.dot(trainX.T,(layer_1_delta))


#%% ----------------------------------------------------------
#UNSCENTED Kalman Filter (ALL)

#INISIALISASI

#inisialisasi faktor skalar
#m = np.sum(trainY)
#L = np.sum(trainX)

#inisialisasi PARAMETER -> van der merwe suggest using beta =2 kappa = 3-n
#def ukF(beta, aplpha, n, kappa)
beta = 2
alpha = np.random.random(1)
n = 1 #n = dimensi dari x
kappa = 3-n #atau 0 oleh dash 2014

##WEIGHTS
lambda_ = alpha**2 * (n + kappa) - n
Wc = np.full(2*n + 1,  1. / (2*(n + lambda_)))
Wm = np.full(2*n + 1,  1. / (2*(n + lambda_)))
Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
Wm[0] = lambda_ / (n + lambda_)

#SIGMA POINTS masih salaah
#a
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

def P_(kmax,n,k):
    P = np.zeros((n, n))
    for i in range(kmax):
        y = sigmas[i] - x
        P += Wc[i] * np.outer(y, y) 
        P += Q
        y = (sigmas[k] - x).reshape(kmax, 1) # convert into 2D array
        P += Wc[k] * np.dot(y, y.T) #P += Wc[K] * np.dot(y, y.T)
        return()

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

#%%
innovation = ((trainY-layer_2).sum()/len(trainY))
#w_concat_new = w_concat + np.dot(K,innovation)

#prediksi data uji
layer_h_deltas = layer_1_delta
context_layer = layer_1

context_layer_p = np.full((testY.shape[0],hidden_dim),0)
layer_lp = tanh(np.dot(testX,synapse_0) +
               np.dot(context_layer_p,synapse_h))
layer_2p = tanh(np.dot(layer_lp,synapse_1))

#denorm
def denormalize(normalized, data, scale):
    denormalized = []
    for i in range(len(normalized)):
        a = ((normalized[i]-min(scale))*(max(data)-
              min(data)))/(max(scale)-min(scale))+min(data)
        denormalized.append(a)
    return np.array(denormalize)
y_pred = denormalize(layer_2p, test_data, scale)
testY = denormalize(testY, test_data, scale)

def mse(x,y):
    mse = []
    for i in range(len(x)):
        a = (x[i]-y[i])**2
        mse.append(a)
    mse = float((sum(mse)/len(y)))
    return mse

def rmse(x,y):
    rmse = []
    for i in range(len(x)):
        a = (x[i]-y[i])**2
        mse.append(a)
    rmse = float((sum(mse)/len(y))**0.5)
    return rmse

def mae(x,y):
    mae = []
    for i in range(len(x)):
        a = abs(x[i]-y[i])
        mae.append(a)
    mae = float((sum(mse))/len(y))
    return mae

def mape(x,y):
    mape = []
    for i in range(len(x)):
        a = abs((x[i]-y[i])/x[i])
        mape.append(a)
    mape = float((sum(mse))/len(y))*100
    return mape

def dstat(x,y):
    dstat = 0
    n = len(y)
    for i in range(n-1):
        if(((x[i+1]-y[i])*(y[i+1]-y[i]))>=0):
            dstat += 1
    Dstat = (1/float(n-1))*float(dstat)*100
    return float(Dstat)

epoch = 1270
start_time = time.time()
for i in range(epoch):
    index = 0
    layer_2_value = []
    context_layer = np.full((batch_dim,hidden_dim),0)
    layer_h_deltas = np.zeros(hidden_dim)
    while(index+batch_dim<=trainX.shape[0]):
        X = trainX[index:index+batch_dim,:]
        Y = trainY[index:index+batch_dim]
        index = index+batch_dim

        #input to hidden
        layer_1 = tanh(np.dot(X,synapse_0)+np.dot(context_layer,synapse_h))
    
        #hidden to output
        layer_2 = tanh(np.dot(layer_1,synapse_1))
        layer_2_value.append(layer_2)
    
        #hitung error output
        layer_2_error = Y - layer_2
    
        #layer 2 deltas
        layer_2_delta = layer_2_error*dtanh(layer_2)
    
        #layer 1 delta
        layer_1_delta = (np.multiply(layer_h_deltas,synapse_h.T) + np.multiply(layer_2_delta,synapse_1.T)) * dtanh(layer_1)
    
        #calculate weight update
        synapse_1_update = np.dot(np.atleast_2d(layer_1).T,(layer_2_delta))
        synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
        synapse_0_update = np.dot(X.T,(layer_1_delta))
        
        #concatenate weight
        synapse_0_c = np.reshape(synapse_0,(-1,1))
        synapse_h_c = np.reshape(synapse_h,(-1,1))
        synapse_1_c = np.reshape(synapse_1,(-1,1))
        w_concat = np.concatenate((synapse_0_c,synapse_h_c,synapse_1_c), axis=0)
        
        #jacobian
        dsynapse_0 = np.reshape(synapse_0_update,(1,-1))
        dsynapse_h = np.reshape(synapse_h_update,(1,-1))
        dsynapse_1 = np.reshape(synapse_1_update,(1,-1))
        H = np.concatenate((dsynapse_0,dsynapse_h,dsynapse_1), axis=1)
        H_transpose = H.T
        
        #Kalman Gain
        K1 = np.dot(H,inv(np.reshape(P,1,8235)))
        K2 = np.dot(K1,H_transpose)+R
        K3 = inv(K2)
        K4 = np.dot(P,H_transpose)
        K = np.dot(K4,K3)
        
        #update weight
        #innovation : selisih output prediksi dan output yang diinginkan
        innovation = ((Y-layer_2).sum()/len(layer_2_error))
        w_concat_new = w_concat + np.dot(K,innovation)
        
        #update P
        P1 = np.dot(K,H)
        P2 = np.dot(P1,P)+Q
        P = P-P2
        
        #assign bobot
        synapse_0 = w_concat_new[0:(input_dim*hidden_dim),0]
        synapse_h = w_concat_new[(input_dim*hidden_dim):(input_dim*hidden_dim)+(hidden_dim*hidden_dim),0]
        synapse_1 = w_concat_new[(input_dim*hidden_dim)+(hidden_dim*hidden_dim):w_concat_new.shape[0],0]
        
        #reshape balik bobot
        synapse_0 = np.reshape(synapse_0,(input_dim,hidden_dim))
        synapse_h = np.reshape(synapse_h,(hidden_dim,hidden_dim))
        synapse_1 = np.reshape(synapse_1,(hidden_dim,output_dim))
    
        #reset update
        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0
    
#        update context layer
        layer_h_deltas = layer_1_delta
        context_layer = layer_1
    
    layer_2_value = np.reshape(layer_2_value,(-1,1))
    mse_epoch = mse(trainY,layer_2_value)
    mse_all.append(mse_epoch)
run_time = time.time() - start_time

#%%
plt.plot(mse_all,label='loss')
plt.title('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

#%% 

#coba predict

batch_predict = testX.shape[0]
context_layer_p = np.full((batch_predict,hidden_dim),0)
y_pred = []
index = 0
while(index+batch_predict<=testX.shape[0]):
    X = testX[index:index+batch_predict,:]
    layer_1p = tanh(np.dot(X,synapse_0)+np.dot(context_layer_p,synapse_h))
    layer_2p = tanh(np.dot(layer_1p,synapse_1))
    y_pred.append(layer_2p)
    context_layer_p = layer_1p
    index = index+batch_predict
    
y_pred = denormalize(np.reshape(y_pred,(-1,1)), data['value'], (-1,1))
testY = denormalize(testY, data['value'], (-1,1))
mse_pred = mse(testY,y_pred)
rmse_pred = rmse(testY,y_pred)
mae_pred = mae(testY,y_pred)
dstat_pred = dstat(testY,y_pred)
scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred, run_time]

plt.plot(testY, label='true')
plt.plot(y_pred, label='prediction')
plt.title('RNN-UKF')
plt.legend()
plt.show()
print(scoring)

#%%
plt.plot(testY[0:50], label='true')
plt.plot(y_pred[0:50], label='prediction')
plt.title('RNN-UKF')
plt.legend()
plt.show()

#%%
np.savetxt('bobot_input.csv', synapse_0, delimiter=',')
np.savetxt('bobot_hidden.csv', synapse_h, delimiter=',')
np.savetxt('bobot_output.csv', synapse_1, delimiter=',')
np.savetxt('loss_ukf.csv', mse_all, delimiter=';')

#%%

