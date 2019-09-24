import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv

def pisahData(data,a,b):
     if((a+b)!=1):
            print("pemisahan tidak valid")
     else:
        train = []
        test = []
        train_size = int(len(data)*a)
        train = data[0:train_size-1]
        test = data[train_size-1:len(data)]
     return np.array(train),np.array(test)

def normalize(data, scale):
    normalized = []
    for i in range(len(data)):
        a = (min(scale))+(data[i]-min(data))*(max(scale)-min(scale))/(max(data)-min(data))
        normalized.append(a)
    return np.array(normalized)

def denormalize(normalized, data, scale):
    denormalized = []
    for i in range(len(normalized)):
        a = ((normalized[i]-min(scale))*(max(data)-min(data)))/(max(scale)-min(scale))+min(data)
        denormalized.append(a)
    return np.array(denormalized)

# =============================================================================
# scale = (-1,1)
# normalized = normalize(data,scale)
# print(normalized)
# plt.show()
# =============================================================================


#persiapan data
data = pd.read_csv('data.csv',usecols=[1],
                    engine='python',
                    delimiter=',',
                    decimal=".",
                    thousands=',',
                    header=None,
                    names=['date','value'] )
data['value'] = data['value'].values
data['value'] = data['value'].astype('float32')
data_raw = normalize(data['value'],(-1,1))

train_data, test_data = pisahData(data_raw, 0.7, 0.3)
train_data = train_data.reshape(-1,1) #reshape data dengan range -1,1 -> satu kolom kebawah
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
input_dim = windowSize
hidden_dim = 6
output_dim = 1

np.random.seed(4)

#fungsi aktivasi dan turunannya
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def dtanh(x):
    return (1-tanh(x)**2)

#inisialisasi random BOBOOOTT awal JST
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 #inisialisasi
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 #bobotjaringan dg
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 #interval[-1,1]

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#log mse tiap epoch
mse_all = []

#inisialisasi sebelum train
jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)+(hidden_dim*output_dim)
Q = 0.01*np.identity(jumlah_w)
R = 1*np.identity(output_dim)
P = 1*np.identity(jumlah_w)

context_layer = np.full((batch_dim,hidden_dim),0)
layer_h_deltas = np.zeros(hidden_dim)

# inisialisasi Q,R wajib buat UKF sbg kovarian proses dan measurement
# inisiallsasi P sbg error kovarian untuk saat ini (state)
# context_layer = np.full((batch_dim,hidden_dim),0) # bedanya batch dengan trainX???
# context_layer = np.full((trainX.shape[0],hidden_dim),0)
# layer_h_deltas = np.zeros(hidden_dim)
# # yang asli, tapi bermasalah di jumlah dimensi P, maka diganti dg jumlah_w "inisialisaasi sebelum train" =============================================================================
# Q = (np.std(data)*np.std(data))*np.eye(windowSize) #kovarian Noise process
# R = np.std(data)*np.std(data) #Kovarian Noise measurement(observasi)
# P = np.eye(windowSize) #kovarian estimasi vektor state
# V = np.zeros((windowSize), dtype=int)
# =============================================================================

#forward pass (hitug NILAI neuron pada layer)
# =============================================================================
# layer_1 = tanh(np.dot(trainX,synapse_0) +
#               np.dot(context_layer,synapse_h)) #di hidden layer (sbg zh di paper)
# layer_2 = tanh(np.dot(layer_1,synapse_1)) #di output layer (sbg yo di paper)
# 
# =============================================================================
#update bobot (GAPERLU) sudah dibawah
# =============================================================================
# layer_2_error = layer_2 - trainY #nilai error dari prediksi buat jacobian
# layer_2_delta = layer_2_error*dtanh(layer_2)
# layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) +
#                  np.multiply(layer_2_delta,synapse_1.T))*dtanh(layer_1)
# synapse_1_update = np.dot(np.atleast_2d(layer_1).T, (layer_2_delta))
# synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
# synapse_0_update = np.dot(trainX.T,(layer_1_delta))
# 
# =============================================================================

#%% ----------------------------------------------------------
#UNSCENTED Kalman Filter (ALL)

#INISIALISASI

#inisialisasi faktor skalar
#m = np.sum(trainY)
#L = np.sum(trainX)

#inisialisasi PARAMETER -> van der merwe suggest using beta =2 kappa = 3-n
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
#innovation = ((trainY-layer_2).sum()/len(trainY))
#w_concat_new = w_concat + np.dot(K,innovation)

#prediksi data uji
# =============================================================================
# layer_h_deltas = layer_1_delta
# context_layer = layer_1
# =============================================================================

# =============================================================================
# batch_predict = testX.shape[0]
# context_layer_p = np.full((testY.shape[0],hidden_dim),0)
# layer_lp = tanh(np.dot(testX,synapse_0) +
#                np.dot(context_layer_p,synapse_h))
# layer_2p = tanh(np.dot(layer_lp,synapse_1))
# 
# y_pred = []
# y_pred = denormalize(np.reshape(y_pred,(-1,1)), data['value'], (-1,1))
# testY = denormalize(testY, data['value'], (-1,1))
# 
# =============================================================================
def mse(x,y):
    mse = []
    for i in range(len(y)):
        a = (x[i]-y[i])**2
        mse.append(a)
    mse = float((sum(mse)/len(y)))
    return mse


def mae(x,y):
    mae = []
    for i in range(len(y)):
        a = abs(y[i]-x[i])
        mae.append(a)
    mae = float(sum(mae)/len(y))
    return mae

def rmse(x,y):
    rmse = []
    for i in range(len(y)):
        a = (x[i]-y[i])**2
        rmse.append(a)
    rmse = float((sum(rmse)/len(y))**0.5)
    return rmse

def mape(x,y):
    mape = []
    for i in range(len(y)):
        a = abs((x[i]-y[i])/x[i])
        mape.append(a)
    mape = float((sum(mse))/len(y))*100
    return mape

def dstat(x,y):
    dstat = 0
    n = len(y)
    for i in range(n-1):
        if(((x[i+1]-y[i])*(y[i+1]-y[i]))>0):
            dstat += 1
    Dstat = (1/float(n-2))*float(dstat)*100
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

        # forward passs -> input to hidden
        layer_1 = tanh(np.dot(X,synapse_0)+np.dot(context_layer,synapse_h))
    
        #hidden to output
        layer_2 = tanh(np.dot(layer_1,synapse_1))
        layer_2_value.append(layer_2)
    
        #hitung error output
        layer_2_error = layer_2 - Y[:,None] #problemnya, y diganti dr Y matrix
    
        #layer 2 deltas
        layer_2_delta = layer_2_error*dtanh(layer_2)
    
        #layer 1 delta
        layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) + np.dot(layer_2_delta,synapse_1.T)) * dtanh(layer_1)
    
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
        K1 = np.dot(H,P)
        K2 = np.dot(K1,H_transpose)+R
        K3 = inv(K2)
        K4 = np.dot(P,H_transpose)
        K = np.dot(K4,K3)
        
        #update weight
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




# =============================================================================
# #%% ----------------------------------------------------------
# #UNSCENTED KALMAN FILTER (ALL)
# 
# #inisialisasi faktor skalar
# #m = np.sum(trainY)
# #L = np.sum(trainX)
# 
# #inisialisasi PARAMETER -> van der merwe suggest using beta =2 kappa = 3-n
# beta = 2
# alpha = np.random.random(1)
# n = 1 #n = dimensi dari x
# kappa = 3-n #atau 0 oleh dash 2014
# 
# #initial mean and covariance
# mean = (0., 0.)
# p = np.array([[32., 15], [15., 40.]])
# 
# # create sigma points and weights
# points = MerweScaledSigmaPoints(n=2, alpha=.3, beta=2., kappa=.1)
# sigmas = points.sigma_points(mean, p)
# 
# ### pass through nonlinear function
# sigmas_f = np.empty((5, 2))
# for i in range(5):
#     sigmas_f[i] = sigmas_f(sigmas[i, 0], sigmas[i ,1]) ##? 
# 
# ### use unscented transform to get new mean and covariance
# ukf_mean, ukf_cov = unscented_transform(sigmas_f, points.Wm, points.Wc)
# 
# #generate random points
# np.random.seed(100)
# xs, ys = trainX(mean=mean, cov=p, size=5000).T
# 
# mean(xs, ys, trainX, ukf_mean, 'Unscented Mean')
# ax = plt.gcf().axes[0]
# ax.scatter(sigmas[:,0], sigmas[:,1], c='r', s=30);
# 
# ##WEIGHTS
# lambda_ = alpha**2 * (n + kappa) - n
# Wc = np.full(2*n + 1,  1. / (2*(n + lambda_)))
# Wm = np.full(2*n + 1,  1. / (2*(n + lambda_)))
# Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
# Wm[0] = lambda_ / (n + lambda_)
# 
# #SIGMA POINTS masih salaah
# #a
# s = np.sqrt(data) #data atau x??? data kok
# np.dot(s, s.T)
# 
# sigmas = np.zeros((2*n+1, n))
# U = np.sqrt((n+lambda_)*P) # sqrt
# 
# def sigmaPoint(data,X,k):
#     sigmas[0] = X
#     for k in range(n):
#         sigmas[k+1]   = X + U[k]
#         sigmas[n+k+1] = X - U[k]
#         x = np.dot(Wm, sigmas) #jumlah sigma mean atau means
#         return np.dot(x,n)
#     
# x = np.dot(Wm, sigmas)
# n = sigmas.shape
# sg = sigmaPoint
# 
# def P_(kmax,n,k):
#     P = np.zeros((n, n))
#     for i in range(kmax):
#         y = sigmas[i] - x
#         P += Wc[i] * np.outer(y, y) 
#         P += Q
#         y = (sigmas[k] - x).reshape(kmax, 1) # convert into 2D array
#         P += Wc[k] * np.dot(y, y.T) #P += Wc[K] * np.dot(y, y.T)
#         return()
# =============================================================================

# =============================================================================
# #PREDIKSI (STEP)
# def predict(sigma_points_fn):
#     """ Performs the predict step of the UKF. On return, 
#     trainY and P contain the predicted state (trainY) 
#     and covariance (P). 'p' stands for prediction.
#     """
# 
#     # calculate sigma points for given mean and covariance
#     sigmas = sigma_points_fn(x,P)
# 
#     for i in range(sigmas):
#         trainX[i] = synapse_0(sigmas[i])
# 
#     xp,Pp = (sigmas_f,Wm,Wc,Q).T #transform
#     return(predict)
#     
# predict
# 
# #UPDATE STEP
# def update(z):
#     # transform sigma points into measurement space
#     for i in range(sigmas):
#         sigmas_h[i] = hx(sigmas_f[i])
# 
#     # mean and covariance of prediction passed through UT
#     zp, Pz = (sigmas_h, Wm,Wc, R).T #transform
# 
#     # compute cross variance of the state and the measurements
#     Pxz = np.zeros(input_dim, hidden_dim)
#     for i in range(20):
#         Pxz += Wc[i] * np.outer(sigmas_f[i] - xp,sigmas_h[i] - zp)
# 
#     K = np.dot(Pxz, np.linalg.inv(Pz)) # Kalman gain
# 
#     x = xp + np.dot(K, z-zp)
#     P = Pp - np.dot(np.dot(K, Pz),np.dot(K.T))
#     return np.array(x),np.array(P)
# =============================================================================

#%%
# =============================================================================
# def transition_funciton(trainX, noise):
#     '''
#     --------
#     The function that evolve the state from k-1 to k.
#     --------
#     Inputs: 
#     state: State vector of the system
#     noise: The noise of the dynamic process
#     '''
#     a = np.sin(trainX[0]) + trainX[1] + noise[0]
#     b = trainX[1] + noise[1]
# 
#     return np.array([a,b])
# 
# def observation_function(trainX, noise):
#     '''
#     The function is about the relationship between the state vector 
#     and the external measurement
#     '''
#     C = np.array([
#         [-1, 0.5],
#         [0.2, 0.1]
#     ])
#     return np.dot(C, state) + noise
# 
# 
# '''----INITIALIZE THE PARAMETERS----'''
# transition_covariance = np.eye(2)
# noise_generator = np.random.RandomState(0)
# observation_covariance = np.eye(2) + noise_generator.randn(2,2) * 0.01
# Initial_state = [0,0]
# intial_covariance = [[1,0.1], [-0.1,1]]
# 
# # UKF
# kf = UnscentedKalmanFilter(
#     transition_funciton, observation_function,
#     transition_covariance, observation_covariance,
#     Initial_state,intial_covariance,
#     random_state=noise_generator
# )
# 
# trainX, observations = kf.sample(500,Initial_state)
# # estimate state with filtering and smoothing
# filtered_state_estimates = kf.filter(observations)[0]
# smoothed_state_estimates = kf.smooth(observations)[0]
# 
# =============================================================================

# =============================================================================
# #UNSCENTED KALMAN FILTER
# 
# #inisialisasi parameter yang dibutuhkan: sigma dan bobot mean & kovarian
# sg1 = np.eye(2) #transition_covariance
# sg2 = np.random.RandomState(0) #noise_generator
# sg3 = np.eye(2) + sg2.randn(2,2) * 0.0 #observation_covariance
# sg4 = [0,0] #Initial_state
# sg5 = [[1,0.1], [-0.1,1]] #intial_covariance
# 
# def ukf(foFx, dataX, P, Nmeas, hmeas, z, Q, R):
#         """ Unscented Kalman Filter for nonlinear dynamic systems
# 
#     :param FofX:    function handle for f(x)
#     :param x:       "a priori" state estimate
#     :param P:       "a priori" estimated state covariance
#     :param hmeas:   function handle for h(x)
#     :param z:       current measurement
#     :param Q:       process noise covariance
#     :param R:       measurement noise covariance
#     :N
#     :return:    x:  "a posteriori" state estimate
#                 P:  "a posteriori" state covariance            
#     """
#         Nstates = trainX.size
# #       Nmeas = trainY.size
#         
#         # tunables
#         sigmas = MerweScaledSigmaPoints(5,
#         alpha = 1e-1  # default, tunable
#         kappa = 0  # default, tunable
#         beta = 2  # default, tunable
# 
#         # params
#         Lambda = (alpha ** 2) * (Nstates + kappa) - Nstates  # scaling factor
#         cc = Nstates + Lambda  # scaling factor
# 
#         # (2) komputasi weights buat titik sigma yang telah ditentukan
#         Wm = (0.5/cc) #Wm ke i sama dengan Wc ke i
#         Wc = Wm.copy() 
#         Wm[0,0] = Lambda / cc  # weights for means
#         Wc[0,0] += Lambda/(1 - alpha ** 2 + beta)  # weights for covariance
# 
#         # (1) sigma points around x
#         sigmas = MerweScaledSigmaPoints
#         X = sigmas(x, P, sqrt(cc))
#         # unscented transformation of process
#         x1, X1, P1, X2 = ut(FofX, X, Wm, Wc, Nstates, Q)
#         # unscented transformation of measurements
#         z1, Z1, P2, Z2 = ut(hmeas, X1, Wm, Wc, Nmeas, R)
#         P12 = X2 @ diagflat(Wc) @ Z2.T  # transformed cross-covariance
#         K = P12 * inv(P2)
#         x = x1 + K @ (z - z1)  # state update
#         P = P1 - K @ P12.T  # covariance update
#     
# =============================================================================
