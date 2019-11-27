import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
# from scipy.linalg import cholesky
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints as Merwe
from filterpy.kalman import unscented_transform as UT

#%% persiapan data
data = pd.read_csv('data.csv',usecols=[1],engine='python',delimiter=',',decimal=".",thousands=',',header=None,names=['date','value'] )
data['value'] = data['value'].values
data['value'] = data['value'].astype('float32')

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

#fungsi aktivasi dan turunannya
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def dtanh(x):
    return (1-tanh(x)**2)

def normalize(data, scale):
    normalized = []
    for i in range(len(data)):
        a = (min(scale))+(data[i]-min(data))*(max(scale)-min(scale))/(max(data)-min(data))
        normalized.append(a)
    return np.array(normalized)
    scale = (-1,1)
    normalized = normalize(data,scale)

def denormalize(normalized, data, scale):
    denormalized = []
    for i in range(len(normalized)):
        a = ((normalized[i]-min(scale))*(max(data)-min(data)))/(max(scale)-min(scale))+min(data)
        denormalized.append(a)
    return np.array(denormalized)

# normalisasi dengan data satu kolom kebawah
data_raw = normalize(data['value'],(-1,1))

plt.plot(data_raw[0:1000], label='data mentah')
plt.title('data_raw')
plt.legend()
plt.show()

# pembagian data latih dan test
train_data, test_data = pisahData(data_raw, 0.7, 0.3)
train_data = train_data.reshape(-1,1) #satu kolom kebawah
test_data = test_data.reshape(-1,1)

# createwindowSize =3 untuk input
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

#%% PELATIHAN ====  gunain trainX & trainY ====
# INISIALISASI banyaknya neuron setiap layer 
alpha = 0.1
batch_dim = trainX.shape[0] # ambil jumlah baris (n) dari trainX(n,m)
input_dim = windowSize
hidden_dim = 6
output_dim = 1
jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)+(hidden_dim*output_dim)

# BOBOT === inisialisasi bobot awal (baris,kolom)
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 # inisialisasi random bobot awal
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # dengan interval [-1,1]
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # random.random ->> interval [0,1]

synapse_0_update = np.zeros_like(synapse_0) #meng-0 kan semua isi array sesuai shape dr variabel (synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# log mse tiap epoch
mse_all = []

# inisialisasi sebelum train
Q = 1*np.identity(jumlah_w) #kovarian Noise process
R = 1*np.identity(output_dim) #Kovarian Noise measurement(observasi)
P = 1*np.identity(jumlah_w) #kovarian estimasi vektor state

#%% Unscented Kalman Filter without filterpy

# inisialisasi
n_fx_output = output_dim
n_nn_hidden = hidden_dim
n_hidden_weights = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)
n_total_weights = jumlah_w
n_output_weights = output_dim
n_total_intercepts = n_nn_hidden + n_fx_output
n_fx_input = input_dim
n_total_weights = n_hidden_weights + n_output_weights
n_total_nn_parameters = n_total_weights + n_total_intercepts
n_joint_ukf_process_states = n_fx_input + n_total_nn_parameters
Xdim = n_joint_ukf_process_states
dimx = Xdim
print('Xdim',Xdim)

# SIGMA POINTS dan Weightnya (dari mean dan kovarian) pada w_concat
points = Merwe(n=1, alpha=.3, beta=2., kappa=0) #makin besar alpha, makin menyebar data[:train_data], range(train_data)
# sigmas = points.sigma_points(x, p) 
# plt.plot(sigmas, marker='x', ls=None)

kf = UKF(dim_x=1, dim_z=1, dt=.1, hx=None, fx=None, points=points) # sigma = points
    

# def titiksigma(mean):

# def bobot_titik_sigma():
    
# def fx():

# def hx():

# def 
    
# initialisasi mean dan kovarian awal
'''
        x =[]
        xx =[]
        batch_x = w_concat.shape[0]
        index_i = 0
        while(index_i+batch_x<=w_concat.shape[0]):
            xx += np.array(len(w_concat[0,:]))
            x.append(xx)
            index_i = index_i+batch_x
        
# =============================================================================
#         for i in w_concat:
#             xx = np.array(len(w_concat[0,:]))
#         x.append(x)
#         # x = np.array(len(w_concat[0,:]))
#         # p_ = np.array([[1],[1]])
# =============================================================================
        p = np.diag([1**2, 1**2]) # agar jadi integer, bukan float
        
        # SIGMA POINTS dan Weightnya (dari mean dan kovarian) pada w_concat
        points = Merwe(n=1, alpha=.3, beta=2., kappa=0) #makin besar alpha, makin menyebar data[:train_data], range(train_data)
        sigmas = points.sigma_points(x, p) 
        plt.plot(sigmas, marker='x', ls=None)
        
        kf = UKF(dim_x=1, dim_z=1, dt=.1, hx=None, fx=None, points=points) # sigma = points
        
        print(points)
        
        # inisialisasi P disini tentukan sndiri karena konstan
        sigmas_i= []
        sigma_0 = []
        for i in w_concat:
            sigma_tg_i = i # tiap i jadi mean tengah
            sigma_tg_around = sigmas(sigma_tg_i,p)
        


        sigmas_point = points.sigma_points()
        sigmas_per_point_mean.append(points.sigma_points(mean,p))
        plt.plot(sigmas_per_point_mean[0], marker='x', ls=None)
        sigmas_per_point_mean = np.array(sigmas_per_point_mean)
        plt.plot(mean_per_point[0], marker='x', ls=None)
        
# =============================================================================
#         uxs = []
#         for z in (w_concat):
#             # kf.predict()
#             # kf.update(z)
#             uxs.append(kf.x.copy())
#         uxs = np.array(uxs)
# =============================================================================
    
        plt.plot(mean,marker='x',ls=None)
        plt.plot(mean_per_point,marker='o',ls=None)
        
# =============================================================================
#         sigmas = np.zeros((2*points.n+1))
#         U = cholesky((points.n+points.kappa)*P)
#     
#         for k in range(points.n):
#             sigmas[k+1] = mean[k] + U[k]
#             sigmas[points.n+k+1] = mean[k] - U[k]
#         # sigmas = points.sigma_points(mean,P)
#       
#         dim_x = 2 # jumlah variabel (dim=1, maka dim_x=2)
#         dim_z = 2 # jumlah input measurement (posisi x,y maka dim_z=2)
#         num_sigmas = 2*points.n+1
# =============================================================================
        sigmas_f = np.zeros((kf.num_sigmas,kf.dim_x)) # dim_x
        sigmas_h = np.zeros((kf.num_sigmas,kf.dim_z)) # dim_x
# =============================================================================
#         for i in range(len(synapse_1)):
#             mean = synapse_1
#             sigma = points(mean)
#         Pz = np.dot(synapse_1[i],points[0])
# =============================================================================
        
        # ukf points (mean dan kovarian) setiap synapse layer
        # hitung bobot tiap points tersebut
        # Unscented transform untuk trasformasi mean dan kovarian ke tuple (measurement space)
        # new mean dan sigmas
        # Unscented Transform
        xp = np.eye(UT(sigmas_f,points.Wc,points.Wm,Q)) # X bar
        zp = np.eye(UT(sigmas_h,points.Wc,points.Wm,R)) # Myu z
        
# =============================================================================
#         # Cross Covariance prediksi ~> UT
#         Pz = np.zeros((dim_x,dim_z))
#         for i in range(num_sigmas):
#             Pz += points.Wc[i] * np.outer((sigmas_h[i]-zp),(sigmas_h[i]-zp)) + R
#         
#         # Cross covariance (P) dari state ke measurement
#         Pxz = np.zeros((dim_x,dim_z))
#         for i in range(num_sigmas):
#             Pxz += points.Wc[i] * np.outer(sigmas_f[i]-xp,sigmas_h[i]-zp) # outer sm dengan transpose?
# =============================================================================
'''

#%% EVALUASI ====
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
    mape = float((sum(mape))/len(y))*100
    return mape

def dstat(x,y):
    dstat = 0
    n = len(y)
    for i in range(n-1):
        if(((x[i+1]-y[i])*(y[i+1]-y[i]))>0):
            dstat += 1
    Dstat = (1/float(n-2))*float(dstat)*100
    return float(Dstat)

#%% MULAI EPOCH ============ PELATIHAN ===================

epoch = 5
start_time = time.time()
num_use_obs = len(synapse_1)
for i in range(epoch):
    index = 0
    layer_2_value = []
    context_layer = np.full((batch_dim,hidden_dim),0) 
    layer_h_deltas = np.zeros(hidden_dim) # context layer (sebelumnya)
    # while(index+batch_dim<=trainX.shape[0]):
    while i in range(index+batch_dim<=trainX.shape[0]):        
        # input dan output
        X = trainX[index:index+batch_dim,:]
        Y = trainY[index:index+batch_dim]
        index = index+batch_dim

        # forwardpass = propagated = bawa ke input ~> prev hidden
        layer_1 = tanh(np.dot(X,synapse_0) + np.dot(context_layer,synapse_h))
    
        # hidden ~> output
        layer_2 = tanh(np.dot(layer_1,synapse_1))
        layer_2_value.append(layer_2)
    
        # hitung error output
        layer_2_error = layer_2 - Y[:,None] #problemnya, y diganti dr Y matrix
        # layer_2_deltas.append((layer_2_error)*dtanh(layer_2))
        
        # error di output layer -> layer 2 deltas (masuk ke context layer dari hidden layer)
        layer_2_delta = layer_2_error*dtanh(layer_2)
        
        # error di hidden layer -> layer 1 delta (masuk ke hidden layer dari context layer)
        layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) + np.dot(layer_2_delta,synapse_1.T)) * dtanh(layer_1)
    
        # calculate weight update
        synapse_1_update = np.dot(np.atleast_2d(layer_1).T,(layer_2_delta))
        synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
        synapse_0_update = np.dot(X.T,(layer_1_delta))
        
        # concatenate weight
        synapse_0_c = np.reshape(synapse_0,(-1,1))
        synapse_h_c = np.reshape(synapse_h,(-1,1))
        synapse_1_c = np.reshape(synapse_1,(-1,1))
        w_concat = np.concatenate((synapse_0_c,synapse_h_c,synapse_1_c), axis=0)
                
        # === UKF ===
        zp, Pz = UT(kf.sigmas_f, points.Wm, points.Wc) # UT buat new mean dan kovarian
        Pxz = kf.cross_variance(w_concat[i], zp ,kf.sigmas_f, kf.sigmas_h)
        print('Pxz',Pxz)
        
        # Kalman gain
        # K = kf.K(w_concat)
        K = np.dot(Pxz,inv(Pz))
        
        # update P
        P1 = np.dot(K,Pz)
        P2 = np.dot(P1,K.T)
        P = P-P2
        
        # update weight
        innovation = ((Y-layer_2).sum()/len(layer_2_error)) # hitung error pred dan yang diharapkan
        w_concat_new = w_concat + np.dot(K,innovation)

        # assign bobot versi ekf
        synapse_0 = w_concat_new[0:(input_dim*hidden_dim),0]
        synapse_h = w_concat_new[(input_dim*hidden_dim):(input_dim*hidden_dim)+(hidden_dim*hidden_dim),0]
        synapse_1 = w_concat_new[(input_dim*hidden_dim)+(hidden_dim*hidden_dim):w_concat_new.shape[0],0]
        
        # reshape balik bobot
        synapse_0 = np.reshape(synapse_0,(input_dim,hidden_dim))
        synapse_h = np.reshape(synapse_h,(hidden_dim,hidden_dim))
        synapse_1 = np.reshape(synapse_1,(hidden_dim,output_dim))
    
        # reset update
        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0
    
        # update context layer
        layer_h_deltas = layer_1_delta
        context_layer = layer_1
               
    layer_2_value = np.reshape(layer_2_value,(-1,1))
    mse_epoch = mse(trainY,layer_2_value)
    mse_all.append(mse_epoch)
    
run_time = time.time() - start_time
#%% seberapa besar lossnya???

plt.plot(mse_all,label='loss')
plt.title('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

#%% mari coba ============ PREDIKSI ===================

batch_predict = testX.shape[0] # mengambil banyaknya baris (n) dari testX(n,m)
context_layer_p = np.full((batch_predict,hidden_dim),0) # return full array yg isinya (0) sebesar dimensi [batch_predict x hidden_dim]
y_pred = [] # hasil akhir Y prediksi
index = 0
while(index+batch_predict<=testX.shape[0]):
    X = testX[index:index+batch_predict,:]
    layer_1p = tanh(np.dot(X,synapse_0)+np.dot(context_layer_p,synapse_h))
    layer_2p = tanh(np.dot(layer_1p,synapse_1))
    y_pred.append(layer_2p)
    context_layer_p = layer_1p
    index = index+batch_predict
    
y_pred = denormalize(np.reshape(y_pred,(-1,1)), data['value'], (-1,1))
testYseb = testY.reshape(-1,1)
testY = denormalize(testY, data['value'], (-1,1))
mse_pred = mse(testY,y_pred)
rmse_pred = rmse(testY,y_pred)
mae_pred = mae(testY,y_pred)
mape_pred = mape(testY,y_pred)
dstat_pred = dstat(testY,y_pred)
scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred,run_time]

plt.plot(testYseb[0:50], label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(layer_2p[0:50], label='prediction')
plt.xlabel('Data ke-')
plt.ylabel('Harga')
plt.title('RNN-UKF sebelun denormalisasi')
plt.legend()
plt.show()

plt.plot(testY[0:50], label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(y_pred[0:50], label='prediction')
plt.xlabel('Data ke-')
plt.ylabel('Harga')
plt.title('RNN-UKF')
plt.legend()
plt.show()

print('synapse_1 dimensi', synapse_1.ndim)
plt.plot(synapse_1[:], label='synapse_1', marker ='o') #testY[0:50] untuk plotting (catatan)
plt.xlabel('bobot ke')
plt.ylabel('value')
plt.title('Bobot')
plt.legend()
plt.show()

#scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred,run_time]
print("runtime : ", run_time)
print("mse : " , mse_pred)
print("rmse : ", rmse_pred) 
print("mape : ", mape_pred) 
print("mae: " , mae_pred)
print("dstat : " , dstat_pred)

#%%
np.savetxt('bobot_input.csv', synapse_0, delimiter=',')
np.savetxt('bobot_hidden.csv', synapse_h, delimiter=',')
np.savetxt('bobot_output.csv', synapse_1, delimiter=',')
np.savetxt('loss_ukf.csv', mse_all, delimiter=';')