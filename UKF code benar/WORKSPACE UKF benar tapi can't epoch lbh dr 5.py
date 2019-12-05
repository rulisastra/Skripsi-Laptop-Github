import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import cholesky

#%% persiapan data

# IDRUSD.csv = 64% | 71#
# data baru  mulai 27sep2017.csv = 0.0
# data baru all = 21%
# data = 56%
# Ether mulai 1jul2017 = 52%
# Ether = 55%
# Litecoin mulai 1jan2017.csv = 55%
# NEO jul2017 = 62%
# NEO all = 63
# BTCIDR 1sept2017 | 74% 
# Currency Converter.csv = 70%

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

def norm(x, scale):
    normalized = []
    for i in range(len(x)):
        a = (min(scale))+(x[i]-min(x))*(max(scale)-min(scale))/(max(x)-min(x))
        normalized.append(a)
    return np.array(normalized)
    scale = (-1,1)
    normalized = normalize(x,scale)

# normalisasi dengan data satu kolom kebawah
data_raw = normalize(data['value'],(-1,1))

# pembagian data latih dan test
train_data, test_data = pisahData(data_raw, 0.8, 0.2) #8:2 = 71%
train_data = train_data.reshape(-1,1) #satu kolom kebawah
test_data = test_data.reshape(-1,1)

plt.plot(data[774:-1], c='r', label='test', ls=None)
plt.plot(data[0:774], c='g', label='training', ls=None)
plt.title('DATA 8:2')
plt.xlabel('Hari-ke')
plt.ylabel('Harga')
plt.legend()
plt.show()

plt.plot(data_raw, c='b', label='test', ls=None)
plt.plot(train_data[0:774], c='r', label='training', ls=None)
plt.title('jangjauan ternormalisasi 8:2')
plt.xlabel('Hari-ke')
plt.ylabel('value ternormalisasi')
plt.legend()
plt.show()

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
windowSize = 5 # 5
trainX, trainY = createDataset(train_data,windowSize)
testX, testY = createDataset(test_data, windowSize)

#%% PELATIHAN ====  gunain trainX & trainY ====
# INISIALISASI banyaknya neuron setiap layer 
alpha = .1
batch_dim = trainX.shape[0] # ambil jumlah baris (n) dari trainX(n,m)
input_dim = windowSize
hidden_dim = 7 # 7,9,
output_dim = 1

np.random.seed(1) # 1 =72%

# BOBOT === inisialisasi bobot awal (baris,kolom)
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 # inisialisasi random bobot awal
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # dengan interval [-1,1]
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # random.random ->> interval [0,1]

synapse_0_update = np.zeros_like(synapse_0) #meng-0 kan semua isi array sesuai shape dr variabel (synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# log mse tiap epoch
mse_all = []
rmse_all = []

# inisialisasi sebelum train
jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)+(hidden_dim*output_dim)
Q = 1*np.identity(jumlah_w) #kovarian Noise process
R = 1*np.identity(output_dim) #Kovarian Noise measurement(observasi)
# P = 1
P = 1*np.identity(jumlah_w) #kovarian estimasi vektor state

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

#%% MULAI EPOCH ============ TRAINING ===================

epoch = 5 # 100
start_time = time.time()
for i in range(epoch):
    index = 0
    layer_2_value = []
    context_layer = np.full((batch_dim,hidden_dim),0) 
    layer_h_deltas = np.zeros(hidden_dim) # context layer (sebelumnya)
    while(index+batch_dim<=trainX.shape[0]):        
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
        synapse_0_update = np.dot(trainX.T,(layer_1_delta))
        
        # concatenate weight
        synapse_0_c = np.reshape(synapse_0,(-1,1))
        synapse_h_c = np.reshape(synapse_h,(-1,1))
        synapse_1_c = np.reshape(synapse_1,(-1,1))
        w_concat = np.concatenate((synapse_0_c,synapse_h_c,synapse_1_c), axis=0)
        
        synapse_0_masuk = np.reshape(synapse_0_update,(1,-1)) # satu baris kesamping
        synapse_h_masuk = np.reshape(synapse_h_update,(1,-1))
        synapse_1_masuk = np.reshape(synapse_1_update,(1,-1))
        masuk = np.concatenate((synapse_0_masuk,synapse_h_masuk,synapse_1_masuk), axis=1)
        masuk_t = masuk.T
        
        n = masuk.size
        beta = 2.
        kappa = 3-n
        # lambda_ = alpha**2 * (n + kappa) - n
        lambda_ = 1
        
        # myu = np.mean(masuk)
        myu = masuk        
        
        # cari sigma
        U = cholesky((n + lambda_)*P) # sama dg np.sqrt
        # sigmas = np.zeros((2*n+1, n)) # filterpy version dengan shape (121,60) karena dikali dg P!
        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = myu
        for k in range(n):
            sigmas[k+1] = np.subtract(myu, -U[k])
            sigmas[n+k+1] = np.subtract(masuk, U[k])
            
# =============================================================================
#         plt.plot(sigmas[0], color='r', marker ='o', ls=None)
#         plt.plot(sigmas[k+1],'x')
#         plt.plot(sigmas[n+k+1],'x')
#         plt.xlabel('sigmas ke')
#         plt.ylabel('value')
#         plt.title('SIGMAS semua')
#         plt.show()
# =============================================================================
        
        # gabung (121,60) kebawah.. jadinya 121,60
        # sigmas_concat = np.concatenate((sigma,s_n,s_n2), axis=0)
        
        # cari pembobotan sigmanya
        #%% BOBOT SIGMA dari Merwe
        c_ = .5 / (n + lambda_)
        Wm = np.full(2*n+1, c_)
        Wc = Wm # size (121,) atau (n,)
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        Wm[0] = lambda_ / (n + lambda_)
        
        '''
        # masuk ke tahap measurement
        '''
        Z = sigmas
        myu_z = np.sum(np.dot(Wm,Z)) # sigma besar artinya jumlah dari semua 
        Z_mz = np.subtract(Z,myu_z)
        Pz1 = np.dot(Z_mz,Z_mz.T)
        Pz2 = np.dot(Wc,Pz1)
        Pz = np.sum(Pz2) + R

# =============================================================================
#         plt.plot(Pz1[0], color='g', marker ='o', ls=None)
#         plt.plot(Pz1[k+1],'x')
#         plt.plot(Pz1[n+k+1],'x')
#         plt.xlabel('KOVARIAN ke')
#         plt.ylabel('value')
#         plt.title('KOVARIAN (P) semua')
#         plt.show()
# =============================================================================
        
        Pz_inv = inv(Pz)
        Tut = Pz2.sum()
        K = np.dot(Tut,Pz_inv)
        
        # masuk ke bobot
        innovation = ((Y-layer_2).sum()/len(layer_2_error)) # hitung error pred dan yang diharapkan
        w_concat_new = w_concat + np.dot(K,innovation)
        
        # update P
        K_transpose = K.T
        P2 = np.dot(Pz,K_transpose)
        P1 = np.dot(K,P2)
        # P = np.subtract(P,P1)
        
        P = P*P1
        # P = np.round(P,2)
        
# =============================================================================
#         # mms = np.reshape(w_concat,(1,-1)) * masuk
#         mms = w_concat
#         ''' 
#         ============= UKF di measurement ============= 
#         '''
#         #%% Unscented Kalman Filter without filterpy
#         n = mms.size # julier versi masalah 'dimension of problem'
#         beta = 2.
#         kappa = 3-n # 0
#         # lambda_ = 1. 
#         lambda_ = alpha**2 * (n + kappa) - n
#         
#         #%% SIGMA POINTS around mean
#         mean = np.sum(w_concat) / n # mean secara keseluruhan
#         U = cholesky((n + lambda_)*P) # sama dg np.sqrt
# 
#         # filterpy version dengan shape (121,60) karena dikali dg P!
#         sigmas = np.zeros((2*n+1, n))
#         # maka....
# 
#         for k in range(n):
#             s_ = np.reshape(mms,(1,-1))
#             s_n = np.subtract(mms, -U)
#             s_n2 = np.subtract(mms, U)
#         # gabung (121,60) kebawah.. jadinya 121,60
#         sigmas_concat = np.concatenate((s_,s_n,s_n2), axis=0)
#         
#         
#         #%% BOBOT SIGMA dari Merwe
#         lambda_ = alpha**2 *(n + kappa) - n
#         c_ = .5 / (n + lambda_)
#         Wm = np.full(2*n+1, c_)
#         Wc = Wm # size (121,) atau (n,)
#         Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
#         Wm[0] = lambda_ / (n + lambda_)
#         
#         # SIGMA Unscented Transform  (biar hold value dg eyeP) ke measurement
#         # Mz = np.reshape(np.dot(Wm,sigmas_concat),(-1,1)) # yang lama computenya
#         Mz = np.dot(np.reshape(Wm,(1,-1)),sigmas_concat) # myu z yang dipake awalnya
#         # Mz = np.dot(np.reshape(Wm,(1,-1)),s_)
#         
#         # KOVARIANCE ke measurement juga
#         # SEMUA BERANTAKAN GARA2 Pz!
#         kmax, n = sigmas.shape
#         Pz = np.zeros(sigmas_concat.shape)
#         for k in range(kmax):
#             c = np.subtract(sigmas_concat[k],Mz)
#             # harusnya +=
#             Pz = Wc[k] * np.outer(c, c) # makin besar nilainya dengan +=
#         Pz += Q
#         
#         # Kalman gain
#         Pxz = np.zeros(sigmas.shape)
#         for k in range(kmax):
#             cc = np.subtract(sigmas[0],Mz)
#             c = np.subtract(sigmas_concat[k],Mz)
#             # harusnya +=
#             Pxz = Wc[k] * np.outer(cc, c) # makin besar nilainya dengan +=
# 
#         K = np.dot(Pxz,inv(Pz)) 
#    
#         # UPDATE weight
#         innovation = ((Y-layer_2).sum()/len(layer_2_error)) # hitung error pred dan yang diharapkan
#         w_concat_new_all = w_concat + np.dot(K,innovation)
#         w_concat_new_slice = np.reshape((w_concat_new_all[0]),(-1,1))
#         w_concat_new = norm(w_concat_new_slice,(-1,1))
# 
#         # UPDATE kovarian
# # =============================================================================
# #         leading the 5th minor of array is  not positive definite
# #         KURANG UPDATE P !
# # =============================================================================
#         P2 = np.dot(Pz,K.T)
#         # P = np.sqrt(np.dot(K,P2))
# =============================================================================
    
        #%%
        # assign bobot
        synapse_0_ukf = np.reshape(w_concat_new[0:(input_dim*hidden_dim),0],(input_dim,hidden_dim))
        synapse_h_ukf = np.reshape(w_concat_new[(input_dim*hidden_dim):(input_dim*hidden_dim)+(hidden_dim*hidden_dim),0],(hidden_dim,hidden_dim))
        synapse_1_ukf = np.reshape(w_concat_new[(input_dim*hidden_dim)+(hidden_dim*hidden_dim):w_concat_new.shape[0],0],(hidden_dim,output_dim))

        # assign
        synapse_0_update = synapse_0_ukf * alpha
        synapse_1_update = synapse_h_ukf * alpha
        synapse_h_update = synapse_1_ukf * alpha
        
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
    
    rmse_epoch = rmse(trainY,layer_2_value)
    rmse_all.append(rmse_epoch)
    
run_time = time.time() - start_time
#%% seberapa besar lossnya???

plt.plot(mse_all,label='loss', marker = 'x')
plt.title('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

plt.plot(layer_2_error, label='besar error', color='g', marker = 'x')
plt.title('Nilai Error')
plt.xlabel('Value ke')
plt.ylabel('layer_2_error')
plt.legend()
plt.show()

plt.plot(layer_2_value, label='layer 2 value', color='y', marker = 'x')
plt.title('layer 2 valuesnya ')
plt.xlabel('Value ke')
plt.ylabel('Nilai error')
plt.legend()
plt.show()

# =============================================================================
# plt.plot(mse_all,label='loss', marker='x')
# plt.title('Loss (MSE)')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.show()
# 
# =============================================================================
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
scoring = [mse_pred,rmse_pred,mae_pred,mape_pred,dstat_pred,run_time]

# =============================================================================
# plt.plot(mse_all,label='mse training', color='g')
# plt.plot(mse_pred,label='mse uji', color='r')
# plt.title('Loss (MSE)')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.show()
# =============================================================================

plt.plot(testYseb, label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(layer_2p, label='prediction')
plt.xlabel('Sebanyak data full')
plt.ylabel('Harga')
plt.title('RNN-UKF sebelum denormalisasi')
plt.legend()
plt.show()

plt.plot(testY[0:50], marker='o', label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(y_pred[0:50], marker='o', label='prediction')
plt.title('RNN-UKF')
plt.legend()
plt.show()

plt.plot(Pz, color='r', marker ='o', ls=None)
plt.plot(Pz2[1:91],'x')
plt.plot(Pz2[92:183],'x')
plt.xlabel('KOVARIAN ke')
plt.ylabel('value')
plt.title('KOVARIAN (P) semua')
plt.show()

plt.plot(K, color='r', marker ='o', ls='-')
plt.xlabel('value')
plt.ylabel('value')
plt.title('Kalman size: ')
plt.show()
print('Kalman dim: ', K.ndim)
print('Kalman size: ', K.size)
print('Kalman shape: ', K.shape)

#scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred,run_time]
print("mse : " , mse_pred)
print("rmse : ", rmse_pred) 
print("mape : ", mape_pred) 
print("mae: " , mae_pred)
print("dstat : " , dstat_pred)
print("runtime : ", run_time)

#%%
np.savetxt('bobot_input.csv', synapse_0, delimiter=',')
np.savetxt('bobot_hidden.csv', synapse_h, delimiter=',')
np.savetxt('bobot_output.csv', synapse_1, delimiter=',')
np.savetxt('loss_ukf.csv', mse_all, delimiter=';')