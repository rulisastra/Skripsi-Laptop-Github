import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#%% Learning Rate dengan windowSize 5, hidden 7, 8:2, epoch 100, seed(1), USDIDR 1000 data.csv
Lr1 = pd.read_csv('loss_ukf_LR(1).csv', usecols=[0])
Lr01 = pd.read_csv('loss_ukf_LR(0.1).csv', usecols=[0])
Lr001 = pd.read_csv('loss_ukf_LR(0.01).csv', usecols=[0])
Lr0001 = pd.read_csv('loss_ukf_LR(0.001).csv', usecols=[0])
Lr_all = pd.concat([Lr1,Lr01,Lr001,Lr0001],axis=1)

plt.plot(Lr1,label='Lr 1', color='r')
plt.plot(Lr01,label='Lr 0.1', color='y')
plt.plot(Lr001,label='Lr 0.01', color='g')
plt.plot(Lr0001,label='Lr 0.001', color='b')
plt.title('Epoch percobaan Learning Rate')
plt.xlabel('Epoch ke-')
plt.ylabel('loss MSE training')
plt.legend()
# plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
plt.show()


#%% Q dengan alpha 0.001
Q1 = pd.read_csv('loss_ukf_Q(1).csv', usecols=[0])
Q01 = pd.read_csv('loss_ukf_Q(0.1).csv', usecols=[0])
Q001 = pd.read_csv('loss_ukf_Q(0.01).csv', usecols=[0])
Q0001 = pd.read_csv('loss_ukf_Q(0.001).csv', usecols=[0])
Q_all = pd.concat([Q1,Q01,Q001,Q0001],axis=1)

plt.plot(Q1,label='Q 1', color='r')
plt.plot(Q01,label='Q 0.1', color='y')
# =============================================================================
# plt.plot(Q001,label='Q 0.01', color='g')
# plt.plot(Q0001,label='Q 0.001', color='b')
# =============================================================================
plt.title('Epoch percobaan Q')
plt.xlabel('Epoch ke-')
plt.ylabel('loss MSE training')
plt.legend()
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
plt.show()

#%% R dengan alpha 0.001
R1 = pd.read_csv('loss_ukf_R(1).csv', usecols=[0])
R01 = pd.read_csv('loss_ukf_R(0.1).csv', usecols=[0])
R001 = pd.read_csv('loss_ukf_R(0.01).csv', usecols=[0])
R0001 = pd.read_csv('loss_ukf_R(0.001).csv', usecols=[0])
R_all = pd.concat([R1,R01,R001,R0001],axis=1)

plt.plot(R1,label='R 1', color='r')
plt.plot(R01,label='R 0.1', color='y')
plt.plot(R001,label='R 0.01', color='g')
plt.plot(R0001,label='R 0.001', color='b')
plt.title('Epoch percobaan R')
plt.xlabel('Epoch ke-')
plt.ylabel('loss MSE training')
plt.legend()
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
plt.show()