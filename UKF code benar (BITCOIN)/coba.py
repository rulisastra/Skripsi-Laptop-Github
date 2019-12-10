import numpy as np
import matplotlib.pyplot as plt
from filterpy import kalman
from filterpy.kalman import unscented_transform as UT
from filterpy.kalman import UnscentedKalmanFilter as UKF

data = np.array([1,2,3,3,5,5])

# data = np.reshape(data,(-1,1))
# P = np.identity(data.size)
P = np.eye(data.size)

points = kalman.MerweScaledSigmaPoints(n=data.size, alpha=.1, beta=.1, kappa=0)

Wm = points.Wm
Wc = points.Wc
banyak_sigma = points.num_sigmas
print(banyak_sigma)
sigmas_ = points.sigma_points(data,P)

kf = UKF(dim_x=data.size,dim_z=1,dt=1, 
         hx=None,fx=None,points=points)

Myu_aksen = kf.x_mean(Wm,sigmas_[k])
# fx = kf.fx(data)

# =============================================================================
# zs = UKF.compute_process_sigmas(data,dt=None)
# x = UKF.batch_filter(data)
# K = kf.K
# =============================================================================


plt.plot(data,marker='x')
plt.plot(sigmas_, marker='o')
# UT_ = UT(data,Wm,Wc)


# =============================================================================
# kf = kalman.UnscentedKalmanFilter(dim_x, dim_z, dt=.1, hx=None, fx=None, points=points) # sigma = points
# 
# data = [1,2,3,4,5,6]
# 
# sigmas = 
# =============================================================================
