# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:21:48 2018

@author: Matjaz
"""
from numpy.linalg import cholesky as chol
from numpy.linalg import inv
from numpy import zeros, sqrt, diagflat, ones
import numpy as np


def ukf(FofX, x, P, hmeas, z, Q, R):
    """ Unscented Kalman Filter for nonlinear dynamic systems

    :param FofX:    function handle for f(x)
    :param x:       "a priori" state estimate
    :param P:       "a priori" estimated state covariance
    :param hmeas:   function handle for h(x)
    :param z:       current measurement
    :param Q:       process noise covariance
    :param R:       measurement noise covariance
    :return:    x:  "a posteriori" state estimate
                P:  "a posteriori" state covariance
    """

    Nstates = x.size  # number of states
    Nmeas = z.size  # number of measurements

    # tunables
    alpha = 1e-1  # default, tunable
    ki = 0  # default, tunable
    beta = 2  # default, tunable

    # params
    Lambda = (alpha ** 2) * (Nstates + ki) - Nstates  # scaling factor
    cc = Nstates + Lambda  # scaling factor

    # weights
    Wm=0.5/cc+zeros((1,1+2*Nstates))
    Wm[0,0] = Lambda / cc  # weights for means
    Wc = Wm.copy()
    Wc[0,0] += (1 - alpha ** 2 + beta)  # weights for covariance

    # sigma points around x
    X = sigmas(x, P, sqrt(cc))
    # unscented transformation of process
    x1, X1, P1, X2 = ut(FofX, X, Wm, Wc, Nstates, Q)
    # unscented transformation of measurements
    z1, Z1, P2, Z2 = ut(hmeas, X1, Wm, Wc, Nmeas, R)
    P12 = X2 @ diagflat(Wc) @ Z2.T  # transformed cross-covariance
    K = P12 * inv(P2)
    x = x1 + K @ (z - z1)  # state update
    P = P1 - K @ P12.T  # covariance update

    return x, P, K


def ut(FofX, X, Wm, Wc, n, R):
    """ Unscented Transformation

    :param FofX:    function handle for f(x)
    :param X:       sigma points
    :param Wm:      weights for mean
    :param Wc:      weights for covraiance
    :param n:       number of outputs of f
    :param R:       additive covariance
    :return:    y: transformed mean
                Y: transformed smapling points
                P: transformed covariance
                Y1: transformed deviations
    """
    L = X.shape[1]  # size(X,2)
    y = zeros((n, 1))
    Y = zeros((n, L))
    for k in range(L):
        Y[:, k:k+1] = FofX(X[:, k:k+1])
        y+=Wm[0,k] * Y[:, k:k+1]

    Y1 = Y - y[:, zeros((L),int)]
    P = Y1 @ diagflat(Wc) @ Y1.T + R

    return y, Y, P, Y1


def sigmas(x, P, cc):
    """ Sigma points around reference point

    :param x:   reference point
    :param P:   covariance
    :param c:   coefficient
    :return:    Sigma points
    """
    A = cc * chol(P).T
    Y = x[:, zeros(x.size,int)]
    X = np.hstack((x, Y + A, Y - A))

    return X
