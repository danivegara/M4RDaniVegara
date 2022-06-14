import numpy as np
from scipy.stats import multivariate_normal as mvn
from reduction import glmRoutine

def DGP(s, a, sigStrength, rho, n, var, d, intercept=0, binary=False, noise=0, separate=False, DGPseed=None):
    '''
    Generate random explanatory and response data according to certain parameters
    
    :s: number of signal variables
    :a: number of noise variables correlated with signal variables
    :sigStrength: signal strength
    :rho: correlation among signal variables and noise variables correlated with signal variables
    :n: sample size
    :var: variance of potential explanatory variables
    :d: number of potential explanatory variables
    :intercept: expected value of the response variable when all potential explanatory variables are at zero
    :binary: boolean variable determining if output should be binary 
    :noise: variance of observations around true regression lines
    :separate: if True, signal variables will be independetly correlated with their own noise variables
    :DGPseed: seed for random number generator

    :return X: simulated design matrix
    :return Y: simulated response variable
    :return trueIdx: indicies of the variables in the true model
    :return aIdx: indicies of the variables correlated with trueIdx
    '''

    q = a+s
    if separate:
        if d<s*(a+1):
            print('Dimensions too small! Changed from', d, ' to ', s*(a+1))
            d = s*(a+1)
        covMatrix = np.zeros((d,d))
        for i in range(s):
            covMatrix[i, s+(a*i):s+(a*(i+1))] = rho
            covMatrix[s+(a*i):s+(a*(i+1)), s+(a*i):s+(a*(i+1))] = rho*np.ones((a,a))/2
        covMatrix +=  covMatrix.T
        np.fill_diagonal(covMatrix, 1)
    else:
        cov1 = rho*np.ones(q) + (1-rho)*np.eye(q)
        covMatrixInit = np.concatenate((np.concatenate((cov1, np.zeros((q, d-q))),axis=1), np.concatenate((np.zeros((d-q,q)), np.eye(d-q)), axis=1)))
        covMatrix = np.diag(np.sqrt(var) * np.ones(d)) @ covMatrixInit @ np.diag(np.sqrt(var) * np.ones(d))
    trueBetaInit = np.concatenate((sigStrength * np.ones(s), np.zeros(d-s)))
    if DGPseed != None:
        np.random.seed(DGPseed)
    permuteVec = np.random.permutation(d)
    trueBeta = trueBetaInit[permuteVec]
    trueIdx = np.where(trueBeta!=0)[0]
    I = np.eye(d)
    permMatrix = I[permuteVec,:]
    covPerm = permMatrix @ covMatrix @ np.linalg.inv(permMatrix)
    
    aIdx = np.unique([i[0] for i in np.argwhere((covPerm-np.eye(d))!=0)])
    aIdx = [i for i in aIdx if i not in trueIdx]

    X = mvn.rvs(mean = np.zeros(d), cov=covPerm, size = n)
    epsilon = np.random.normal(0,noise,n)
    if binary==True: 
        Y = 1/(1+np.exp(-intercept * np.ones(n) - X @ trueBeta - epsilon))
        Y[Y>=0.5]=1
        Y[Y<0.5] =0
    else: Y = intercept * np.ones(n) + X @ trueBeta + epsilon

    return X, Y, trueIdx, aIdx
