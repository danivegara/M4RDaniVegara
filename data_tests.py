import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from names import *
import time
import collections
from DGP import DGP
from reduction import reduce, testVariables

def loadTestData(title, names, alpha, s, sigStrength, intercept=0, noise=1, n=None):
    data = np.loadtxt(title, delimiter=" ", skiprows=1)[:,1:]
    X = np.delete(data, 3, 1)
    ind = np.arange(X.shape[1])
    P = np.corrcoef(X, rowvar=False)
    P = np.argwhere(np.triu(np.isclose(np.abs(P),1,rtol=0,atol=alpha),1))
    Xdict = collections.defaultdict(list)
    for i in P:
        Xdict[i[0]].append(i[1])
    if alpha!=0: 
        ind = np.delete(ind, P[:,1])
        X = np.delete(X, P[:,1], axis=1)
    else: ind = np.arange(X.shape[1])
    
    if n is None: n = X.shape[0]
    else: X = X[:n]
    v = np.concatenate((sigStrength*np.ones(s), np.zeros(X.shape[1]-s)))
    Beta = np.random.permutation(v)
    trueIdx = np.where(Beta!=0)[0]
    epsilon = np.random.normal(0,noise,n)
    Y = 1/(1+np.exp(-intercept * np.ones(n) - X @ Beta - epsilon))
    Y[Y<=np.median(Y)]=0.
    Y[Y!=0.]=1.
    return X, Y, Xdict, ind, trueIdx

def tests(title, names, alpha, N, rB, nS, tol, n=None):
    TPR = np.zeros(len(rB))
    PPV = 1.0*TPR

    CORR = np.zeros((20,len(rB)))
    HCORR = np.zeros((20,len(rB)))
    OVERALL = 1.0*CORR
    HOVERALL = 1.0*HCORR

    for i in range(len(rB)):
            print(i)
            pltCorr = []
            K=[]
            for iter in range(N):
                print('______')
                print('Beta', rB[i])
                print(iter)
                X, Y, Xdict, ind, trueIdx = loadTestData(title, names, alpha, s=nS, sigStrength=rB[i], n=n)
                l = reduce(X, Y, nIter=20, threshold=tol*(10**(-i/5)), multiplier=0.9, minSurvivors=20)
                print('len', len(l))
                
                P = np.corrcoef(X, rowvar=False)
                corr = {}
                for a in trueIdx:
                    for b in l:
                        c = np.abs(P[a,b])
                        if b in corr:
                            if c > corr[b]: corr[b] = c
                        elif b not in trueIdx: corr[b] = c
                Kd = {}
                for a in trueIdx:
                    for b in range(len(P)):
                        c = np.abs(P[a,b])
                        if b in Kd:
                            if c > Kd[b]: Kd[b] = c
                        else: Kd[b] = c

                for key in corr.keys() :
                    pltCorr.append(corr[key])

                for key in Kd.keys() :
                    K.append(Kd[key])

                f = 1.0*l

                if alpha !=0:
                    for k in l:
                        if k in Xdict: 
                            print('WRONG')
                            f=np.concatenate((f,Xdict[k]))
                           #corr[k]=Xdict[k]
                    l = np.concatenate((l, [Xdict[i] for i in l]))
                    l = [x for y in l for x in y]

                TP = len(np.intersect1d(f, trueIdx))
                FN = len([i for i in trueIdx if i not in f])
                FP = len([i for i in f if i not in trueIdx])
                #TN = d - (TP+FN+FP)

                if TP+FN!=0: TPR[i] += (TP/(TP+FN))/N
                if TP+FP!=0: PPV[i] += (TP/(TP+FP))/N
                #if PPV[i,j]+TPR[i,j]!=0: FSC[i,j] += 2*(PPV[i,j]*TPR[i,j])/(PPV[i,j]+TPR[i,j])/N
            
            K1 = np.array(K)
            tx = np.array(pltCorr)

            K1h = np.histogram(K1, bins=20)[0]
            txh = np.histogram(tx, bins=20)[0]
            K1h = np.cumsum(K1h)
            txh = np.cumsum(txh)
            
            if np.max(txh)>0: CORR[:, i] = txh/np.max(txh)#/g0
            if np.max(K1h)>0: OVERALL[:, i] = K1h/np.max(K1h)
            
            t2 = tx[tx>0.9]
            t2h = np.histogram(t2, bins=20)[0]
            t2h = np.cumsum(t2h)
            if np.max(t2h)>0: HCORR[:, i] = t2h/np.max(t2h)
            
            K2 = K1[K1>0.9]
            K2h = np.histogram(K2, bins=20)[0]
            K2h = np.cumsum(K2h)
            if np.max(K2h)>0: HOVERALL[:, i] = K2h/np.max(K2h)

    tx = np.mean(CORR, axis=1)
    K1 = np.mean(OVERALL, axis=1)
    bins = np.arange(0, 1, 0.05)
    plt.bar(range(20), K1, label='overall', alpha=0.45)
    plt.bar(range(20), tx, label='model', alpha=0.45)
    plt.xticks(range(20), np.round(bins, 2))
    plt.legend()
    plt.xlabel('correlation')
    plt.title('Empirical CDF')
    plt.show()

    K2 = np.mean(HOVERALL, axis=1)
    t2 = np.mean(HCORR, axis=1)
    bins = np.arange(0.9, 1, 0.005)
    plt.bar(range(20), K2, label='overall', alpha=0.45)
    plt.bar(range(20), t2, label='model', alpha=0.45)
    plt.xticks(range(20), np.round(bins, 2))
    plt.legend()
    plt.xlabel('correlation')
    plt.title('Empirical CDF - High correlation setting')
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(rB, np.arange(0,1,0.05), CORR, levels=np.arange(0,1,0.1))
    ax.clabel(cp, fontsize=10)
    ax.set_title('Correlation bins - Simulation')
    ax.set_ylabel('correlation')
    ax.set_xlabel(r'$\beta$ for signal variables')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(rB, np.arange(0.9,1,0.005), HCORR, levels=np.arange(0,1,0.1)) #.9,1,.005
    ax.clabel(cp, fontsize=10)
    ax.set_title('High correlation bins - Simulation')
    ax.set_ylabel('correlation')
    ax.set_xlabel(r'$\beta$ for signal variables')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(rB, np.arange(0,1,0.05), OVERALL, levels=np.arange(0,1,0.1))
    ax.clabel(cp, fontsize=10)
    ax.set_title('Correlation bins - Overall')
    ax.set_ylabel('correlation')
    ax.set_xlabel(r'$\beta$ for signal variables')
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(rB, np.arange(0.9,1,0.005), HOVERALL, levels=np.arange(0,1,0.1))
    ax.clabel(cp, fontsize=10)
    ax.set_title('High correlation bins - Overall')
    ax.set_ylabel('correlation')
    ax.set_xlabel(r'$\beta$ for signal variables')
    plt.show()

    return TPR, PPV


def plots(title, names, alpha, N, rB, nS, tol, n=None):
    TPR, PPV, = tests(title, names, alpha, N, rB, nS, tol, n)
    P = [TPR, PPV]
    ylabels = ['TPR', 'PPV']
    titles = ['Sensitivity', 'Precision']

    for i in range(len(P)):
        plt.plot(rB, P[i])
        plt.xlabel(r'$\beta$')
        plt.ylabel(ylabels[i])
        plt.title(titles[i])
        plt.show()

#sigStrength = np.linspace(0.4, 1, 12) 
#plots(title='combi14_dos.txt', names=namescombi14, alpha=0, N=50, rB=sigStrength, nS=10, tol=0.04)