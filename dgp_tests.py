import numpy as np
import matplotlib.pyplot as plt
import collections
from DGP import DGP
from reduction import glmRoutine, reduce, testVariables


def loadTest(alpha, s, a, sigStrength, rho, n, var, d, intercept, noise, binary, DGPseed=None):
    X, Y, trueIdx, aIdx = DGP(s, a, sigStrength, rho, n, var, d, intercept, noise, binary, separate= True, DGPseed=None)
    ind = np.arange(X.shape[1])
    P = np.corrcoef(X, rowvar=False)
    P = np.argwhere(np.triu(np.isclose(np.abs(P),1,rtol=0,atol=alpha),1))
    Xdict = collections.defaultdict(list)
    for i in P:
        Xdict[i[0]].append(i[1])
    ind = np.delete(ind, P[:,1])
    X = np.delete(X, P[:,1],axis=1)

    return X, Y, Xdict, ind, trueIdx, aIdx

nIter = 10
tol = 1.0e-1
m = 28/30
mS = 5
aI = None
th = 0.4
alpha = 0

def tests(N, rB, rCorr, nS, nA, d, n):
    pT = np.zeros((len(rB), len(rCorr)))
    pC = 1.0*pT
    TPR = 1.0*pT
    PPV = 1.0*pT
    FSC = 1.0*pT
    from sklearn.linear_model import Lasso
    from regressors import stats    

    for i in range(len(rB)):
        for j in range(len(rCorr)):
            print(i,j)
            for _ in range(N):
                X, Y, Xdict, ind, trueIdx, aIdx = loadTest(alpha=alpha, s=nS, a=nA, sigStrength=rB[i], rho=rCorr[j], n=n, var=1, d=d, intercept=0, binary=True, noise=1)
                l = reduce(X, Y, nIter=30, threshold=tol, multiplier=m, minSurvivors=mS)

                #l = glmRoutine(X, Y, family='linear')
                #l = np.argwhere(l<0.1)
                print('_____')
                print('Beta ', rB[i])
                print('corr', rCorr[j])

                TP = len(np.intersect1d(l, trueIdx))
                FN = len([i for i in trueIdx if i not in l])
                FP = len([i for i in l if i not in trueIdx])
                #TN = d - (TP+FN+FP)

                pC[i,j] += len(np.intersect1d(l, aIdx))/(nA*nS*N) #assumes separate=True
                if len(l)!=0: TPR[i,j] += (TP/(TP+FN))/N
                if TP+FP!=0: 
                    PPV[i,j] += (TP/(TP+FP))/N
                    pT[i,j] += len([i for i in l if (i not in trueIdx and i not in aIdx)])/((TP+FP)*N)
                if PPV[i,j]+TPR[i,j]!=0: FSC[i,j] += 2*(PPV[i,j]*TPR[i,j])/(PPV[i,j]+TPR[i,j])/N
                #if TN+FP!=0: TNR[i,j] += (TN/(TN+FP))/N
                #ACC[i,j] += (TP+TN)/(d*N)
    
    return TPR, PPV, pT, pC


def plots(N, rB, rCorr, nS, nA, d, n):

    TPR, PPV, tP, tC = tests(N, rB, rCorr, nS, nA, d, n)
    P = [TPR, PPV, tC, tP]
    titles = ['Sensitivity', 'Precision', 'Proportion of correlated variables', 'Proportion of FP not in corr']
    X, Y = np.meshgrid(rCorr, rB)
    for i in range(len(P)):
        fig, ax = plt.subplots(1, 1)
        min = round(np.min(P[i]),1)
        if min > 0.8 or np.max(P[i])<0.4: step = 0.05
        else: step=0.1
        cp = ax.contour(X, Y, P[i], levels=np.arange(0, 1, step))
        ax.clabel(cp, fontsize=10)
        ax.set_title(titles[i])
        ax.set_xlabel('correlation')
        ax.set_ylabel(r'$\beta$ for signal variables')
        plt.show()

#sigStrength = np.linspace(0.00001, 1.5, 7)
#corr = np.linspace(0.9, 0.999, 7)
#plots(45, sigStrength, corr, 1, 2, 10, 50)