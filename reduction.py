import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sb
import matplotlib.pyplot as plt
from names import *
import itertools
import collections


def loadData(title, names, alpha): #loads data and separates into explanatory and response variables
    #delete variables with > 1-aplha corr and have one represent them to be reintroduced
    
    data = np.loadtxt(title, delimiter=" ", skiprows=1)[:,1:]
    Y = data[:,3]
    X = np.delete(data, (2,3), 1)
    ind = np.arange(X.shape[1])
    print('sample size', len(X))
    print('N0', X.shape[1])
    P = np.corrcoef(X, rowvar=False)
    P = np.argwhere(np.triu(np.isclose(np.abs(P),1,rtol=0,atol=alpha),1))
    Xdict = collections.defaultdict(list)
    for i in P:
        Xdict[names[i[0]]].append(names[i[1]])
    ind = np.delete(ind, P[:,1])
    X = np.delete(X, P[:,1],axis=1)
    print('N1', X.shape[1])
    
    return X, Y, Xdict, ind



def glmRoutine(X, Y, family=sm.families.Gaussian()): #fits GLM to data and returns p-values
    if family == 'linear':
        stats = sm.OLS(Y, X)
    elif family == 'log':
        stats = sm.Logit(Y, X)
    else:
        stats = sm.GLM(Y, X, family=family)

    res = stats.fit()
    return res.pvalues


def include(X, subsetX, alwaysInclude, idx): #includes "alwaysInclude" indicies in subsetX and idx
    if len(alwaysInclude)==1:
        subsetX = np.concatenate((subsetX, np.reshape(X[:,alwaysInclude], (len(X),1))), axis=1)
    else:
        subsetX = np.concatenate((subsetX, X[:, alwaysInclude]), axis=1)
    idx = np.concatenate((idx, alwaysInclude))
    return subsetX, idx

def cubePhase(cube, s, X, Y, alwaysInclude, p):
    cube = cube.ravel()
    np.random.shuffle(cube)
    cube = cube.reshape((s,s,s))
    consider = np.zeros((3*(s**2), 2))
    for k in range(s):
        for i in range(s):
            #iterate over columns
            idx = cube[:,i,k]
            idx = idx[idx!=p]
            subsetX = X[:,idx]
            if alwaysInclude != None:
                subsetX, idx = include(X, subsetX, alwaysInclude, idx)
            if subsetX.shape[1] > 1:
                pVals = glmRoutine(subsetX, Y, family='linear')
                A = np.argpartition(pVals,1)[0:2]
                consider[k*s + i, :] = idx[A]
            #iterate over rows
            idx = cube[i,:,k]
            idx = idx[idx!=p]
            subsetX = X[:,idx]
            if alwaysInclude != None:
                subsetX, idx = include(X, subsetX, alwaysInclude, idx)
            if subsetX.shape[1] > 1:
                pVals = glmRoutine(subsetX, Y, family='linear')
                A = np.argpartition(pVals,1)[0:2]
                consider[s**2 + k*s + i, :] = idx[A]
            #iterate over tube fibers
            idx = cube[i,k,:]
            idx = idx[idx!=p]
            subsetX = X[:,idx]
            if alwaysInclude != None:
                subsetX, idx = include(X, subsetX, alwaysInclude, idx)
            if subsetX.shape[1] > 1:
                pVals = glmRoutine(subsetX, Y, family='linear')
                A = np.argpartition(pVals,1)[0:2]
                consider[2*(s**2) + k*s + i, :] = idx[A]

    consider.reshape(6*(s**2))
    #keep only variables that appear in at least 2/3 models
    u, c = np.unique(consider, return_counts=True)
    final = u[c>1]
    final = [int(i) for i in final]
    final = list(set(final))
    
    return final

def squarePhase(square, sq, X, Y, threshold, p):
    final = []
    square = square.ravel()
    np.random.shuffle(square)
    square = square.reshape((sq,sq))
    for i in range(sq):
        #iterate over columns
        idx = square[:,i]
        idx = idx[idx!=p]
        subsetX = X[:,idx]
        if subsetX.shape[1] > 1:
            pVals = glmRoutine(subsetX, Y) #family='log')
            asdf = [idx[i] for i in range(len(pVals)) if pVals[i] < threshold]
            if len(asdf)!=0: final.append([idx[i] for i in range(len(pVals)) if pVals[i] < threshold])
        #iterate over rows
        idx = square[i,:]
        idx = idx[idx!=p]
        subsetX = X[:,idx]
        if subsetX.shape[1] > 1:
            pVals = glmRoutine(subsetX, Y) #family='log')
            asdf = [idx[i] for i in range(len(pVals)) if pVals[i] < threshold]
            if len(asdf)!=0: final.append([idx[i] for i in range(len(pVals)) if pVals[i] < threshold])
    final = [int(i) for x in final for i in x]
    return final


def reduce(X, Y, nIter, threshold, multiplier, minSurvivors=None, alwaysInclude=None):
    """
    Runs data through cube then square, reducing as explained in paper

    :X: explanatory variables
    :Y: response variable
    :nIter: the number of randomizations of cube and square
    :threshold: variables with p-values < threshold are kept in cube phase
    :multiplier: variables that appear in >= multiplier*nIter iterations of square are kept
    :alwaysInclude: indicies of variables that are included in every model in cube phase

    :return final: list of variables that are kept through the reduction
    """
    
    #-----------------CUBE PHASE-----------------
    #build smallest perfect cube which fits all variable indicies
    s = 1
    while s**3 <= X.shape[1]:
            s+=1
    cube = np.reshape(np.arange(s**3), (s,s,s))
    p = X.shape[1]
    cube[cube >= p] = p

    #commence iteration through cube phase
    ff = []
    for _ in range(nIter):
        if p > 100: 
            final = cubePhase(cube, s, X, Y, alwaysInclude, p)
        else: final = np.arange(X.shape[1])
        
        #if number of variables surviving is < minSurvivors, don't do square phase 
        if minSurvivors is None: minSurvivors = 1

        if len(final) >= minSurvivors:
            #-----------------SQUARE PHASE-----------------
            #create smallest perfect square that fits all remaining indicies
            sq = 1
            while sq**2 <= len(final):
                    sq+=1
            final = np.concatenate((final, p*np.ones(sq**2 - len(final), dtype=int))) 
            square = np.reshape(final, (sq,sq))

            #commence iteration through square phase
            final = squarePhase(square, sq, X, Y, threshold, p)
        ff.append(final)

    #keep only values that appear multiplier*nIter times 
    ff = [int(i) for x in ff for i in x]
    u, c = np.unique(ff, return_counts=True)
    ff = u[c >= multiplier*nIter] 
    ff = ff[ff!=p]
    if len(ff) == 0:
        print('No values in the final phase: thresholds are too harsh')

    return ff


def pltCorr(l, f, X): #plots a correlation map of final variables
    df = pd.DataFrame(X[:,l], columns=f)
    dataplot = sb.heatmap(df.corr(), cmap="YlGnBu", annot=True)
    plt.show()


#------------------------ MODEL SELECTION PHASE ----------------------------#

def createSubsets(L, s=1, e=8): #create all subsets of sizes from s to e
    N = min(e, len(L))
    F = []
    for i in range(s, N+1):
        F.append(list(itertools.combinations(L, i)))
    return F
    
def testVariables(X, Y, L, ind, tol = 1.0e-3): #uses LRT to compare nested models
    stats = sm.OLS(Y, X[:,L]).fit()
    F = createSubsets(L)
    M = []
    n0 = len(F[0][0])
    F=F[:8-n0]
    for i in F:
        print(len(i[0]))
        for j in i:
            s = sm.OLS(Y, X[:,j]).fit()
            pVal = stats.compare_lr_test(s)[1]
            if pVal > 1-tol:
                k = ind[np.array(j)]
                M.append([i for i in k])
    return M


#------------------------ PARAMETERS --------------------------------------#

plot = 0       #binary variable saying if to plot
alpha = 0.005#.05   #variables with corr > 1-alpha get eliminated   
nIter = 100    #number of iterations
m = 0.9        #multiplier
aI = None#[0,1]     #alwaysInclude
mS = 15        #min number of survivors from cube phase
th = 0.7   #p-val threshold for lrt


def compute(file, names, tol, M=False, plot=False, Mi=False): 
    """
    Final function

    :file: filename for SNP data
    :names: filename for names of SNPs
    :tol: variables with p-value <tol are kept in square phase
    :M: binary variable
    
    :return M: if M True, returns plausible models in array M
    """

    X, Y, Xdict, ind = loadData(file, names, alpha=alpha)
    l = reduce(X, Y, nIter, tol, m, mS, aI)
    f0 = ind[l]
    f = []
    for i in f0:
        f.append(names[i])

    if len(l) != 0 and plot:
        pltCorr(l, f, X)
    
    z=0
    if M: 
        M = testVariables(X, Y, l, ind, tol=th)
        N = []
        corr = {}
        B = np.zeros(len(l))
        for i in M:
            for k in i: 
                B[np.argwhere(f0==k)]+=1
            n = [names[k] for k in i]
            for i in n:
                if i in Xdict and (i not in corr): 
                    corr[i] = Xdict[i]
                    z+=len(corr[i])
            N.append(n)
        a = np.unique([x for y in N for x in y])
        B = B/B.max()
        plt.bar(f, B)
        plt.title('Frequency of variables')
        plt.show()
        return M, N, corr

def final(file, names):
    N = 3
    rP = np.linspace(0.03, 0.07, N)
    rT = np.linspace(0.1, 0.3, N)
    rA = np.linspace(0.0001, 0.03, N)
    F = {}
    lM = 0
    for a in range(N):
        for j in range(N):
            for k in range(N):
                print('ijk', a, j, k)
                X, Y, Xdict, ind = loadData(file, names, alpha=rA[k])
                l = reduce(X, Y, 100, rP[j], m, mS, aI)
                print('l', len(l))
                f0 = ind[l]
                f = []
                for i in f0:
                    k = names[i]
                    f.append(k)
                    if k not in F:
                        F[k] = 1
                if len(l)>0:
                    M = testVariables(X, Y, l, ind, tol=rT[a])
                    if len(M)==0: M = [f0]
                    lM += len(M)
                    for i in M:
                        n = [names[k] for k in i]
                        for i in n:
                            F[i] += 1
                            if i in Xdict:
                                for k in Xdict[i]: 
                                    if k not in F: F[k] = 1
                                    else: F[k] +=1

    names = list(F.keys())
    values = np.array(list(F.values())) 

    colors = {} 
    x = 1
    y = 0
    for i in F:
        if i not in colors: colors[i] = 0
        if i in Xdict:
            for j in Xdict[i]:
                if j in F and j not in colors: 
                    colors[i] = 1*x
                    colors[j] = 1*x 
                    print('YEEES', i, j)
                    y = 1
            if y==1: x+=1
            print('x', x)
            y=0

    c = {}
    for i in F:
        c[i] = colors[i]
    colours = ['tab:orange', 'tab:blue', 'tab:green', 'tab:cyan', 'tab:red', 'tab:gray', 'tab:olive', 'navajowhite', 'tab:pink', 'tab:purple', 'firebrick', 'steelblue', 'yellow', 'darkslateblue', 'plum', 'honeydew']
    c = list(c.values())
    colorss = [colours[i] for i in c]

    plt.bar(range(len(F)), values/lM, tick_label=names, color=colorss)
    plt.xticks(rotation=45, ha="right")
    plt.show()

    x = []
    d = []
    for i in range(len(colorss)):
        k = colorss[i]
        if k in x and k!='tab:orange': d.append(i)
        if k not in x: x.append(k)
    colorss = np.delete(colorss, d)

    F0 = F.copy()
    for i in d:
        F0.pop(list(F.keys())[i])
    
    names = list(F0.keys())
    values = np.array(list(F0.values()))
    plt.bar(range(len(F0)), values/lM, tick_label=names, color=colorss)
    plt.xticks(rotation=45, ha="right")
    plt.show()

    F = dict(sorted(F.items(), key=lambda i: i[1], reverse=True)[:20])
    c = {}
    for i in F:
        c[i] = colors[i]
    c = list(c.values())
    colorss = [colours[i] for i in c]
    names = list(F.keys())
    values = np.array(list(F.values())) 
    plt.bar(range(len(F)), values/lM, tick_label=names, color=colorss)
    plt.xticks(rotation=45, ha="right")
    plt.show()

 
#final('mela14_dos.txt', namesmela14)
#'combi14_dos.txt', namescombi14
#'combi9_dos.txt', namescombi9
#'mela9_dos.txt', namesmela9