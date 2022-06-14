import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from names import *
import time
import collections
from DGP import DGP
from reduction import reduce, testVariables


#compute('combi14_dos.txt', namescombi14, tol=0.0001, M=True)
#compute('mela14_dos.txt', namesmela14, tol=0.000001, M=True)   AZ
#compute('combi9_dos.txt', namescombi9, tol= 0.0001, M=True)
#'mela9_dos.txt', namesmela9
M, N, corr = compute('combi9_dos.txt', namescombi9, tol=0.01, M=True, plot=False)