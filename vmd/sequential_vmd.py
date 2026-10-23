
import argparse
import os
import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD  
import glob
import scipy.io
import csv
from joblib import Parallel, delayed
import time

path = '/your_directory/'
savepath = '/your_savepath/'
samples = np.load(path+'samples.npz')
# f = samples[10000,:1096]

#. VMD parameters
alpha = 2000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 4            # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  

#
idx= 0
initial_index = 312
end_index = 1200


def vmd_append(t):
    f = samples[I,:t];    # 783: ~2014
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  
    return u[:,-1].reshape(-1,1)


# Start timing
start_time = time.time()

# Initial decomposition
f = samples[i,:initial_index]
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  
u_all = u;

u0 = Parallel(n_jobs=args.n)(delayed(vmd_append)(i) for i in range(initial_index,end_index))

u0 = np.concatenate(u0, axis=1)
u_all = np.concatenate((u_all, u0), axis=1)

for m in range(K):
    mylist = np.append(np.array([i]), u_all[m-1,:])
    with open(savepath+'mode'+str(m)+'/append_mode'+str(m)+'_'+str(i)+'.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(mylist)
    

# End timing
end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")

