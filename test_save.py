import scipy.io as sio
from cProfile import label
from re import A
import pandas as pd
import numpy as np
import h5py
from multiprocessing import cpu_count, Pool
import time
from datetime import datetime
from tqdm import tqdm
import datetime as dt
import os
from random import shuffle

import matplotlib.pyplot as plt
from matplotlib import rc

from ipdb import set_trace as st

import warnings
warnings.filterwarnings('ignore')

N_sample = 100000 # all:85455l
# N_sample = 85455 # all:85455
train_divide = 5/4
mode_max=4
IC = 10
# save_mat = '/media/faraday/andong/GONG_NN/RK4_MF_5days_'+str(int(IC))+'_'+str(int(boost_num))+'_'+str(int(N_sample))+'.mat'
# save_mat = "/media/faraday/andong/GONG_NN/RK4_MF_5days_"+str(IC)+'_'+str(mode_max)+'_'+str(N_sample)+".mat"
# save_mat = '/media/faraday/andong/GONG_NN/RK4_MF_5days_WL_'+str(int(IC))+'_'+str(int(N_sample))+'_best.h5'
# save_mat = '/media/faraday/andong/GONG_NN/RK4_MF_5days_init_'+str(int(IC))+'_'+str(int(N_sample))+'.h5'
load_mat = '/media/faraday/andong/GONG_NN/RK4_MF_5days_WL_'+str(int(IC))+'_'+str(int(N_sample))+'.h5'
save_mat = '/media/faraday/andong/GONG_NN/RK4_MF_5days_WL_'+str(int(IC))+'_'+str(int(N_sample))+'_test.h5'
# save_mat = '/media/faraday/andong/GONG_NN/RK4_MF_5days_'+str(int(IC))+'_'+str(int(N_sample))+'.mat'
# print(save_mat)
# st()
# save_mat = "/media/faraday/andong/GONG_NN/RK4_MF_5days_"+str(N_sample)+"_origin.mat"
with h5py.File(load_mat, 'r') as f:
    test_idx = np.array(f['test_idx']).squeeze()
    train_idx = np.array(f['train_idx']).squeeze()
    date = np.array(f['Time'])
    vr = np.array(f['vr'])
    vr_5days = np.array(f['vr_5days'])
    vr_5days_test_final = np.array(f['vr_5days_train_final']).T
    f.close()

# st()
with h5py.File(save_mat, 'w') as f:
    f.create_dataset('Time', data=date[test_idx])
    f.create_dataset('vr', data=vr[test_idx, :24, -1][:, ::-1])
    f.create_dataset('vr_5days', data=vr_5days[test_idx, :24, -1][:, ::-1])
    f.create_dataset('vr_5days_pred', data=vr_5days_test_final[:24, test_idx][::-1].T)
    f.close()

############################## sub-RMSE (test) ###########################

Vr_clu = np.arange(200, 800, 20)
figname = 'Figs/exm/RMSE_test.png'


RMSE = 0
RMSE_v0 = 0
RMSE_clu = []
RMSE_v0_clu = []

SE_Per = np.zeros([len(Vr_clu), len(test_idx)])
SE = np.zeros([len(Vr_clu), len(test_idx)])
num = np.zeros([len(Vr_clu), len(test_idx)])

for ii, i in enumerate(test_idx):

    # st()
    error = vr_5days_test_final[:24, i] - vr_5days[i, :24, -1].T
    error_v0 = vr[i, :24, -1] - vr_5days[i, :24, -1].T

    for n, Vr_thr in enumerate(Vr_clu):

        # st()

        idx_t = np.where(vr_5days[i, :24, -1]>Vr_thr)[0]
        
        num[n, ii] = len(idx_t)
        SE[n, ii] = np.sqrt(np.nanmean(error[idx_t]**2))
        SE_Per[n, ii] = np.sqrt(np.nanmean(error_v0[idx_t]**2))
        # SE_Per[n, ii] += np.sum(error_v0[idx_t]**2)
    
    RMSE_clu.append(error)
    RMSE_v0_clu.append(error_v0)
    RMSE += np.nanmean(error**2)
    RMSE_v0 += np.nanmean(error_v0**2)

# st()
# RMSE_clu = np.asarray(RMSE_clu)
# RMSE_v0_clu = np.asarray(RMSE_v0_clu)

print('RMSE of prediction & Vr is {} & {}'.format(
        np.sqrt(RMSE/test_idx.shape[0]), np.sqrt(RMSE_v0/test_idx.shape[0])))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(Vr_clu, np.nanmean(SE,axis=1), 'kx-', label='Proposed method')
ax.plot(Vr_clu, np.nanmean(SE_Per
,axis=1), 'ro-', label='Persistence')

ax.set_xlabel('Vr(km/s)')
ax.set_ylabel('RMSE(km/s)')
ax.set_title('final')
ax.legend()    

# plt.show()

fig.savefig('Figs/RMSE_all_'+str(N_sample)+'_'+str(IC)+'.png')
