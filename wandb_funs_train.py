import os
import sys
import torch
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torch import erf, erfinv
from torch.autograd import Variable
import datetime as dt
from ipdb import set_trace as st
import scipy.io as sio

import numpy as np
from tqdm import tqdm
import concurrent.futures
import h5py
import warnings
import sys
import random

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.dates as mdates
import matplotlib.units as munits

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import lightning.pytorch as lp
import wandb

from Model.Modules import V_FNO_DDP, V_FNO_long, dV_FNO_long
# from funs import est_beta, est_beta_in, seed_torch
# import asymmLaplace_accrue_torch



font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            # Apply Xavier initialization
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv2d):
            # Apply Kaiming initialization
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            # Initialize BatchNorm
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
            

def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

################# rk42 model #######################

def apply_rk42_model(v_init, dr_vec, dp_vec, 
                     mode, omega_rot=27.27):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    backwards model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param v_init: 1d array, initial velocity for forward/backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """
    # st()
    if mode == 'f':
        dr_vec = -1*dr_vec
    elif mode == 'b':
        dr_vec = dr_vec
    else:
        warnings.warn("Invalid mode. Please use 'f' for forward or 'b' for backward.", UserWarning)
        sys.exit("Exiting the program due to invalid mode.")
    
    omega_rot=(2 * np.pi) / (omega_rot * 86400)
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = np.clip(v_init, 150, 950)

    for i in range(1, len(dr_vec)+1):

        dvdphi_j1 = np.hstack((v[i-1, 1]-v[i-1, -1], 
                               v[i-1, 2:]-v[i-1, :-2],
                               v[i-1, 0]-v[i-1, -2])) # periodic BC
                               
        # dvdphi_j1 = np.log(v[i-1]+1)
        
        k1 = dvdphi_j1/v[i-1]
        k2 = v[i-1] - q*k1/2
        k2 = 0.5 * np.hstack((k2[1] - k2[-1], 
                              k2[2:] - k2[:-2],
                              k2[0] - k2[-2]))/k2 # periodic BC
        
        k3 = v[i-1] - q*k2/2
        k3 = 0.5 * np.hstack((k3[1] - k3[-1], 
                              k3[2:] - k3[:-2],
                              k3[0] - k3[-2]))/k3 # periodic BC

        k4 = v[i-1] - q*k3
        k4 = 0.5 * np.hstack((k4[1] - k4[-1], 
                              k4[2:] - k4[:-2],
                              k4[0] - k4[-2]))/k4 # periodic BC
        
        v[i] = v[i-1] -1/6*q*(k1+2*k2+2*k3+k4)

        if np.isnan(v).sum() != 0:
            st() 

    return v



def apply_rk42_model_tensor(v_init, dr_vec, dp_vec,
                      mode, omega_rot=27.27):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param v_init: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """
    if mode == 'f':
        dr_vec = -1*dr_vec
    elif mode == 'b':
        dr_vec = dr_vec
    else:
        warnings.warn("Invalid mode. Please use 'f' for forward or 'b' for backward.", UserWarning)
        sys.exit("Exiting the program due to invalid mode.")
    
    omega_rot=(2 * np.pi) / (omega_rot * 86400)
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = torch.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = torch.clip(v_init, 150, 950)

    for i in range(1, len(dr_vec)+1):

        # vv = torch.clip(v[i-1].clone(), 150, 950)
        vv = v[i-1].clone()
        
        dvdphi_j1 = torch.hstack((vv[1]-vv[-1], 
                               vv[2:]-vv[:-2],
                               vv[0]-vv[-2])) # periodic BC
        # dvdphi_j1 = torch.log(vv)
        k1 = dvdphi_j1/vv
        k2 = vv + q*k1/2
        k2 = 0.5 * torch.hstack((k2[1] - k2[-1], 
                              k2[2:] - k2[:-2],
                              k2[0] - k2[-2]))/k2 # periodic BC
        
        k3 = vv + q*k2/2
        k3 = 0.5 * torch.hstack((k3[1] - k3[-1], 
                              k3[2:] - k3[:-2],
                              k3[0] - k3[-2]))/k3 # periodic BC

        k4 = vv + q*k3
        k4 = 0.5 * torch.hstack((k4[1] - k4[-1], 
                              k4[2:] - k4[:-2],
                              k4[0] - k4[-2]))/k4 # periodic BC
        
        v[i] = vv + 1/6*q*(k1+2*k2+2*k3+k4)
        
    return v


################# rk42log model #######################

def apply_rk42log_model(v_init, dr_vec, dp_vec, 
                        mode, omega_rot=27.27):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    backwards model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param v_init: 1d array, initial velocity for forward/backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """
    # st()

    if mode == 'f':
        dr_vec = -1*dr_vec
    elif mode == 'b':
        dr_vec = dr_vec
    else:
        warnings.warn("Invalid mode. Please use 'f' for forward or 'b' for backward.", UserWarning)
        sys.exit("Exiting the program due to invalid mode.")
    
    omega_rot=(2 * np.pi) / (omega_rot * 86400)
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = np.clip(v_init, 150, 950)

    for i in range(1, len(dr_vec)+1):

        vv = v[i-1]
        lnv = np.log(vv+1)
        
        k1 = 0.5 * np.hstack((lnv[1] - lnv[-1], 
                              lnv[2:] - lnv[:-2],
                              lnv[0] - lnv[-2])) # periodic BC
        k2 = np.log(vv - q*k1/2)
        k2 = 0.5 * np.hstack((k2[1] - k2[-1], 
                              k2[2:] - k2[:-2],
                              k2[0] - k2[-2])) # periodic BC
        k3 = np.log(vv - q*k2/2)
        k3 = 0.5 * np.hstack((k3[1] - k3[-1], 
                              k3[2:] - k3[:-2],
                              k3[0] - k3[-2])) # periodic BC

        k4 = np.log(vv - q*k3)
        k4 = 0.5 * np.hstack((k4[1] - k4[-1], 
                              k4[2:] - k4[:-2],
                              k4[0] - k4[-2])) # periodic BC
        
        v[i] = vv - 1/6*q*(k1+2*k2+2*k3+k4)

    return v


def apply_rk42log_model_tensor(v_init, dr_vec, dp_vec, 
                               mode, omega_rot=27.27):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    backwards model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param v_init: 1d array, initial velocity for forward/backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """
    # st()

    if mode == 'f':
        dr_vec = -1*dr_vec
    elif mode == 'b':
        dr_vec = dr_vec
    else:
        warnings.warn("Invalid mode. Please use 'f' for forward or 'b' for backward.", UserWarning)
        sys.exit("Exiting the program due to invalid mode.")
    
    omega_rot=(2 * np.pi) / (omega_rot * 86400)
    
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = torch.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = torch.clip(v_init, 150, 950)

    for i in range(1, len(dr_vec)+1):

        vv = v[i-1].clone()
        # vv = torch.clip(vv, 150, 950)
        lnv = torch.log(vv+1)
        
        k1 = 0.5 * torch.hstack((lnv[1] - lnv[-1], 
                            lnv[2:] - lnv[:-2],
                            lnv[0] - lnv[-2])) # periodic BC
        k2 = torch.log(vv - q*k1/2)
        k2 = 0.5 * torch.hstack((k2[1] - k2[-1], 
                            k2[2:] - k2[:-2],
                            k2[0] - k2[-2])) # periodic BC
        k3 = torch.log(vv - q*k2/2)
        k3 = 0.5 * torch.hstack((k3[1] - k3[-1], 
                            k3[2:] - k3[:-2],
                            k3[0] - k3[-2])) # periodic BC

        k4 = torch.log(vv - q*k3)
        k4 = 0.5 * torch.hstack((k4[1] - k4[-1], 
                            k4[2:] - k4[:-2],
                            k4[0] - k4[-2])) # periodic BC
        
        v[i] = vv - 1/6*q*(k1+2*k2+2*k3+k4)
            
    return v



def batch_read_chunk(file_path, vari, start_idx, chunk_size):
    # Open the HDF5 file inside the worker process
    with h5py.File(file_path, 'r') as f:
        # Read the chunk of data for the specified variable
        return np.array(f[vari][start_idx:start_idx + chunk_size])

def batch_read(file_path, vari, N_sample, num_workers=96):
    """
    Multi-CPU version of batch_read function using ProcessPoolExecutor
    to parallelize the reading process, safely handling HDF5 files.
    """
    # Define chunk size (the size of data each process will handle)
    chunk_size = N_sample // 1000
    
    # Split the indices into chunks
    indices = np.arange(0, N_sample, chunk_size)
    
    out = []
    
    # Use ProcessPoolExecutor for multi-core processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each chunk of data to be read in parallel
        futures = [executor.submit(batch_read_chunk, file_path, vari, idx, chunk_size) for idx in indices]
        
        # Collect the results as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            out.append(future.result())
    
    # Concatenate all the chunks together
    out = np.concatenate(out)
    
    return out
'''


def batch_read(file_path, vari, N_sample):

    out = []
    with h5py.File(file_path, 'r') as f:
        for idx in tqdm(np.arange(0, N_sample, N_sample//1000)):
            out_t = np.array(f[vari][idx:idx+N_sample//1000])
            out.append(out_t)
        f.close()
    out = np.concatenate(out)

    return out
'''


def plot_com(x1, x2, x3=[], index = []):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x1, 'r.-', label='vr_5days_pred')
    ax.plot(x2, 'b.-', label='vr_5days')
    if len(x3) != 0:
        ax.plot(x3, 'g.-', label='vr')

    ax.legend()
    if len(index) != 0:
        line = np.arange(0, 1000, 10)
        ax.plot(np.tile(index[0], line.shape[0]), line)
        ax.plot(np.tile(index[-1], line.shape[0]), line)
    
    fig.savefig('Figs/test_com_WL.png')
    plt.close()

    

def fill_nan_nearest(input_tensor):

    # st()
    nan_mask = torch.isnan(input_tensor)

    # Find the nearest non-NaN values
    non_nan_indices = torch.arange(input_tensor.size(-1))[None, None, :].to('cuda:'+str(input_tensor.get_device()))
    nearest_indices = torch.argmin(torch.abs(non_nan_indices - nan_mask.float()), dim=-1)

    # Use indexing to replace NaN values with the nearest non-NaN values
    filled_tensor = torch.where(nan_mask, input_tensor[nearest_indices], input_tensor)

    return filled_tensor


def fill_nan_linear(vector):
    # Convert vector to torch tensor
    # vector = torch.tensor(vector, dtype=torch.float)

    # Find non-NaN indices
    non_nan_indices = torch.logical_not(torch.isnan(vector))

    st()
    # Create a tensor with linearly spaced values for interpolation
    x = torch.arange(len(vector), dtype=torch.float).to('cuda:'+str(vector.get_device()))

    # Interpolate NaN values using linear interpolation
    interpolated_values = interpolate(x[non_nan_indices], vector[non_nan_indices].unsqueeze(0), x, mode='linear').squeeze(0)

    # Replace NaN values with interpolated values
    vector[torch.isnan(vector)] = interpolated_values[torch.isnan(vector)]

    return vector.numpy()


def ML_data_read(filename_data,
                 N_sample,
                 shuffle_flag
                 ):

    with h5py.File(filename_data, 'r') as f:

        # torch.manual_seed(2023)
        np.random.seed(2023)
        # seed_torch(seed=2333)

        if shuffle_flag:
            shuffled_indices = np.random.permutation(101000)
        else:
            shuffled_indices = np.arange(101000)

        vr_5days = np.array(f['Vr_5days'])[shuffled_indices][:N_sample].T    
        vr = np.array(f['Vr'])[shuffled_indices][:N_sample].T
        r_end_5day = np.array(f['r_5days'])[shuffled_indices][:N_sample]*1.496e8
        r_end = np.array(f['r'])[shuffled_indices][:N_sample]*1.496e8
        images = np.array(f['GONG'])[shuffled_indices][:N_sample]
        date_clu = np.array(f['Time'])[shuffled_indices][:N_sample]
        idx = np.where(np.abs(np.diff(vr_5days[:121], axis = 1).max(axis=0))<400)[0]
        # print(idx.shape[0])
        
        # st()
        vr_5days = vr_5days[:, idx]
        vr = vr[:, idx]
        r_end_5day = r_end_5day[idx]
        r_end = r_end[idx]
        images = images[idx]
        date_clu = date_clu[idx]
        # f.close()
    
    return vr, vr_5days, r_end_5day, r_end, images, date_clu


def score_sort(std_Y_all, 
               vr, vr_5days, 
               mode='z_score'):

    num_std = np.zeros(std_Y_all.shape[0])

    # st()
    for i in tqdm(range(std_Y_all.shape[0])):
        
        if mode == 'z_score':
            num_std[i] = np.mean(np.abs(vr[i].squeeze() - \
                vr_5days[i].squeeze())/std_Y_all[i])
        elif mode == 'error':
            num_std[i] = np.mean(np.abs(vr[i].squeeze() - \
                vr_5days[i].squeeze()))
        elif mode == 'max':
            num_std[i] = np.max(vr_5days[i].squeeze())
        elif mode == 'mean':
            num_std[i] = np.mean(vr_5days[i].squeeze())
        
    # st()
    std_Y_sort = np.sort(num_std)[::-1]
    idx_sort = np.argsort(num_std)[::-1]

    return idx_sort, std_Y_sort


def plot_example(vr_pred, vr_real, 
                 v0_pred, v0_real,
                 weights):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(vr_pred.cpu().detach().numpy(), 'ro-', label='vr_pred')
    ax.plot(vr_real.cpu().detach().numpy(), 'b*-', label='vr_real')
    plt.legend()
    ax1 = ax.twinx()
    ax.plot(v0_pred.cpu().detach().numpy(), 'r*-', label='v0_pred')
    ax.plot(v0_real.cpu().detach().numpy(), 'bo-', label='v0_real')    
    fig.savefig('Figs/test.jpg')


class ProgressBar_tqdm(lp.callbacks.TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(            
            disable=True,            
        )
        return bar

class LitProgressBar(lp.callbacks.ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        percent = (batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def GONG_init(GONG_data, N_sample):

    ############## read image 
    with h5py.File(GONG_data, 'r') as f:

        images = np.array(f['image'][:N_sample])
        date_GONG = np.array(f['date'][:N_sample]).astype(int)
        f.close()

    ############## convert date to datetime format 
    # st()
    date_clu = []
    for i, date_tt in enumerate(tqdm(date_GONG)):
                
        t = dt.datetime(int(date_tt[0]),
                        int(date_tt[1]),
                        int(date_tt[2]),
                        int(date_tt[3]),
                        # int(date_tt[4]),
                        )
        date_clu.append(t) 

    date_clu = np.asarray(date_clu)

    return images, date_clu, date_GONG


def v02vr(pred_Y_train, v0, IC, r_end_5day):

    ######### convert v0_5days to vr_5days (training set)

    p = np.linspace(0, 360, 129)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi

    v0_5days_train = np.zeros([pred_Y_train.shape[0], 128])
    vr_5days_train = np.zeros([pred_Y_train.shape[0], 128])
    # st()
    for i in tqdm(range(r_end_5day.shape[0])):  
        r_vector=np.arange(695700*IC,r_end_5day[i], 695700*10) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:] - r_vector[0:-1]
        v0_5days_train[i] = np.abs(pred_Y_train[i].squeeze()+v0[:, i])
        vr_5days_train[i] = apply_rk42log_f_model(
            v0_5days_train[i], 
            dr_vec, dp_vec, 
            r0=695700*IC)[-1]

    return vr_5days_train


@rank_zero_only
def data_preprocess(init_data, 
                    vr,
                    vr_5days,
                    ML_file
                    ):

    _, _, r_end_5day, r_end, images, date_clu = init_data

    # st()
    ############## form Y by all variables required by training process
    Y = np.vstack([np.swapaxes(vr, 0, 2),
                np.expand_dims(np.tile(r_end_5day, (656, 1)), axis=0),
                # np.expand_dims(np.tile(HC_lon_now, (128, 1)), axis=0),
                # np.expand_dims(np.tile(HC_lon_5days, (128, 1)), axis=0),
                np.swapaxes(vr_5days, 0, 2), 
                np.expand_dims(np.tile(np.arange(vr_5days.shape[0]), (656, 1)), axis=0)
                # np.tile(np.arange(vr_5days.shape[0]), (100, 656, 1)), 
                ]).T 
    
    # st()

    ############## normalized X 
    X = (images - images.min())/(images.max() - images.min())
    # X = (images - images.mean())/images.std()
    # st()
    
    ############## reformat v0 for adding it to X
    y0 = vr[:, :, 0]
    X_reshape = np.random.randn(X.shape[0], 130, 656)   # Shape: (100000, 130, 656)
    X_reshape[:, :, :180] = X

    # Reshape y0 to (100000, 1, 656)
    y0_reshaped = y0[:, np.newaxis, :]

    # Concatenate X1_reshaped and X2 along the second axis
    X = np.concatenate([X_reshape, y0_reshaped], axis=1)
    X = np.swapaxes(X, 1, 2)
    # st()

    ############## save a ML-ready dataset (X and Y)
    with h5py.File(ML_file, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)
        f.close()
        

def lon_generate(HC_lon_now, HC_lon_5days):

    p = np.linspace(0, 360, 129)[:-1]
    lon_idx = np.zeros([HC_lon_now.shape[0], 24])
    for i in range(HC_lon_now.shape[0]):
        if HC_lon_now[i] > HC_lon_5days[i]:
            idx_sta = np.where((p > HC_lon_5days[i]))[0][0]
            # st()
            if idx_sta + 24 > 127:
                idx = np.arange(idx_sta-1, idx_sta+23)
            else:    
                idx = np.arange(idx_sta, idx_sta+24)
        else:
            idx_sta = np.where(p > HC_lon_5days[i])[0]
            
            if len(idx_sta) >= 1:
                idx_sta = idx_sta[0]
            # st()
                idx = np.hstack((np.arange(idx_sta, 128), 
                                    np.arange(24-128+idx_sta)))
            elif len(idx_sta) == 0:
                idx = np.arange(24)

        lon_idx[i] = idx

    return lon_idx


def data_split(date_ACE, idx_clu, test_year):

    # st()
    # idx_test = np.where((date_ACE[idx_clu, 0] == 2018))[0]
    idx_test = np.where((date_ACE[idx_clu, 1] == 8))[0]
    # idx_test = np.where((date_ACE[idx_clu, 0] == 2015))[0]
    # idx_test = np.where((date_ACE[idx_clu, 0] == 2015) & (date_ACE[idx_clu, 1] >= 10))[0]
    idx_valid = np.where((date_ACE[idx_clu, 1] == 9))[0]
    
    # st()
    # idx_valid = np.where((date_ACE[idx_clu, 0] == test_year))[0]
    # idx_valid = np.where((date_ACE[idx_clu, 0] == test_year-2) | (date_ACE[idx_clu, 0] == test_year-1))[0]
    # idx_valid = np.where((date_ACE[idx_clu, 0] == 2014) | (date_ACE[idx_clu, 0] == 2016))[0]
    # idx_valid = np.where((date_ACE[idx_clu, 0] == 2015) & (date_ACE[idx_clu, 1] < 8))[0]
    # st()
    idx_train = np.arange(len(idx_clu))
    idx_train = list(idx_train)
    for idx in idx_test:
        idx_train.remove(idx)
    for idx in idx_valid:
        idx_train.remove(idx)

    idx_train = np.asarray(idx_train)
    idx_valid = np.asarray(idx_valid)
    idx_test = np.asarray(idx_test)

    return idx_train, idx_valid, idx_test


def V2logV_all(vr, vr_5days, 
           r_end, r_end_5day, 
           IC, idx_clu, 
           vr_mat):

    # if os.path.exists(vr_mat)==0:
    # st()
    ############# boundary angle
    p = np.linspace(0, 360, 129)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    v0 = np.zeros(vr[:, idx_clu].shape)
    vr_tmp = np.zeros([vr[:, idx_clu].shape[1], 128, 100])
    v0_5days = np.zeros(vr[:, idx_clu].shape)
    vr_5days_tmp = np.zeros([vr[:, idx_clu].shape[1], 128, 100])

    # st()
    for i, idx in enumerate(tqdm(idx_clu)):
        # st()
        # r_vector=np.arange(695700*IC, r_end[i], 695700*5)
        r_vector=np.linspace(695700*IC,r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:]-r_vector[0:-1]

        # st()
        # v0[:, i] = apply_rk42_b_model(vr[:, idx], dr_vec, dp_vec, alpha=0.15,
        v0[:, i] = apply_rk42log_b_model(vr[:, idx], dr_vec, dp_vec, alpha=0.15,
                        rh=r_end[i], add_v_acc=True,
                        r0=695700*IC, omega_rot=(2 * np.pi) / (27.27 * 86400))[-1]

        # st()
        # vr_tmp[:, i] = apply_rk42_f_model(v0[:, i], dr_vec, dp_vec,
        vr_tmp[i] = apply_rk42log_f_model(v0[:, i], dr_vec, dp_vec,
                                r0=695700*IC).T

        # v0_5days[:, i] = apply_rk42_b_model(vr_5days[:, idx], dr_vec, dp_vec, alpha=0.15,
        v0_5days[:, i] = apply_rk42log_b_model(vr_5days[:, idx], dr_vec, dp_vec, alpha=0.15,
                        rh=r_end_5day[i], add_v_acc=True,
                        r0=695700*IC, omega_rot=(2 * np.pi) / (27.27 * 86400))[-1]

        # vr_5days_tmp[:, i] = apply_rk42_f_model(v0_5days[:, i], dr_vec, dp_vec,
        vr_5days_tmp[i] = apply_rk42log_f_model(v0_5days[:, i], dr_vec, dp_vec,
                                r0=695700*IC).T
        
    # st()
    with h5py.File(vr_mat, 'w') as f:
        
        # f.create_dataset('v0', data=v0)
        # f.create_dataset('v0_5days', data=v0_5days)
        f.create_dataset('vr', data=vr_tmp)
        f.create_dataset('vr_5days', data=vr_5days_tmp)
        f.close()


@rank_zero_only
def V2logV_long(init_data,
           IC, idx_clu, 
           vr_mat):

    vr, vr_5days, r_end_5day, r_end, images, date_clu = init_data

    # if os.path.exists(vr_mat)==0:
    # st()
    ############# boundary angle
    p = np.linspace(0, 360, 657)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    # v0 = np.zeros(vr[:, idx_clu].shape)
    v0 = np.zeros([vr[:, idx_clu].shape[1], 656, 100])
    # v0_5days = np.zeros(vr[:, idx_clu].shape)
    v0_5days = np.zeros([vr[:, idx_clu].shape[1], 656, 100])

    # st()
    for i, idx in enumerate(tqdm(idx_clu)):
        # st()
        # r_vector=np.arange(695700*IC, r_end[i], 695700*5)
        r_vector=np.linspace(695700*IC,r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:]-r_vector[0:-1]

        # st()
        # v0[:, i] = apply_rk42_b_model(vr[:, idx], dr_vec, dp_vec, alpha=0.15,
        v0[i] = apply_rk42log_b_model(vr[:, idx], dr_vec, dp_vec, alpha=0.15,
                        rh=r_end[i], add_v_acc=True,
                        r0=695700*IC, omega_rot=(2 * np.pi) / (27.27 * 86400))[::-1].T

        # st()
        # vr_tmp[:, i] = apply_rk42_f_model(v0[:, i], dr_vec, dp_vec,
        # vr_tmp[i] = apply_rk42log_f_model(v0[:, i], dr_vec, dp_vec,
        #                         r0=695700*IC).T

        # v0_5days[:, i] = apply_rk42_b_model(vr_5days[:, idx], dr_vec, dp_vec, alpha=0.15,
        v0_5days[i] = apply_rk42log_b_model(vr_5days[:, idx], dr_vec, dp_vec, alpha=0.15,
                        rh=r_end_5day[i], add_v_acc=True,
                        r0=695700*IC, omega_rot=(2 * np.pi) / (27.27 * 86400))[::-1].T
        
        # st()

        # vr_5days_tmp[:, i] = apply_rk42_f_model(v0_5days[:, i], dr_vec, dp_vec,
        # vr_5days_tmp[i] = apply_rk42log_f_model(v0_5days[:, i], dr_vec, dp_vec,
        #                         r0=695700*IC).T
        
    # st()
    with h5py.File(vr_mat, 'w') as f:
        
        f.create_dataset('r', data=r_end)
        f.create_dataset('r_5days', data=r_end_5day)
        f.create_dataset('vr', data=v0)
        f.create_dataset('vr_5days', data=v0_5days)
        # f.create_dataset('vr', data=vr_tmp)
        # f.create_dataset('vr_5days', data=vr_5days_tmp)
        # f.create_dataset('vr_mean', data=vr_5days_tmp[:, -1, -1])
        # f.create_dataset('vr_std', data=vr_5days_tmp[:, -1, -1])
        f.create_dataset('date_clu', data=date_clu)
        f.close()
        

@rank_zero_only
def V2V_long(init_data,
           IC, idx_clu, 
           vr_mat):

    vr, vr_5days, r_end_5day, r_end, images, date_clu = init_data

    # if os.path.exists(vr_mat)==0:
    # st()
    ############# boundary angle
    p = np.linspace(0, 360, 657)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    # v0 = np.zeros(vr[:, idx_clu].shape)
    v0 = np.zeros([vr[:, idx_clu].shape[1], 656, 100])
    vr_tmp = np.zeros([vr[:, idx_clu].shape[1], 656, 100])
    # v0_5days = np.zeros(vr[:, idx_clu].shape)
    v0_5days = np.zeros([vr[:, idx_clu].shape[1], 656, 100])
    vr_5days_tmp = np.zeros([vr[:, idx_clu].shape[1], 656, 100])

    # st()
    for i, idx in enumerate(tqdm(idx_clu)):
        # st()
        r_vector=np.linspace(695700*IC,r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:]-r_vector[0:-1]

        # st()
        v0[i] = apply_rk42log_model(vr[:, idx], dr_vec, dp_vec, 
                                 mode='b', omega_rot=27.27)[::-1].T

        v0_5days[i] = apply_rk42log_model(vr_5days[:, idx], dr_vec, dp_vec, 
                                       mode='b', omega_rot=27.27)[::-1].T
        
        # st()
        vr_tmp[i] = apply_rk42log_model(v0[i, :, 0], dr_vec, dp_vec, 
                                 mode='f', omega_rot=27.27).T

        vr_5days_tmp[i] = apply_rk42log_model(v0_5days[i, :, 0], dr_vec, dp_vec, 
                                       mode='f', omega_rot=27.27).T
        
        
    with h5py.File(vr_mat, 'w') as f:
        
        f.create_dataset('r', data=r_end)
        f.create_dataset('r_5days', data=r_end_5day)
        f.create_dataset('vr', data=v0)
        f.create_dataset('vr_5days', data=v0_5days)
        # f.create_dataset('vr', data=vr_tmp)
        # f.create_dataset('vr_5days', data=vr_5days_tmp)
        # f.create_dataset('vr_mean', data=vr_5days_tmp[:, -1, -1])
        # f.create_dataset('vr_std', data=vr_5days_tmp[:, -1, -1])
        f.create_dataset('date_clu', data=date_clu)
        f.close()


def V2logV(vr, vr_5days, 
           r_end, r_end_5day, 
           IC, idx_clu, 
           vr_mat):

    # if os.path.exists(vr_mat)==0:

    ############# boundary angle
    p = np.linspace(0, 360, 129)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    v0 = np.zeros(vr[:, idx_clu].shape)
    vr_tmp = np.zeros(vr[:, idx_clu].shape)
    v0_5days = np.zeros(vr[:, idx_clu].shape)
    vr_5days_tmp = np.zeros(vr[:, idx_clu].shape)

    for i, idx in enumerate(tqdm(idx_clu)):
        # st()
        r_vector=np.arange(695700*IC, r_end[i], 695700*5)
        # r_vector=np.linspace(695700*IC,r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:]-r_vector[0:-1]

        # st()
        # v0[:, i] = apply_rk42_b_model(vr[:, idx], dr_vec, dp_vec, alpha=0.15,
        v0[:, i] = apply_rk42log_b_model(vr[:, idx], dr_vec, dp_vec, alpha=0.15,
                        rh=r_end[i], add_v_acc=True,
                        r0=695700*IC, omega_rot=(2 * np.pi) / (25.38 * 86400))[-1]

        # st()
        # vr_tmp[:, i] = apply_rk42_f_model(v0[:, i], dr_vec, dp_vec,
        vr_tmp[:, i] = apply_rk42log_f_model(v0[:, i], dr_vec, dp_vec,
                                r0=695700*IC)[-1]

        # v0_5days[:, i] = apply_rk42_b_model(vr_5days[:, idx], dr_vec, dp_vec, alpha=0.15,
        v0_5days[:, i] = apply_rk42log_b_model(vr_5days[:, idx], dr_vec, dp_vec, alpha=0.15,
                        rh=r_end_5day[i], add_v_acc=True,
                        r0=695700*IC, omega_rot=(2 * np.pi) / (25.38 * 86400))[-1]

        # vr_5days_tmp[:, i] = apply_rk42_f_model(v0_5days[:, i], dr_vec, dp_vec,
        vr_5days_tmp[:, i] = apply_rk42log_f_model(v0_5days[:, i], dr_vec, dp_vec,
                                r0=695700*IC)[-1]

    ############## remove CMEs ###############
    # idx = np.where(np.abs(np.diff(vr_5days_tmp, axis=0)).max(axis=0) < 400)[0]
    # st()
    # if save_flag:
    with h5py.File(vr_mat, 'w') as f:
        
        f.create_dataset('v0', data=v0)
        f.create_dataset('v0_5days', data=v0_5days)
        f.create_dataset('vr', data=vr_tmp)
        f.create_dataset('vr_5days', data=vr_5days_tmp)
        f.close()



class GONG_Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.Y[idx]
        return x_sample, y_sample
    

class netcdf_Dataset(Dataset):
    def __init__(self, filename, index_list):
        # def __init__(self, X, Y, filename, start_idx, end_idx):
        
        self.filename = filename
        self.list_idx = index_list
        # self.X = X
        # self.Y = Y

    def __len__(self):
        return len(self.list_idx)

    def __getitem__(self, idx):
        # st()
        with h5py.File(self.filename, 'r') as f:
            x_sample = torch.from_numpy(np.array(f['X'][self.list_idx[idx]])).float()
            y_sample = torch.from_numpy(np.array(f['Y'][self.list_idx[idx]])).float()
            idx_sel = self.list_idx[idx]
            # f.close()

        return x_sample, y_sample, idx_sel


def V_loss(y_pred, 
           y_true, 
           vr_mean,
           vr_std,
           IC,
           weight_flag=True
           ):

    table = wandb.Table(columns=['ID', 'Image'])

    # y_pred = (y_pred - vr_mean)
    # st()
    
    device = 'cuda:'+str(y_pred.get_device())
    y0 = y_true[:, :128] # v0
    r_end = y_true[:, 128] # r_end
    HC_lon_now = y_true[:, 129].detach().cpu()
    HC_lon_5days = y_true[:, 130].detach().cpu()
    yr = y_true[:, 131:259].to(device) # v0_5days
    yr_5days = y_true[:, 259:387].to(device) # vr_5days
    yr_5days = y_true[:, 259:387].to(device) # vr_5days

    RMSE = torch.tensor(0).float().to(device)
    p = torch.linspace(0, 360, 129)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    # tmp_clu = []
    for i in range(r_end.shape[0]):

        # r_vector=torch.arange(695700*IC, r_end[i], 695700*5) # solve the backward propagation all the way to 1 solar radius
        r_vector=torch.linspace(695700*IC, r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:] - r_vector[0:-1]
        tmp = torch.zeros(y_pred[i].shape).to(device)
        tmp[:] = np.nan

        if HC_lon_now[i] > HC_lon_5days[i]:
            idx = np.where((p < HC_lon_now[i])
                            & (p > HC_lon_5days[i])
                )[0]
        else:
            idx = np.where(p > HC_lon_5days[i])[0]
            idx = np.hstack((idx, 
                            np.where(p < HC_lon_now[i])[0]))


        # v_init = y_pred[i].squeeze()
        # v_init = torch.abs(y_pred[i].squeeze()+y0[i].squeeze())
        v_init = torch.abs(y_pred[i].squeeze()*5+y0[i].squeeze())
        # v_init = torch.abs(y_pred[i].squeeze()*vr_mean+y0[i].squeeze())
        # idx_900 = torch.where(v_init > 900)[0]
        # idx_150 = torch.where(v_init < 150)[0]
        # v_init[idx_900] = 900
        # v_init[idx_150] = 150
        
        # import ipdb;ipdb.set_trace()
        
        # v_init = fill_nan_nearest(v_init)
        # v_init = fill_nan_linear(v_init)

        # tmp = apply_rk42_f_model_tensor(v_init, 
        tmp = apply_rk42log_f_model_tensor(v_init, 
                    dr_vec, dp_vec, 
                    r0=695700*IC)[-1]
        # st()

        # tmp_clu.append(tmp)
        weights = 5+5*torch.tanh((yr_5days[i, idx]-350)/150)
        # print('yr_5days in device: {}'.format(yr_5days.get_device()))
        # print('tmp in device: {}'.format(tmp.get_device()))
        RMSE_vr = yr_5days[i] - tmp.to(device)

        if i == 0:
            plot_com(tmp.cpu().detach().numpy(), 
                        yr_5days[i].cpu(),
                        yr[i].cpu(),
                        index = idx)
        
        # fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        # ax[0].plot(yr_5days[i].cpu().detach().numpy(), 'ro-', label='vr_5days')
        # ax[0].plot(yr[i].cpu().detach().numpy(), 'k.-', label='vr')
        # ax[0].plot(tmp.cpu().detach().numpy(), 'bx-', label='vr_5days_pred')
        # plt.legend()
        # ax[1].plot(RMSE_vr.cpu().detach().numpy(), 'bx')
        # fig.savefig('Figs/test.jpg')
        # st()

        # RMSE_v0 = y0_5days[i, :] - v_init.to(device)
        # idx_v0 = torch.where(torch.abs(y0[i, :] - y0_5days[i, :])>10)[0]

        # st()
        if weight_flag:
            # plot_example(tmp[idx], yr_5days[i, idx],
            #              v_init, y0_5days[i], 
            #              weights)
            # st()
            RMSE += torch.mean(weights*(RMSE_vr[idx]**2)) 
            # RMSE += torch.mean((RMSE_v0[idx_v0]**2)) 
        else:
            RMSE += torch.mean(RMSE_vr[idx]**2) 
            # RMSE += torch.mean(RMSE_v0**2) 

    return RMSE/(i+1)



def V_loss_all(y_pred, 
               y_true, 
               vr_mean,
               vr_std,
               IC,
               ratio,
               weight_flag=True
               ):

    # y_pred = (y_pred - vr_mean)
    # y_pred_mag = y_pred[:, :128]
    # y_pred_t = y_pred[:, 128:]
    y_pred_t = y_pred
    # y_pred_t = (y_pred_t - vr_mean)/vr_std
    # st()
    
    device = 'cuda:'+str(y_pred.get_device())
    # print('max of y_pred {}'.format(y_pred.max()))
    # print('min of y_pred {}'.format(y_pred.min()))
    # st()
    yr = y_true[:, :, :100]
    r_end = y_true[:, 0, 100] # r_end
    HC_lon_now = y_true[:, 0, 101].detach().cpu()
    HC_lon_5days = y_true[:, 0, 102].detach().cpu()
    # y0_5days = y_true[:, 131:259].to(device) # v0_5days
    yr_5days = y_true[:, :, 103:] # vr_5days

    RMSE = torch.tensor(0).float().to(device)
    # RMSE = Variable(RMSE, requires_grad=True)
    p = torch.linspace(0, 360, 129)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    # tmp_clu = []
    for i in range(r_end.shape[0]):
        tmp = torch.zeros(y_pred_t[i].shape).to(device)
        tmp[:] = np.nan
        r_vector=torch.linspace(695700*IC, r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:] - r_vector[0:-1]

        if HC_lon_now[i] > HC_lon_5days[i]:
            idx = np.where((p < HC_lon_now[i])
                            & (p > HC_lon_5days[i])
                )[0]
        else:
            idx = np.where(p > HC_lon_5days[i])[0]
            idx = np.hstack((idx, 
                            np.where(p < HC_lon_now[i])[0]))

        # v_init = torch.abs(y_pred[i].squeeze()+y0[i].squeeze())
        # v_init = y_pred[i].squeeze()*vr_mean+400
        # v_init = y_pred_t[i].squeeze()*(y_pred_mag[i]+1)*100 #+yr[i, :, 0].squeeze()
        # v_init = y_pred_t[i].squeeze()*(y_pred_mag[i]+1)*10+yr[i, :, 0].squeeze()
        # v_init = y_pred[i].squeeze()+yr[i, :, 0].squeeze()
        # st()
        # v_init = y_pred_t[i].squeeze()*vr_std+yr[i, :, 0].squeeze()*1
        # v_init = torch.abs(y_pred_t[i].squeeze()*y_pred_mag[i].squeeze())
        # v_init = torch.abs(y_pred_t[i].squeeze())
        # v_init = torch.abs(y_pred_t[i].squeeze()*vr_mean)
        # v_init = torch.abs(y_pred_t[i].squeeze()*50+yr[i, :, 0].squeeze())
        # v_init = torch.tanh((y_pred_t[i].squeeze() - vr_mean)/300)*250+550
        v_init = torch.abs(y_pred_t[i].squeeze()*vr_std*2+yr[i, :, 0].squeeze())
        # v_init = torch.abs(y_pred_t[i].squeeze()*vr_std*ratio+yr[i, :, 0].squeeze())
        # v_init = torch.abs(y_pred_t[i].squeeze()*vr_std+yr[i, :, 0].squeeze())
        # v_init = torch.abs(y_pred_t[i].squeeze()*y_pred_mag[i].squeeze()+yr[i, :, 0].squeeze())
        # st()
        # tmp = apply_rk42_f_model_tensor(v_init, 
        tmp = apply_rk42log_f_model_tensor(v_init, 
                    dr_vec, dp_vec, 
                    r0=695700*IC).to(device)
        # if torch.isnan(tmp).sum()>0:
        #     st()
        #     continue
        
        # st()
        # if i == 0:
        #     plot_com(tmp[-1].cpu().detach().numpy(), 
        #              yr_5days[i, :, -1].cpu(),
        #              yr[i, :, -1].cpu(),
        #              index = idx)
            # print(v_init.max())
            # print(v_init.min())
            # st()
        # st()
        if weight_flag:
            st()
            # for j in range(tmp.shape[0]):
            for j in range(tmp.shape[0]-1, tmp.shape[0]):
                diff = torch.abs(yr_5days[i, :, j] - yr[i, :, j])
                idx = diff.argsort()[-24:]
                # idx = torch.where(>10)[0][:24]
                # idx = torch.where(torch.abs(yr_5days[i, :, j] - yr[i, :, j])>10)[0][:24]
                weights = (5+5*torch.tanh((yr_5days[i, idx, j]-350)/150))
                weights = weights * (0.01 + j / 100)
                # weights = weights * (5*np.tanh(j/40)+1)
                # st()
                RMSE_vr = yr_5days[i, idx, j] - tmp[j, idx]

                # fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                # ax.plot(yr_5days[i, :, j].cpu().detach().numpy(), 'ro-', label='vr_5days')
                # ax.plot(yr[i, :, j].cpu().detach().numpy(), 'k.-', label='vr')
                # ax.plot(tmp[j].cpu().detach().numpy(), 'bx-', label='vr_5days_pred')
                # plt.legend()
                # ax1 = ax.twinx()
                # ax1.plot(np.abs(yr_5days[i, :, j].cpu().detach().numpy() - yr[i, :, j].cpu().detach().numpy()))
                
                # fig.savefig('Figs/test.jpg')
            
                # st()
                RMSE += torch.nanmean(weights*(RMSE_vr**2)) 
                # RMSE += torch.nanmean(weights*(RMSE_vr**2)) 
                # RMSE += torch.mean((RMSE_v0[idx_v0]**2)) 
            # RMSE = RMSE/tmp.shape[0]
            
        else:
            # diff = torch.abs(yr_5days[i, :, -1] - yr[i, :, -1])
            # # st()
            # idx = diff.argsort()[-24:]
            RMSE_vr = (yr_5days[i, idx, -1] - tmp[-1, idx])*yr_5days[i, idx, -1]/yr_5days[i, idx, -1].max()
            # RMSE_vr = RMSE_vr*yr_5days[i, idx, -1]/yr_5days[i, idx, -1].max()
            RMSE += torch.nanmean((RMSE_vr**2)) 

    return RMSE/(i+1)/j
    # return Variable(RMSE/(i+1)/tmp.shape[1], requires_grad = True)


def V_loss_long(y_pred, 
               y_true, 
               vr_mean,
               vr_std,
               IC,
               ratio,
               weight_flag=True
               ):
    
    device = 'cuda:'+str(y_pred.get_device())

    if len(y_pred.shape) == 1:
        y_pred = y_pred.unsqueeze(axis=0)
    # print(y_pred.max())
    # st()
    yr = y_true[:, :, :100]
    r_end = y_true[:, 0, 100] # r_end
    yr_5days = y_true[:, :, 101:-1] # vr_5days
    event_idx = y_true[:, 0, -1]
    
    # st()

    RMSE = torch.tensor(0).float().to(device)
    RMSE_ori = torch.tensor(0).float().to(device)
    p = torch.linspace(0, 360, 657)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    # tmp_clu = []
    for i in range(r_end.shape[0]):
        r_vector=torch.linspace(695700*IC, r_end.cpu().detach().numpy()[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:] - r_vector[0:-1]
        
        # st()
        # v_init = y_pred[i].squeeze()*vr_std/10+yr[i, :, 0].squeeze()
        v_init = torch.exp(y_pred[i].squeeze()/5)*yr[i, :, 0].squeeze()
        # tmp = v_init
        tmp = apply_rk42log_model_tensor(v_init, dr_vec, dp_vec, 
                    mode='f', omega_rot=27.27).to(device)[-1]
        tmp_ori = apply_rk42log_model_tensor(yr_5days[i, :, 0], dr_vec, dp_vec, 
                    mode='f', omega_rot=27.27).to(device)[-1]
        # st()
        # print('event_idx: {}'.format(int(event_idx[i])))
        # st()
        if int(event_idx[i]) == 0:
            
            for n in range(500):
                figname = 'Figs/test/event0_epoch'+str(n)+'.png'
                if os.path.exists(figname):
                    continue
                else:
                    fig, ax = plt.subplots(figsize=(16, 8))
                    # st()
                    ax.plot(tmp[:150].detach().cpu().numpy(), label='vr_5days_pred')
                    ax.plot(tmp_ori[:150].detach().cpu().numpy(), label='vr_5days_ori')
                    ax.plot(yr_5days[i, :150, -1].detach().cpu().numpy(), label='vr_5days')
                    ax.plot(yr[i, :150, -1].detach().cpu().numpy(), label='vr')
                    plt.legend()
                    
                    fig.savefig(figname)
                    # st()
                    plt.close()
                    break
        
        if weight_flag:
            # st()

            weights = (5+5*torch.tanh((yr_5days[i, :120, -1]-350)/150))
            weights_diff = torch.log((yr[i, :120, -1] - yr_5days[i, :120, -1])**2+1)
            # RMSE_vr = yr_5days[i, :24, -1] - tmp[:24]
            RMSE_vr = yr_5days[i, :120, -1] - tmp[:120]
            # RMSE += torch.nanmean((RMSE_vr**2)) 
            # RMSE += torch.mean(weights**3*(RMSE_vr**2))
            # st() 
            # RMSE += torch.nanmean(weights**2.5*(RMSE_vr**2)) 
            # RMSE += torch.nanmean(weights*(RMSE_vr**2)) 
            # st()
            
            
            # Fisher Z-Transformation
            vx = tmp[:120] - torch.mean(tmp[:120])
            vy = yr_5days[i, :120, -1] - torch.mean(yr_5days[i, :120, -1])
            
            r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            z = 0.5 * torch.log((1 + r**2) / (1 - r**2))  # Fisher Z-transformation
            
            scale_z = 1/(1+torch.exp(-1*z))
            # RMSE += torch.mean(weights_diff*weights*(RMSE_vr**2)) * (1 - scale_z) 
            # RMSE += torch.mean(weights_diff**2*weights*(RMSE_vr**2)) * (1 - scale_z) 
            # RMSE_ori += torch.mean((RMSE_vr**2)) 
            
            RMSE_v0 = yr_5days[i, :, 0] - v_init
            RMSE += torch.nanmean((RMSE_v0**2)) 
            
            if torch.isinf(RMSE).any() or torch.isnan(RMSE).any():
                st()
        else:
            RMSE_v0 = yr_5days[i, :, 0] - v_init
            RMSE_vr = yr_5days[i, :120, -1] - tmp[:120]
            RMSE += torch.nanmean((RMSE_vr**2)) 
            # RMSE += torch.nanmean((RMSE_v0**2)) 
            if torch.isinf(RMSE).any() or torch.isnan(RMSE).any():
                st()
            # test_fig_plot(event_idx[i].cpu().detach().numpy(), tmp[:120], yr[i, :120, -1], yr_5days[i, :120, -1])
    
    # print('i: {}, RMSE: {}'.format(i+1, torch.sqrt(RMSE/(i+1))))
    # print('i: {}, no weight RMSE: {}'.format(i+1, torch.sqrt(RMSE_ori/(i+1))))
    
    return RMSE/(i+1)
    # return Variable(RMSE/(i+1)/tmp.shape[1], requires_grad = True)


def V_loss_update(y_pred, 
               y_true, 
               vr_mean,
               vr_std,
               IC,
               ratio,
               weight_flag=True
               ):
    
    device = 'cuda:'+str(y_pred.get_device())
    # st()
    yr = y_true[:, :, :100]
    r_end = y_true[:, 0, 100] # r_end
    yr_5days = y_true[:, :, 103:] # vr_5days

    RMSE = torch.tensor(0).float().to(device)
    p = torch.linspace(0, 360, 129)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    # tmp_clu = []
    for i in range(r_end.shape[0]):
        # tmp = torch.zeros(y_pred[i].shape).to(device)
        # tmp[:] = np.nan
        # st()
        r_vector=torch.linspace(695700*IC, r_end.cpu().detach().numpy()[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:] - r_vector[0:-1]

        idx = torch.where(y_pred>0.7)[0]
        y_pred[idx] = 0.7

        # diff = torch.abs(yr_5days[i, :, 0] - yr[i, :, 0])
        # idx = diff.argsort()[-24:]
        # v_init = yr[i, :, 0].float().to(device)
        # v_init[idx] = torch.exp(y_pred[i, idx].squeeze())*yr[i, idx, 0].squeeze()
        # v_init = torch.exp(y_pred[i].squeeze())*yr[i, :24, -1].squeeze()
        # v_init = torch.exp(y_pred[i].squeeze())*yr[i, :, -1].squeeze()
        v_init = torch.exp(y_pred[i].squeeze())*yr[i, :, -1].squeeze()
        # st()
        # v_init = y_pred[i].squeeze()*vr_std*2+yr[i, :, 0].squeeze()
        # if 
        v_init = apply_rk42log_f_model_tensor(v_init, 
                    dr_vec, dp_vec, 
                    r0=695700*IC).to(device)[-1]
        # st()
        
        # for j in range(0, 1):
        # for j in range(tmp.shape[0]-1, tmp.shape[0]):
        # for j in range(tmp.shape[0]):
            # diff = torch.abs(yr_5days[i, :, j] - yr[i, :, j])
            # idx = diff.argsort()[:]
            # idx = np.arange(128)
        if weight_flag:
            
            # weights = (5+5*torch.tanh((yr_5days[i, :, 0]-350)/150))
            # weights = torch.sqrt((yr_5days[i, :, 0]-yr[i, :, 0])**2)
            # # RMSE_vr = yr_5days[i, :, 0] - tmp[0, :]
            # RMSE_vr = yr_5days[i, :, 0] - v_init
            # RMSE += torch.nanmean(weights*(RMSE_vr**2)) 
            # st()
            # weights_diff = torch.zeros([yr_5days.shape[0], 24])
            # weights_diff[1:] = torch.diff(yr_5days[i, :24, -1])
            # weights_diff = weights_diff.to(device)

            weights = (5+5*torch.tanh((yr_5days[i, :24, -1]-350)/150))
            # RMSE_vr = yr_5days[i, :24, -1] - tmp[:24]
            RMSE_vr = yr_5days[i, :24, -1] - v_init[:24]
            # RMSE += torch.nanmean((RMSE_vr**2)) 
            RMSE += torch.nanmean(weights*(RMSE_vr**2)) 
            # RMSE += torch.nanmean(weights*weights_diff/5*(RMSE_vr**2)) 
            # st()
        
        else:
            RMSE_vr = (yr_5days[i, :24, -1] - v_init[:24])*yr_5days[i, :24, -1]/yr_5days[i, :24, -1].max()
            RMSE += torch.nanmean((RMSE_vr**2)) 

    return RMSE/(i+1)
    # return Variable(RMSE/(i+1)/tmp.shape[1], requires_grad = True)


def dV_loss(y_pred, 
            y_true, 
            vr_mean,
            vr_std,
            CRPS_min,
            RS_min,
            IC,
            weight_flag=True
            ):

    device = 'cuda:'+str(y_pred.get_device())

    # st()
    lon_now = y_true[:, 0, 2].cpu().detach().numpy()
    lon_5days = y_true[:, 0, 3].cpu().detach().numpy()
    y_pred = (y_pred - vr_mean)/vr_std
    d = torch.abs(y_true[:, :, 1] - y_true[:, :, 0])
    d = d.to(device)
    N = d.shape[0]
    # st()
    # sigma = torch.abs(y_pred).squeeze().to(device)
    sigma = torch.exp(y_pred).squeeze().to(device)
    x = torch.zeros(sigma.shape)
    loss = 0

    x = d/(sigma+1e-5)
    x = x/np.sqrt(2)
    x = x.to(device)

    CRPS = torch.zeros([x.shape[0], 24]).to(device)
    RS = torch.zeros([x.shape[0], 24]).to(device)

    p = np.linspace(0, 360, 129)[:-1]
    lon_idx = np.zeros([lon_now.shape[0], 24])
    # st()
    for i in range(lon_now.shape[0]):
        if lon_now[i] > lon_5days[i]:
            idx_sta = np.where((p > lon_5days[i]))[0][0]
            # st()
            if idx_sta + 24 > 127:
                idx = np.arange(idx_sta-1, idx_sta+23)
            else:    
                idx = np.arange(idx_sta, idx_sta+24)
        else:
            idx_sta = np.where(p > lon_5days[i])[0]
            
            if len(idx_sta) >= 1:
                idx_sta = idx_sta[0]
                idx = np.hstack((np.arange(idx_sta, 128), 
                                    np.arange(24-128+idx_sta)))
            elif len(idx_sta) == 0:
                idx = np.arange(24)

        if len(idx) == 23:
            st()
        # st()
        lon_idx[i] = idx

    for idx in range(24): ## 0 or 1
        # st()
        x_t = torch.diagonal(x[:, lon_idx[:, idx]])
        ind = torch.argsort(x_t)
        ind_orig = torch.argsort(ind)+1
        ind_orig = ind_orig.to(device)
        CRPS_1 = np.sqrt(2)*x_t*erf(x_t)
        CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x_t**2) 
        CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
        CRPS_3 = CRPS_3.to(device)
        # st()
        CRPS[:, idx] = torch.diagonal(sigma[:, lon_idx[:, idx]])*(CRPS_1 + CRPS_2 - CRPS_3)
        # st()
        if torch.isinf(CRPS[i, idx]):
            st() 

        # import ipdb;ipdb.set_trace()
        RS[:, idx] = N*(x_t/N*(erf(x_t)+1) - 
            x_t*(2*ind_orig-1)/N**2 + 
            torch.exp(-x_t**2)/np.sqrt(np.pi)/N)
    
        RS = RS.to(device)
        
        weights = torch.exp((torch.diagonal(d[:, lon_idx[:, idx]])-vr_mean)/vr_std)
        weights = weights.to(device)
        loss += torch.nanmean((CRPS[:, idx]/CRPS_min\
            +RS[:, idx]/RS_min
            )
            *weights
            )
    # st()
    
    if torch.isnan(loss.cpu()).sum() > 0:
        st()

    # print(loss)
    # st()
    return loss/24


def dV_loss_24(y_pred, 
            y_true, 
            vr_mean,
            vr_std,
            CRPS_min,
            RS_min,
            IC,
            weight_flag=True
            ):

    device = 'cuda:'+str(y_pred.get_device())

    # st()
    lon_now = y_true[:, 0, 2].cpu().detach().numpy()
    lon_5days = y_true[:, 0, 3].cpu().detach().numpy()
    # y_pred = (y_pred - vr_mean)/vr_std
    d = torch.abs(y_true[:, :, 1] - y_true[:, :, 0])
    d = d.to(device)
    N = d.shape[0]*d.shape[1]

    # sigma = torch.abs(y_pred).squeeze().to(device)
    sigma = torch.exp(y_pred*10).squeeze().to(device)
    # x = torch.zeros(sigma.shape)
    # print('max sigma = {}'.format(sigma.max()))
    # print('min sigma = {}'.format(sigma.min()))
    # st()
    loss = 0

    p = np.linspace(0, 360, 129)[:-1]
    lon_idx = np.zeros([lon_now.shape[0], 24])
    # st()
    for i in range(lon_now.shape[0]):
        if lon_now[i] > lon_5days[i]:
            idx_sta = np.where((p > lon_5days[i]))[0][0]
            # st()
            if idx_sta + 24 > 127:
                idx = np.arange(idx_sta-1, idx_sta+23)
            else:    
                idx = np.arange(idx_sta, idx_sta+24)
        else:
            idx_sta = np.where(p > lon_5days[i])[0]
            
            if len(idx_sta) >= 1:
                idx_sta = idx_sta[0]
                idx = np.hstack((np.arange(idx_sta, 128), 
                                    np.arange(24-128+idx_sta)))
            elif len(idx_sta) == 0:
                idx = np.arange(24)

        if len(idx) == 23:
            st()
        # st()
        lon_idx[i] = idx

    CRPS = torch.zeros([d.shape[0], 24]).to(device)
    # RS = torch.zeros([d.shape[0], 24]).to(device)
    d_save = torch.zeros([d.shape[0], 24]).to(device)
    x_t_clu = torch.zeros([d.shape[0], 24]).to(device)
        
    for j in range(d.shape[0]):
        d_save[j] = d[j, lon_idx[j]]
        x = d[j, lon_idx[j]]/np.sqrt(2)
        x = x.to(device)
        x_t = x/sigma[j]
        CRPS_1 = np.sqrt(2)*x_t*erf(x_t)
        CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x_t**2) 
        CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
        CRPS_3 = CRPS_3.to(device)
        # st()
        CRPS[j] = sigma[j]*(CRPS_1 + CRPS_2 - CRPS_3)
        x_t_clu[j] = x_t

    # for j in range(d.shape[1]):
    x_t1 = x_t_clu.reshape(-1, 1).squeeze()
    ind = torch.argsort(x_t1)
    ind_orig = torch.argsort(ind)+1
    ind_orig = ind_orig.to(device)
    # import ipdb;ipdb.set_trace()
    RS = torch.mean(N*(x_t1/N*(erf(x_t1)+1) - 
        x_t1*(2*ind_orig-1)/N**2 + 
        torch.exp(-x_t1**2)/np.sqrt(np.pi)/N)
    )

    RS = RS.to(device)

    # st()

    beta, CRPS_min, RS_min = est_beta_in(d_save.cpu())

    # print('beta: {}'.format(beta))
    # print('sigma max: {}'.format(sigma.max()))
    # print('error max: {}'.format(d_save.max()))

    # st()
    # weights = torch.exp((torch.diagonal(d[:, lon_idx[:, idx]])-vr_mean)/vr_std)
    # weights = weights.to(device)
    loss = torch.nanmean(torch.nanmean(CRPS))/CRPS_min + RS/RS_min
    # st()
    
    # if torch.isnan(loss.cpu()).sum() > 0:
    #     st()

    # print(loss)
    # st()
    return loss


def dV_loss_120_update(y_pred, 
            y_true, 
            vr_mean,
            vr_std,
            CRPS_min,
            RS_min,
            IC,
            weight_flag=True
            ):

    device = 'cuda:'+str(y_pred.get_device())

    # st()
    lon_now = y_true[:, 0, 2].cpu().detach().numpy()
    lon_5days = y_true[:, 0, 3].cpu().detach().numpy()
    # y_pred = (y_pred - vr_mean)/vr_std
    d = torch.abs(y_true[:, :, 1] - y_true[:, :, 0])
    d = d.to(device)
    N = d.shape[0]*d.shape[1]

    sigma = torch.exp(y_pred+1).squeeze().to(device)
    loss = 0

    CRPS = torch.zeros([d.shape[0], 120]).to(device)
    # RS = torch.zeros([d.shape[0], 24]).to(device)
    d_save = torch.zeros([d.shape[0], 120]).to(device)
    x_t_clu = torch.zeros([d.shape[0], 120]).to(device)
        
    for j in range(d.shape[0]):

        # d_save[j] = d[j, lon_idx[j]]
        # x = d[j, lon_idx[j]]/np.sqrt(2)
        d_save[j] = d[j, :120]
        x = d[j, :120]/np.sqrt(2)
        x = x.to(device)
        x_t = x/sigma[j]
        CRPS_1 = np.sqrt(2)*x_t*erf(x_t)
        CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x_t**2) 
        CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
        CRPS_3 = CRPS_3.to(device)
        # st()
        CRPS[j] = sigma[j]*(CRPS_1 + CRPS_2 - CRPS_3)
        x_t_clu[j] = x_t

    # for j in range(d.shape[1]):
    x_t1 = x_t_clu.reshape(-1, 1).squeeze()
    ind = torch.argsort(x_t1)
    ind_orig = torch.argsort(ind)+1
    ind_orig = ind_orig.to(device)
    # import ipdb;ipdb.set_trace()
    RS = torch.mean(N*(x_t1/N*(erf(x_t1)+1) - 
        x_t1*(2*ind_orig-1)/N**2 + 
        torch.exp(-x_t1**2)/np.sqrt(np.pi)/N)
    )

    RS = RS.to(device)

    beta, CRPS_min, RS_min = est_beta_in(d_save.cpu())
    loss = torch.nanmean(torch.nanmean(CRPS))/CRPS_min + RS/RS_min
    return loss


def dV_loss_24_update(y_pred, 
            y_true, 
            vr_mean,
            vr_std,
            CRPS_min,
            RS_min,
            IC,
            weight_flag=True
            ):

    device = 'cuda:'+str(y_pred.get_device())

    # st()
    lon_now = y_true[:, 0, 2].cpu().detach().numpy()
    lon_5days = y_true[:, 0, 3].cpu().detach().numpy()
    # y_pred = (y_pred - vr_mean)/vr_std
    d = torch.abs(y_true[:, :, 1] - y_true[:, :, 0])
    d = d.to(device)
    N = d.shape[0]*d.shape[1]

    sigma = torch.exp(y_pred*10).squeeze().to(device)
    loss = 0

    p = np.linspace(0, 360, 129)[:-1]
    lon_idx = np.zeros([lon_now.shape[0], 24])

    CRPS = torch.zeros([d.shape[0], 24]).to(device)
    # RS = torch.zeros([d.shape[0], 24]).to(device)
    d_save = torch.zeros([d.shape[0], 24]).to(device)
    x_t_clu = torch.zeros([d.shape[0], 24]).to(device)
        
    for j in range(d.shape[0]):

        # d_save[j] = d[j, lon_idx[j]]
        # x = d[j, lon_idx[j]]/np.sqrt(2)
        d_save[j] = d[j, :24]
        x = d[j, :24]/np.sqrt(2)
        x = x.to(device)
        x_t = x/sigma[j]
        CRPS_1 = np.sqrt(2)*x_t*erf(x_t)
        CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x_t**2) 
        CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
        CRPS_3 = CRPS_3.to(device)
        # st()
        CRPS[j] = sigma[j]*(CRPS_1 + CRPS_2 - CRPS_3)
        x_t_clu[j] = x_t

    # for j in range(d.shape[1]):
    x_t1 = x_t_clu.reshape(-1, 1).squeeze()
    ind = torch.argsort(x_t1)
    ind_orig = torch.argsort(ind)+1
    ind_orig = ind_orig.to(device)
    # import ipdb;ipdb.set_trace()
    RS = torch.mean(N*(x_t1/N*(erf(x_t1)+1) - 
        x_t1*(2*ind_orig-1)/N**2 + 
        torch.exp(-x_t1**2)/np.sqrt(np.pi)/N)
    )

    RS = RS.to(device)

    # st()

    beta, CRPS_min, RS_min = est_beta_in(d_save.cpu())

    # print('beta: {}'.format(beta))
    # print('sigma max: {}'.format(sigma.max()))
    # print('error max: {}'.format(d_save.max()))

    # st()
    # weights = torch.exp((torch.diagonal(d[:, lon_idx[:, idx]])-vr_mean)/vr_std)
    # weights = weights.to(device)
    loss = torch.nanmean(torch.nanmean(CRPS))/CRPS_min + RS/RS_min
    # st()
    
    # if torch.isnan(loss.cpu()).sum() > 0:
    #     st()

    # print(loss)
    # st()
    return loss


def dV_loss_NG(y_pred, 
            y_true, 
            vr_mean,
            vr_std,
            CRPS_min,
            RS_min,
            IC,
            weight_flag=True,
            mode='AL',
            ):

    device = 'cuda:'+str(y_pred.get_device())

    # st()
    lon_now = y_true[:, 0, 2].cpu().detach().numpy()
    lon_5days = y_true[:, 0, 3].cpu().detach().numpy()
    # y_pred = (y_pred - vr_mean)/vr_std
    d = y_true[:, :, 1] - y_true[:, :, 0]
    d = d.to(device)
    N = d.shape[0]

    out = torch.exp(y_pred).squeeze() # guarantee kappa >= 0
    k = out[:, :128]
    l = out[:, 128:]

    x = torch.zeros(d.shape)
    loss = 0

    CRPS = torch.zeros([x.shape[0], 24]).to(device)
    RS = torch.zeros([x.shape[0], 24]).to(device)

    p = np.linspace(0, 360, 129)[:-1]
    lon_idx = np.zeros([lon_now.shape[0], 24])
    # st()
    for i in range(lon_now.shape[0]):
        if lon_now[i] > lon_5days[i]:
            idx_sta = np.where((p > lon_5days[i]))[0][0]
            # st()
            if idx_sta + 24 > 127:
                idx = np.arange(idx_sta-1, idx_sta+23)
            else:    
                idx = np.arange(idx_sta, idx_sta+24)
        else:
            idx_sta = np.where(p > lon_5days[i])[0]
            
            if len(idx_sta) >= 1:
                idx_sta = idx_sta[0]
                idx = np.hstack((np.arange(idx_sta, 128), 
                                    np.arange(24-128+idx_sta)))
            elif len(idx_sta) == 0:
                idx = np.arange(24)

        if len(idx) == 23:
            st()
        # st()
        lon_idx[i] = idx

    for idx in range(24): ## 0 or 1

        st()
        mean_CRPS = asymmLaplace_accrue_torch.get_avg_CRPS_torch(k, d, lam=l)
        RS = asymmLaplace_accrue_torch.analytical_RS_torch(d, k, l)

        x_t = torch.diagonal(x[:, lon_idx[:, idx]])
        ind = torch.argsort(x_t)
        ind_orig = torch.argsort(ind)+1
        ind_orig = ind_orig.to(device)
        CRPS_1 = np.sqrt(2)*x_t*erf(x_t)
        CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x_t**2) 
        CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
        CRPS_3 = CRPS_3.to(device)
        # st()
        CRPS[:, idx] = torch.diagonal(sigma[:, lon_idx[:, idx]])*(CRPS_1 + CRPS_2 - CRPS_3)
        # st()
        if torch.isinf(CRPS[i, idx]):
            st() 

        # import ipdb;ipdb.set_trace()
        RS[:, idx] = N*(x_t/N*(erf(x_t)+1) - 
            x_t*(2*ind_orig-1)/N**2 + 
            torch.exp(-x_t**2)/np.sqrt(np.pi)/N)
    
        RS = RS.to(device)
        
        weights = torch.exp((torch.diagonal(d[:, lon_idx[:, idx]])-vr_mean)/vr_std)
        weights = weights.to(device)
        loss += torch.nanmean((CRPS[:, idx]/CRPS_min\
            +RS[:, idx]/RS_min
            )
            # *weights
            )
    # st()
    
    if torch.isnan(loss.cpu()).sum() > 0:
        st()

    # print(loss)
    # st()
    return loss/24


############## form ML-ready dataset ######################  

def GONG_read_all(ML_file, N_sample):

    X = batch_read(ML_file, 'X', N_sample)
    Y = batch_read(ML_file, 'Y', N_sample)
    
    return X, Y


def GONG_read(ML_file):

    with h5py.File(ML_file, 'r') as f:
        # st()
        # labels = np.array(f['label'])
        X = np.array(f['X'])
        Y = np.array(f['Y'])

        X = np.zeros([X.shape[0], 130, 181])
        # X = np.zeros([labels.shape[0], 128, 44, 18])
        Y = np.zeros([X.shape[0], 387])
        X_all = f['X']
        Y_all = f['Y']

        # X[icnt, j] = zoom(X_all[icnt, j, :, :], 1/1, order=3)

        for icnt in tqdm(range(X.shape[0])):
            # for j in range(X.shape[1]):
            #     X[icnt, j] = zoom(X_all[icnt, j, :, :], 1/1, order=3)
            X[icnt] = X_all[icnt]
            Y[icnt] = Y_all[icnt]
        f.close()
    
    return X, Y


class GONG_Model(lp.LightningModule):

    def __init__(self, 
                 lr,
                 weight_decay,
                 loss_func,
                #  test_func,
                 vr_mean,
                 vr_std,
                 dropout,
                 width,
                 mode,
                 IC,
                 optim,
                 weight_flag,
                 ratio
                 ):

        super().__init__()
        self.save_hyperparameters()
        
        # self.model = V_FNO_DDP(dropout,
        self.model = V_FNO_long(dropout,
                  width,
                  mode,
                  vr_mean,
                  vr_std,
                  hidden_size=16,
                  num_layers=2,
                  outputs=656,
                  )
        
        initialize_weights(self.model)

        self.lr = lr
        self.momentum = 0.3
        self.weight_decay = weight_decay
        self.loss_func = loss_func
        # self.test_test = test_func
        self.vr_mean = vr_mean
        self.vr_std = vr_std
        self.IC = IC
        self.optim = optim
        self.weight_flag=weight_flag
        self.ratio=ratio

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y, idx_sel = batch
        # print(idx_sel)
        y_hat = self.forward(x)
        # print('max/min of y_pred is {}/{}'.format(y_hat.max(), y_hat.min()))
        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std, 
                              self.IC,
                              self.ratio,
                            #   False,
                            #   True,
                              self.weight_flag
                              )

        self.log("train_loss", torch.sqrt(loss), 
                 sync_dist=True, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True)
        # self.log('train_loss', )

        return torch.sqrt(loss)

    # def training_epoch_end(self, outputs) -> None:
    #     gathered = self.all_gather(outputs)
    #     if self.global_rank == 0:
    #         # print(gathered)
    #         loss = sum(output['loss'].mean() for output in gathered) / len(outputs)
    #         print(loss.item())

    def validation_step(self, batch):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std,
                              self.IC,
                              self.ratio,
                            #   self.ratio,
                            #   False
                              self.weight_flag
                            #   True
                              )
        self.log('valid_loss', torch.sqrt(loss), sync_dist=True)
        return torch.sqrt(loss)

    def test_step(self, batch):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std,
                              self.IC,
                              self.ratio,
                            #   self.ratio,
                              self.weight_flag
                            #   weight_flag=True
                              )
        self.log('test_loss', torch.sqrt(loss), sync_dist=True)
        # self.log('test_loss', torch.sqrt(loss), sync_dist=True)
        return loss

    
    # def on_validation_epoch_end(self):
    #     ## F1 Macro all epoch saving outputs and target per batch
    #     val_all_outputs = self.val_step_outputs
    #     val_all_targets = self.val_step_targets
    #     val_loss_epoch = self.loss_func(val_all_outputs, val_all_targets,
    #                           self.vr_mean,
    #                           self.vr_std,
    #                           self.IC,
    #                           2,
    #                         #   self.ratio,
    #                           False
    #                         #   True
    #                           )
    #     self.log("val_f1_epoch", val_loss_epoch, on_step=False, on_epoch=True, prog_bar=True)

    #     # free up the memory
    #     # --> HERE STEP 3 <--
    #     self.val_step_outputs.clear()
    #     self.val_step_targets.clear()
    

    def configure_optimizers(self):

        if self.optim == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optim == 'lbfgs':
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
            
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=5, 
                                                    gamma=0.1)

        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler, 
                'monitor': 'train_loss'}

class dV_Model(lp.LightningModule):

    def __init__(self, 
                 lr,
                 weight_decay,
                 loss_func,
                #  test_func,
                 vr_mean,
                 vr_std,
                 CRPS_min,
                 RS_min,
                 dropout,
                 width,
                 mode,
                 IC,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.model = dV_FNO_long(dropout,
                  mode,
                  width,
                  vr_mean,
                  vr_std,
                #   hidden_size=16,
                #   num_layers=2,
                #   outputs=128,
                  outputs=120,
                  )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_func = loss_func
        # self.test_test = test_func
        self.vr_mean = vr_mean
        self.vr_std = vr_std
        self.CRPS_min = CRPS_min
        self.RS_min = RS_min
        self.IC = IC

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std,
                              self.CRPS_min,
                              self.RS_min, 
                              self.IC,
                              True)

        self.log("train_loss", torch.sqrt(loss), 
                 sync_dist=True, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True)
        # self.log('train_loss', )
        return torch.sqrt(loss)

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std,
                              self.CRPS_min,
                              self.RS_min,
                              self.IC,
                              True)
        self.log('valid_loss', torch.sqrt(loss), sync_dist=True)
        return torch.sqrt(loss)

    def test_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std,
                              self.CRPS_min,
                              self.RS_min,
                              self.IC,
                              weight_flag=False)
        self.log('test_loss', torch.sqrt(loss), sync_dist=True)
        # self.log('test_loss', torch.sqrt(loss), sync_dist=True)
        return loss

    def configure_optimizers(self):
        
        # optimizer = torch.optim.SGD(self.model.parameters(), 
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                lr=self.lr,
                                weight_decay=self.weight_decay
                                )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=10, 
                                                    gamma=0.1)

        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler, 
                'monitor': 'train_loss'}


def V_train(X, Y,
            idx_train, idx_valid, idx_test,
            vr_mean, vr_std,
            iter,
            config, 
            checkpoint_dir, checkpoint_file,
            initfile,
            flag=False
            ):

    V_checkpoint_name = checkpoint_dir+'/'+checkpoint_file+'_'+str(iter)+'_V.ckpt'
    V_checkpoint_init = checkpoint_dir+'/'+initfile+'_'+str(iter)+'_V.ckpt'
    if os.path.exists(V_checkpoint_name) & flag:
        # if os.path.exists(checkpoint_name) & train_flag & Overwrite:
            os.remove(V_checkpoint_name)
            pass
    
    # st()

    X, Y = torch.fGONG_Model.hstack((idx_train, idx_valid))
    idx_valid = np.hstack((idx_train, idx_valid))
    train_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_train], Y[idx_train]), 
                        batch_size=config['batch'], 
                        num_workers=8)
    val_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_valid], Y[idx_valid]), 
                        batch_size=config['batch'], 
                        num_workers=8)
    test_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X, Y), 
                        # GONG_Dataset(X[idx_test], Y[idx_test]), 
                        batch_size=X.shape[0]//100, 
                        num_workers=8)

    logger = WandbLogger(project="GONG_lightning")
    # print("Defining model")

    model = GONG_Model(config['lr'],
                       config['weight_decay'],
                       V_loss,
                    #    GONG_test,
                       vr_mean,
                       vr_std,
                       config['dropout'],
                       config['width'],
                       config['mode'],
                       config['IC']
                       )

    bar = ProgressBar_tqdm()
    # bar = LitProgressBar()
    early_stopping_callback = lp.callbacks.EarlyStopping('train_loss', patience=10)
    accumulator = lp.callbacks.GradientAccumulationScheduler(scheduling={4: 2})

    checkpoint_callback = lp.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_file+"_"+str(iter)+"_V",
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        mode="min")
 
    trainer = lp.Trainer(
        # accelerator="auto",
        # accelerator='cpu', devices=1,
        accelerator='gpu', devices=config['dev_num'],
        default_root_dir=os.getcwd(),
        strategy='ddp_find_unused_parameters_true',
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        max_epochs=config['max_epochs'],
        # no_validation=True,
        callbacks=[bar,
                   checkpoint_callback, 
                #    early_stopping_callback,
                   accumulator
                   ],
        # callbacks=[early_stopping_callback],
        # weights_summary=None,
        logger=logger,
    )
    # import ipdb;ipdb.set_trace()
    if flag:
        if os.path.exists(V_checkpoint_init):
            # model = GONG_Model.load_from_checkpoint(V_checkpoint_init)
            pass
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
        # st()
        # trainer.save_checkpoint(V_checkpoint_name)
        # trainer.test(model, dataloaders=test_data)
    
    # import ipdb; ipdb.set_trace()

    best_model = GONG_Model.load_from_checkpoint(V_checkpoint_name).to('cpu') #.to('cuda')
    pred_Y_test = best_model.forward(X)
    # st()

    # return (pred_Y_test.detach().numpy())*200
    # return (pred_Y_test.detach().numpy())*200
    return (pred_Y_test.detach().numpy())*5
    # return (pred_Y_test.detach().numpy())*vr_mean


def V_train_all(X, Y,
            idx_train, idx_valid, idx_test,
            vr_mean, vr_std,
            iter,
            config, 
            checkpoint_dir, checkpoint_file,
            initfile,
            # weight_flag=False,
            flag=False
            ):

    V_checkpoint_name = checkpoint_dir+'/'+checkpoint_file+'_'+str(iter)+'_V.ckpt'
    V_checkpoint_init = initfile
    # V_checkpoint_init = checkpoint_dir+'/'+initfile+'_'+str(iter)+'_V.ckpt'
    if os.path.exists(V_checkpoint_name) & flag:
        # if os.path.exists(checkpoint_name) & train_flag & Overwrite:
            os.remove(V_checkpoint_name)
            pass
    
    # st()

    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    # print("Getting data loaders")

    # st()
    # idx_train = np.hstack((idx_train, idx_valid, idx_test))
    # idx_train = np.hstack((idx_train, idx_valid))
    # idx_valid = np.hstack((idx_train, idx_valid))
    train_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_train], Y[idx_train]), 
                        batch_size=config['batch'], 
                        num_workers=8)
    val_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_valid], Y[idx_valid]), 
                        batch_size=config['batch'], 
                        num_workers=8)
    test_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X, Y), 
                        # GONG_Dataset(X[idx_test], Y[idx_test]), 
                        batch_size=X.shape[0]//10, 
                        num_workers=8)

    logger = WandbLogger(project="GONG_lightning")
    # print("Defining model")
    # st()
    model = GONG_Model(config['lr'],
                       config['weight_decay'],
                    #    V_loss_update,
                       V_loss_all,
                    #    GONG_test,
                       vr_mean,
                       vr_std,
                       config['dropout'],
                       config['width'],
                       config['mode'],
                       config['IC'],
                    #    config['weight_flag'],
                    #    config['ratio'],
                       )

    bar = ProgressBar_tqdm()
    # bar = LitProgressBar()
    early_stopping_callback = lp.callbacks.EarlyStopping('valid_loss', patience=5)
    # accumulator = lp.callbacks.GradientAccumulationScheduler(scheduling={4: 2})

    checkpoint_callback = lp.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_file+"_"+str(iter)+"_V",
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        # monitor="train_loss",
        mode="min")
 
    trainer = lp.Trainer(
        # accelerator="auto",
        # accelerator='cpu', devices=1,
        # auto_lr_find='my_value',
        accelerator='gpu', devices=config['dev_num'],
        default_root_dir=os.getcwd(),
        strategy='ddp_find_unused_parameters_true',
        num_sanity_val_steps=1,        
        log_every_n_steps=5,
        max_epochs=config['max_epochs'],
        inference_mode=False,
        # no_validation=True,
        callbacks=[bar,
                   checkpoint_callback, 
                   early_stopping_callback,
                #    accumulator
                   ],
        # callbacks=[early_stopping_callback],
        # weights_summary=None,
        logger=logger,
    )
    # import ipdb;ipdb.set_trace()
    if flag:
        if os.path.exists(V_checkpoint_init):
            # model = GONG_Model.load_from_checkpoint(V_checkpoint_init)
            # model = GONG_Model.load_from_checkpoint(V_checkpoint_init)
            pass
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
        # st()
        # trainer.save_checkpoint(V_checkpoint_name)
        # trainer.test(model, dataloaders=test_data)
    
    # import ipdb; ipdb.set_trace()

    # best_model = GONG_Model.load_from_checkpoint(V_checkpoint_init).to('cpu') #.to('cuda')
    best_model = GONG_Model.load_from_checkpoint(V_checkpoint_name).to('cpu') #.to('cuda')
    pred_Y_test = best_model.forward(X)
    # pred_Y_test = (pred_Y_test - vr_mean)/vr_std
    # print(pred_Y_test[:, 128:].detach().numpy()*vr_std+Y[:, :, 0].numpy()*1)
    # print(pred_Y_test[:, :128].detach().numpy()*10+Y[:, :, 0].numpy())
    # print(pred_Y_test[:, :128].detach().numpy()*pred_Y_test[:, 128:].detach().numpy()+Y[:, :, 0].numpy())
    # print(pred_Y_test[:, :128].detach().numpy()*(pred_Y_test[:, 128:].detach().numpy()+1)*100)
    # return (pred_Y_test.detach().numpy())*200
    # return (pred_Y_test.detach().numpy())*200
    # return (pred_Y_test.detach().numpy())-vr_mean
    # return pred_Y_test.detach().numpy()
    # return pred_Y_test.detach().numpy()*vr_mean
    # return pred_Y_test.detach().numpy()*50+Y[:, :, 0].numpy()
    
    # return np.tanh((pred_Y_test.detach().numpy() - vr_mean)/300)*250+550
    return pred_Y_test.detach().numpy()*vr_std*2+Y[:, :, 0].numpy()
    # return pred_Y_test.detach().numpy()*vr_std*config['ratio']+Y[:, :, 0].numpy()
    # return pred_Y_test[:, :128].detach().numpy()*pred_Y_test[:, 128:].detach().numpy()+Y[:, :, 0].numpy()
    # return pred_Y_test[:, 128:].detach().numpy()*vr_std+Y[:, :, 0].numpy()*1
    # return pred_Y_test[:, :128].detach().numpy()*vr_mean+Y[:, :, 0]


def V_train_update(X, Y,
            idx_train, idx_valid, idx_test,
            vr_mean, vr_std,
            iter,
            config, 
            checkpoint_dir, checkpoint_file,
            initfile,
            # weight_flag=False,
            flag=False
            ):

    V_checkpoint_name = checkpoint_dir+'/'+checkpoint_file+'_'+str(iter)+'_V.ckpt'
    V_checkpoint_init = initfile
    # V_checkpoint_init = checkpoint_dir+'/'+initfile+'_'+str(iter)+'_V.ckpt'
    if os.path.exists(V_checkpoint_name) & flag:
        # if os.path.exists(checkpoint_name) & train_flag & Overwrite:
            os.remove(V_checkpoint_name)
            pass
    
    # st()

    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    # print("Getting data loaders")

    # st()
    idx_train = np.hstack((idx_train, idx_valid, idx_test))
    # idx_test = np.hstack((idx_train, idx_valid, idx_test))
    # idx_train = np.hstack((idx_train, idx_test))
    idx_valid = idx_test
        
    train_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_train], Y[idx_train]), 
                        # batch_size=len(idx_train), 
                        batch_size=config['batch'], 
                        num_workers=96)
    val_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_valid], Y[idx_valid]), 
                        batch_size=config['batch'], 
                        # batch_size=len(idx_valid), 
                        num_workers=96)
    test_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X, Y), 
                        # GONG_Dataset(X[idx_test], Y[idx_test]), 
                        batch_size=config['batch'], 
                        num_workers=96)

    logger = WandbLogger(project="GONG_lightning")
    # print("Defining model")
    # st()
    model = GONG_Model(config['lr'],
                       config['weight_decay'],
                       V_loss_long,
                    #    V_loss_update,
                    #    V_loss_all,
                    #    GONG_test,
                       vr_mean,
                       vr_std,
                       config['dropout'],
                       config['width'],
                       config['mode'],
                       config['IC'],
                       config['Optimize'],
                    #    config['weight_flag'],
                    #    config['ratio'],
                       )

    bar = ProgressBar_tqdm()
    # bar = LitProgressBar()
    early_stopping_callback = lp.callbacks.EarlyStopping('valid_loss', patience=5)
    # accumulator = lp.callbacks.GradientAccumulationScheduler(scheduling={4: 2})

    checkpoint_callback = lp.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_file+"_"+str(iter)+"_V",
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        # monitor="train_loss",
        mode="min")
 

    trainer = lp.Trainer(
        # accelerator="auto",
        # accelerator='cpu', devices=1,
        # auto_lr_find='my_value',
        accelerator='gpu', devices=config['dev_num'],
        default_root_dir=os.getcwd(),
        strategy='ddp_find_unused_parameters_true',
        num_sanity_val_steps=1,        
        log_every_n_steps=5,
        max_epochs=10,
        # max_epochs=config['max_epochs'],
        inference_mode=False,
        # no_validation=True,
        callbacks=[bar,
                   checkpoint_callback, 
                   early_stopping_callback,
                #    accumulator
                   ],
        # callbacks=[early_stopping_callback],
        # weights_summary=None,
        logger=logger,
        precision=16,
    )
    # import ipdb;ipdb.set_trace()
    if flag:
        if os.path.exists(V_checkpoint_init):
            model = GONG_Model.load_from_checkpoint(V_checkpoint_init)
            # model = GONG_Model.load_from_checkpoint(V_checkpoint_init)
            pass
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
        # st()
        # trainer.save_checkpoint(V_checkpoint_name)
        # trainer.test(model, dataloaders=test_data)
    
    # import ipdb; ipdb.set_trace()
    best_model = GONG_Model.load_from_checkpoint(V_checkpoint_name).to('cuda')
    # best_model = GONG_Model.load_from_checkpoint(V_checkpoint_init).to('cuda')
    best_model.eval()
    pred_Y_test = []

    with torch.no_grad():
        for batch in tqdm(test_data):
            # st()
            X_batch = batch[0].to('cuda')  # Assuming your model is on the 'device' (e.g., 'cuda' or 'cpu')
            preds = best_model(X_batch).cpu()
            pred_Y_test.append(preds)
            # pred_Y_test.append(preds.unsqueeze(axis=0))
    # st()
    pred_Y_test = torch.cat(pred_Y_test)

    # print('first 10 of pred_out: {}'.format(pred_Y_test[10, :10]))
    # print('first 10 of v0_out: {}'.format(Y[10, :10, 0]))
    # best_model = GONG_Model.load_from_checkpoint(V_checkpoint_init).to('cpu') #.to('cuda')
    # best_model = GONG_Model.load_from_checkpoint(V_checkpoint_name).to('cpu') #.to('cuda')
    # pred_Y_test = best_model.forward(X)

    # pred_Y_test = (pred_Y_test - vr_mean)/vr_std
    # print(pred_Y_test[:, 128:].detach().numpy()*vr_std+Y[:, :, 0].numpy()*1)
    # print(pred_Y_test[:, :128].detach().numpy()*10+Y[:, :, 0].numpy())
    # print(pred_Y_test[:, :128].detach().numpy()*pred_Y_test[:, 128:].detach().numpy()+Y[:, :, 0].numpy())
    # print(pred_Y_test[:, :128].detach().numpy()*(pred_Y_test[:, 128:].detach().numpy()+1)*100)
    # return (pred_Y_test.detach().numpy())*200
    # return (pred_Y_test.detach().numpy())*200
    # return (pred_Y_test.detach().numpy())-vr_mean
    # return pred_Y_test.detach().numpy()
    # return pred_Y_test.detach().numpy()*vr_mean
    # return pred_Y_test.detach().numpy()*50+Y[:, :, 0].numpy()
    # st()
    # return np.tanh((pred_Y_test.detach().numpy() - vr_mean)/300)*250+550
    # return np.exp(pred_Y_test.detach().numpy())*Y[:, :24, 99].numpy()
    # return pred_Y_test.detach().numpy()+Y[:, :120, 99].numpy()
    return pred_Y_test.detach().numpy()*1.3+Y[:, :, 0].numpy()
    # return np.exp(pred_Y_test[:, :120].detach().numpy())*Y[:, :120, 99].numpy()
    # return pred_Y_test.detach().numpy()*vr_std*2+Y[:, :, 0].numpy()
    # return pred_Y_test.detach().numpy()*vr_std*config['ratio']+Y[:, :, 0].numpy()
    # return pred_Y_test[:, :128].detach().numpy()*pred_Y_test[:, 128:].detach().numpy()+Y[:, :, 0].numpy()
    # return pred_Y_test[:, 128:].detach().numpy()*vr_std+Y[:, :, 0].numpy()*1
    # return pred_Y_test[:, :128].detach().numpy()*vr_mean+Y[:, :, 0]


def V_train_filebatch(filename,
            idx_train, idx_valid, idx_test,
            vr_mean, vr_std,
            iter,
            config, 
            checkpoint_dir, checkpoint_file,
            initfile,
            # weight_flag=False,
            flag=False
            ):

    V_checkpoint_name = checkpoint_dir+'/'+checkpoint_file+'_'+str(iter)+'_V.ckpt'
    V_checkpoint_init = initfile
    # V_checkpoint_init = checkpoint_dir+'/'+initfile+'_'+str(iter)+'_V.ckpt'
    if (os.path.exists(V_checkpoint_name) & flag):
        # if os.path.exists(checkpoint_name) & train_flag & Overwrite:
        os.remove(V_checkpoint_name)
        pass
    
    # idx_train_all = np.hstack((idx_train[::10], idx_valid, idx_test))
    # idx_train_all = np.hstack((idx_train[:10], idx_valid, idx_test))
    idx_train_all = np.hstack((idx_train, idx_valid, idx_test))
    idx_all = np.hstack((idx_train, idx_valid, idx_test))
    # idx_valid = np.hstack((idx_train, idx_valid, idx_test))
    # idx_test = np.hstack((idx_train, idx_valid, idx_test))
    # idx_train = np.hstack((idx_train, idx_test))
    # idx_valid = idx_test
    
    # N_sample = idx_train.shape[0]
    # np.random.shuffle(idx_train)
    
    # dataset = netcdf_Dataset(filename, idx_train)

    # for i in range(len(dataset)):
    #     x_sample, y_sample, idx_sel = dataset[i]
    #     print(idx_sel)  # Check if idx_sel is in the correct order

    # st()
        
    train_data = torch.utils.data.DataLoader(
                        netcdf_Dataset(filename, idx_train_all), 
                        # batch_size=len(idx_train), 
                        batch_size=config['batch'], 
                        shuffle=True,  # Ensure no shuffling of data
                        num_workers=11)
    val_data = torch.utils.data.DataLoader(
                        netcdf_Dataset(filename, idx_train_all), 
                        batch_size=config['batch'], 
                        shuffle=False,  # Ensure no shuffling of data
                        # batch_size=len(idx_valid), 
                        num_workers=11)
    test_data = torch.utils.data.DataLoader(
                        netcdf_Dataset(filename, idx_all), 
                        # GONG_Dataset(X[idx_test], Y[idx_test]), 
                        batch_size=config['batch'], 
                        shuffle=False,  # Ensure no shuffling of data
                        # batch_size=idx_train.shape[0], 
                        num_workers=11)

    logger = WandbLogger(project="GONG_lightning")
    # print("Defining model")
    # st()
    model = GONG_Model(config['lr'],
                       config['weight_decay'],
                       V_loss_long,
                    #    V_loss_update,
                    #    V_loss_all,
                    #    GONG_test,
                       vr_mean,
                       vr_std,
                       config['dropout'],
                       config['width'],
                       config['mode'],
                       config['IC'],
                       config['Optimize'],
                       config['weight_flag'],
                       config['ratio'],
                       )

    bar = ProgressBar_tqdm()
    # bar = LitProgressBar()
    early_stopping_callback = lp.callbacks.EarlyStopping('valid_loss', patience=15)
    # accumulator = lp.callbacks.GradientAccumulationScheduler(scheduling={4: 2})

    checkpoint_callback = lp.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_file+"_"+str(iter)+"_V",
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        # monitor="train_loss",
        mode="min")

    trainer = lp.Trainer(
        # accelerator="auto",
        # accelerator='gpu', devices=1,
        # auto_lr_find='my_value',
        accelerator='cuda', devices=config['dev_num'],
        default_root_dir=os.getcwd(),
        # strategy='auto',
        strategy='ddp_find_unused_parameters_true',
        num_sanity_val_steps=1,        
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        # max_epochs=10,
        # max_epochs=2,
        max_epochs=config['max_epochs'],
        inference_mode=False,
        # no_validation=True,
        callbacks=[bar,
                   checkpoint_callback, 
                   early_stopping_callback,
                #    accumulator
                   ],
        # callbacks=[early_stopping_callback],
        # weights_summary=None,
        logger=logger,
        # shuffle=True,   # Enable shuffling
        # precision='16-mixed',
        precision=32
    )
    # import ipdb;ipdb.set_trace()
    if flag:
        if os.path.exists(V_checkpoint_init):
            # model = GONG_Model.load_from_checkpoint(V_checkpoint_init)
            # model = GONG_Model.load_from_checkpoint(V_checkpoint_init)
            pass
        # st()
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
        # st()
        # trainer.save_checkpoint(V_checkpoint_name)
        # trainer.test(model, dataloaders=test_data)
    
    # import ipdb; ipdb.set_trace()
    best_model = GONG_Model.load_from_checkpoint(V_checkpoint_name).to('cuda')
    # best_model = GONG_Model.load_from_checkpoint(V_checkpoint_init).to('cuda')
    best_model.eval()
    pred_Y_test = []
    Y_clu = []

    with torch.no_grad():
        for batch in tqdm(test_data):
            # st()
            X_batch = batch[0].to('cuda')  # Assuming your model is on the 'device' (e.g., 'cuda' or 'cpu')
            preds = best_model(X_batch).cpu()
            # pred_Y_test.append(preds*vr_std*config['ratio']+batch[1][:, :, 0])
            pred_Y_test.append(np.exp(preds)*batch[1][:, :, 0])
            # pred_Y_test.append(preds*batch[1][:, :, 0])
            Y_clu.append(batch[1])
            
            # pred_Y_test.append(preds.unsqueeze(axis=0))
    # st()
    pred_Y_test = torch.cat(pred_Y_test)
    Y_clu = torch.cat(Y_clu)
    return pred_Y_test.detach().numpy(), Y_clu.detach().numpy()


def V_train_pred(X, Y, iter, checkpoint_dir, checkpoint_file):

    V_checkpoint_name = checkpoint_dir+'/'+checkpoint_file+'_'+str(iter)+'_V.ckpt'    
    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    best_model = GONG_Model.load_from_checkpoint(V_checkpoint_name).to('cpu') #.to('cuda')
    pred_Y_test = best_model.forward(X)
    return np.exp(pred_Y_test.detach().numpy())*Y[:, :, 0].numpy()
    

def dV_train(X, Y, y_pred,
            idx_train, idx_valid, idx_test,
            vr_mean, vr_std,
            iter,
            config, 
            checkpoint_dir, 
            checkpoint_file, 
            initfile,
            flag=False
            ):

    dV_checkpoint_name = checkpoint_dir+'/'+checkpoint_file+'_'+str(iter)+'_dV.ckpt'
    dV_checkpoint_init = checkpoint_dir+'/'+initfile+'_'+str(iter)+'_dV.ckpt'
    if os.path.exists(dV_checkpoint_name) & flag:
        # if os.path.exists(checkpoint_name) & train_flag & Overwrite:
            os.remove(dV_checkpoint_name)
            pass

    
    # st()
    HC_lon_now = Y[:, 0, 101]
    HC_lon_5days = Y[:, 0, 102]
    y0_5days = Y[:, :, :100]
    yr_5days = Y[:, :, 103:]
    # st()
    beta, CRPS_min, RS_min = est_beta(X, y_pred, yr_5days)

    Y = np.stack([y_pred.squeeze(), 
                  yr_5days[:, :, -1].squeeze(),
                  np.tile(HC_lon_now, (128, 1)).T,
                  np.tile(HC_lon_5days, (128, 1)).T,
                  ], 
                  axis=-1)

    # st()

    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    # print("Getting data loaders")

    # idx_train = np.hstack((idx_train, idx_valid, idx_test))
    # idx_valid = np.hstack((idx_train, idx_valid))
    train_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_train], Y[idx_train]), 
                        batch_size=config['batch']*10, 
                        num_workers=8)
    val_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_valid], Y[idx_valid]), 
                        batch_size=config['batch']*10, 
                        num_workers=8)
    test_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X, Y), 
                        # GONG_Dataset(X[idx_test], Y[idx_test]), 
                        batch_size=X.shape[0]*10, 
                        num_workers=8)

    logger = WandbLogger(project="GONG_lightning")
    # print("Defining model")
    
    # st()
    # print('max CRPS: {}'.format(CRPS_min))
    # print('max RS: {}'.format(RS_min))

    model = dV_Model(config['lr'],
                       config['weight_decay'],
                    #    dV_loss,
                       dV_loss_24,
                    #    dV_loss_NG,
                    #    GONG_test,
                       vr_mean,
                       vr_std,
                       CRPS_min,
                       RS_min,
                       config['dropout'],
                       config['width'],
                       config['mode'],
                       config['IC'])

    bar = ProgressBar_tqdm()
    # bar = LitProgressBar()
    early_stopping_callback = lp.callbacks.EarlyStopping('valid_loss', patience=10)
    accumulator = lp.callbacks.GradientAccumulationScheduler(scheduling={4: 2})

    checkpoint_callback = lp.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_file+"_"+str(iter)+"_dV",
        # filename=checkpoint_file,
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        mode="min")
    # lp.callbacks.ModelCheckpoint.CHECKPOINT_NAME_LAST=dV_checkpoint_name

    trainer = lp.Trainer(
        # accelerator="auto",
        # accelerator='cpu', devices=1,
        accelerator='gpu', 
        # devices=1,
        devices=config['dev_num'],
        default_root_dir=os.getcwd(),
        strategy='ddp_find_unused_parameters_true',
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        max_epochs=config['max_epochs'],
        # no_validation=True,
        callbacks=[bar,
                   checkpoint_callback, 
                   early_stopping_callback,
                   accumulator],
        # callbacks=[early_stopping_callback],
        # weights_summary=None,
        logger=logger,
    )
    # import ipdb;ipdb.set_trace()

    if flag:
        if os.path.exists(dV_checkpoint_init):
            # model = GONG_Model.load_from_checkpoint(dV_checkpoint_init)
            pass
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
        # st()
        # trainer.save_checkpoint(dV_checkpoint_name)
        # trainer.test(model, dataloaders=test_data)
    
    # st()
    # import ipdb; ipdb.set_trace()
    # best_model = model.load_from_checkpoint(dV_checkpoint_name).to('cpu') #.to('cuda')
    best_model = dV_Model.load_from_checkpoint(dV_checkpoint_name).to('cpu') #.to('cuda')
    pred_Y_test = best_model.forward(X)
    # aa = np.exp(pred_Y_test.detach().numpy()*30)
    # print('aa[296]: {}'.format(aa[296]))
    # print('X[296]: {}'.format(X[296, 0]))
    # st()

    # return np.exp(pred_Y_test.detach().numpy()*30)
    return np.exp(pred_Y_test.detach().numpy()*10)
    # return np.abs(pred_Y_test.detach().numpy()*vr_std)
    # return np.exp((pred_Y_test.detach().numpy()))
    # return np.exp((pred_Y_test.detach().numpy() - vr_mean)/vr_std)


def dV_train_update(X, Y, y_pred,
            idx_train, idx_valid, idx_test,
            vr_mean, vr_std,
            iter,
            config, 
            checkpoint_dir, 
            checkpoint_file, 
            initfile,
            flag=False
            ):

    dV_checkpoint_name = checkpoint_dir+'/'+checkpoint_file+'_'+str(iter)+'_dV.ckpt'
    dV_checkpoint_init = checkpoint_dir+'/'+initfile+'_'+str(iter)+'_dV.ckpt'
    if os.path.exists(dV_checkpoint_name) & flag:
        # if os.path.exists(checkpoint_name) & train_flag & Overwrite:
            os.remove(dV_checkpoint_name)
            pass

    
    # st()
    HC_lon_now = Y[:, 0, 101]
    HC_lon_5days = Y[:, 0, 102]
    y0_5days = Y[:, :, :100]
    yr_5days = Y[:, :, 103:]
    # st()
    beta, CRPS_min, RS_min = est_beta(X, y_pred, yr_5days)

    Y = np.stack([y_pred.squeeze(), 
                  yr_5days[:, :, -1].squeeze(),
                  np.tile(HC_lon_now, (656, 1)).T,
                  np.tile(HC_lon_5days, (656, 1)).T,
                  ], 
                  axis=-1)

    # st()

    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    # print("Getting data loaders")

    # idx_train = np.hstack((idx_train, idx_valid, idx_test))
    # idx_valid = np.hstack((idx_train, idx_valid))
    train_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_train], Y[idx_train]), 
                        batch_size=config['batch']*10, 
                        num_workers=8)
    val_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X[idx_valid], Y[idx_valid]), 
                        batch_size=config['batch']*10, 
                        num_workers=8)
    test_data = torch.utils.data.DataLoader(
                        GONG_Dataset(X, Y), 
                        # GONG_Dataset(X[idx_test], Y[idx_test]), 
                        batch_size=X.shape[0]//100, 
                        num_workers=8)

    logger = WandbLogger(project="GONG_lightning")
    # print("Defining model")
    
    # st()
    # print('max CRPS: {}'.format(CRPS_min))
    # print('max RS: {}'.format(RS_min))

    model = dV_Model(config['lr'],
                       config['weight_decay'],
                    #    dV_loss,
                       dV_loss_120_update,
                    #    dV_loss_NG,
                    #    GONG_test,
                       vr_mean,
                       vr_std,
                       CRPS_min,
                       RS_min,
                       config['dropout'],
                       config['width'],
                       config['mode'],
                       config['IC'])

    bar = ProgressBar_tqdm()
    # bar = LitProgressBar()
    early_stopping_callback = lp.callbacks.EarlyStopping('valid_loss', patience=10)
    accumulator = lp.callbacks.GradientAccumulationScheduler(scheduling={4: 2})

    checkpoint_callback = lp.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_file+"_"+str(iter)+"_dV",
        # filename=checkpoint_file,
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        mode="min")
    # lp.callbacks.ModelCheckpoint.CHECKPOINT_NAME_LAST=dV_checkpoint_name

    trainer = lp.Trainer(
        # accelerator="auto",
        # accelerator='cpu', devices=1,
        accelerator='gpu', 
        # devices=1,
        devices=config['dev_num'],
        default_root_dir=os.getcwd(),
        strategy='ddp',
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        max_epochs=config['max_epochs'],
        # no_validation=True,
        callbacks=[bar,
                   checkpoint_callback, 
                   early_stopping_callback,
                   accumulator],
        # callbacks=[early_stopping_callback],
        # weights_summary=None,
        logger=logger,
    )
    # import ipdb;ipdb.set_trace()

    if flag:
        if os.path.exists(dV_checkpoint_init):
            # model = GONG_Model.load_from_checkpoint(dV_checkpoint_init)
            pass
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
        # st()
        # trainer.save_checkpoint(dV_checkpoint_name)
        # trainer.test(model, dataloaders=test_data)
    
    # st()
    # import ipdb; ipdb.set_trace()
    # best_model = model.load_from_checkpoint(dV_checkpoint_name).to('cpu') #.to('cuda')
    best_model = dV_Model.load_from_checkpoint(dV_checkpoint_name).to('cpu') #.to('cuda')
    pred_Y_test = best_model.forward(X)
    # aa = np.exp(pred_Y_test.detach().numpy()*30)
    # print('aa[296]: {}'.format(aa[296]))
    # print('X[296]: {}'.format(X[296, 0]))
    # st()

    # return np.exp(pred_Y_test.detach().numpy()*30)
    return np.exp(pred_Y_test.detach().numpy()+1)
    # return np.abs(pred_Y_test.detach().numpy()*vr_std)
    # return np.exp((pred_Y_test.detach().numpy()))
    # return np.exp((pred_Y_test.detach().numpy() - vr_mean)/vr_std)

def v02vr_5days(pred_Y_test, 
                r_end, 
                IC):
        
    p = np.linspace(0, 360, 657)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    v0_5days_test = np.zeros([pred_Y_test.shape[0], 656])
    vr_5days_test = np.zeros([pred_Y_test.shape[0], 656])
    for i in tqdm(range(r_end.shape[0])):  
        r_vector=np.linspace(695700*IC,r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:] - r_vector[0:-1]
        v0_5days_test[i] = pred_Y_test[i].squeeze()

        vr_5days_test[i] = apply_rk42log_model(
            v0_5days_test[i], 
            dr_vec, dp_vec, 
            mode='f')[-1]
    
    return vr_5days_test


def plot_vr5days(date_ACE, 
                 idx_clu, 
                 i,
                 IC,
                 r_end_5day,
                 ACE_data,
                 vr_5days_test_pred,
                 std_vr_5days_test_pred,
                 vr,
                 vr_5days,
                 HC_lon_now,
                 HC_lon_5days
                 ):

    date = date_ACE[idx_clu][i].astype(int)
    date = dt.datetime(date[0], date[1], date[2], date[3], date[4])
    print('by {}, {}th max predictions are {}'.format(date, i, np.abs(vr_5days_test_pred[i]).max()))

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # st()
    ax.plot(ACE_data['long_grid'], vr_5days_test_pred[i], 'r-', label='vr_5days_pred')
    ax.plot(ACE_data['long_grid'], vr[:, i], 'g-', label='vr')
    ax.plot(ACE_data['long_grid'],
            vr_5days[:, i], 
            'k-', label='vr_5days')
    ax.plot(np.tile(HC_lon_now[i], (30)), 
            np.linspace(np.min(vr[:, i]), 
                        np.max(vr[:, i]), 30), '--',
            label='long_now')
    ax.plot(np.tile(HC_lon_5days[i], (30)), 
            np.linspace(np.min(vr[:, i]), 
                        np.max(vr[:, i]), 30), '--',
            label='long_5days')
    # ax.fill_between(ACE_data['long_grid'].squeeze(),
    #                 vr_5days_test_pred[i]-std_vr_5days_test_pred[i],
    #                 vr_5days_test_pred[i]+std_vr_5days_test_pred[i], 
    #                 interpolate=True, alpha=.5,
    #                 label='Uncertainty')
    ax.legend()
    ax.set_ylabel('$V_{r} 5days$')
    ax.set_title(date)
    
    fig.savefig('Figs/wandb/Vr_example_'+str(IC)+'_'+str(i)+'.jpg', dpi=300)
    plt.close()


def plot_vr5days_all(date_ACE, 
                 idx_clu, 
                 i,
                 IC,
                 r_end_5day,
                 ACE_data,
                 vr_5days_test_pred,
                 std_vr_5days_test_pred,
                 vr,
                 vr_5days,
                 HC_lon_now,
                 HC_lon_5days
                 ):

    date = date_ACE[idx_clu][i].astype(int)
    date = dt.datetime(date[0], date[1], date[2], date[3], date[4])
    p = np.arange(0, 360, 128)
    print('by {}, {}th max predictions are {}'.format(date, i, np.abs(vr_5days_test_pred[i]).max()))

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    # st()
    ax.plot(ACE_data['long_grid'], vr_5days_test_pred[i], 'r-', label='vr_5days_pred')
    ax.plot(ACE_data['long_grid'], vr[i, :, -1], 'g-', label='vr')
    ax.plot(ACE_data['long_grid'],
            vr_5days[i, :, -1], 
            'k-', label='vr_5days')
    ax.plot(np.tile(HC_lon_now[i], (30)), 
            np.linspace(np.min(vr[i, :, -1]), 
                        np.max(vr[i, :, -1]), 30), '--',
            label='long_now')
    ax.plot(np.tile(HC_lon_5days[i], (30)), 
            np.linspace(np.min(vr[i, :, -1]), 
                        np.max(vr[i, :, -1]), 30), '--',
            label='long_5days')
    ax.fill_between(ACE_data['long_grid'].squeeze(),
                    vr_5days_test_pred[i]-std_vr_5days_test_pred[i],
                    vr_5days_test_pred[i]+std_vr_5days_test_pred[i], 
                    interpolate=True, alpha=.5,
                    label='Uncertainty')
    ax.legend()
    ax.set_ylabel('$V_{r} 5days$')
    ax.set_title(date)
    
    fig.savefig('Figs/wandb/Vr_example_'+str(IC)+'_'+str(i)+'.jpg', dpi=300)
    plt.close()


def plot_vr5days_update(date_ACE, 
                 idx_clu, 
                 i,
                 IC,
                 vr_5days_test_pred,
                #  dvr_5days_test_pred,
                 vr,
                 vr_5days,
                 ):


    date = date_ACE[idx_clu][i].astype(int)
    date = dt.datetime(date[0], date[1], date[2], date[3], date[4])
    print('by {}, {}th max predictions are {}'.format(date, i, np.abs(vr_5days_test_pred[i]).max()))
    time_points = [date + dt.timedelta(hours=i) for i in range(121)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.plot(time_points, vr_5days_test_pred[i][::-1], 'r-', label='vr_5days_pred')
    # ax.fill_between(time_points,
    #             vr_5days_test_pred[i][::-1]-dvr_5days_test_pred[i][::-1],
    #             vr_5days_test_pred[i][::-1]+dvr_5days_test_pred[i][::-1], 
    #             interpolate=True, alpha=.5,
    #             label='Uncertainty')
    # ax.plot(time_points, dvr_5days_test_pred[i][::-1], 'r-', label='vr_5days_pred')
    vr_idx = vr[i, :121]
    vr_5days_idx = vr_5days[i, :121]
    ax.plot(time_points, vr_idx[::-1], 'g-', label='vr')
    ax.plot(time_points, vr_5days_idx[::-1], 'k-', label='vr_5days')
    # Set x-axis major formatter to display short time format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    # Rotate x-axis labels for better readability
    fig.autofmt_xdate()
    ax.legend()
    ax.set_ylabel('$V_{sw}$')
    ax.set_title(date)
    
    fig.savefig('Figs/Vr_example_'+str(IC)+'_'+str(i)+'.jpg', dpi=300)
    plt.close()


def plot_vr5days_temp(date_ACE, 
                 idx_clu, 
                 iter_boost,
                 i,
                 IC,
                 r_end_5day,
                 ACE_data,
                 vr_5days_test_pred,
                 std_vr_5days_test_pred,
                 vr,
                 vr_5days,
                 HC_lon_now,
                 HC_lon_5days
                 ):

    date = date_ACE[idx_clu][i].astype(int)
    date = dt.datetime(date[0], date[1], date[2], date[3], date[4])
    print('by {}, {}th max predictions are {}'.format(date, i, np.abs(vr_5days_test_pred[i]).max()))

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    # st()
    ax.plot(ACE_data['long_grid'], vr_5days_test_pred[i], 'r-', label='vr_5days_pred')
    ax.plot(ACE_data['long_grid'], vr[i, :, -1], 'g-', label='vr')
    ax.plot(ACE_data['long_grid'],
            vr_5days[i, :, -1], 
            'k-', label='vr_5days')
    ax.plot(np.tile(HC_lon_now[i], (30)), 
            np.linspace(np.min(vr[i, :, -1]), 
                        np.max(vr[i, :, -1]), 30), '--',
            label='long_now')
    ax.plot(np.tile(HC_lon_5days[i], (30)), 
            np.linspace(np.min(vr[i, :, -1]), 
                        np.max(vr[i, :, -1]), 30), '--',
            label='long_5days')
    ax.fill_between(ACE_data['long_grid'].squeeze(),
                    vr_5days_test_pred[i]-std_vr_5days_test_pred[i],
                    vr_5days_test_pred[i]+std_vr_5days_test_pred[i], 
                    interpolate=True, alpha=.5,
                    label='Uncertainty')
    ax.legend()
    ax.set_ylabel('$V_{r}^{5days}$')
    ax.set_title(date)
    
    fig.savefig('Figs/wandb/Vr_example_'
                +str(IC)+'_'
                +str(i)+'_'
                +str(iter_boost)
                +'.jpg', dpi=300)
    plt.close()


def plot_vr5days_show(date_ACE, 
                 idx_clu, 
                 i,
                 IC,
                 r_end_5day,
                 ACE_data,
                 vr_5days_test_pred,
                 std_vr_5days_test_pred,
                 vr,
                 vr_5days,
                 HC_lon_now,
                 HC_lon_5days
                 ):

    date = date_ACE[i].astype(int)
    date = dt.datetime(date[0], date[1], date[2], date[3], date[4])
    print('by {}, {}th max predictions are {}'.format(date, i, np.abs(vr_5days_test_pred[i]).max()))

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # st()
    ax.plot(np.arange(24), vr_5days_test_pred[i, idx_clu[i].astype(int)], 'r-', label='vr_5days_pred')
    ax.plot(np.arange(24), vr[i, idx_clu[i].astype(int), -1], 'g-', label='vr')
    ax.plot(np.arange(24), vr_5days[i, idx_clu[i].astype(int), -1], 'k-', label='vr_5days')
    ax.fill_between(np.arange(24), 
                    vr_5days_test_pred[i, idx_clu[i].astype(int)]-std_vr_5days_test_pred[i],
                    vr_5days_test_pred[i, idx_clu[i].astype(int)]+std_vr_5days_test_pred[i], 
                    interpolate=True, alpha=.5,
                    label='Uncertainty')
    ax.legend()
    ax.set_ylabel('$V_{r} 5days$')
    ax.set_title(date)
    
    fig.savefig('Figs/wandb/Vr_example_'+str(IC)+'_'+str(i)+'_0.jpg', dpi=300)
    plt.close()


def plot_vr5days_clu(date_ACE, 
                 idx_clu, 
                 i,
                 IC,
                 r_end_5day,
                 ACE_data,
                 V_array,
                 dV_array,
                #  0,
                 vr,
                 vr_5days):

    date = date_ACE[i].astype(int)
    date = dt.datetime(date[0], date[1], date[2], date[3], date[4])
    # print(date_ACE[idx_clu][i].astype(int))
    # print('by {}, {}th max predictions are {}'.format(date, i, np.abs(y_t_clu_t[i]).max()))

    r_vector=np.arange(695700*IC, r_end_5day[i], 695700*10) # solve the backward propagation all the way to 1 solar radius
    # r_vector=np.linspace(695700*IC,r_end_5day[i], 40) # solve the backward propagation all the way to 1 solar radius
    dr_vec = r_vector[1:] - r_vector[0:-1]

    fig, ax = plt.subplots(V_array.shape[0], figsize=(15, 20))  

    converter = mdates.ConciseDateConverter()
    munits.registry[np.datetime64] = converter
    munits.registry[dt.date] = converter
    munits.registry[dt.datetime] = converter

    # st()
    for j in range(V_array.shape[0]):

        # st()
        ax[j].plot(np.arange(24), V_array[j, i], 'r', label='vr_5days_pred')
        ax[j].plot(np.arange(24), vr[i, idx_clu[i].astype(int), -1], 'g-', label='vr')
        ax[j].plot(np.arange(24),
                vr_5days[i, idx_clu[i].astype(int), -1], 
                'k-', label='vr_5days')

        ax[j].fill_between(np.arange(24).squeeze(),
                    V_array[j, i]-dV_array[j, i],
                    V_array[j, i]+dV_array[j, i], 
                    interpolate=True, alpha=.5,
                    label='Uncertainty')
        
        if j == 0:
            ax[j].legend(loc=4, fontsize='xx-small')
            # ax[j].set_ylabel('$V_{r} 5days$')
            ax[j].set_title('Persistence')
            ax[j].get_xaxis().set_visible(False)
        elif j == V_array.shape[0]-1:
            ax[j].set_title('Final')
            ax[j].set_xlabel(date)
        else:
            ax[j].set_title('ensemble:'+str(j))
            ax[j].get_xaxis().set_visible(False)
        for label in ax[j].get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
        plt.xticks(fontsize=20)

    fig.savefig('Figs/'+str(V_array.shape[0])+'/Vr_ensemble_'+str(IC)+'_'+str(i)+'.jpg', dpi=300)
    plt.close()


@rank_zero_only
def save_model_result(save_h5, 
                      date_clu,
                      r_end,
                      r_end_5day,
                      vr,
                      vr_5days,
                      vr_5days_pred,
                      idx_train,
                      idx_test,
                      idx_clu
                      ):
    
    # st()
    with h5py.File(save_h5, 'w') as f:
        f.create_dataset("Time", data=date_clu[idx_clu])
        f.create_dataset("r", data=r_end)
        f.create_dataset("r_5days", data=r_end_5day)
        f.create_dataset("vr", data=vr)
        f.create_dataset("vr_5days", data=vr_5days)
        f.create_dataset("vr_5days_train_final", data=vr_5days_pred)
        f.create_dataset("train_idx", data=idx_train)
        f.create_dataset("test_idx", data=idx_test)
        f.close()
        
    matdata = {
        'Time': date_clu[idx_test],
        'vr': vr[idx_test],
        'vr_5days': vr_5days[idx_test],
        'vr_5days_pred': vr_5days_pred[idx_test]
    }
    
    # st()
    sio.savemat(save_h5[:-3]+'.mat', matdata)
    


@rank_zero_only
def test_fig_plot(idx, tmp, yr, yr_5days):

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(tmp.cpu().detach().numpy(), label='Vr_pred_5days')
    ax.plot(yr.cpu().detach().numpy(), label='Vr')
    ax.plot(yr_5days.cpu().detach().numpy(), label='Vr_5days')
    ax.legend()
    fig.savefig('Figs/valid/event_'+str(idx)+'.png')
    plt.close()

    return None