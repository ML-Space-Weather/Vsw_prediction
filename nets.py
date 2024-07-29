import numpy as np
import cmath
import os
import random
from skorch import NeuralNetRegressor
from skorch.callbacks import ProgressBar, Checkpoint
from skorch.callbacks import EarlyStopping, LRScheduler, WarmRestartLR
import torch.nn.functional as F
import torch
from torch.nn.functional import interpolate
from torch import nn
from torch import erf, erfinv
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from matplotlib import rc

from skorch import NeuralNet, NeuralNetBinaryClassifier, NeuralNetClassifier
from ipdb import set_trace as st
from tqdm import tqdm 
from numpy.polynomial.hermite import Hermite
from scipy.fft import ifft, fft, fftfreq, fftshift
from scipy.stats import norm

# from nets import apply_hux_f_model

# torch.autograd.set_detect_anomaly(True)

__all__ = [
    'seed_torch',
    'init_weights',
    'shuffle_n'
]


def plot_com(x1, x2, x3=[], index = []):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x1, 'ro-', label='vr_5days_pred')
    ax.plot(x2, 'bo-', label='vr_5days')
    if len(x3) != 0:
        ax.plot(x3, 'go-', label='vr')

    ax.legend()
    if len(index) != 0:
        line = np.arange(0, 1000, 10)
        ax.plot(np.tile(index[0], line.shape[0]), line)
        ax.plot(np.tile(index[-1], line.shape[0]), line)
    
    fig.savefig('Figs/test_com.png')
    plt.close()

def fill_nan_nearest_array(input_array):

    # st()
    if np.isnan(input_array[0]):
        input_array[0] = np.nanmean(input_array)
    nan_mask = np.isnan(input_array)

    input_array[nan_mask] = np.interp(np.flatnonzero(nan_mask), 
                            np.flatnonzero(~nan_mask), 
                            input_array[~nan_mask]
                            )

    if np.isnan(input_array).sum() > 0:
        st()
    return input_array


def fill_nan_nearest(input_tensor):

    # st()
    if torch.isnan(input_tensor[0]):
        input_tensor[0] = torch.nanmean(input_tensor.clone())
    if torch.isnan(input_tensor[-1]):
        input_tensor[-1] = torch.nanmean(input_tensor.clone())
    nan_mask = torch.isnan(input_tensor)

    # Find the nearest non-NaN values
    non_nan_indices = torch.arange(input_tensor.size(-1))[None, None, :].to('cuda:'+str(input_tensor.get_device()))
    nearest_indices = torch.argmin(torch.abs(non_nan_indices - nan_mask.float()), dim=-1)

    # Use indexing to replace NaN values with the nearest non-NaN values
    filled_tensor = torch.where(nan_mask, input_tensor[nearest_indices], input_tensor)

    # if torch.isnan(filled_tensor).sum() > 0:
    #     st()
    return filled_tensor

def my_custom_loss_func(y_true, y_pred):

    y_pred = torch.from_numpy(y_pred).cuda()
    y_true = torch.from_numpy(y_true)
    # import ipdb;ipdb.set_trace()
    loss = my_weight_rmse_CB(y_pred, y_true)
    return loss

def shuffle_n(input, n, verbose=False, seed=2333):
    
    seed_torch(seed)
    # import ipdb; ipdb.set_trace()
    idx = np.arange(len(input)//n)
    np.random.shuffle(idx)

    out = input[n*idx[0]:np.minimum(n*(idx[0]+1), len(input))]
    for i in np.arange(1, len(idx)):
        out = np.hstack((out,
                         input[n*idx[i]:np.minimum(n*(idx[i]+1),
                                                   len(input))]))
    # import ipdb; ipdb.set_trace()
    if verbose:
        print('Shape after shuffling:', out.shape)
    return out


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def norm_cdf(cdf, thres_cdf):
    cdf_05 = np.zeros(cdf.shape)
    cdf_05[np.where(cdf == thres_cdf)[0]] = 0.5
    # import ipdb;ipdb.set_trace()
    idx_pos = np.where(cdf >= thres_cdf)[0]
    idx_neg = np.where(cdf < thres_cdf)[0]

    cdf_05[idx_pos] = (cdf[idx_pos] - thres_cdf)/(cdf.max() - thres_cdf)*0.5+0.5
    cdf_05[idx_neg] = 0.5 - (cdf[idx_neg] - thres_cdf)/(cdf.min() - thres_cdf)*0.5

    return cdf_05

def cdf_AH(X):
    X = X.astype(np.int)
    cdf = np.zeros(X.max() - X.min()+1)
    ccdf = np.zeros(X.max() - X.min()+1)
    CDF = np.zeros(X.shape[0])
    CCDF = np.zeros(X.shape[0])
    x = np.arange(X.min(), X.max()+1)

    for i in x:
        idx = np.where(X <= i)[0]
        cdf[i-X.min()] = len(idx)/len(X)
        ccdf[i-X.min()] = 1 - len(idx)/len(X)

    for i, dst_clu in enumerate(X):
        try:
            idx = np.where(x == dst_clu)[0]
            CDF[i] = cdf[idx]
            CCDF[i] = ccdf[idx]
        except:
            import ipdb;ipdb.set_trace()

        CDF[i] = cdf[idx]
        CCDF[i] = ccdf[idx]

    # import ipdb;ipdb.set_trace()
    return CCDF, CDF, ccdf[np.where(x == -100)[0]]

def maxmin_scale(X, X_t):

    X_max = X.max(axis=0)
    X_min = X.min(axis=0)

    X = (X - X_min)/(X_max-X_min)
    X_t = (X_t - X_min)/(X_max-X_min)

    return X, X_t

def std_scale(X, X_t):

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    X = (X - X_mean)/X_std
    X_t = (X_t - X_mean)/X_std

    return X, X_t


def Hermite_pred(coef, x):

    sum = np.tile(coef[:, 0], (60, 1)).T
    # st()

    for idx, coef_i in enumerate(coef[:, 1:].T):
        sum += np.tile(coef_i, (60, 1)).T*np.tile(Hermite.basis(idx+1)(x), (1024, 1))

    return sum


def case_pred(date, y, HC_lon, figname):

    fig, ax  = plt.subplots(1, 1, figsize=(10, 12))

    ax.plot(date, y[0], 'r-', label='vr_5days_pred')
    ax.plot(date, y[1], 'g-', label='vr')
    ax.plot(date, y[2], 'k-', label='vr_5days')
    ax.plot(np.tile(HC_lon[0], (30)), 
            np.linspace(np.min(y[2]), 
                        np.max(y[2]), 30), '--',
            label='long_now')
    ax.plot(np.tile(HC_lon[1], (30)), 
            np.linspace(np.min(y[2]), 
                        np.max(y[2]), 30), '--',
            label='long_5days')
    ax.legend()
    ax.set_ylabel('$V_{r} 5days$')
    
    fig.savefig(figname, dpi=300)
    plt.close()


def torch_flip(X, dim, device):

    inv_idx = torch.arange(X.size(dim)-1, -1, -1).long()
    inv_tensor = X.cpu().index_select(dim, inv_idx)
    if dim == 0:
        inv_tensor = X[inv_idx]
    elif dim == 1:
        inv_tensor = X[:, inv_idx]

    return inv_tensor.to(device)

my_callbacks = [
    Checkpoint(),
    EarlyStopping(patience=5),
    LRScheduler(WarmRestartLR),
    ProgressBar(),
]

class MLP(torch.nn.Module):
    def __init__(self, out=12, in_dims=5):
        super(MLP, self).__init__()

        self.drop = torch.nn.Dropout(p=0.6)
        # dropout
        self.fc1 = torch.nn.Linear(in_dims, 64)
        # Fully-connected classifier layer
        self.fc2 = torch.nn.Linear(64, 32)
        # Fully-connected classifier layer
        self.fc3 = torch.nn.Linear(32, out)
        # Fully-connected classifier layer

    def forward(self, x):
        # point B

        # out = np.zeros(X.shape)
        # for i in np.arange(out):
        # x = X[:, i]
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return out


class MLP_EN(torch.nn.Module):
    def __init__(self, out=12, in_dims=5):
        super(MLP_EN, self).__init__()

        self.drop = torch.nn.Dropout(p=0.6)
        # dropout
        self.fc1 = torch.nn.Linear(in_dims, 12)
        # Fully-connected classifier layer
        self.fc2 = torch.nn.Linear(12, 12)
        # Fully-connected classifier layer
        self.fc3 = torch.nn.Linear(12, out)
        # Fully-connected classifier layer

    def forward(self, x):
        # point B

        # out = np.zeros(X.shape)
        # for i in np.arange(out):
        # x = X[:, i]
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return out


class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model
        
    def forward(self, x):
        return self.model(x)


class CNN_1D(torch.nn.Module):
    def __init__(self, num_channel=9, out=1):
        super(CNN_1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(num_channel, 18, kernel_size=2)
        # 9 input channels, 18 output channels
        self.conv2 = torch.nn.Conv1d(18, 36, kernel_size=2)
        # 18 input channels from previous Conv. layer, 36 out
        self.conv2_drop = torch.nn.Dropout2d(p=0.2)
        # dropout
        self.fc1 = torch.nn.Linear(36*1, 36)
        # Fully-connected classifier layer
        self.fc2 = torch.nn.Linear(36, 16)
        # Fully-connected classifier layer
        self.fc3 = torch.nn.Linear(16, out)
        # Fully-connected classifier layer

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        # print(x.shape)
        # x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape)


        # point A
        x = x.view(x.shape[0], -1)

        # point B
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return x.float()


class lstm_gp(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_gp, self).__init__()
        # ipdb.set_trace()
        self.rnn = torch.nn.GRU(input_size, hidden_size,
                                num_layers, batch_first=True)  # , bidirectional=True
        self.gru1 = torch.nn.GRU(input_size,hidden_size,hidden_size, bidirectional=True) #, bidirectional=True
        self.gru2 = torch.nn.GRU(hidden_size*2,hidden_size,num_layers, bidirectional=True) #, bidirectional=True
        self.rnn1 = torch.nn.LSTM(input_size,hidden_size,hidden_size, bidirectional=True) #, bidirectional=True
        self.rnn2 = torch.nn.LSTM(hidden_size,hidden_size,num_layers, bidirectional=False) #, bidirectional=True
        
        self.reg = torch.nn.Linear(hidden_size, output_size)

    def forward(self,x):

        # import ipdb;ipdb.set_trace()
        x, _ = self.rnn(x)
        
        s,b,h = x.shape
        x = x.view(s*b, h)
        #ipdb.set_trace()
        x = self.reg(x)
        x = x.view(s,b,-1)
        # import ipdb;ipdb.set_trace()
        return x[:, -1]
    
class lstm_reg(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.6,
                 output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        # ipdb.set_trace()
        self.gru1 = torch.nn.GRU(input_size-1, hidden_size,
                                num_layers,
                                # bidirectional=True,
                                batch_first=True
        )  # , bidirectional=True
        self.gru2 = torch.nn.GRU(input_size,hidden_size,
                                 num_layers,
                                 bidirectional=True,
                                 batch_first=True) #, bidirectional=True
        
        self.gru3 = torch.nn.GRU(hidden_size*2,hidden_size,num_layers, bidirectional=True) #, bidirectional=True
        self.lstm1 = torch.nn.LSTM(input_size-1,
                                  hidden_size,
                                  num_layers,
                                  bidirectional=True,
                                  batch_first=True) #, bidirectional=True
        
        self.lstm2 = torch.nn.LSTM(hidden_size,hidden_size,num_layers, bidirectional=False) #, bidirectional=True
        
        self.reg = torch.nn.Linear(hidden_size, 1)
        self.reg2 = torch.nn.Linear(2, 1)
        self.reg3 = torch.nn.Linear(128, output_size)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(256, 64),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(20, 1),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self,X):

        # check https://meetonfriday.com/posts/d9cbeda0/
        # x = X.squeeze()
        # if X.shape[0] != 64:
        #     st()
        
        x = X[:, :, :-1].squeeze()
        x0 = X[:, :, -1].unsqueeze(2)
        # x = x[:, :, :-1]
        # y0 = X[:, 0, :128, -1].squeeze()
        # st()
        x, _ = self.gru1(x)
        # x, _ = self.lstm1(x)
        
        try:
            s, b, h = x.shape
        except:
            x = x.unsqueeze(0)
            s, b, h = x.shape
            # st()
        # x = x.reshape([s*b, h])
        x = self.out(x) #.squeeze().unsqueeze(2)
        
        # st()
        x = torch.vstack([x.permute(2, 1, 0), x0.permute(2, 1, 0)]).permute(2, 1, 0)
        x = self.reg2(x).squeeze()
        x = x.reshape([s, b])
        # st()
        # x = x.view(s,b, -1)
        # x = self.reg3(x).squeeze()
        if torch.isnan(x).sum() != 0:
            st() 
            pass

        return x


class cnn_lstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.6,
                 output_size=1, num_layers=2):
        super(cnn_lstm, self).__init__()
        # ipdb.set_trace()
        self.gru1 = torch.nn.GRU(input_size, hidden_size,
                                num_layers,
                                batch_first=True
        )  # , bidirectional=True
        self.gru2 = torch.nn.GRU(input_size,hidden_size,
                                 num_layers,
                                 bidirectional=True,
                                 batch_first=True) #, bidirectional=True
        
        self.gru3 = torch.nn.GRU(hidden_size*2,hidden_size,num_layers, bidirectional=True) #, bidirectional=True
        self.lstm1 = torch.nn.LSTM(input_size,
                                  hidden_size,
                                  num_layers,
                                  # bidirectional=True,
                                  batch_first=True) #, bidirectional=True
        
        self.lstm2 = torch.nn.LSTM(hidden_size,hidden_size,num_layers, bidirectional=False) #, bidirectional=True
        
        self.reg = torch.nn.Linear(hidden_size, 1)
        self.reg2 = torch.nn.Linear(2, 1)
        self.reg3 = torch.nn.Linear(128, output_size)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(256, 64),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(20, output_size),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self,X):

        # check https://meetonfriday.com/posts/d9cbeda0/
        x = X.squeeze()
        x0 = x[:, :, -1].reshape(-1, 1)
        # x = x[:, :, :-1]
        # y0 = X[:, 0, :128, -1].squeeze()

        self.gru1.flatten_parameters() 
        # import ipdb;ipdb.set_trace()
        # st()    
        x, _ = self.gru1(x)
        # x, _ = self.lstm1(x)
        
        s,b,h = x.shape
        x = x.reshape([s*b, h])
        #ipdb.set_trace()
        # x = self.out(x)
        x = self.reg(x).squeeze()
        # x = self.reg(x)

        # st()
        # x = x.view(s,b,-1)
        x = torch.vstack([x.T, x0.T]).T
        x = self.reg2(x).squeeze()
        # x = x.reshape([s, b])
        # st()
        x = x.view(s,b, -1)
        # x = self.reg3(x).squeeze()

        return x


class PhysinformedNet_simple(NeuralNetRegressor):
    def __init__(self,  *args, 
                 Y_mean,
                 Y_std,
                 thres=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        self.thres = thres
        self.Y_mean = Y_mean
        self.Y_std = Y_std

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        
        aa = (y_pred - y_true.to(self.device))**2
        bb = aa*y_true.to(self.device)**2
        loss_RMS = torch.mean(bb)
        # st()
        # loss_RMS = torch.mean(bb[:, -1, :])
        loss = loss_RMS.to(self.device)
        
        return loss

class PhysinformedNet(NeuralNetRegressor):
    def __init__(self,  *args,
                 N_tar,
                 lon_now,
                 lon_5days,
                 Y_mean,
                 Y_std,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        # self.thres = thres
        # self.Y_mean = Y_mean
        self.N_tar = N_tar
        self.lon_now = lon_now
        self.lon_5days = lon_5days
        self.Y_mean = Y_mean
        self.Y_std = Y_std

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)

        # st()
        y_real = y_true[:, :128]
        r_end = y_true[:, 128]
        HC_lon_now = y_true[:, 129]
        HC_lon_5days = y_true[:, 130]
        y0 = y_true[:, 131:]
        # y_real = (y_real - self.Y_mean)/self.Y_std
        # y0 = (y0 - self.Y_mean)/self.Y_std

        yf_t = torch.zeros(y_real.shape, dtype=torch.cfloat)
        target_real = y_pred[:, 2:self.N_tar+2]
        target_imag = y_pred[:, self.N_tar+2:]

        # st()
        yf_t[:, 0] = y_pred[:, 0]+y_pred[:, 1]*1j
        yf_t[:, 1:self.N_tar+1] = target_real+target_imag*1j
        yf_t[:, -self.N_tar:] = torch_flip(target_real, 1, self.device)\
            -torch_flip(target_imag, 1, self.device)*1j
        y_aft = torch.abs(torch.fft.ifft(yf_t))
        
        # print('y_aft:', y_aft.shape)
        # st()

        vr_pred = torch.zeros(y_aft.shape)
        RMSE = torch.zeros(y_aft.shape[0])
        p = torch.linspace(0, 2*np.pi, 128)
        for i in range(r_end.shape[0]):
            # r_vector=torch.linspace(695700,r_end[i],200) # solve the backward propagation all the way to 1 solar radius
            # dr_vec = r_vector[0:-1] - r_vector[1:]
            # dp_vec = p[0:-1] - p[1:]

            # # st()

            # tmp = apply_hux_f_model(y_aft[i], dr_vec, dp_vec) 
            # # print(tmp.shape)
            # vr_pred[i, :] = tmp[-1]

            if HC_lon_now[i] > HC_lon_5days[i]:
                idx = np.where((p < HC_lon_now[i]*np.pi/180)
                            & (p > HC_lon_5days[i]*np.pi/180)
                    )[0]
            else:
                idx = np.where((p < HC_lon_now[i]*np.pi/180)
                            | (p > HC_lon_5days[i]*np.pi/180)
                    )[0]

            # st()
            RMSE[i] = torch.sum((y0[i, idx] - y_aft[i, idx])**2)

            # tmp_final = tmp[-1, idx]*torch.from_numpy(self.Y_std[idx])\
            #     +torch.from_numpy(self.Y_mean[idx])
            # RMSE[i] = torch.sum((tmp[-1, idx] - y_real[i, idx])**2)
            # st()
        
            # print(torch.mean(RMSE[:i]))
        # st()

        return torch.sqrt(torch.mean(RMSE))


class PhysinformedNet_nofft(NeuralNetRegressor):
    def __init__(self,  *args,
                 N_tar,
                 lon_now,
                 lon_5days,
                #  Y_mean,
                #  Y_std,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        # self.thres = thres
        # self.Y_mean = Y_mean
        self.N_tar = N_tar
        self.lon_now = lon_now
        self.lon_5days = lon_5days
        # self.Y_mean = Y_mean[:128]
        # self.Y_std = Y_std[:128]

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)

        # y_true = (y_true - self.Y_mean)/self.Y_std

        y_real = y_true[:, :128] # vr
        r_end = y_true[:, 128] 
        HC_lon_now = y_true[:, 129]
        HC_lon_5days = y_true[:, 130]
        y0_5days = y_true[:, 131:259].to(self.device) # v0
        y_real_5days = y_true[:, 259:].to(self.device) # vr_5days

        # st()

        # RMSE = torch.zeros(y_real.shape[0])
        RMSE = 0
        p = torch.linspace(0, 360, 129)[:-1]
        dp_vec = (p[1:]-p[0:-1])/180*np.pi
        for i in range(r_end.shape[0]):
            # st()
            r_vector=torch.linspace(695700,r_end[i], 200) # solve the backward propagation all the way to 1 solar radius
            dr_vec = r_vector[1:] - r_vector[0:-1]

            # st()

            # tmp = apply_hux_f_model_change(y_pred[i].squeeze(), dr_vec, dp_vec, 
            #                         r0=695700) 
            # st()
            # +y0[i].squeeze()
            # tmp = apply_rk42_f_model_tensor(y_pred[i].squeeze()+y0[i].squeeze(), dr_vec, dp_vec, 
            #                         r0=695700) 
            # st()
            # print(tmp.shape)
            # vr_pred[i, :] = tmp[-1]

            if HC_lon_now[i] > HC_lon_5days[i]:
                idx = np.where((p < HC_lon_now[i])
                             & (p > HC_lon_5days[i])
                    )[0]
            else:
                idx = np.where(p > HC_lon_5days[i])[0]
                idx = np.hstack((idx, 
                                np.where(p < HC_lon_now[i])[0]))

            weights = 1/(torch.arange(1, len(idx)+1))
            weights = weights/weights.sum()*len(idx)
            # weights = torch_flip(weights.to(self.device), 0, self.device)
            # st()
            # RMSE[i] = torch.sum((y_real[i].to(self.device) - y_pred[i])**2)
            # RMSE[i] = torch.sqrt(torch.mean((y_real[i, idx].to(self.device) \
            #     - tmp[-1, idx].to(self.device))**2))
            # RMSE[i] = torch.mean((y_real[i, idx].to(self.device) - y_pred[i, idx])**2)
            # RMSE += torch.mean((y_real[i].to(self.device) - y_pred[i] - y0[i])**2)
            # RMSE_t = y_real[i, idx].to(self.device) - y_pred[i, idx].squeeze()
            RMSE_tt = y0_5days[i, idx].to(self.device) - y_pred[i, idx].squeeze().to(self.device) - y_real[i, idx].squeeze().to(self.device)
            # RMSE_tt = y_real_5days[i, idx].to(self.device) - tmp[0, idx].to(self.device) - y0[i, idx].squeeze()
            if (torch.isnan(RMSE_tt).sum() > 0):
                print('RMSE_tt: {}'.format(RMSE_tt))
                # st()
                continue
            # RMSE_t = y_real[i, idx].to(self.device) - y_pred[i, idx].squeeze()
            # RMSE_t = y_real[i, idx].to(self.device) - y_pred[i, idx].squeeze() - y0[i, idx].squeeze()
            # st()
            idx_tt = np.where((y_real[i, idx] < 1000)
                            & (y_pred[i, idx].squeeze().cpu().T + y_real[i, idx].squeeze().cpu() > 50)
                            # & (torch.abs(RMSE_t.cpu()) < 1000)
                            # & (torch.abs(RMSE_tt) < 1000)
                            )[0] 
            # st()
            if (len(idx_tt) == 0) | (torch.isnan(RMSE_tt).sum() > 0):
                # st()
                continue
            # idx_tt1 = torch.where(torch.abs(RMSE_t) > 1500)[0] 
            # RMSE += torch.mean(weights**2*RMSE_t**2) 
            # st()
            RMSE += torch.mean((RMSE_tt[idx_tt]**2)) 
            # RMSE += torch.mean(weights[idx_tt].to(self.device)*(RMSE_tt[idx_tt]**2)) 
            # RMSE += torch.mean(weights[idx_tt]*(RMSE_t[idx_tt]**2+RMSE_tt[idx_tt]**2)) 
            # st()
            # RMSE += torch.mean((y_real[i, idx].to(self.device) - y_pred[i, idx] - y0[i, idx])**2)
        
        # st()
        if RMSE == 0:
            st()
            pass
        # print('RMSE: {}'.format(torch.sqrt(RMSE/(i+1))))
            # st()
        # st()
        # RMSE = torch.mean((y_pred.squeeze() - y_true.to(self.device))**2)

        return torch.sqrt(RMSE/r_end.shape[0])


class PhysinformedNet_all(NeuralNetRegressor):
    def __init__(self,  *args,
                 N_tar,
                 lon_now,
                 lon_5days,
                #  Y_mean,
                #  Y_std,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        # self.thres = thres
        # self.Y_mean = Y_mean
        self.N_tar = N_tar
        self.lon_now = lon_now
        self.lon_5days = lon_5days
        # self.Y_mean = Y_mean[:128]
        # self.Y_std = Y_std[:128]

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)

        # y_true = (y_true - self.Y_mean)/self.Y_std

        y0 = y_true[:, :128] # v0
        r_end = y_true[:, 128] # r_end
        HC_lon_now = y_true[:, 129]
        HC_lon_5days = y_true[:, 130]
        y0_5days = y_true[:, 131:259].to(self.device) # v0_5days
        yr_5days = y_true[:, 259:].to(self.device) # vr_5days

        # RMSE = torch.zeros(y_real.shape[0])
        RMSE = 0
        p = torch.linspace(0, 360, 129)[:-1]
        dp_vec = (p[1:]-p[0:-1])/180*np.pi
        for i in range(r_end.shape[0]):
            # st()
            r_vector=torch.linspace(695700,r_end[i], 50) # solve the backward propagation all the way to 1 solar radius
            dr_vec = r_vector[1:] - r_vector[0:-1]
            # st()

            if HC_lon_now[i] > HC_lon_5days[i]:
                idx = np.where((p < HC_lon_now[i])
                                & (p > HC_lon_5days[i])
                    )[0]
            else:
                idx = np.where(p > HC_lon_5days[i])[0]
                idx = np.hstack((idx, 
                                np.where(p < HC_lon_now[i])[0]))

            try:
                tmp = apply_rk42log_f_model_tensor(y_pred[i].squeeze()+y0[i].squeeze().to(self.device), 
                            dr_vec, dp_vec, 
                            r0=695700)[-1]
                if torch.isnan(tmp).sum() != 0:
                    print(tmp)
                    st()
            except:
                st()

            RMSE_vr = yr_5days[i, idx] - tmp[idx].to(self.device)
            RMSE += torch.mean((RMSE_vr**2)) 

        return torch.sqrt(RMSE/(i+1))


class PhysinformedNet_weight(NeuralNetRegressor):
    def __init__(self,  *args,
                 lon_now,
                 lon_5days,
                 vr_mean,
                 vr_std,
                 mode,
                #  Y_mean,
                #  Y_std,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        # self.thres = thres
        # self.Y_mean = Y_mean
        self.lon_now = lon_now
        self.lon_5days = lon_5days
        self.vr_mean = vr_mean
        self.vr_std = vr_std
        self.mode = mode
        # self.Y_mean = Y_mean[:128]
        # self.Y_std = Y_std[:128]

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)

        # y_true = (y_true - self.Y_mean)/self.Y_std

        y0 = y_true[:, :128] # v0
        r_end = y_true[:, 128] # r_end
        HC_lon_now = y_true[:, 129]
        HC_lon_5days = y_true[:, 130]
        y0_5days = y_true[:, 131:259].to(self.device) # v0_5days
        yr_5days = y_true[:, 259:].to(self.device) # vr_5days

        # RMSE = torch.zeros(y_real.shape[0])
        RMSE = 0
        p = torch.linspace(0, 360, 129)[:-1]
        dp_vec = (p[1:]-p[0:-1])/180*np.pi
        for i in range(r_end.shape[0]):
            # st()
            r_vector=torch.linspace(695700,r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
            dr_vec = r_vector[1:] - r_vector[0:-1]
            # st()

            if HC_lon_now[i] > HC_lon_5days[i]:
                idx = np.where((p < HC_lon_now[i])
                                & (p > HC_lon_5days[i])
                    )[0]
            else:
                idx = np.where(p > HC_lon_5days[i])[0]
                idx = np.hstack((idx, 
                                np.where(p < HC_lon_now[i])[0]))

            try:
                # st()
                tmp = apply_rk42log_f_model_list(y_pred[i].squeeze()+y0[i].squeeze().to(self.device), 
                            dr_vec, dp_vec, 
                            r0=695700)[-1]

                # yr = apply_rk42log_f_model_tensor(y0[i].squeeze().to(self.device), 
                #             dr_vec, dp_vec, 
                #             r0=695700)[-1]
                # st()
                # figname = 'Figs/case/'+str(i)+'.jpg'
                # y_plot = np.vstack([np.expand_dims(tmp.detach().numpy(), 0),
                #             np.expand_dims(yr.cpu().detach().numpy(), 0),
                #             np.expand_dims(yr_5days[i].cpu().detach().numpy(), 0),
                # ])
                # HC_lon = y_true[i, 129:131].T
                # case_pred(p, y_plot, HC_lon, figname)
                # st()

                # if torch.isnan(tmp).sum() != 0:
                #     print(tmp)
                #     st()
            except:
                st()

            weights = 5+5*torch.tanh((yr_5days[i, idx]-650)/150)
            # weights = torch.from_numpy(self.weight).to(self.device)
            RMSE_vr = yr_5days[i, idx] - tmp[idx].to(self.device)
                
            # weights = weights.float().to(self.device)
            # st()  
            weights = weights ** self.mode 
            # RMSE += torch.nanmean((RMSE_vr**2)) 

            
            RMSE += torch.nanmean(weights*(RMSE_vr**2)) 
            # st()
        # if torch.isnan().sum() != 0:
        #     st()

        return torch.sqrt(RMSE/(i+1))



class PhysinformedNet_V(NeuralNetRegressor):
    def __init__(self,  *args,
                 lon_now,
                 lon_5days,
                 vr_mean,
                 vr_std,
                 mode,
                 IC,
                #  Y_mean,
                #  Y_std,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        # self.thres = thres
        # self.Y_mean = Y_mean
        self.lon_now = lon_now
        self.lon_5days = lon_5days
        self.vr_mean = vr_mean
        self.vr_std = vr_std
        self.mode = mode
        self.IC = IC
        # self.Y_mean = Y_mean[:128]
        # self.Y_std = Y_std[:128]

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # st()
        # y_pred = (y_pred - y_pred.min())/self.vr_std
        # y_pred = (y_pred - y_pred.min())/y_pred.std()
        y_pred_ori = y_pred
        y_pred = (y_pred - self.vr_mean)/self.vr_std
        # y_pred = (y_pred - y_pred.min())/(y_pred.max()-y_pred.min())
        # y_pred = y_pred * self.vr_std

        # st()
        y0 = y_true[:, :128] # v0
        r_end = y_true[:, 128] # r_end
        HC_lon_now = y_true[:, 129]
        HC_lon_5days = y_true[:, 130]
        y0_5days = y_true[:, 131:259].to(self.device) # v0_5days
        yr_5days = y_true[:, 259:387].to(self.device) # vr_5days
        yr = y_true[:, 387:].to(self.device) # vr_5days

        # st()
        # RMSE = torch.zeros(y_real.shape[0])
        RMSE = 0

        p = torch.linspace(0, 360, 129)[:-1]
        dp_vec = (p[1:]-p[0:-1])/180*np.pi
        for i in range(r_end.shape[0]):

            # st()
            # if torch.isnan(y_pred[i]).sum() >= 1:
            #     print(i)
            #     continue
            # st()
            vr_std = self.vr_std

            # r_vector=torch.arange(695700*self.IC, r_end[i], 695700*10) # solve the backward propagation all the way to 1 solar radius
            r_vector=torch.linspace(695700*self.IC, r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
            dr_vec = r_vector[1:] - r_vector[0:-1]
            # st()
            tmp = torch.zeros(y_pred[i].shape).to(self.device)
            tmp[:] = np.nan
            # st()

            if HC_lon_now[i] > HC_lon_5days[i]:
                idx = np.where((p < HC_lon_now[i])
                                & (p > HC_lon_5days[i])
                    )[0]
            else:
                idx = np.where(p > HC_lon_5days[i])[0]
                idx = np.hstack((idx, 
                                np.where(p < HC_lon_now[i])[0]))

            # while torch.isnan(tmp).sum() >= 1:

            # st()
            # del tmp
            # vr_std = vr_std/100
            # vr_std = 1
            # tmp = apply_rk42log_f_model_list(y_pred[i].squeeze(), 
            # tmp = apply_rk42log_f_model_list(y_pred[i].squeeze()*y0[i].squeeze().to(self.device), 
            # tmp = apply_rk42log_f_model_list(y_pred[i].squeeze()*y0[i].squeeze().to(self.device), 
            # tmp = apply_rk42_f_model_tensor(y_pred[i].squeeze()*self.vr_mean+y0[i].squeeze().to(self.device), 
            # tmp = apply_rk42_f_model_tensor(y_pred[i].squeeze()+y0[i].squeeze().to(self.device), 
            # v_init = y_pred[i].squeeze()+y0[i].squeeze().to(self.device)
            # v_init = (y_pred[i].squeeze()+1e-2)*y0[i].squeeze().to(self.device)
            # v_init = torch.abs(y_pred[i].squeeze())
            # v_init = torch.abs(y_pred[i].squeeze()+y0[i].squeeze().to(self.device))
            # v_init = torch.abs(y_pred[i].squeeze()+y0[i].squeeze().to(self.device))
            # v_init = torch.abs(y_pred[i].squeeze()*5+y0[i].squeeze().to(self.device))
            v_init = torch.abs(y_pred[i]).squeeze()*self.vr_std+y0[i].squeeze().to(self.device)
            # v_init = torch.abs(y_pred[i]).squeeze()*self.vr_mean+y0[i].squeeze().to(self.device)
            # print(v_init.max())
            # idx_1000 = torch.where(v_init > 900)[0]
            # idx_200 = torch.where(v_init < 200)[0]
            # # idx_200 = torch.where(v_init < 200)[0]
            # v_init[idx_1000] = 900
            # v_init[idx_200] = 200

            # st()
            # tmp_lb = apply_rk42log_b_model(v_init[0].cpu().detach().numpy(), 
            #         # dr_vec, dp_vec,
            #         np.array(dr_vec), np.array(dp_vec), 
            #         rh=50 * 695700,
            #         r0=r_end[i])[-1]
            # tmp_lb = apply_rk42_b_model(100, 
            #             np.array(dr_vec), np.array(dp_vec), 
            #             r0=695700*self.IC)[-1]
            # tmp_ub = apply_rk42_b_model(1000, 
            #             np.array(dr_vec), np.array(dp_vec), 
            #             r0=695700*self.IC)[-1]
            # tmp = apply_rk42_f_model_tensor(v_init, 
            tmp = apply_rk42log_f_model_tensor(v_init, 
                        dr_vec, dp_vec, 
                        r0=695700*self.IC)[-1]
            # st()

            if torch.isnan(tmp).sum() > 0:
                st()

            weights = 5+5*torch.tanh((yr_5days[i, idx]-350)/150)
            # w_idx = torch.where((tmp[idx] > 1000) | (tmp[idx] < 100))[0]
            # weights[w_idx] = 0
            # st()
            # weights = weights ** 1.5 
            # weights = weights ** self.mode 
            RMSE_vr = yr_5days[i, idx] - tmp[idx].to(self.device)

            if i == 0:
                plot_com(tmp.cpu().detach().numpy(), 
                         yr_5days[i].cpu(),
                         yr[i].cpu(),
                         index = idx)
                # st()
            # idx = np.where((tmp>100) & (tmp<1000))[0]
            # RMSE += torch.nanmean((RMSE_vr**2)) 
            # st()
            RMSE += torch.nanmean((RMSE_vr**2)) 
            # RMSE += torch.nanmean(weights*(RMSE_vr**2)) 
            # print("EMSE_vr:", RMSE_vr)
            # print('RMSE:', RMSE)
            # st()

        # print('RMSE: {}'.format(torch.sqrt(RMSE/(i+1))))
        # st()
        return torch.sqrt(RMSE/24)
        # return torch.sqrt(RMSE/(i+1))


class PhysinformedNet_V_nolog(NeuralNetRegressor):
    def __init__(self,  *args,
                 lon_now,
                 lon_5days,
                 vr_mean,
                 vr_std,
                 mode,
                 IC,
                #  Y_mean,
                #  Y_std,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        # self.thres = thres
        # self.Y_mean = Y_mean
        self.lon_now = lon_now
        self.lon_5days = lon_5days
        self.vr_mean = vr_mean
        self.vr_std = vr_std
        self.mode = mode
        self.IC = IC
        # self.Y_mean = Y_mean[:128]
        # self.Y_std = Y_std[:128]

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # st()
        # y_pred = (y_pred - y_pred.min())/self.vr_std
        y_pred = (y_pred - self.vr_mean)/self.vr_std
        # y_pred = (y_pred - y_pred.min())/(y_pred.max()-y_pred.min())*3
        # y_pred = y_pred * self.vr_std

        y0 = y_true[:, :128] # v0
        r_end = y_true[:, 128] # r_end
        HC_lon_now = y_true[:, 129]
        HC_lon_5days = y_true[:, 130]
        y0_5days = y_true[:, 131:259].to(self.device) # v0_5days
        yr_5days = y_true[:, 259:].to(self.device) # vr_5days

        # st()
        # RMSE = torch.zeros(y_real.shape[0])
        RMSE = 0

        p = torch.linspace(0, 360, 129)[:-1]
        dp_vec = (p[1:]-p[0:-1])/180*np.pi
        for i in range(r_end.shape[0]):

            # st()
            # if torch.isnan(y_pred[i]).sum() >= 1:
            #     print(i)
            #     continue
            # st()
            vr_std = self.vr_std

            r_vector=torch.arange(695700*self.IC, r_end[i], 695700*5) # solve the backward propagation all the way to 1 solar radius
            # r_vector=torch.linspace(695700*self.IC, r_end[i], 50) # solve the backward propagation all the way to 1 solar radius
            dr_vec = r_vector[1:] - r_vector[0:-1]
            # st()
            tmp = torch.zeros(y_pred[i].shape).to(self.device)
            tmp[:] = np.nan
            # st()

            if HC_lon_now[i] > HC_lon_5days[i]:
                idx = np.where((p < HC_lon_now[i])
                                & (p > HC_lon_5days[i])
                    )[0]
            else:
                idx = np.where(p > HC_lon_5days[i])[0]
                idx = np.hstack((idx, 
                                np.where(p < HC_lon_now[i])[0]))

            # while torch.isnan(tmp).sum() >= 1:

            # st()
            # del tmp
            # vr_std = vr_std/100
            # vr_std = 1
            # tmp = apply_rk42log_f_model_list(y_pred[i].squeeze(), 
            # tmp = apply_rk42log_f_model_list(y_pred[i].squeeze()*y0[i].squeeze().to(self.device), 
            # tmp = apply_rk42log_f_model_list(y_pred[i].squeeze()*y0[i].squeeze().to(self.device), 
            # tmp = apply_rk42_f_model_tensor(y_pred[i].squeeze()*self.vr_mean+y0[i].squeeze().to(self.device), 
            # tmp = apply_rk42_f_model_tensor(y_pred[i].squeeze()+y0[i].squeeze().to(self.device), 
            # v_init = y_pred[i].squeeze()+y0[i].squeeze().to(self.device)
            # v_init = (y_pred[i].squeeze()+1e-2)*y0[i].squeeze().to(self.device)
            v_init = torch.abs(y_pred[i].squeeze()*self.vr_std+y0[i].squeeze().to(self.device))
            # v_init = torch.abs(y_pred[i]).squeeze()*self.vr_mean+y0[i].squeeze().to(self.device)
            # print(v_init.max())
            idx_1000 = torch.where(v_init > 900)[0]
            idx_200 = torch.where(v_init < 200)[0]
            # idx_200 = torch.where(v_init < 200)[0]
            v_init[idx_1000] = 900
            v_init[idx_200] = 200

            # tmp_lb = apply_rk42log_b_model(v_init[0].cpu().detach().numpy(), 
            #         # dr_vec, dp_vec,
            #         np.array(dr_vec), np.array(dp_vec), 
            #         rh=50 * 695700,
            #         r0=r_end[i])[-1]
            # tmp_lb = apply_rk42_b_model(100, 
            #             np.array(dr_vec), np.array(dp_vec), 
            #             r0=695700*self.IC)[-1]
            # tmp_ub = apply_rk42_b_model(1000, 
            #             np.array(dr_vec), np.array(dp_vec), 
            #             r0=695700*self.IC)[-1]
            # st()
            tmp = apply_rk42_f_model_tensor(v_init, 
            # tmp = apply_rk42log_f_model_tensor(v_init, 
                        dr_vec, dp_vec, 
                        r0=695700*self.IC)[-1]

            if torch.isnan(tmp).sum() > 0:
                st()

            weights = 5+5*torch.tanh((yr_5days[i, idx]-350)/150)
            # w_idx = torch.where((tmp[idx] > 1000) | (tmp[idx] < 100))[0]
            # weights[w_idx] = 0
            # st()
            # weights = weights ** 1.5 
            # weights = weights ** self.mode 
            RMSE_vr = yr_5days[i, idx] - tmp[idx].to(self.device)

            # idx = np.where((tmp>100) & (tmp<1000))[0]
            # RMSE += torch.nanmean((RMSE_vr**2)) 
            # st()
            # RMSE += torch.nanmean((RMSE_vr**2)) 
            RMSE += torch.mean(weights*(RMSE_vr**2)) 
            # print("EMSE_vr:", RMSE_vr)
            # print('RMSE:', RMSE)
            # st()

        # print('RMSE: {}'.format(torch.sqrt(RMSE/(i+1))))
        # st()
        return torch.sqrt(RMSE/(i+1))



class PhysinformedNet_EN(NeuralNet):
    def __init__(self, *args, pre_weight='False',
                 num_output, weight, alpha, l1_ratio,
                 # P,
                 loss,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = nn.ReLU()
        # self.ReLU6 = nn.ReLU6()
        # self.criterion = criterion
        # self.X = 0
        # self.loss = my_BCEloss
        # self.loss = my_weight_BCEloss
        # self.loss = my_L1_loss
        # self.P = P
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.pre_weight = pre_weight
        self.weight = weight
        self.num_output = num_output

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # loss_ori = torch.zeros(1).cuda()

        if np.isnan(y_pred.cpu().detach().numpy().sum()):

            pass

            # import ipdb;ipdb.set_trace()
        '''
        print('BCE value:', self.loss(y_pred.type(torch.FloatTensor).cuda(),
                                      y_true.type(torch.FloatTensor).cuda()))
        '''        
       
             
        loss_ori = self.loss(y_pred=y_pred.type(torch.FloatTensor).to(self.device),
                             y=y_true.type(torch.FloatTensor).to(self.device),
                             pre_weight=self.pre_weight,
                             # alpha=self.alpha,
                             # l1_ratio=self.l1_ratio,
                             # P=self.P,
                             weight=self.weight)
        # import ipdb; ipdb.set_trace()
        
        l1_lambda = self.alpha*self.l1_ratio
        l1_reg = torch.tensor(0.)
        l1_reg = l1_reg.to(self.device)
        for param in self.module.parameters():
            l1_reg += torch.sum(torch.abs(param))
        loss1 = l1_lambda * l1_reg
        loss_ori += loss1
        
        l2_lambda = self.alpha*(1-self.l1_ratio) 
        l2_reg = torch.tensor(0.)
        l2_reg = l2_reg.to(self.device)
        for param in self.module.parameters():
            l2_reg += torch.norm(param).sum()
        loss_ori += l2_lambda * l2_reg
        
        # loss_ori += loss_ori_t

        # loss_ori = loss_ori
        # import ipdb; ipdb.set_trace()
        return loss_ori



class PhysinformedNet_1D(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ReLU = torch.nn.ReLU()
        self.ReLU6 = torch.nn.ReLU6()

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # import ipdb;ipdb.set_trace()
        
        loss_ori = super().get_loss(y_pred.squeeze(),
                                    y_true.squeeze(), 
                                    X=X, 
                                    training=training)
        # loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
        # X = X.cuda()

        '''

        loss_RMS = torch.mean((y_pred.sum(axis=1) -
                          y_true.cuda())**2)

        loss_RMS = torch.mean((y_pred.std(axis=1) -
                          y_true.cuda())**2)

        loss_RMS = torch.mean((y_pred[:,:, 0] -
                          y_true[:,:].cuda())**2)
        loss_RE = torch.mean((y_pred[:,:, 0] -
                          y_true[:,:].cuda())**2)
        '''

        # ipdb.set_trace()

        # print('loss_term1:', loss_term1)
        # print('loss:', loss)
        # ipdb.set_trace()
        loss = loss_ori

        # loss += loss_term0.mean()
        # print('loss+0:', loss)
        # loss += loss_term1.squeeze()
        # print('loss+1:', loss)
        # loss += loss_term2.squeeze()
        # print('loss+2:', loss)
        # loss += loss_term3.squeeze()
        # print('loss+3:', loss)

        return loss

class PhysinformedNet_single(NeuralNet):
    def __init__(self, *args, pre_weight='False',
                 num_output, weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.ReLU = nn.ReLU()
        self.ReLU6 = nn.ReLU6()
        # self.criterion = criterion
        self.X = 0
        # self.loss = my_BCEloss
        # self.loss = my_weight_BCEloss
        # self.loss = my_L1_loss
        # self.P = P
        self.loss = my_weight_rmse_CB
        self.pre_weight = pre_weight
        self.weight = weights

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # loss_ori = torch.zeros(1).cuda()

        if np.isnan(y_pred.cpu().detach().numpy().sum()):

            pass

            # import ipdb;ipdb.set_trace()
        '''
        print('BCE value:', self.loss(y_pred.type(torch.FloatTensor).cuda(),
                                      y_true.type(torch.FloatTensor).cuda()))
        '''        
        # import ipdb; ipdb.set_trace()
             
        loss_ori = self.loss(y_pred=y_pred.type(torch.FloatTensor).to(self.device),
                             y=y_true.type(torch.FloatTensor).to(self.device),
                             pre_weight=self.pre_weight,
                             # P = self.P,
                             weight=self.weight)
        # loss_ori += loss_ori_t

        # loss_ori = loss_ori

        return loss_ori


def my_weight_rmse_CB(y_pred, y, thres=0.5,
                      pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    # import ipdb;ipdb.set_trace()
    resi = y_pred.cpu() - y
    resi = resi**2

    # a = l1_ratio*alpha
    # b = (1-l1_ratio)*alpha

    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] >= thres)[0]
        idx_neg = torch.where(y[:, i] < thres)[0]
        beta = 0.9999
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            E_pos = (1-beta**len(idx_pos))/(1-beta) 
            MSE_pos = torch.sum(resi[idx_pos, i])/E_pos
            loss += MSE_pos
        if len(idx_neg) != 0:
            E_neg = (1-beta**len(idx_neg))/(1-beta) 
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            MSE_neg = torch.sum(resi[idx_neg, i])/E_neg
            loss += MSE_neg
    # loss += a*np.sum(np.abs(weight))

    return loss

def my_weight_rmse2(weight, y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    # weights = torch.from_numpy(weight).type(torch.FloatTensor).cuda()
    resi = y_pred - y
    resi = resi**2

    # import ipdb;ipdb.set_trace()
    # P = torch.cos(0.9*np.pi*(y - 0.5))
    
    # resi = resi*P
    for i in range(y.shape[1]):
        loss += torch.mean(resi[:, i])*weight[i]

    '''
    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] >= 0.5)[0]
        idx_neg = torch.where(y[:, i] < 0.5)[0]
        beta = 0.9999
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            E_pos = (1-beta**len(idx_pos))/(1-beta) 
            MSE_pos = torch.sum(resi[idx_pos, i])/E_pos
            loss += MSE_pos*weight[i]/2
        if len(idx_neg) != 0:
            E_neg = (1-beta**len(idx_neg))/(1-beta) 
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            MSE_neg = torch.sum(resi[idx_neg, i])/E_neg
            loss += MSE_neg*weight[i]/2
    '''
    
    return loss


def norm_1d(data):
    return data/data.sum()

        
class PhysinformedNet_AR_cpu(NeuralNetRegressor):
    def __init__(self,  *args, beta, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        self.beta = beta
        # self.d = d

    def get_loss(self, y_pred, y_true, X, training=False):
        
        # import ipdb;ipdb.set_trace()
        d = y_true[:, 1] - y_true[:, 0]
        # d = (d - d.mean())/d.std()

        # d = d.cuda()
        N = d.shape[0]
        sigma = torch.exp(y_pred).squeeze()
        
        x = torch.zeros(sigma.shape[0])
        CRPS = torch.zeros(sigma.shape[0])
        RS = torch.zeros(sigma.shape[0])
        
        for i in range(N):
            x[i] = d[i]/np.sqrt(2)/sigma[i]
            
        # import ipdb;ipdb.set_trace()
        
        ind = torch.argsort(x)
        ind_orig = torch.argsort(ind)+1
        
        # x = x.cuda()
        # import ipdb;ipdb.set_trace()


        def AR(i):
            
            CRPS = sigma[i]*(np.sqrt(2)*x[i]*erf(x)[i]
                + np.sqrt(2/np.pi)*torch.exp(-x[i]**2) 
                - 1/np.sqrt(np.pi))

            # import ipdb;ipdb.set_trace()
            RS = N*(x[i]/N*(erf(x)[i]+1) - 
                x[i]*(2*ind_orig[i]-1)/N**2 + 
                torch.exp(-x[i]**2)/np.sqrt(np.pi)/N)

        
        for i in range(N):
                    
            CRPS[i] = sigma[i]*(np.sqrt(2)*x[i]*erf(x)[i]
                + np.sqrt(2/np.pi)*torch.exp(-x[i]**2) 
                - 1/np.sqrt(np.pi))

            # import ipdb;ipdb.set_trace()
            RS[i] = N*(x[i]/N*(erf(x)[i]+1) - 
                x[i]*(2*ind_orig[i]-1)/N**2 + 
                torch.exp(-x[i]**2)/np.sqrt(np.pi)/N)
        # loss_CB = 
        
    
        loss = self.beta*CRPS+(1-self.beta)*RS
        loss = torch.nanmean(loss)
                
        return loss

class PhysinformedNet_AR(NeuralNetRegressor):
    
    def __init__(self,  *args, beta, 
                 mean, std, CRPS_min, RS_min,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        self.beta = beta
        self.CRPS_min = CRPS_min
        self.RS_min = RS_min
        self.mean = mean
        self.std = std
        # self.d = d

    def get_loss(self, y_pred, y_true, X, training=False):
        
        # import ipdb;ipdb.set_trace()
        # st()
        d = y_true[:, 1] - y_true[:, 0]
        d = (d - self.mean)/self.std
        # y_pred = (y_pred - self.mean)/self.std

        d = d.cuda()
        N = d.shape[0]
        sigma = torch.exp(y_pred).squeeze().to(self.device)
        
        weights = 0.3 - 0.7*y_true[:, 1]/400
        weights = weights.to(self.device)
        
        x = torch.zeros(sigma.shape[0])
        CRPS = torch.zeros(sigma.shape[0])
        RS = torch.zeros(sigma.shape[0])
        
        # st()
        x = d.to(self.device)/sigma
        x = x/np.sqrt(2)
            
        # import ipdb;ipdb.set_trace()
        
        ind = torch.argsort(x.T).T
        ind_orig = torch.argsort(ind)+1
        ind_orig = ind_orig.to(self.device)
        x = x.to(self.device)
        CRPS_1 = np.sqrt(2)*x*erf(x)
        CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x**2) 
        CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
        # CRPS_1 = CRPS_1.to(self.device)
        # CRPS_2 = CRPS_2.to(self.device)
        CRPS_3 = CRPS_3.to(self.device)
                  
        CRPS = sigma*(CRPS_1 + CRPS_2 - CRPS_3)

        # import ipdb;ipdb.set_trace()
        RS = N*(x/N*(erf(x)+1) - 
            x*(2*ind_orig-1)/N**2 + 
            torch.exp(-x**2)/np.sqrt(np.pi)/N)
        
        RS = RS.to(self.device)
        # import ipdb;ipdb.set_trace()

        # for i in range(N):
                    
        #     CRPS[i] = sigma[i]*(np.sqrt(2)*x[i]*erf(x)[i]
        #         + np.sqrt(2/np.pi)*torch.exp(-x[i]**2) 
        #         - 1/np.sqrt(np.pi))

        #     # import ipdb;ipdb.set_trace()
        #     RS[i] = N*(x[i]/N*(erf(x)[i]+1) - 
        #         x[i]*(2*ind_orig[i]-1)/N**2 + 
        #         torch.exp(-x[i]**2)/np.sqrt(np.pi)/N)
        # # loss_CB = 
        
        # beta = RS_min/(RS_min+CRPS_min)
        # 1-beta = CRPS_min/(RS_min+CRPS_min)
        # loss = self.beta*CRPS+(1-self.beta)*RS
        loss = torch.zeros(CRPS.shape)
        for i in range(CRPS.shape[1]):
            loss[:, i] = CRPS[:, i]/self.CRPS_min[i]+RS[:, i]/self.RS_min[i]
        loss = torch.nanmean(loss, axis=0)
        # loss = torch.mean(loss*weights)
        st()
        return loss.mean()


class PhysinformedNet_AR_2D(NeuralNetRegressor):
    
    def __init__(self,  *args, beta, 
                 mean, std, CRPS_min, RS_min,
                 lon_now,
                 lon_5days,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        self.beta = beta
        self.CRPS_min = CRPS_min
        self.RS_min = RS_min
        self.mean = mean
        self.std = std
        self.lon_now = lon_now
        self.lon_5days = lon_5days
        # self.d = d

    def get_loss(self, y_pred, y_true, X, training=False):
        
        # import ipdb;ipdb.set_trace()
        # print('y_true max & min: {} & {}'.format(y_true.max(), y_true.min()))
        # print('y_pred max & min: {} & {}'.format(y_pred.max(), y_pred.min()))
        # print('y_pred shape: {}'.format(y_pred.shape))
        # print(y_pred.shape)
        # st()

        lon_now = y_true[:, 0, 2].numpy()
        lon_5days = y_true[:, 0, 3].numpy()
        # y_true[:, :, 0] = (y_true[:, :, 0] - self.mean)/self.std
        # y_true[:, :, 0] = (y_true[:, :, 0] - self.mean)/self.std
        # y_true[:, :, 1] = (y_true[:, :, 1] - self.mean)/self.std
        # st()
        # y_pred = (y_pred - self.mean)
        y_pred = (y_pred - self.mean)/self.std
        # y_pred = (y_pred - y_pred.mean())/y_pred.std()
        # y_pred = y_pred
        d = y_true[:, :, 1] - y_true[:, :, 0]
        # d = (d - self.mean)/self.std
        d = d.to(self.device)

        N = d.shape[0]
        # sigma = torch.exp(y_pred).squeeze().to(self.device)
        sigma = torch.abs(y_pred).squeeze().to(self.device)
        # sigma = torch.abs(y_pred).squeeze().to(self.device)*self.mean
        # sigma = torch.exp(torch.abs(y_pred)).squeeze().to(self.device)
        # print(sigma.max())
        # st()
        
        x = torch.zeros(sigma.shape)
        # CRPS = torch.zeros(sigma.shape)
        # RS = torch.zeros(sigma.shape)
        loss = 0
        
        # st()
        x = d/(sigma+1e-5)
        x = x/np.sqrt(2)
        x = x.to(self.device)

        CRPS = torch.zeros([x.shape[0], 24]).to(self.device)
        RS = torch.zeros([x.shape[0], 24]).to(self.device)

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
        
        # weights = np.zeros(y_true[:, :, 1])

        # weights = (y_true[:, :, 1] - y_true.min())/(y_true.max()-y_true.min())
        # weights = np.exp(y_true[:, :, 1])
        # weights = weights.to(self.device)
        # st()
        for idx in range(24): ## 0 or 1
            x_t = torch.diagonal(x[:, lon_idx[:, idx]])
            ind = torch.argsort(x_t)
            ind_orig = torch.argsort(ind)+1
            ind_orig = ind_orig.to(self.device)
            CRPS_1 = np.sqrt(2)*x_t*erf(x_t)
            CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x_t**2) 
            CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
            CRPS_3 = CRPS_3.to(self.device)
            # st()
            CRPS[:, idx] = torch.diagonal(sigma[:, lon_idx[:, idx]])*(CRPS_1 + CRPS_2 - CRPS_3)
            # st()
            if torch.isinf(CRPS[i, idx]):
                st() 

            # import ipdb;ipdb.set_trace()
            RS[:, idx] = N*(x_t/N*(erf(x_t)+1) - 
                x_t*(2*ind_orig-1)/N**2 + 
                torch.exp(-x_t**2)/np.sqrt(np.pi)/N)
        
            RS = RS.to(self.device)
            # st()

            # for i in range(lon_idx.shape[0]):
            #     # st()
            #     Y_t = y_true[i, lon_idx[i], 1]
            #     weights = (Y_t - Y_t.min())/(Y_t.max()-Y_t.min())
                
            #     CRPS[i, idx] = weights[idx]*CRPS[i, idx]
            #     RS[i, idx] = weights[idx]*RS[i, idx]
                
            # st()
            weights = torch.exp((torch.diagonal(d[:, lon_idx[:, idx]])-self.mean)/self.std)
            weights = weights.to(self.device)
            loss += torch.nanmean((CRPS[:, idx]/self.CRPS_min\
                +RS[:, idx]/self.RS_min
                )
                *weights
                )
        # st()
        
        if torch.isnan(loss.cpu()).sum() > 0:
            st()

        # print(loss)
        # st()
        return loss/24


def apply_hux_f_model(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 1d upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """

    # r0 = torch.from_numpy(r0).float()
    # rh = torch.from_numpy(rh).float()

    # st()
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial.squeeze()
    # st()

    if add_v_acc:
        # st()
        v_acc = alpha * (v[0, :] * (1 - np.exp(-r0 / rh)))
        v[0, :] = v_acc + v[0, :]


    for i in range(len(dr_vec)):
        for j in range(len(dp_vec) + 1):

            if j == len(dp_vec):  # force periodicity
                v[i + 1, j] = v[i + 1, 0]
                # pass
            else:
                if (omega_rot * dr_vec[i]) / (dp_vec[j] * v[i, j]) > 1:
                    # print(dr_vec[i] - dp_vec[j] * v[i, j] / omega_rot)
                    # print(i, j)  # courant condition
                    pass

                frac1 = (v[i, j + 1] - v[i, j]) / v[i, j]
                frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
                # v[i + 1, j] = frac1 * frac2
                v[i + 1, j] = v[i, j] + frac1 * frac2

    return v


def apply_hux_f_model_change(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 1d upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """

    # r0 = torch.from_numpy(r0).float()
    # rh = torch.from_numpy(rh).float()

    # st()
    v = torch.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial.squeeze()
    # st()

    if add_v_acc:
        # st()
        v_acc = alpha * (v[0, :] * (1 - np.exp(-r0 / rh)))
        v[0, :] = v_acc + v[0, :].clone()

    # st()
    for i in range(len(dr_vec)):
        # st()
        frac1 = v[i, 1:].clone()/v[i, :-1].clone() - 1
        frac2 = (omega_rot * dr_vec[i]) / dp_vec 
        # st()
        v[i + 1, :-1] = v[i, :-1].clone() + frac1 * frac2

        # for j in range(len(dp_vec)):

            # frac1 = (v[i, j + 1].clone() - v[i, j].clone()) / v[i, j].clone()
            # frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
            # v[i + 1, j] = frac1 * frac2
            # v[i + 1, j] = v[i, j].clone() + frac1[j] * frac2[j]

        v[i + 1, 127] = v[i + 1, 0].clone()

    return v


def apply_hux_f_model_np(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 1d upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """

    # r0 = torch.from_numpy(r0).float()
    # rh = torch.from_numpy(rh).float()

    # st()
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial.squeeze()
    # st()

    if add_v_acc:
        # st()
        v_acc = alpha * (v[0, :] * (1 - np.exp(-r0 / rh)))
        v[0, :] = v_acc + v[0, :]

    # st()
    for i in range(len(dr_vec)):
        # st()
        frac1 = v[i, 1:]/v[i, :-1] - 1
        frac2 = (omega_rot * dr_vec[i]) / dp_vec 
        # st()
        v[i + 1, :-1] = v[i, :-1] + frac1 * frac2

        # for j in range(len(dp_vec)):

            # frac1 = (v[i, j + 1].clone() - v[i, j].clone()) / v[i, j].clone()
            # frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
            # v[i + 1, j] = frac1 * frac2
            # v[i + 1, j] = v[i, j].clone() + frac1[j] * frac2[j]

        v[i + 1, 127] = v[i + 1, 0]

    return v


def apply_hux_f_model(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 1d upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial

    if add_v_acc:
        v_acc = alpha * (v[0, :] * (1 - np.exp(-r0 / rh)))
        v[0, :] = v_acc + v[0, :]

    for i in range(len(dr_vec)):
        for j in range(len(dp_vec) + 1):

            if j == len(dp_vec):  # force periodicity
                v[i + 1, j] = v[i + 1, 0]

            else:
                if (omega_rot * dr_vec[i]) / (dp_vec[j] * v[i, j]) > 1:
                    print(dr_vec[i] - dp_vec[j] * v[i, j] / omega_rot)
                    print(i, j)  # courant condition

                frac1 = (v[i, j + 1] - v[i, j]) / v[i, j]
                frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
                v[i + 1, j] = v[i, j] + frac1 * frac2

    return v


def apply_hux_b_model(r_final, dr_vec, dp_vec, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      r0=30 * 695700, omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """ Apply 1d backwards propagation.
    :param r_final: 1d array, initial velocity for backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh:  float, hyper parameter for acceleration (default r=50 rs). units: (km)
    :param add_v_acc: bool, True will add acceleration boost.
    :param r0: float, initial radial location. units = (km).
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """

    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[-1, :] = r_final

    for i in range(len(dr_vec)):
        for j in range(len(dp_vec) + 1):

            if j != len(dp_vec):
                # courant condition
                if (omega_rot * dr_vec[i]) / (dp_vec[j] * v[-(i + 1), j]) > 1:
                    print("CFL violated", dr_vec[i] - dp_vec[j] * v[-(i + 1), j] / omega_rot)
                    raise ValueError('CFL violated')

                frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
            else:
                frac2 = (omega_rot * dr_vec[i]) / dp_vec[0]

            frac1 = (v[-(i + 1), j - 1] - v[-(i + 1), j]) / v[-(i + 1), j]
            v[-(i + 2), j] = v[-(i + 1), j] + frac1 * frac2

    # add acceleration after upwind.
    if add_v_acc:
        v_acc = alpha * (v[0, :] * (1 - np.exp(-r0 / rh)))
        v[0, :] = -v_acc + v[0, :]

    return v


def apply_ab2_b_model(r_final, dr_vec, dp_vec, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      r0=30 * 695700, omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    backwards model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_final: 1d array, initial velocity for backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh:  float, hyper parameter for acceleration (default r=50 rs). units: (km)
    :param add_v_acc: bool, True will add acceleration boost.
    :param r0: float, initial radial location. units = (km).
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """
    # st()
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_final
    # st()
    v[1, :] = v[0, :] - 0.5*q*np.hstack((v[0, 1]-v[0, -1],
                                         v[0, 2:]-v[0, :-2],
                                         v[0, 0]-v[0, -2]))/v[0, :]
    for i in range(2, len(dr_vec)+1):

        dvdphi_j2 = np.hstack((v[i-2, 1]-v[i-2, -1], 
                               v[i-2, 2:]-v[i-2, :-2],
                               v[i-2, 0]-v[i-2, -2])) # periodic BC

        dvdphi_j1 = np.hstack((v[i-1, 1]-v[i-1, -1], 
                               v[i-1, 2:]-v[i-1, :-2],
                               v[i-1, 0]-v[i-1, -2])) # periodic BC
        v[i, :] = v[i-1, :] - 0.5*q*(
            3/2*dvdphi_j1/v[i-1, :] - 0.5*dvdphi_j2/v[i-2, :])
    # st()

    return v


def apply_ab2_f_model(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """
    # st()
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial
    # st()
    v[1, :] = v[0, :] + 0.5*q*np.hstack((v[0, 1]-v[0, -1],
                               v[0, 2:]-v[0, :-2],
                               v[0, 0]-v[0, -2]))/v[0, :]

    # st()
    for i in range(2, len(dr_vec)+1):
        # st()
        dvdphi_j2 = np.hstack((v[i-2, 1]-v[i-2, -1], 
                               v[i-2, 2:]-v[i-2, :-2],
                               v[i-2, 0]-v[i-2, -2])) # periodic BC
        
        dvdphi_j1 = np.hstack((v[i-1, 1]-v[i-1, -1], 
                               v[i-1, 2:]-v[i-1, :-2],
                               v[i-1, 0]-v[i-1, -2])) # periodic BC
        # st()
        v[i, :] = v[i-1, :] + 0.5*q*(
            3/2*dvdphi_j1/v[i-1, :] - 0.5*dvdphi_j2/v[i-2, :])
    # st()
    return v


def apply_ab22_b_model(r_final, dr_vec, dp_vec, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      r0=30 * 695700, omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    backwards model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_final: 1d array, initial velocity for backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh:  float, hyper parameter for acceleration (default r=50 rs). units: (km)
    :param add_v_acc: bool, True will add acceleration boost.
    :param r0: float, initial radial location. units = (km).
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """
    # st()
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1), dtype = 'complex_')  # initialize array vr.
    v_sq = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_final
    v_sq[0, :] = r_final**2
    # st()
    v_sq[1, :] = v_sq[0, :] - q*np.hstack((v[0, 1]-v[0, -1],
                                         v[0, 2:]-v[0, :-2],
                                         v[0, 0]-v[0, -2]))
    v[1, :] = list(map(cmath.sqrt, v_sq[1, :]))

    for i in range(2, len(dr_vec)+1):

        dvdphi_j2 = np.hstack((v[i-2, 1]-v[i-2, -1], 
                               v[i-2, 2:]-v[i-2, :-2],
                               v[i-2, 0]-v[i-2, -2])) # periodic BC

        dvdphi_j1 = np.hstack((v[i-1, 1]-v[i-1, -1], 
                               v[i-1, 2:]-v[i-1, :-2],
                               v[i-1, 0]-v[i-1, -2])) # periodic BC
        
        v_sq[i, :] = v_sq[i-1, :] - q*(3/2*dvdphi_j1 - 0.5*dvdphi_j2)
        v[i, :] = list(map(cmath.sqrt, v_sq[i, :]))
        if np.isnan(v[i, :]).sum() > 0:
            print(i)
            st()

    return v



def apply_ab22_f_model(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """
    # st()
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1), dtype = 'complex_')  # initialize array vr.
    v_sq = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial
    v_sq[0, :] = r_initial**2
    # st()
    v_sq[1, :] = v_sq[0, :] + q*np.hstack((v[0, 1]-v[0, -1],
                                           v[0, 2:]-v[0, :-2],
                                           v[0, 0]-v[0, -2]))
    v[1, :] = list(map(cmath.sqrt, v_sq[1, :]))
    # st()
    for i in range(2, len(dr_vec)+1):
        # st()
        dvdphi_j2 = np.hstack((v[i-2, 1]-v[i-2, -1], 
                               v[i-2, 2:]-v[i-2, :-2],
                               v[i-2, 0]-v[i-2, -2])) # periodic BC
        
        dvdphi_j1 = np.hstack((v[i-1, 1]-v[i-1, -1], 
                               v[i-1, 2:]-v[i-1, :-2],
                               v[i-1, 0]-v[i-1, -2])) # periodic BC
        # st()
        v_sq[i, :] = v_sq[i-1, :] + q*(3/2*dvdphi_j1 - 0.5*dvdphi_j2)
        v[i, :] = list(map(cmath.sqrt, v_sq[i, :]))
        
    # st()
    return np.abs(v)



def apply_ab22_f_model_tensor(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """
    # st()
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = torch.zeros((len(dr_vec) + 1, len(dp_vec) + 1), dtype=torch.cfloat)  # initialize array vr.
    v_sq = torch.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial
    v_sq[0, :] = r_initial**2
    # st()
    v_sq[1, :] = v_sq[0, :] + q*torch.hstack((v[0, 1]-v[0, -1],
                                           v[0, 2:]-v[0, :-2],
                                           v[0, 0]-v[0, -2]))
    v[1, :] = torch.sqrt(v_sq[1, :])
    # st()
    for i in range(2, len(dr_vec)+1):
        # st()
        dvdphi_j2 = torch.hstack((v[i-2, 1]-v[i-2, -1], 
                               v[i-2, 2:]-v[i-2, :-2],
                               v[i-2, 0]-v[i-2, -2])) # periodic BC
        
        dvdphi_j1 = torch.hstack((v[i-1, 1]-v[i-1, -1], 
                               v[i-1, 2:]-v[i-1, :-2],
                               v[i-1, 0]-v[i-1, -2])) # periodic BC
        # st()
        v_sq[i, :] = v_sq[i-1, :] + q*(3/2*dvdphi_j1 - 0.5*dvdphi_j2)
        v[i, :] = v_sq[i, :]
        
    # st()
    return v


def apply_rk42_b_model(r_final, dr_vec, dp_vec, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      r0=30 * 695700, omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    backwards model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_final: 1d array, initial velocity for backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh:  float, hyper parameter for acceleration (default r=50 rs). units: (km)
    :param add_v_acc: bool, True will add acceleration boost.
    :param r0: float, initial radial location. units = (km).
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """
    # st()

    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_final

    idx_low = np.where(v[0] < 100)[0]
    idx_high = np.where(v[0] > 900)[0]
    v[0, idx_low] = 100
    v[0, idx_high] = 900

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



def apply_rk42_f_model(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """

    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial

    idx_low = np.where(v[0] < 100)[0]
    idx_high = np.where(v[0] > 900)[0]
    v[0, idx_low] = 100
    v[0, idx_high] = 900
    # st()
    for i in range(1, len(dr_vec)+1):

        dvdphi_j1 = np.hstack((v[i-1, 1]-v[i-1, -1], 
                               v[i-1, 2:]-v[i-1, :-2],
                               v[i-1, 0]-v[i-1, -2])) # periodic BC
        # dvdphi_j1 = np.log(v[i-1])
        
        k1 = dvdphi_j1/v[i-1]
        k2 = v[i-1] + q*k1/2
        k2 = 0.5 * np.hstack((k2[1] - k2[-1], 
                              k2[2:] - k2[:-2],
                              k2[0] - k2[-2]))/k2 # periodic BC
        
        k3 = v[i-1] + q*k2/2
        k3 = 0.5 * np.hstack((k3[1] - k3[-1], 
                              k3[2:] - k3[:-2],
                              k3[0] - k3[-2]))/k3 # periodic BC

        k4 = v[i-1] + q*k3
        k4 = 0.5 * np.hstack((k4[1] - k4[-1], 
                              k4[2:] - k4[:-2],
                              k4[0] - k4[-2]))/k4 # periodic BC
        
        v[i] = v[i-1] + 1/6*q*(k1+2*k2+2*k3+k4)
        
    return v



def apply_rk42_f_model_tensor(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """
    
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = torch.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial

    # idx_low = np.where(v[0] < 150)[0]
    # idx_high = np.where(v[0] > 900)[0]
    # v[0, idx_low] = 150
    # v[0, idx_high] = 900

    # st()
    for i in range(1, len(dr_vec)+1):

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


def apply_rk42log_b_model(r_final, dr_vec, dp_vec, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      r0=30 * 695700, omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    backwards model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_final: 1d array, initial velocity for backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh:  float, hyper parameter for acceleration (default r=50 rs). units: (km)
    :param add_v_acc: bool, True will add acceleration boost.
    :param r0: float, initial radial location. units = (km).
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """
    # st()

    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_final

    # idx_low = np.where(v[0] < 200)[0]
    # idx_high = np.where(v[0] > 900)[0]
    # v[0, idx_low] = 200
    # v[0, idx_high] = 900

    for i in range(1, len(dr_vec)+1):

        vv = v[i-1]
        idx_nan = np.where((vv<=200) | (vv>=900))[0]
        vv[idx_nan] = np.nan
        vv = fill_nan_nearest_array(vv).squeeze()
        
        # st()
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

        if np.isnan(v).sum() != 0:
            st() 

    # st()

    return v



def apply_rk42log_f_model(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """

    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = r_initial

    idx_low = np.where(v[0] < 200)[0]
    idx_high = np.where(v[0] > 900)[0]
    v[0, idx_low] = 200
    v[0, idx_high] = 900

    # st()
    for i in range(1, len(dr_vec)+1):

        vv = v[i-1]
        # idx_nan = np.where((vv<=200) | (vv>=900))[0]
        # vv[idx_nan] = np.nan
        # vv = fill_nan_nearest_array(vv).squeeze()
    
        lnv = np.log(vv+1)
        
        k1 = 0.5 * np.hstack((lnv[1] - lnv[-1], 
                              lnv[2:] - lnv[:-2],
                              lnv[0] - lnv[-2])) # periodic BC
        k2 = np.log(vv + q*k1/2)
        k2 = 0.5 * np.hstack((k2[1] - k2[-1], 
                              k2[2:] - k2[:-2],
                              k2[0] - k2[-2])) # periodic BC
        
        k3 = np.log(vv + q*k2/2)
        k3 = 0.5 * np.hstack((k3[1] - k3[-1], 
                              k3[2:] - k3[:-2],
                              k3[0] - k3[-2])) # periodic BC

        k4 = np.log(vv + q*k3)
        k4 = 0.5 * np.hstack((k4[1] - k4[-1], 
                              k4[2:] - k4[:-2],
                              k4[0] - k4[-2])) # periodic BC
        
        v[i] = vv + 1/6*q*(k1+2*k2+2*k3+k4)
        
    return v



def apply_rk42log_f_model_tensor(r_initial, dr_vec, dp_vec, r0=10 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """
    
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v = torch.zeros((len(dr_vec) + 1, len(dp_vec) + 1)).to('cuda:'+str(r_initial.get_device()))  # initialize array vr.
    # v = []
    # for i in np.arange(v):
    #     v.append()
    # st()
    v[0, :] = torch.abs(r_initial)

    idx_nan = torch.where((v[0, :]<=150) | (v[0, :]>=900))[0]
    v[0, idx_nan] = float('nan')
    
    v[0, :] = fill_nan_nearest(v[0, :]).squeeze()
    # st()

    # idx_low = torch.where(v[0, :]<=200)[0]
    # idx_high = torch.where(v[0, :]>=900)[0]
    # v[0, idx_low] = 200
    # v[0, idx_high] = 900
    
    # st()
    for i in range(1, len(dr_vec)+1):
            
        # idx_low = torch.where(v[i-1, :]<=200)[0]
        # idx_high = torch.where(v[i-1, :]>=900)[0]
        # v[i-1, idx_low] = 200
        # v[i-1, idx_high] = 900
        
        # st()
        # vv = v[i-1]
        vv = v[i-1].clone()

        # st()
        lnv = torch.log(vv+1)
        k1 = 0.5 * torch.hstack((lnv[1] - lnv[-1], 
                              lnv[2:] - lnv[:-2],
                              lnv[0] - lnv[-2])) # periodic BC
        k2 = torch.log(vv + q*k1/2)
        k2 = 0.5 * torch.hstack((k2[1] - k2[-1], 
                              k2[2:] - k2[:-2],
                              k2[0] - k2[-2])) # periodic BC
        
        k3 = torch.log(vv + q*k2/2)
        k3 = 0.5 * torch.hstack((k3[1] - k3[-1], 
                              k3[2:] - k3[:-2],
                              k3[0] - k3[-2])) # periodic BC

        k4 = torch.log(vv + q*k3)
        k4 = 0.5 * torch.hstack((k4[1] - k4[-1], 
                              k4[2:] - k4[:-2],
                              k4[0] - k4[-2])) # periodic BC
        
        v[i] = vv + 1/6*q*(k1+2*k2+2*k3+k4)
    
    
        # print('vv: {}'.format(vv))
        # print('k1: {}'.format(k1))
        # print('k2: {}'.format(k2))
        # print('k3: {}'.format(k3))
        # print('k4: {}'.format(k4))
        # if torch.isnan(k4).sum() > 0:
        #     st()
        
    return v


def apply_rk42log_f_model_list(r_initial, dr_vec, dp_vec, r0=30 * 695700, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """Apply 2nd order Adams-Bashfort in r, and 2nd order central finite difference in phi 
    upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.
    :param r_initial: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np)
    """
    
    q = dr_vec[0]*omega_rot/dp_vec[0]
    v0 = np.zeros((len(dp_vec) + 1))  # initialize array vr.
    v = []
    # st()
    for i in np.arange(len(dr_vec)+1):
        if i == 0:
            v.append(np.abs(r_initial))
        else:
            v.append(v0)

    # st()
    for i in range(1, len(dr_vec)+1):

        # st()

        vv = v[i-1].clone()

        lnv = np.log(vv+1)
        k1 = 0.5 * np.hstack((lnv[1] - lnv[-1], 
                              lnv[2:] - lnv[:-2],
                              lnv[0] - lnv[-2])) # periodic BC
        k2 = np.log(v[i-1] + q*k1/2+1)
        k2 = 0.5 * np.hstack((k2[1] - k2[-1], 
                              k2[2:] - k2[:-2],
                              k2[0] - k2[-2])) # periodic BC
        
        k3 = np.log(v[i-1] + q*k2/2+1)
        k3 = 0.5 * np.hstack((k3[1] - k3[-1], 
                              k3[2:] - k3[:-2],
                              k3[0] - k3[-2])) # periodic BC

        k4 = np.log(v[i-1] + q*k3+1)
        k4 = 0.5 * np.hstack((k4[1] - k4[-1], 
                              k4[2:] - k4[:-2],
                              k4[0] - k4[-2])) # periodic BC
        
        v[i] = vv + 1/6*q*(k1+2*k2+2*k3+k4)

    # st()
        
    return v