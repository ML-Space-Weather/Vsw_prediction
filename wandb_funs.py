import torch
import numpy as np
import h5py
from tqdm import tqdm
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.dates as mdates

import lightning.pytorch as lp
from Model import V_FNO_DDP

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)



def data_split(date_ACE, idx_clu, test_year):

    # st()
    idx_test = np.where((date_ACE[idx_clu, 0] >= test_year))[0]
    # idx_test = np.where((date_ACE[idx_clu, 0] == 2015))[0]
    # idx_test = np.where((date_ACE[idx_clu, 0] == 2015) & (date_ACE[idx_clu, 1] >= 10))[0]
    idx_valid = np.where((date_ACE[idx_clu, 0] == test_year-2) | (date_ACE[idx_clu, 0] == test_year-1))[0]
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

    ############# boundary angle
    p = np.linspace(0, 360, 129)[:-1]
    dp_vec = (p[1:]-p[0:-1])/180*np.pi
    v0 = np.zeros(vr[:, idx_clu].shape)
    vr_tmp = np.zeros([vr[:, idx_clu].shape[1], 128, 100])
    v0_5days = np.zeros(vr[:, idx_clu].shape)
    vr_5days_tmp = np.zeros([vr[:, idx_clu].shape[1], 128, 100])

    # st()
    for i, idx in enumerate(tqdm(idx_clu)):
        r_vector=np.linspace(695700*IC,r_end[i], 100) # solve the backward propagation all the way to 1 solar radius
        dr_vec = r_vector[1:]-r_vector[0:-1]
        v0[:, i] = apply_rk42log_b_model(vr[:, idx], dr_vec, dp_vec, alpha=0.15,
                        rh=r_end[i], add_v_acc=True,
                        r0=695700*IC, omega_rot=(2 * np.pi) / (25.38 * 86400))[-1]

        vr_tmp[i] = apply_rk42log_f_model(v0[:, i], dr_vec, dp_vec,
                                r0=695700*IC).T

        v0_5days[:, i] = apply_rk42log_b_model(vr_5days[:, idx], dr_vec, dp_vec, alpha=0.15,
                        rh=r_end_5day[i], add_v_acc=True,
                        r0=695700*IC, omega_rot=(2 * np.pi) / (25.38 * 86400))[-1]

        vr_5days_tmp[i] = apply_rk42log_f_model(v0_5days[:, i], dr_vec, dp_vec,
                                r0=695700*IC).T
        
    with h5py.File(vr_mat, 'w') as f:
        
        f.create_dataset('vr', data=vr_tmp)
        f.create_dataset('vr_5days', data=vr_5days_tmp)
        f.close()

############## form ML-ready dataset ######################  

def GONG_read_all(ML_file):

    with h5py.File(ML_file, 'r') as f:
        X = np.array(f['X'])
        Y = np.array(f['Y'])

        f.close()
    
    return X, Y


def V_loss_update(y_pred, 
               y_true, 
               vr_mean,
               vr_std,
               IC,
               ratio,
               weight_flag=True
               ):
    
    device = 'cuda:'+str(y_pred.get_device())
    yr = y_true[:, :, :100]
    r_end = y_true[:, 0, 100] # r_end
    yr_5days = y_true[:, :, 103:] # vr_5days

    RMSE = torch.tensor(0).float().to(device)
    p = torch.linspace(0, 360, 129)[:-1]
    # tmp_clu = []
    for i in range(r_end.shape[0]):
        tmp = torch.zeros(y_pred[i].shape).to(device)
        tmp[:] = np.nan
        v_init = torch.exp(y_pred[i].squeeze())*yr[i, :, 0].squeeze()
        if weight_flag:
        # st()
            
            weights = (5+5*torch.tanh((yr_5days[i, :24, -1]-350)/150))
            RMSE_vr = yr_5days[i, :24, -1] - v_init[:24]
            RMSE += torch.nanmean(weights/5*(RMSE_vr**2)) 

    return RMSE/(i+1)


class GONG_Model(lp.LightningModule):

    def __init__(self, 
                 lr,
                 weight_decay,
                 loss_func,
                 vr_mean,
                 vr_std,
                 dropout,
                 width,
                 mode,
                 IC,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.model = V_FNO_DDP(dropout,
                  width,
                  mode,
                  vr_mean,
                  vr_std,
                  hidden_size=16,
                  num_layers=2,
                  outputs=128,
                  )
        self.lr = lr
        self.momentum = 0.3
        self.weight_decay = weight_decay
        self.loss_func = loss_func
        self.vr_mean = vr_mean
        self.vr_std = vr_std
        self.IC = IC

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std, 
                              self.IC,
                              2,
                              True,
                              )

        self.log("train_loss", torch.sqrt(loss), 
                 sync_dist=True, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True)

        return torch.sqrt(loss)

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std,
                              self.IC,
                              2,
                              True
                              )
        self.log('valid_loss', torch.sqrt(loss), sync_dist=True)
        return torch.sqrt(loss)

    def test_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y,
                              self.vr_mean,
                              self.vr_std,
                              self.IC,
                              2,
                              weight_flag=True
                              )
        self.log('test_loss', torch.sqrt(loss), sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                momentum=self.momentum
                                )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=5, 
                                                    gamma=0.1)

        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler, 
                'monitor': 'train_loss'}


def fill_nan_nearest_array(input_array):

    if np.isnan(input_array[0]):
        input_array[0] = np.nanmean(input_array)
    nan_mask = np.isnan(input_array)

    input_array[nan_mask] = np.interp(np.flatnonzero(nan_mask), 
                            np.flatnonzero(~nan_mask), 
                            input_array[~nan_mask]
                            )

    return input_array


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

    for i in range(1, len(dr_vec)+1):

        vv = v[i-1]
        idx_nan = np.where((vv<=200) | (vv>=900))[0]
        vv[idx_nan] = np.nan
        vv = fill_nan_nearest_array(vv).squeeze()
        
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


def V_train_pred(X, Y, iter, checkpoint_dir, checkpoint_file):

    V_checkpoint_name = checkpoint_dir+'/'+checkpoint_file+'_'+str(iter)+'_V.ckpt'    
    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    # import ipdb; ipdb.set_trace()
    best_model = GONG_Model.load_from_checkpoint(V_checkpoint_name).to('cpu') #.to('cuda')
    pred_Y_test = best_model.forward(X)
    return np.exp(pred_Y_test.detach().numpy())*Y[:, :, 0].numpy()



def plot_vr5days_update(date_ACE, 
                 idx_clu, 
                 i,
                 IC,
                 vr_5days_test_pred,
                 vr,
                 vr_5days
                 ):


    date = date_ACE[idx_clu][i].astype(int)
    date = dt.datetime(date[0], date[1], date[2], date[3], date[4])
    print('by {}, {}th max predictions are {}'.format(date, i, np.abs(vr_5days_test_pred[i]).max()))

    time_points = [date + dt.timedelta(hours=i*5) for i in range(24)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    # st()
    ax.plot(time_points, vr_5days_test_pred[i][::-1], 'r-', label='vr_5days_pred')
    ax.plot(time_points, vr[i, :24, -1][::-1], 'g-', label='vr')
    ax.plot(time_points, vr_5days[i, :24, -1][::-1], 'k-', label='vr_5days')
    # Set x-axis major formatter to display short time format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    # Rotate x-axis labels for better readability
    fig.autofmt_xdate()
    ax.legend()
    ax.set_ylabel('$V_{sw}$')
    ax.set_title(date)
    
    fig.savefig('Figs/Vr_example_'+str(IC)+'_'+str(i)+'.jpg', dpi=300)
    plt.close()
    
