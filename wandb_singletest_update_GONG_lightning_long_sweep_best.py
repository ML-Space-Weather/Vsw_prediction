####################### basic functions ###############
import os
import h5py
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from ipdb import set_trace as st
import pandas as pd

####################### ML functions ###############
import torch
import lightning.pytorch as lp
from wandb_funs_train import * 
from nets import seed_torch

################################## configuration #######################

torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_grad_enabled(True) 

os.environ["WANDB_MODE"] = "online"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

def run(config):

    batch = config['batch']
    max_epochs=config['max_epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    IC = config['IC']
    N_sample = config['N_sample']
    train_per = config['train_per']
    boost_num = config['boost']
    dropout = config['dropout']
    width = config['width']
    mode = config['mode']
    gap = config['gap']
    weight_flag = config['weight_flag']
    train_flag = config['train_flag']
    std_flag = config['std_flag']
    Overwrite = config['Overwrite']
    shuffle_flag = config['Shuffle']

    ######## change the input ####################################
    Inter_data = '/media/faraday/andong/Dataspace/GONG_NN/Data/new_long_data.h5' # This where GONG data will be downloaded

    ######## intermediate data (can be change to other directory) ####################################
    ML_file = '/media/faraday/andong/Dataspace/GONG_NN/Data/long_XY_CNN_WL_'+str(int(IC))+'_'+str(int(N_sample))+'.h5'
    vr_mat = '/media/faraday/andong/Dataspace/GONG_NN/Data/long_rk42log_WL_'+str(int(IC))+'_'+str(N_sample)+'.h5'
    save_mat = '/media/faraday/andong/GONG_NN/long_MF_5days_WL_'+str(int(IC))+'_'+str(int(N_sample))+'.mat'
    save_h5 = '/media/faraday/andong/GONG_NN/long_MF_5days_WL_'+str(int(IC))+'_'+str(int(N_sample))+'.h5'
    initfile = "long_best-checkpoint_"+str(IC)+'_init'

    ######## save the model ####################################
    checkpoint_dir = 'checkpoints'
    checkpoint_file = "long_best-checkpoint_"+str(IC)+'_'+str(N_sample)
    checkpoint_init = checkpoint_dir+'/'+initfile+'_V.ckpt'

    ################################## read data #######################

    init_data = ML_data_read(Inter_data, 
                             N_sample,
                             shuffle_flag)
    
    vr_ori, vr_5days_ori, r_end_5day, r_end, images, date_clu = init_data
    vr_mean = np.mean(vr_5days_ori[:, -1])
    vr_std = np.std(vr_5days_ori[:, -1])
    idx_clu = np.arange(images.shape[0])

    ############## calculate vr_5days  

    # st()
    if ((os.path.exists(vr_mat)==0) | (Overwrite)):

        V2logV_long(init_data, 
            IC, idx_clu, 
            vr_mat)
    
    # # st()
    with h5py.File(vr_mat, 'r') as f:
        
        ############## normalized vr_5days  

        # vr_mean = np.mean(f['vr_5days'][:][:, -1, -1])
        # vr_std = np.std(f['vr_5days'][:][:, -1, -1])
        
        # vr_mean = np.array(f['vr_mean'])
        # vr_std = np.array(f['vr_std'])
        vr = np.array(f['vr'])
        vr_5days = np.array(f['vr_5days'])
        # f.close()

    # st()
    # torch.manual_seed(2023)
    seed_torch(seed=2333)

    # st()
    ############## devide train/test set  
    idx_train, idx_valid, idx_test = data_split(date_clu, idx_clu, 2023)

    # st()
    ############## form ML-ready dataset ######################  

    if ((os.path.exists(ML_file)==0) | (Overwrite)):
        
        # vr = batch_read(vr_mat, 'vr', N_sample)
        # vr_5days = batch_read(vr_mat, 'vr_5days', N_sample)
        
        # set random seed
        data_preprocess(init_data, vr, vr_5days, ML_file)

    # with h5py.File(ML_file, 'r') as f:
    
    #     ############## normalized vr_5days  

    #     Y = np.array(f['Y'])
    #     # f.close()
        
    # st()

    # idx_all = np.hstack((idx_train, idx_valid, idx_test))
    # date_shuffled = date_clu[idx_all]
    # ideally, if GPU training is required, and if cuda is not available, we can raise an exception
    # however, as we want this algorithm to work locally as well (and most users don't have a GPU locally), we will fallback to using a CPU
    # use_cuda = torch.cuda.is_available()
    # print(f"Use cuda: {use_cuda}")
    # load data
    # st()
        
    ############## V model training ################
    pred_Y_test, Y_sel = V_train_filebatch(ML_file,
            idx_train, idx_valid, idx_test,
            vr_mean, vr_std,
            0,
            config, 
            checkpoint_dir,  checkpoint_file,
            checkpoint_init,
            flag=train_flag
            )
    
    # print('first 10 elements of v_init[100] {}'.format(pred_Y_test[100, :10]))
    # st()
    ############# convert solar wind velocity from 10 solar radius to 1 AU
    pred_Y_test = v02vr_5days(pred_Y_test, 
                                r_end, 
                                IC)
    
    # print('first 10 elements of tmp[100] {}'.format(pred_Y_test[100, :10]))
    # st()

    vr_5days_pred = np.array(Y_sel[:, :, -1])
    vr_5days_pred[:, :121] = pred_Y_test[:, :121]

    RMSE_train = np.sqrt(np.nanmean((Y_sel[idx_train, :121, -1] - pred_Y_test[idx_train, :121])**2))
    RMSE_valid = np.sqrt(np.nanmean((Y_sel[idx_valid, :121, -1] - pred_Y_test[idx_valid, :121])**2))
    RMSE_test = np.sqrt(np.nanmean((Y_sel[idx_test, :121, -1] - pred_Y_test[idx_test, :121])**2))
    RMSE_test_per = np.sqrt(np.nanmean((Y_sel[idx_test, :121, -1] - Y_sel[idx_test, :121, 99])**2))
    print('RMSE of the train set is {}'.format(RMSE_train))
    print('RMSE of the vali set is {}'.format(RMSE_valid))
    print('RMSE of the test set is {}'.format(RMSE_test))
    print('RMSE of the per set is {}'.format(RMSE_test_per))

    ############## save all variables into a .h5 file and predictions ########

    # st()
    save_model_result(save_h5, 
                      date_clu,
                      r_end,
                      r_end_5day,
                      Y_sel[:, :, 99],
                      Y_sel[:, :, -1],
                      vr_5days_pred,
                      idx_train,
                      idx_test,
                      idx_clu,
                      )

    ############## plot and save figures 

    # st()
    
    for i in idx_test[::gap]:
        plot_vr5days_update(date_clu, 
                 idx_clu, 
                 i,
                 IC,
                 vr_5days_pred[:, :121],
                #  pred_dY_test[:, :121],
                 Y_sel[:, :, 99],
                 Y_sel[:, :, -1],
                 )


############## configuration ########


sweep_config = {
    'batch': 10,
    'max_epochs': 30,
    'lr': 3e-4,
    'dropout': 0.8,
    'width': 3,
    # 'Optimize': 'sgd',
    'Optimize': 'adam',
    # 'Optimize': 'rmsprop',

    'thres_up': 900,  # Needs to be a list of values
    'thres_low': 200,  # Needs to be a list of values
    'weight_decay': 1e-4,  # Needs to be a list of values
    'IC': 10,  # Needs to be a list of values
    'N_sample': 101000,  # Needs to be a list of values
    'train_per': 6,  # Needs to be a list of values
    'dev_num': 8,  # Needs to be a list of values
    'ratio': 1.5,  # Needs to be a list of values
    'boost': 3,  # Needs to be a list of values
    'mode': 3,  # Needs to be a list of values
    'gap': 10,  # Needs to be a list of values
    'weight_flag': False,  # Needs to be a list of values
    'train_flag': False,  # Needs to be a list of values
    'std_flag': False,  # Needs to be a list of values
    'Overwrite': False,  # Needs to be a list of values
    'Shuffle': False  # Needs to be a list of values
    }

run(sweep_config)

'''
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'valid_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch': {'values': [5, 10, 30]},
        'max_epochs': {'values': [30]},
        'lr': {'values': [3e-4, 3e-3, 3e-2]},
        'dropout': {'values': [0.8]},
        'width': {'values': [3]},
        'Optimize': {'values': ['sgd', 'adam', 'rmsprop']},
        'ratio': {'values': [1.1, 1.2, 1.3, 1.4, 1.5]},  # Needs to be a list of values
        # 'Optimize': {'values': ['adam']},

        'thres_up': {'values': [900]},  # Needs to be a list of values
        'thres_low': {'values': [200]},  # Needs to be a list of values
        'weight_decay': {'values': [1e-4]},  # Needs to be a list of values
        'IC': {'values': [10]},  # Needs to be a list of values
        'N_sample': {'values': [10600]},  # Needs to be a list of values
        'train_per': {'values': [5]},  # Needs to be a list of values
        'dev_num': {'values': [8]},  # Needs to be a list of values
        'boost': {'values': [3]},  # Needs to be a list of values
        'mode': {'values': [3]},  # Needs to be a list of values
        'gap': {'values': [100]},  # Needs to be a list of values
        'weight_flag': {'values': [False]},  # Needs to be a list of values
        'train_flag': {'values': [True]},  # Needs to be a list of values
        'std_flag': {'values': [False]},  # Needs to be a list of values
        'Overwrite': {'values': [False]},  # Needs to be a list of values
        'Shuffle': {'values': [False]}  # Needs to be a list of values
    }
}


def main():
    wandb.init(project="GONG_lightning")
    run(wandb.config)
    
sweep_id = wandb.sweep(sweep_config, project="GONG_lightning")
wandb.agent(sweep_id, main, count=5)
wandb.alert(title='Completed', text='Sweep has been completed')

run(sweep_config)

'''
