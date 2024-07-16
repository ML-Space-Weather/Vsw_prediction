####################### basic functions ###############
import os
import h5py
import numpy as np

####################### ML functions ###############
import torch
from wandb_funs import V2logV_all, GONG_read_all
from wandb_funs import V_train_pred, plot_vr5days_update

################################## configuration #######################

torch.set_float32_matmul_precision('medium')
torch.set_grad_enabled(True) 

def run(config):

    IC = config['IC']
    N_sample = config['N_sample']
    gap = config['gap']
    Overwrite = config['Overwrite']

    ######## change the input ####################################
    Inter_data = '/media/faraday/andong/Dataspace/GONG_NN/Data/new_data.h5' # This where GONG data will be downloaded

    ######## intermediate data (can be change to other directory) ####################################
    ML_file = '/media/faraday/andong/Dataspace/GONG_NN/Data/XY_CNN_WL_'+str(int(IC))+'_'+str(int(N_sample))+'.h5'
    vr_mat = '/media/faraday/andong/Dataspace/GONG_NN/Data/rk42log_WL_'+str(int(IC))+'_'+str(N_sample)+'.h5'
    save_h5 = '/media/faraday/andong/GONG_NN/RK4_MF_5days_WL_'+str(int(IC))+'_'+str(int(N_sample))+'.h5'

    ######## save the model ####################################
    checkpoint_dir = 'checkpoints'
    checkpoint_file = "best-checkpoint_"+str(IC)+'_'+str(N_sample)

    ################################## read data #######################

    with h5py.File(Inter_data, 'r') as f:
        vr_5days = np.array(f['Vr_5days'][:N_sample]).T
        vr = np.array(f['Vr'][:N_sample]).T
        r_end_5day = np.array(f['r_5days'][:N_sample])*1.496e8
        r_end = np.array(f['r'][:N_sample])*1.496e8
        HC_lon_now = np.array(f['HC_lon_now'][:N_sample])
        HC_lon_5days = np.array(f['HC_lon_5days'][:N_sample])
        images = np.array(f['GONG'][:N_sample])
        date_clu = np.array(f['Time'][:N_sample])
        f.close()

    ############## devide train/test set  
    idx_clu = np.arange(date_clu.shape[0])

    ############## calculate vr_5days  

    if (os.path.exists(vr_mat)==0) | (Overwrite):

        V2logV_all(vr, vr_5days, 
            r_end, r_end_5day, 
            IC, idx_clu, 
            vr_mat)
    
    # st()
    with h5py.File(vr_mat, 'r') as f:
        vr = np.array(f['vr'])
        vr_5days = np.array(f['vr_5days'])
        f.close()

    ############## normalized vr_5days  

    # set random seed
    torch.manual_seed(2023)

    ############## form ML-ready dataset ######################  

    if (os.path.exists(ML_file)==0) | (Overwrite):

        ############## form Y by all variables required by training process
        Y = np.vstack([np.swapaxes(vr, 0, 2),
                    np.expand_dims(np.tile(r_end_5day, (128, 1)), axis=0),
                    np.expand_dims(np.tile(HC_lon_now, (128, 1)), axis=0),
                    np.expand_dims(np.tile(HC_lon_5days, (128, 1)), axis=0),
                    np.swapaxes(vr_5days, 0, 2), 
                    ]).T 

        # st()
        ############## normalized X 
        X = (images - images.mean())/images.std()
        
        ############## reformat v0 for adding it to X
        y0 = vr[:, :, 0]
        y0 = np.vstack([y0.T, np.zeros([2, vr[:, :, 0].shape[0]])])
        X = np.vstack([np.swapaxes(X, 0, 2), np.expand_dims(y0, axis=0)])
        X = np.swapaxes(X, 0, 2)

        ############## save a ML-ready dataset (X and Y)
        with h5py.File(ML_file, 'w') as f:
            f.create_dataset('X', data=X)
            f.create_dataset('Y', data=Y)
            f.close()

    else:
        X, Y = GONG_read_all(ML_file)

    # ideally, if GPU training is required, and if cuda is not available, we can raise an exception
    # however, as we want this algorithm to work locally as well (and most users don't have a GPU locally), we will fallback to using a CPU
    # use_cuda = torch.cuda.is_available()
    # print(f"Use cuda: {use_cuda}")
    # load data
    ############## V model training ################
    pred_Y_test = V_train_pred(X, Y, 0,
            checkpoint_dir,  checkpoint_file,
            )
    
    vr_5days_pred = np.array(vr_5days[:, :, -1])
    vr_5days_pred[:, :24] = pred_Y_test[:, :24]

    ############## save all variables into a .h5 file and predictions ########
    with h5py.File(save_h5, 'w') as f:
        f.create_dataset("HC_lon_5days", data=HC_lon_5days)
        f.create_dataset("HC_lon_now", data=HC_lon_now)
        f.create_dataset("r", data=r_end)
        f.create_dataset("r_5days", data=r_end_5day)
        f.create_dataset("vr", data=vr)
        f.create_dataset("vr_5days", data=vr_5days)
        f.create_dataset("vr_5days_train_final", data=vr_5days_pred)
        f.close()

    ############## plot and save figures 

    for i in np.arange(X.shape[0])[::gap]:
        plot_vr5days_update(date_clu, 
                 idx_clu, 
                 i,
                 IC,
                 vr_5days_pred[:, :24],
                 vr,
                 vr_5days,
                 )


############## configuration ########

sweep_config = {
    'IC': 10,
    'N_sample': 1000,
    'width': 3,
    'mode': 3,
    'gap': 100,
    'weight_flag':False,
    'Overwrite': False,
    }
    
run(sweep_config)
