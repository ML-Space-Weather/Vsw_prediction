# GONG_NN - Solar wind velocity forecast

Notebooks for a project with aim of predicting the Solar wind velocity ($V_{sw}$) 5 days ahead using the PIFNN method.

If you have any questions w.r.t the code, please contact andong.hu@colorado.edu or enrico.camporeale@colorado.edu.

## Overview

This project is conducted primarily in python that present theory alongside application. 

Two folders 'Figs' (for saving figures) and 'checkpoints' (for trained model) should be constructed in advance. The trained model is too large to upload to git. Hence, it needs to be downloaded and saved in Folder 'checkpoints'

To be able to run the python/notebooks themselves, it is most convenient to use the supplied conda environment file ([environment.yml](environment.yml)) to create the corresponding conda environment as described in Section 'Tutorial' or [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

'preprocess_LiveWind_pred.py' is used to create the input data, including GONG image and solar wind data. The configurations need to be customized in advance. 'wandb_single_update_GONG_lightning_sweep_v0vr_pred.py' is an prediction to generate the final predictions from the model.

### Python modules

The following files contain useful functions and classes used throughout the notebooks.

- [Dataset Preparation](preprocess_LiveWind_pred.py) : find and save ACE data corresponding to GONG map to form a ML-ready dataset.
- [main function](wandb_single_update_GONG_lightning_sweep_v0vr_pred.py) : main function, including forecasting and plotting.
- [Functions](wandb_funs.py) : Various functions ranging from model training to plotting.
- [networks](Model.py) : arritectures used for training.

The notebooks are used to display the outputs.

# Tutorial

## environment install 

    conda env create -f environment.yml
    
Then activate virtue environment by

    conda activate GONG_pred

## Create the input files

    python3 preprocess_LiveWind_pred.py

## end-to-end predictions

    python3 wandb_single_update_GONG_lightning_sweep_v0vr_pred.py 


