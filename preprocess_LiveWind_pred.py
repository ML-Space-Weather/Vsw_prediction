# This is the Python Notebook that creates the pre-processed datasets to be used in the LiveWind model
# 
# The Notebook is divided in the following sections:
# 
# * Download ACE data
# * Download GONG data
# * Transform coordinate position of ACE from GSE to Heliospheric Carrington coordinates
# * 

import os
import subprocess
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import astropy.units as u
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from datetime import datetime
from astropy.io import fits
from ipdb import set_trace as st
from tqdm import tqdm
import concurrent.futures
from multiprocessing import cpu_count, Pool
import h5py


download_ACE = False
download_GONG = False
download_AIA = True
transform_ACE_coordinate = False
Synchnization = False

# Define start and end time
end_time = datetime(2024,6,1)
start_time = datetime(2010, 1, 1)

# Define directories (customize the directories)
current_dir = os.getcwd()
data_dir = '/media/faraday/andong/Dataspace/GONG_NN/Data' # This is where auxiliary data files will be saved
ACE_data_dir  = '/media/faraday/andong/Dataspace/GONG_NN/Data/ACE_data' # This is where ACE data will be downloaded
GONG_data_dir = '/media/faraday/andong/Dataspace/GONG_NN/Data/GONG_data' # This where GONG data will be downloaded
AIA_data_dir = '/media/faraday/andong/Dataspace/GONG_NN/Data/AIA_data' # This where GONG data will be downloaded
Inter_data = '/media/faraday/andong/Dataspace/GONG_NN/Data/new_data.h5' # This where GONG data will be downloaded

# Download ACE dataset (might take a while)

def main():

    # Auxiliary routines
    def pol2cart(phi,rho):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(phi, rho)

    if download_ACE:

        print('START downloading ACE...') 
        os.makedirs(ACE_data_dir, exist_ok=True)
        os.chdir(ACE_data_dir)

        # Iterate over months from start_time to end_time
        t = start_time -relativedelta(months = 1) ## need a bit more ACE data than GONG images
        while t <= end_time:
            yr = t.year
            mo = t.month
        
            st1 = f"{yr}{mo:02d}"
            
            # Construct the command
            command = f"wget -r -nc --random-wait --no-parent -e robots=off --no-directories -A '{st1}*_ace_swepam_1h.txt' https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/"
            
            # Execute the command
            subprocess.run(command, shell=True)
                
            # Move to the next month
            t += relativedelta(months=1)
        
        # Change back to the original directory
        os.chdir(current_dir)
        print('END downloading ACE...') 


    # Download GONG data (might take a while)

    if download_GONG:

        print('Start downloading GONG...') 

        # Define directories
        
        os.makedirs(GONG_data_dir, exist_ok=True)
        os.chdir(GONG_data_dir)
        
        # Define start and end time
        end_time = datetime(2024,6,1)
        start_time = datetime(2010, 1, 1)
        
        # Iterate over months from start_time to end_time
        t = start_time
        while t <= end_time:
            yr = t.year
            mo = t.month
        
            st1 = f"{yr}{mo:02d}"
            
            # Construct the command
            command = f"wget -r -nc --random-wait --no-parent -e robots=off --no-directories -A '*.fits.gz' https://gong.nso.edu/archive/oQR/zqs/{st1}"
            # import ipdb;ipdb.set_trace()
            # Execute the command
            subprocess.run(command, shell=True)
                
            # Move to the next month
            t += relativedelta(months=1)
        
        # Gunzip all the .gz files
        command = "find . ! -name . -prune -name '*.gz' -exec gunzip {} +"
        subprocess.run(command, shell=True)
        
        # Change back to the original directory
        os.chdir(current_dir)
        print('END downloading GONG...') 

    # st()

    if transform_ACE_coordinate:

        print('Start transform_ACE_coordinate...') 

        # Download ACE position in GSE coordinates
        os.chdir(data_dir)
        
        subprocess.run("wget https://izw1.caltech.edu/ACE/ASC/DATA/pos_att/ACE_GSE_position.txt", shell=True)
        # subprocess.run("wget https://izw1.caltech.edu/ACE/ASC/DATA/pos_att/ACE_GSE_position.txt", shell=True)
        df = pd.read_csv("ACE_GSE_position.txt", sep="\t")

        lon_helio=np.zeros(len(df))
        lat_helio=np.zeros(len(df))
        r_helio=np.zeros(len(df))
        Earth_lon=np.zeros(len(df))
        Earth_lat=np.zeros(len(df))

        for i in tqdm(range(0,len(df))):  # For some reason this takes some time!
            obstime=datetime.strftime(datetime.strptime(np.array2string(df.loc[i,"Year"])+np.array2string(df.loc[i,"DOY"]), "%Y%j"),"%Y-%m-%d")
            c=SkyCoord(x=df.loc[i,"GSE_X(km)"]*u.km, y=df.loc[i,"GSE_y(km)"]*u.km, z=df.loc[i,"GSE_z(km)"]*u.km, representation_type='cartesian',obstime=obstime,frame=frames.GeocentricSolarEcliptic)
            h=c.transform_to(frames.HeliographicCarrington(observer='sun'))
            lon_helio[i]=h.lon/u.deg
            lat_helio[i]=h.lat/u.deg
            r_helio[i]=h.radius.to(u.AU)/u.AU
        
            c=SkyCoord(x=0*u.km, y=0*u.km, z=0*u.km, representation_type='cartesian',obstime=obstime,frame=frames.GeocentricSolarEcliptic)
            h=c.transform_to(frames.HeliographicCarrington(observer='sun'))
            Earth_lon[i]=h.lon/u.deg
            Earth_lat[i]=h.lat/u.deg

        df['HC_lon'] = np.array(lon_helio)
        df['HC_lat'] = np.array(lat_helio)
        df['HC_radius'] = np.array(r_helio)
        df['Earth_lon'] = np.array(Earth_lon)
        df['Earth_lat'] = np.array(Earth_lat)
        df['date'] = pd.to_datetime(df.apply(lambda row: datetime.strptime(f"{row['Year']} {row['DOY']}", "%Y.0 %j.0"), axis=1))
        df=df[['date','HC_lon','HC_lat','HC_radius','Earth_lon','Earth_lat']]

        df.head()
        filename = os.path.join(data_dir,'ACE_GSE_HC_position.txt')
        df.to_csv(filename, index=False)
        print('END transform_ACE_coordinate...') 
        print('File ACE_GSE_HC_position.txt created')


        # Interpolate HC coordinate from daily to hourly cadence


        print('Start Interpolate HC coordinate from daily to hourly cadence') 

        # Change to the directory containing data
        os.chdir(data_dir)

        # Read ACE position data
        ace_pos = pd.read_csv('ACE_GSE_HC_position.txt')
        ace_pos['date'] = pd.to_datetime(ace_pos['date'])
        ace_pos.head()

        # Change to the directory containing ACE data
        os.chdir(ACE_data_dir)

        # Read all .txt files
        file_list = [file for file in os.listdir() if file.endswith('.txt')]
        T_list = []

        for file in file_list:
            tmp = pd.read_csv(file, header=16, sep = "\s+|\t+|\s+\t+|\t+\s+", engine='python')
            T_list.append(tmp.iloc[1:, :])

        T = pd.concat(T_list, ignore_index=True)
        T=T.iloc[:,0:-1]
        T.columns=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Day.1', 'S', 'Density', 'Speed','Temperature']

        T['HHMM'] = T['HHMM'].astype(int) / 100
        time = pd.to_datetime(T[['YR', 'MO', 'DA', 'HHMM']].astype(int).rename(columns={'YR': 'year', 'MO': 'month', 'DA': 'day', 'HHMM': 'hour'}))
        T = pd.DataFrame({'Time': time, 'Speed': T['Speed']})
        T = T.sort_values(by='Time',ignore_index=True)    
        T.head()

        start_time_speed = T.Time.iloc[0]
        end_time_speed = T.Time.iloc[-1]

        print('Matching Speed with position for all hourly timestamps between',start_time_speed, ' and ', end_time_speed)


        # Interpolate HC_lon and HC_lat to hourly cadance

        X, Y = pol2cart(np.deg2rad(ace_pos['HC_lon']), ace_pos['HC_radius'])
        ace_pos['X']=X
        ace_pos['Y']=Y
        ace_pos = ace_pos.drop(ace_pos[ace_pos.date<start_time_speed].index)
        ace_pos = ace_pos.drop(ace_pos[ace_pos.date>end_time_speed].index)
        time_pos = ace_pos['date']
        time_pos_hr = pd.date_range(start=time_pos.iloc[0], end=time_pos.iloc[-1], freq='h')
        ace = ace_pos[['date','X','Y','HC_lat']].reset_index(drop=True)

        ace_pos_hr = ace.set_index('date').reindex(time_pos_hr).interpolate().reset_index()
        ace_pos_hr = ace_pos_hr.rename(columns={"index": "date"})

        HC_lon_hr, HC_radius_hr = cart2pol(ace_pos_hr['X'],ace_pos_hr['Y']) 
        HC_lon_hr = np.rad2deg(HC_lon_hr)

        f=np.where(HC_lon_hr<0)
        HC_lon_hr.iloc[f]=HC_lon_hr.iloc[f]+360

        ace_pos_hr['HC_lon']=HC_lon_hr
        ace_pos_hr['HC_radius']=HC_radius_hr

        ace_pos_hr = ace_pos_hr.drop(ace_pos_hr[ace_pos_hr.date<start_time_speed].index)
        ace_pos_hr = ace_pos_hr.drop(ace_pos_hr[ace_pos_hr.date>end_time_speed].index)
        ace_pos_hr = ace_pos_hr.drop(columns=['X', 'Y']).reset_index(drop=True)

        ace_pos_hr.head()


        # Match ACE speed with position and save to file

        ACE_speed_lon_lat_hr = ace_pos_hr.set_index('date').join(T.set_index('Time')).reset_index()
        ACE_speed_lon_lat_hr = ACE_speed_lon_lat_hr.rename(columns={"index": "Time"})


        # Save file ACE_speed_lon_lat_hr.txt
        filename = os.path.join(data_dir,'ACE_speed_lon_lat_hr.txt')
        ACE_speed_lon_lat_hr.to_csv(filename, index=False)
        ACE_speed_lon_lat_hr.head()

        print('End Interpolate HC coordinate from daily to hourly cadence') 
        print('File ACE_speed_lon_lat_hr.txt created') 


    # for each GONG image do the following:
    # * find the timestamp of the image T_gong
    # * crop the image to the usable 180 degrees
    # * find two corresponding vr profile:
    #  1) one that ends at time T_gong and goes back in time until HC_lon covers a full 360 degree angle
    #  2) another one that ends at time T_gong + 5 days

    # Read ace file 

    filename = os.path.join(data_dir,'ACE_speed_lon_lat_hr.txt')
    ace = pd.read_csv(filename)

    # Create longitude grid and define nan_label
    long_grid = np.linspace(0, 360, 129)[:-1]
    nan_label = -9999.9

    # Create output directory for cropped GONG data
    output_dir = os.path.join(data_dir,'GONG_data_cropped')
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(GONG_data_dir)

    # List all .fits files
    files = [file for file in os.listdir() if file.endswith('.fits')]

    ind = 0
    Time = []

    new_V_5days = []
    mean_R_5days = []
    HC_lon_now = []
    HC_lon_5days = []
    GONG_image = []

    new_V = []
    mean_R = []

    print('Start matching ACE with GONG')
    if Synchnization:
        for ii, filename in enumerate(tqdm(files)):
            time_GONG = pd.to_datetime(datetime.strptime(filename[5:16], '%y%m%d''t''%H%M') )
            # st()
            if time_GONG < pd.to_datetime(ace['Time'][0]) + timedelta(days=27):
                continue
            elif time_GONG < datetime(2010, 1, 1):
                continue
            elif time_GONG > datetime(2024, 6, 1):
                continue
            try:
                end_index = ace.index[pd.to_datetime(ace['Time'])> time_GONG + timedelta(days=5)][0]
                now_index = ace.index[pd.to_datetime(ace['Time'])<= time_GONG][-1]
            
                final_long = ace.loc[end_index, 'HC_lon']
                start_index = ace.index[pd.to_datetime(ace['Time'])<= time_GONG - timedelta(days=30)][-1]

                # st()
                tmp=np.unwrap(ace['HC_lon'].iloc[start_index:end_index+1]/180*np.pi)
                start_index += np.where(tmp>tmp[-1]+2*np.pi)[0][-1]+1  

                # st()
                V = ace.loc[start_index:end_index+1, 'Speed'].values
                ace_lon = ace.loc[start_index:end_index+1, 'HC_lon'].values
                ace_lat = ace.loc[start_index:end_index+1, 'HC_lat'].values
                R = ace.loc[start_index:end_index+1, 'HC_radius'].values
            
                # Remove nans
                valid_indices = V != nan_label
                V = V[valid_indices]
                ace_lon = ace_lon[valid_indices]
                ace_lat = ace_lat[valid_indices]
                R = R[valid_indices]
            
                # Interpolate the speed data
                sorted_indices = np.argsort(ace_lon)
                V = V[sorted_indices]
                ace_lon = ace_lon[sorted_indices]
                
                unique_indices = np.unique(ace_lon, return_index=True)[1]
                V = V[unique_indices]
                ace_lon = ace_lon[unique_indices]
            
                V = np.concatenate([[V[-1]], V, [V[0]]])
                ace_lon = np.concatenate([[ace_lon[-1] - 360], ace_lon, [ace_lon[0] + 360]])

                shift = np.where(long_grid <= final_long)[0][-1]
                # shift = np.where(long_grid <= final_long)[0][-1] - 1
                interp_V = np.roll(np.interp(long_grid, ace_lon, V), -shift)
                
                new_V_5days.append(interp_V)
            
                mean_R_5days.append(np.mean(R))
                HC_lon_now.append(ace.loc[now_index, 'HC_lon'])
                HC_lon_5days.append(final_long)

                # Process profile ending at time_GONG
                end_index = now_index
                final_long = ace.loc[end_index, 'HC_lon']
                
                # st()
                start_index = ace.index[pd.to_datetime(ace['Time'])<= time_GONG - timedelta(days=30)][-1]

                tmp=np.unwrap(ace['HC_lon'].iloc[start_index:end_index+1]/180*np.pi)
                start_index += np.where(tmp>tmp[-1]+2*np.pi)[0][-1]+1      

                V = ace.loc[start_index:end_index+1, 'Speed'].values
                ace_lon = ace.loc[start_index:end_index+1, 'HC_lon'].values
                R = ace.loc[start_index:end_index+1, 'HC_radius'].values
                valid_indices = V != nan_label 
                V = V[valid_indices]
                ace_lon = ace_lon[valid_indices]
            
                sorted_indices = np.argsort(ace_lon)
                V = V[sorted_indices]
                ace_lon = ace_lon[sorted_indices]
            
                unique_indices = np.unique(ace_lon, return_index=True)[1]
                V = V[unique_indices]
                ace_lon = ace_lon[unique_indices]
            
                V = np.concatenate([[V[-1]], V, [V[0]]])
                ace_lon = np.concatenate([[ace_lon[-1] - 360], ace_lon, [ace_lon[0] + 360]])
            
                interp_V = np.roll(np.interp(long_grid, ace_lon, V), -shift)
            
                data = fits.getdata(filename)
                data = data[24:154, 0:180].astype(np.float32)  # Reduce from double to single

                new_V.append(interp_V)
                mean_R.append(np.mean(R))
                Time.append(time_GONG)
                GONG_image.append(data)

                dd = new_V[-1] - new_V_5days[-1]
                if dd[30] > 0:
                    st() 
                # st()
                # fits.writeto(os.path.join(output_dir, filename), data, overwrite=True)
                ind += 1

            except Exception:
                print('Failed on GONG file: ',filename)  
                # st()
                pass     

        print('End matching ACE with GONG')

        ######################## date ################  

        # date_idx = np.arange(0, len(Time))
        date_clu = []
        for i, date_tt in tqdm(enumerate(Time)):
                    
            t = [date_tt.year,
                date_tt.month,
                date_tt.day,
                date_tt.hour,
                date_tt.minute,
            ]
            date_clu.append(t) 

        # st()
        # Convert lists to arrays
        new_V = np.array(new_V)
        new_V_5days = np.array(new_V_5days)
        mean_R = np.array(mean_R)
        mean_R_5days = np.array(mean_R_5days)
        HC_lon_now = np.array(HC_lon_now)
        HC_lon_5days = np.array(HC_lon_5days)
        date_clu = np.array(date_clu)

        # st()
        # Sort by time
        i=np.argsort(Time)
        new_V = new_V[i]
        new_V_5days = new_V_5days[i]
        mean_R = mean_R[i]
        mean_R_5days = mean_R_5days[i]
        Time = np.asarray(Time)[i]
        HC_lon_now = HC_lon_now[i]
        HC_lon_5days=HC_lon_5days[i]


        with h5py.File(Inter_data, 'w') as f:
            f.create_dataset('Vr_5days', data=new_V_5days)
            f.create_dataset('Vr', data=new_V)
            f.create_dataset('r_5days', data=mean_R_5days)
            f.create_dataset('r', data=mean_R)
            f.create_dataset('HC_lon_now', data=HC_lon_now)
            f.create_dataset('HC_lon_5days', data=HC_lon_5days)
            f.create_dataset('GONG', data=GONG_image)
            f.create_dataset('Time', data=date_clu)
            f.close()

    else:
        with h5py.File(Inter_data, 'r') as f:
            new_V_5days = np.array(f['Vr_5days'])
            new_V = np.array(f['Vr'])
            mean_R_5days = np.array(f['r_5days'])
            mean_R = np.array(f['r'])
            HC_lon_now = np.array(f['HC_lon_now'])
            HC_lon_5days = np.array(f['HC_lon_5days'])
            GONG_image = np.array(f['GONG'])
            date_clu = np.array(f['Time'])

            # f.create_dataset('Vr_5days', data=new_V_5days)
            # f.create_dataset('Vr', data=new_V)
            # f.create_dataset('r_5days', data=mean_R_5days)
            # f.create_dataset('r', data=mean_R)
            # f.create_dataset('HC_lon_now', data=HC_lon_now)
            # f.create_dataset('HC_lon_5days', data=HC_lon_5days)
            # f.create_dataset('GONG', data=GONG_image)
            # f.create_dataset('Time', data=date_clu)
            f.close()

        # Download SDO data (might take a while)
        ######################## date ################  

        date_idx = np.arange(0, date_clu.shape[0])
        Time = []
        for i, date_tt in tqdm(enumerate(date_clu[date_idx])):
                    
            t = datetime(int(date_tt[0]),
                            int(date_tt[1]),
                            int(date_tt[2]),
                            int(date_tt[3]),
                            int(date_tt[4]),
                            )
            Time.append(t) 

if __name__ == "__main__":
    
    main()