# script to show F-T data and FFT results from dataset GM10-ts-Plus
# 
# Z. Zhang
# 2025/03

import pandas as pd
import numpy as np
import os, os.path, glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.animation as animation
from scipy.fftpack import fft,ifft,fftfreq,rfft,irfft,rfftfreq

def plot_fft_per_log3(GMType,file,figFFTPath,fd_arr_1,isSaveFig=False):
    print(f'total #: {len(fd_arr_1)}')
    
    ## FFT
    SAMPLE_RATE = 62.5  # Hertz
    fxf = rfft(fd_arr_1)
    tf  = rfftfreq(fd_arr_1.size, 1/SAMPLE_RATE)

    abs_fxf=np.abs(fxf)
    normalization_y=abs_fxf/fd_arr_1.size
    plt.figure(figsize=(5.2,5))
    plt.plot(tf, normalization_y,'g')
    # plt.xlim((-0.2,4))
    plt.xlim((0,4))
    plt.ylim((0,3))
    plt.xlabel('Frequence (Hz)', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    plt.title(f'GM Class: {GMType}', fontsize=20)
    plt.tight_layout()
    if isSaveFig == 0:
        plt.show()
    else:
        save_path = figFFTPath
        fig_name  = file[:-4]
        plt.savefig(save_path+'/'+fig_name+ f'_[0,End]_fft.png')

## define switch
isPlotFd  = 1
isSaveFig = 0
isPlotFFT = 1
isSaveFFTfig = 0 
isSaveVideo = 0 
class_ls = ['baysalt', 'broad bean', 'cassia seed', 'cat litter', 'crushed peanut', 'gravel', 'in-shell peanut', 'long-grain rice', 'refined salt', 'sand']

start_expid = 28
avd = (0.7,3,6) # diameter(cm), mv, depth(cm)
PENE_DEPTH = -1*0.01*avd[2]
spiral_mv = 0.1*avd[1]
diameter = avd[0]*10 # mm <<<<<< diameter
diameter = np.around(diameter,2)
spiral_mv = np.around(spiral_mv,2)
PENE_DEPTH = np.around(PENE_DEPTH,3)
print(f"diAmeter: {diameter} mm; mV: {spiral_mv}; Depth: {PENE_DEPTH} m.")

gmType = 'long-grain rice'
NutStorePath = 'path/to/dataset/GM10-ts-Plus/' # !! change it to the path of dataset 'GM10-ts-Plus' 
expFolderName = '/' + gmType + '/' + gmType + f'_dia{diameter}mm' + f'_mv{spiral_mv}' + f'_dept{np.round(PENE_DEPTH*100,1)}' #
csv_count = len([f for f in os.listdir(NutStorePath+expFolderName) if f.endswith('.csv')])

for expid in range(start_expid, csv_count + 1):    
    ## data (Fd-itd) / figure dir
    dataPath = NutStorePath+expFolderName
    figPath  = NutStorePath+expFolderName+'/fig'
    figFFTPath = NutStorePath+expFolderName+'/figFFT'
    vidoePath = NutStorePath+expFolderName+'/video'

    if not os.path.exists(figFFTPath) and isSaveFFTfig == 1:
        os.makedirs(figFFTPath)
    if not os.path.exists(figPath) and isSaveFig == 1:
        os.makedirs(figPath)
    if not os.path.exists(vidoePath) and isSaveVideo == 1:
        os.makedirs(vidoePath)    
    
    logId = f'*Exp{expid}'
    # logId = '-1' # <<<<<< all logs
    os.chdir(dataPath)

    if logId == '-1':
        file_suffix = '*timefxfy*.csv'
    else:
        file_suffix = logId + '*timefxfy*.csv'
    num_found = len(glob.glob(file_suffix))
    print(f'---- Find {num_found} log files ----')

    if num_found == 0:
        raise Exception('Not Found Files!')

    for file_id, file in enumerate(glob.glob(file_suffix)):
        print('---- {} ----'.format(file))
        expID_retrieve = file[:10]
        if file.find('Flt') != -1:
            FRLable = 'Flt'
        elif file.find('Raw') != -1:
            FRLable = 'Raw'
        else:
            FRLable = 'NotFound'
        ## double check GM type
        GMType = file[file.find('gm')+2:file.find('.csv')].lower()
        if GMType != gmType:
            raise Exception('GM type not match!')
        ## read content from log file 
        fd_data_ls = pd.read_csv(file,names=['tt','fx','fy'])
        total_ite = len(fd_data_ls)
        print(f'-- Total ite: {total_ite} --')
        
        ## plot Fd-ite
        if isPlotFd == True:
            plt.figure(figsize=(10,5))
            fx_arr = np.array(fd_data_ls.fx)
            fy_arr = np.array(fd_data_ls.fy)
            fd_arr = np.sqrt((fx_arr)**2+(fy_arr)**2)
            tt_arr = np.array(fd_data_ls.tt)
            plt.plot(tt_arr,fd_arr, 'b-', linewidth=2, markevery = 1)
            plt.ylim([0, 15])
            plt.xlim([0, 10])
            plt.grid()
            plt.title(f'GM Class: {GMType}', fontsize=20)
            plt.xlabel('Time (s)', fontsize=20)
            plt.ylabel('Force (N)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            if isSaveFig == 0:
                plt.show()
            else:
                save_path = figPath
                fig_name  = file[:-4]
                plt.savefig(save_path+'/'+fig_name+'.png')
        
        ## data from log file
        fx_arr   = np.array(fd_data_ls.fx)
        fy_arr   = np.array(fd_data_ls.fy)
        time_arr = np.array(fd_data_ls.tt)
        fd_arr = np.sqrt((fx_arr)**2+(fy_arr)**2)

        ## force animation
        if isSaveVideo == True:
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10,5))
            # Set up the line plot
            line, = ax.plot([], [], 'b-', linewidth=2)

            # Set the axis limits
            ax.set_xlim(0, np.max(time_arr))
            ax.set_ylim([0, 15])
            ax.set_title(f'GM Class: {GMType}', fontsize=20)
            ax.set_xlabel('Time (s)', fontsize=20)
            ax.set_ylabel('Force (N)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            # Initialize the animation function
            def init():
                line.set_data([], [])
                return line,

            # Define the update function for each frame of the animation
            def update(frame):
                x = time_arr[:frame+1]  # Time steps up to the current frame
                y = fd_arr[:frame+1]  # Force values up to the current frame
                line.set_data(x, y)
                return line,

            # Calculate the interval between frames based on the desired fps
            fps = 62.5
            interval = 1000 / fps

            # Create the animation
            ani = animation.FuncAnimation(fig, update, frames=len(time_arr), init_func=init, blit=True, interval=interval)
            ani.save(vidoePath+f'/force_animation_{file_id+1}.mp4', writer='ffmpeg')


        ## plot FFT results for this log file
        if isPlotFFT == True:
            plot_fft_per_log3(GMType,file,figFFTPath,fd_arr,0)
