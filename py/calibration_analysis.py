import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import json
from matplotlib.ticker import AutoMinorLocator
from bokeh.plotting import figure, show
from bokeh.io import output_file
import sqlite3
import sys
import os
args = sys.argv


def get_root_data(file):
    '''
    Extracts data from root file into a pandas data frame
    '''
    if file.split(':')[-1] != 'Data':
        file = f'{file}:Data' 
    
    with uproot.open(file) as f:
        print(f.keys())
        
        data = f.arrays(f.keys(), library='pd')
        
        sorted_data = data.sort_values(by='iEvent')
           
    return sorted_data

def ADC_vs_chan(file, stdev=True):
    data = get_root_data(file)
    
    for measure in range(32):
        data_meas = data[(data['Measurement'] == measure)]
        chans = []
        means = []
        rms = []
        for chan in range(128):
            data_chan = data_meas[data_meas['febChannel']==chan]
            ADC = data_chan['ADC'].to_numpy().flatten()
            chans.append(chan)
            means.append(np.mean(ADC))
            rms.append(np.std(ADC))
            
        #plt.figure(figsize=(13,13))
        fig, ax1 = plt.subplots(figsize=(13,13))   
        plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
        plt.tick_params(axis='both', which='minor', direction='in', length=14)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
        # Add major ticks on all sides
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax1.set_title(f'ADC Mean per Channel for Measurement {measure}', fontsize=30) 
        ax1.set_xlabel('febChannel', fontsize=30, loc='right')
        ax1.set_ylabel('Mean ADC', fontsize=30, loc='top')
        ax1.set_xlim(0,127)
        ax1.set_ylim(np.min(means), np.max(means) + 40)
        ax1.step(chans,means,linewidth=2,label='Mean ADC')
        
        if stdev:
            ax2 = ax1.twinx()
            ax2.set_ylim(0,np.max(rms)+0.1)
            ax2.step(chans,rms,linewidth=2, color='orange', label='STDEV')
            ax2.set_ylabel('STDEV', fontsize=15)
            ax2.legend(loc='upper left', fontsize=15)
        
        ax1.legend(loc='upper right', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'plots_{file.split('/')[-1]}/ADC_vs_chan/measurement_{measure}.png')
        plt.clf()
        
def ADC_vs_measure(file):
    data = get_root_data(file)
    
    for chan in range(128):
        data_chan = data[(data['febChannel'] == chan)]
        measure_lo = []
        means_lo = []
        rms_lo = []
        measure_hi = []
        means_hi = []
        rms_hi = []
        means = []
        
        for meas in range(32):
            data_meas = data_chan[data_chan['Measurement']==meas]
            ADC = data_meas['ADC'].to_numpy().flatten()
            means.append(np.mean(ADC))
            if meas == 16:
                continue
            elif meas%2 != 0:
                measure_lo.append(meas)
                means_lo.append(np.mean(ADC))
                rms_lo.append(np.std(ADC))
            else:
                measure_hi.append(meas)
                means_hi.append(np.mean(ADC))
                rms_hi.append(np.std(ADC))
            
        
        # ws = []
        # for i in range(16):
        #     w1 = means[i] - means[i+1]
        #     w2 = means[i+15] - means[i+16]
        #     ws.append((w1+w2)/2)
            
        # print(ws)
        plt.figure(figsize=(15,12))
        plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
        plt.tick_params(axis='both', which='minor', direction='in', length=14)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
        # Add major ticks on all sides
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.xlim(0,31)
        plt.title(f'ADC Mean per Measurement for channel {chan}', fontsize=30)
        plt.xlabel('Measurement', fontsize=30, loc='right')
        plt.ylabel('Mean ADC', fontsize=30, loc='top')
        plt.plot(list(range(32)), means, label='Mean ADC')
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'plots_{file.split('/')[-1]}/ADC_vs_measurement/channel_{chan}.png')
        # fig, ax1 = plt.subplots(figsize=(13,13))   
        # ax1.set_title(f'ADC Mean per Measurement for channel {chan}', fontsize=20) 
        # ax1.set_xlabel('Measurement', fontsize=15)
        # ax1.set_ylabel('Mean ADC for Hi Measurements', fontsize=15)
        # ax1.set_xlim(0,31)
        # ax1.set_ylim(5820,5915)
        # #ax1.set_ylim(np.min(means), np.max(means) + 40)
        # ax1.scatter(measure_hi,means_hi,label='Hi Measurements', color='blue')
        # ax1.legend(loc='upper left', fontsize=15)
        # ax1.spines['left'].set_color('blue')
        # ax1.spines['left'].set_linewidth(2)
        # ax1.tick_params(labelsize=12)
        # ax2 = ax1.twinx()
        # ax2.set_ylim(2005, 2100)
        # ax2.set_ylabel('Mean ADC for Lo Measurements', fontsize=15)
        # ax2.scatter(measure_lo, means_lo, color='red', label='Lo Measurements')
        # ax2.legend(loc='upper right', fontsize=15)
        # ax2.spines['right'].set_color('red')
        # ax2.spines['right'].set_linewidth(2)
        # ax2.tick_params(labelsize=12)
        
        # if stdev:
        #     ax2 = ax1.twinx()
        #     #ax2.set_ylim(0,np.max(rms)+0.1)
        #     ax2.plot(measures,rms,linewidth=2, color='orange', label='STDEV')
        #     ax2.set_ylabel('STDEV', fontsize=15)
        #     ax2.legend(loc='upper left', fontsize=15)
        
        # ax1.legend(loc='upper right', fontsize=15)
        #plt.clf()

def ADC_meas_2dhist(file):
    data = get_root_data(file)
    matrix = np.zeros((32,128))
    
    mins = []
    meas = []
    for measure in range(32):
        data_meas = data[(data['Measurement'] == measure)]
        for chan in range(np.max(data_meas['febChannel']+1)):
            data_chan = data_meas[data_meas['febChannel']==chan]
            ADC = data_chan['ADC'].to_numpy().flatten()
            rms = np.std(ADC)
            matrix[measure][chan] = rms    
        mins.append(np.min(matrix[measure]))
        meas.append(measure)
    
    matrix[matrix == 0] = np.nan
    # output_file("image_plot.html")

    # p = figure(title="2D Image Plot", x_range=(0, 127), y_range=(0, 31),
    #        toolbar_location=None, tools="")
    # p.image(image=[matrix], x=0, y=0, dw=127, dh=31, palette="Spectral11")
    # show(p)


    
    plt.figure(figsize=(15, 12))
    plt.title('ADC RMS for all Channels and Measurements', fontsize=30)
    plt.xlabel('febChannel', fontsize=30, loc='right')
    plt.ylabel('Measurement', fontsize=30, loc='top')
    plt.ylim(0, 31)
    # Set major ticks
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    # Set minor ticks
    plt.tick_params(axis='both', which='minor', direction='in', length=15)
    # Create minor ticks
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(MultipleLocator(1))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    # Create the imshow
    plt.imshow(matrix, extent=[0, matrix.shape[1], matrix.shape[0],0], aspect=3)
    cbar = plt.colorbar(shrink=0.69)
    cbar.ax.tick_params(labelsize=20)  # Set the fontsize for the colorbar ticks
    cbar.set_label('RMS', fontsize=30)
    plt.tight_layout()
    plt.savefig(f'plots_{file.split('/')[-1]}/2dhis_measure_vs_chan_rms.png', bbox_inches='tight')
    plt.clf()
    
    # plt.figure(figsize=(13,12))
    # plt.title('Min ADC RMS for all channels per Measurement')
    # plt.xlabel('Measurement')
    # plt.ylabel('Min ADC RMS')
    # plt.xlim(0,31)
    # plt.step(meas,mins)
    # plt.tight_layout()
    # plt.savefig(f'../plots/minrms_vs_meas.png')
    # plt.clf()
    
    return matrix

def meas_mean_hist(file):
    data = get_root_data(file)
    matrix = np.zeros((32,128))
    
    meas_bar = []
    measure = []
    for meas in range(32):
        data_meas = data[data['Measurement'] == meas]
        meas_mean = np.mean(data_meas['ADC'])
        meas_bar.append(meas_mean)
        measure.append(meas)
        for chan in range(128):
            data_chan = data_meas[data_meas['febChannel'] == chan]
            matrix[meas][chan] = np.mean(data_chan['ADC'] - meas_mean)
        
    # plt.figure(figsize=(13,13))
    # plt.step(measure, meas_bar)
    # plt.title('Mean ADC vs Measurment')
    # plt.xlabel('Measurement')
    # plt.ylabel('Mean ADC')
    # plt.savefig(f'../plots/1d_mean_meas_hist.png')
    # plt.clf()
        
    matrix[matrix == 0] = np.nan
    plt.figure(figsize=(13, 12))
    plt.title('2D Histogram of Mean ADC for all Channels and Measurements', fontsize=30)
    plt.xlabel('febChannel', fontsize=30,loc='right')
    plt.ylabel('Measurement', fontsize=30, loc='top')
    plt.ylim(0, 31)
    # Set major ticks
    plt.tick_params(axis='both', which='major', direction='in', length=10)
    # Set minor ticks
    plt.tick_params(axis='both', which='minor', direction='in', length=5)
    # Create minor ticks
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(MultipleLocator(1))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    # Create the imshow
    plt.imshow(matrix, extent=[0, matrix.shape[1], matrix.shape[0],0], aspect=3)
    cbar = plt.colorbar(shrink=0.69)
    cbar.ax.tick_params(labelsize=20)  # Set the fontsize for the colorbar ticks
    cbar.set_label('Mean_chan-Mean_measure', fontsize=30)
    plt.tight_layout()
    plt.savefig(f'plots_{file.split('/')[-1]}/2d_mean_meas_hist.png', bbox_inches='tight')

def MDAC_const(file, gain, json_file=False):
    data = get_root_data(file)
    ws_dict = {}
    if json_file != False:
        with open(json_file, 'r') as f:
            json_data = json.load(f)['protoBoardCalibs']['186974']

    conn = sqlite3.connect('../data/FEB2v2Mapping-MX-hct22l.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    cursor.execute(f"SELECT * FROM {tables[0][0]}")
    rows = cursor.fetchall()
    for chan in range(128):
        mapping = [tup for tup in rows if chan == tup[3] and gain == tup[1]][0]
        coluta = mapping[4]
        colutachan = mapping[5]
        json_chan = json_data[f'coluta{coluta}'][f'channel{colutachan}']['mdacVals']
        data_chan = data[data['febChannel'] == chan]
        ws = []
        for i in range(8):
            json_mdac = float(json_chan[f'MDACCorrectionCode{i}'])
            w_lo = np.mean(data_chan[data_chan['Measurement'] == 2*i]['ADC']) - np.mean(data_chan[data_chan['Measurement'] == 2*i + 1]['ADC'])
            w_hi = np.mean(data_chan[data_chan['Measurement'] == 2*i + 16]['ADC']) - np.mean(data_chan[data_chan['Measurement'] == 2*i + 17]['ADC'])
            ws.append(float((w_lo + w_hi)/2) - json_mdac)
        ws_dict[chan] = ws
        
    return ws_dict

if __name__ == "__main__":
    file_name = args[1]
    if not os.path.exists(f'plots_{file_name.split('/')[-1]}'):
        os.mkdir(f'plots_{file_name.split('/')[-1]}')
        
    ADC_meas_2dhist(file_name)
    meas_mean_hist(file_name)
    if not os.path.exists(f'plots_{file_name.split('/')[-1]}/ADC_vs_chan'):
        os.mkdir(f'plots_{file_name.split('/')[-1]}/ADC_vs_chan')
    ADC_vs_chan(file_name)
    if not os.path.exists(f'plots_{file_name.split('/')[-1]}/ADC_vs_measurement'):
        os.mkdir(f'plots_{file_name.split('/')[-1]}/ADC_vs_measurement')
    ADC_vs_measure(file_name)
    