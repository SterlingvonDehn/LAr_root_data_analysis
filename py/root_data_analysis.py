#import ROOT
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from scipy.fft import fft, fftfreq
from matplotlib.ticker import MultipleLocator
import json
from matplotlib.ticker import AutoMinorLocator
from bokeh.plotting import figure, show
from bokeh.io import output_file

file_1 = "../data/output-data-fullfeb2v2-20240910.root:Data" #First data set -- data taken in 4 "chuncks"

file_2 = "../data/output-data-MXSX-hct20l-2024-09-18.root:Data" #Second data set -- Full feb data taken, issue with iEvent

file_3 = "../data/output-data-MXSX-hct20l-2024-09-23.root:Data" #Third data set -- Same as second but iEvent error corrected

file_4 = "../data/output-data-hct22l-MDACCalib20241010.root:Data" #Fourth data set -- EMF caibration instead of NEVIS

calib_file_1 = "../data/output-CalibrationHG-mxhct22l-32Measurements-merged.root:Data"  #Fourth data set -- calibration testing with 32 measurements. Many issues such as 0 stdev on many entries.

calib_file_2 = "../data/output-dataHG-mxhct22l-calibset-2024-10-07.root:Data"

calib_file_3_HG = "../data/output-MDACCALIBRATION-HG-mxhct22l-2024-10-10.root:Data"

calib_file_3_LG = "../data/output-MDACCALIBRATION-LG-mxhct22l-2024-10-10.root:Data"

json_file_1 = "../data/calibFile-MDACCalib-2024-10-10.json"

def get_root_data(file):
    '''
    Extracts data from root file into a pandas data frame
    '''
    
    with uproot.open(file) as f:
        print(f.keys())
        
        data = f.arrays(f.keys(), library='pd')
        
        sorted_data = data.sort_values(by='iEvent')
           
    return sorted_data

def EMF_system_test(file):
    '''
    Plots mean pedestal ADC, stdev ADC, and number of entries vs febChannel for hi and lo gain. 
    '''
    
    data = get_root_data(file) #getting data
    
    #split hi-lo gain
    data_high = data[data['gain'] == 1]
    data_low = data[data['gain'] == 0]
    
    #calculating means,stdevs, and num entries for each channel
    ADC_mean_h = []
    ADC_std_h = []
    ADC_mean_l = []
    ADC_std_l = []
    num_ent_h = []
    num_ent_l = []
    for chan in range(128): #looping over all channels
        data_chan_h = data_high[data_high['febChannel']==chan] #taking data for specified channel
        
        ADC_mean_h.append(np.mean(data_chan_h['ADC'])) 
        ADC_std_h.append(np.std(data_chan_h['ADC']))
        num_ent_h.append(len(data_chan_h['ADC']))
        
        data_chan_l = data_low[data_low['febChannel']==chan]
        ADC_mean_l.append(np.mean(data_chan_l['ADC']))
        ADC_std_l.append(np.std(data_chan_l['ADC']))
        num_ent_l.append(len(data_chan_l['ADC']))
    
    febs = list(range(128)) #feb list
    
    #plotting mean vs feb
    plt.figure(figsize=(12,8))
    plt.title('ATLAS Internal/EMF System Test', fontsize=30)
    plt.xlabel('FebChannel', fontsize=30, loc='right')
    plt.ylabel('Mean Pedestal [ADC]', fontsize=30, loc='top')
    plt.ylim(5000,7500)
    plt.xlim(0,127)
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.step(febs, ADC_mean_h, label='Hi gain', linewidth=2)
    plt.step(febs, ADC_mean_l, label='Lo gain', linewidth=2)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig('../plots/mean_system_test.png')

    #plotting stdev vs feb
    plt.figure(figsize=(12,8))
    plt.title('ATLAS Internal/EMF System Test', fontsize=30)
    plt.xlabel('FebChannel', fontsize=30, loc='right')
    plt.ylabel('Standard Deveation', fontsize=30, loc='top')
    plt.ylim(0,20)
    plt.xlim(0,127)
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.step(febs, ADC_std_h, label='Hi gain', linewidth=2)
    plt.step(febs, ADC_std_l, label='Lo gain', linewidth=2)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig('../plots/std_system_test.png')
    plt.clf()
    
    #plotting num entries vs feb
    plt.figure(figsize=(12,8))
    plt.title('ATLAS Internal/EMF System Test')
    plt.xlabel('FebChannel')
    plt.ylabel('# of entries')
    plt.xlim(0,127)
    plt.step(febs,num_ent_h, color='pink', linewidth=2, label=f'High Gain, total: {len(data_high)}')
    plt.step(febs,num_ent_l, color='blue', linewidth=2, label=f'Low Gain, total: {len(data_low)}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../plots/num_entries_system_test.png')
    plt.clf()
    
def correlation(file, gain, auto_range=True):
    '''
    Produces a Pearson Correlation matrix plot for febChannels in the data set.
    ONLY TAKES FIRST TIME ENTRY FOR EACH EVENT --> needs to be adjusted to take all data
    file == root file path
    gain == 0 or 1 -> 0 (lo) 1 (hi)
    auto_range == True or False -> removes 1s from diagonal adding contrast for all other values 
    '''
    data = get_root_data(file) #extracting data
    matrix = np.zeros((128,128)) #creating empty matrix
    
    #calculating correlation coefficient for all channel combinations
    for i in range(128):
        for j in range(i + 1):
            data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain)]
            data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain)]
            #Used if only one time entrie is desired
            
            '''
            #taking first time entry for each event
            for n in data_chan_i:
                A.append(n[0])
            for m in data_chan_j:
                B.append(m[0])
            
            '''
            #creating arrays of data
            A = np.array(data_chan_i['ADC'].to_numpy().flatten(), dtype=np.float64)
            B = np.array(data_chan_j['ADC'].to_numpy().flatten(), dtype=np.float64)

            if len(A) != len(B) or not np.array_equal(data_chan_i['iEvent'].to_numpy(),data_chan_j['iEvent'].to_numpy()): #skipping if arrays are different lenghts 
                continue
            
            #corr_coefficient = (np.mean(A*B) - np.mean(A)*np.mean(B))/(np.std(A)*np.std(B)) #calculating Pearson correlation coefficient
            corr_coefficient = pearsonr(A,B)[0]
            
            #populating matrix depending on auto_range settings
            if auto_range:
                if i != j:
                    matrix[i][j] = corr_coefficient
                    matrix[j][i] = corr_coefficient
            else:
                matrix[i][j] = corr_coefficient
                matrix[j][i] = corr_coefficient

        print(f'{round((i/127)*100, 1)}%') #progress tracker
    
    #defining color range if auto_range == True
    if auto_range:
        max_v = max([abs(np.max(matrix)), abs(np.min(matrix))])
    else:
        max_v = 1
    
    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi'
    
    #plotting
    plt.figure(figsize=(15,15))
    plt.title(f'febChannel Correlation, {gain_title} gain', fontsize=30)
    plt.ylim(0, 127)
    plt.xlabel('febChannel', fontsize=30, loc='right')
    plt.ylabel('febChannel', fontsize=30, loc='top')
    plt.tick_params(axis='both', which='major', direction='in', length=20)
    # Set minor ticks
    plt.tick_params(axis='both', which='minor', direction='in', length=15)
    # Create minor ticks
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    # Create the imshow
    plt.imshow(matrix, extent=[ 0,matrix.shape[1], matrix.shape[0],0], vmin=-max_v,vmax=max_v, cmap='bwr')
    cbar = plt.colorbar(shrink=0.78)
    cbar.ax.tick_params(labelsize=20)  # Set the fontsize for the colorbar ticks
    cbar.set_label('Correlation Coeff', fontsize=30)
    plt.tick_params(labelsize=25)
    plt.tight_layout()
    plt.savefig(f'../plots/{auto_range}_{file.split('/')[-1]}_channel_gain{gain}_correlation_fullavg.png')

    return matrix    
    
def correlation_diff_event(file, gain):
    '''
    '''
    data = get_root_data(file) #extracting data
    matrix = np.zeros((128,128)) #creating empty matrix
    
    #calculating correlation coefficient for all channel combinations
    data = data[data['gain']==gain]
    # half_event = np.max(data['iEvent'])/2
    corr_hist = []
    for i in range(128):
        for j in range(i + 1):
            data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain)].iloc[1::2]
            data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain)].iloc[::2]
            
            # data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain) & (data['iEvent'] < event)]
            # data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain) & (data['iEvent'] > event) & (data['iEvent'] < 2*event)]
            
            #Used if only one time entrie is desired

            #creating arrays of data
            A = np.array(data_chan_i['ADC'].to_numpy().flatten(), dtype=np.float64)
            B = np.array(data_chan_j['ADC'].to_numpy().flatten(), dtype=np.float64)
            # print(data_chan_i['iEvent'])
            # print(data_chan_j['iEvent'])
            # print(len(A), len(B))
            # return
            if len(A) != len(B): #skipping if arrays are different lenghts 
                continue
            
            #corr_coefficient = (np.mean(A*B) - np.mean(A)*np.mean(B))/(np.std(A)*np.std(B)) #calculating Pearson correlation coefficient
            corr_coefficient = pearsonr(A,B)[0]
            #populating matrix depending on auto_range settings
            matrix[i][j] = corr_coefficient
            matrix[j][i] = corr_coefficient
            
            corr_hist.append(corr_coefficient)

        print(f'{round((i/127)*100, 1)}%') #progress tracker
    
    N = len(A)
    #defining color range if auto_range == True
    max_v = max([abs(np.max(matrix)), abs(np.min(matrix))])
    
    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi'
    
    #plotting
    plt.figure(figsize=(15,15))
    plt.title(f'febChannel Correlation Unrelated, {gain_title} gain', fontsize=30)
    plt.ylim(0, 127)
    plt.xlabel('febChannel', fontsize=30, loc='right')
    plt.ylabel('febChannel', fontsize=30, loc='top')
    plt.tick_params(axis='both', which='major', direction='in', length=20)
    # Set minor ticks
    plt.tick_params(axis='both', which='minor', direction='in', length=15)
    # Create minor ticks
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    # Create the imshow
    plt.imshow(matrix, extent=[ 0,matrix.shape[1], matrix.shape[0],0], vmin=-max_v,vmax=max_v, cmap='bwr')
    cbar = plt.colorbar(shrink=0.78)
    cbar.ax.tick_params(labelsize=20)  # Set the fontsize for the colorbar ticks
    cbar.set_label('Correlation Coeff', fontsize=30)
    plt.tick_params(labelsize=25)
    plt.tight_layout()
    plt.savefig(f'../plots/{file.split('/')[-1]}_channel_gain{gain}_correlation_unrelate.png')
    
    
    plt.figure(figsize=(12,9))
    plt.title(f'Correlation Coefficients, {gain_title} gain', fontsize=30)
    plt.xlabel('Correlation Coefficients', fontsize=30, loc='right')
    plt.ylabel('# Entries', fontsize=30, loc='top')
    #plt.yscale('log')
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.hist(corr_hist, bins=70, edgecolor='black',label=f'N = {N}, $\sigma$ = {round(np.std(corr_hist),3)}, Mean = {round(np.mean(corr_hist),3)}')
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'../plots/{N}_corr_coeff_hist_unrelate_{gain}.png')
    
    return [N, np.std(corr_hist)]

def corr_vs_n(events, gain):
    ns = []
    stds = []
    for i in events:
        data = correlation_diff_event(file_3, gain, i)
        ns.append(data[0])
        stds.append(data[1])
        print(i)
        
    a = ((np.log(stds[0]) + np.log(ns[0])/2) + (np.log(stds[-1]) + np.log(ns[-1])/2))/2
    y = lambda x: -x/2 + a

    plt.figure(figsize=(13,13))
    plt.plot(np.log(ns), np.log(stds), label='ln(RMS)')
    plt.plot(np.log(ns),y(np.log(ns)),linestyle=':', label='$y=\\frac{-x}{2}$ ' + f'+ {round(a,2)}')
    plt.xlabel('ln(N)', fontsize=15)
    plt.ylabel('ln(RMS)', fontsize=15)
    plt.title(f'ln(RMS) vs ln(N) for Correlation Coefficients of Unrelated Events, gain: {gain}', fontsize=20)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'../plots/{gain}_lnrms_lnN.png')
    
    return(ns,stds)

def ADC_event(file, chan, gain, event_range=False):
    '''
    Plots a 2d histogram of ADC vs events for specified channel and gain
    chan = channel #
    gain = 1 or 0
    event_range = (start_entry,end_entry), automatic if not specified
    '''
    data = get_root_data(file) #extracts data
    data_feb = data[(data['febChannel']==chan) & (data['gain']==gain)] #selects channel and gain
    data_adc = data_feb['ADC'].to_numpy()
    
    #selecting range
    if not event_range:
        start = 0
        end = np.max(data_feb['iEvent'])
    else:
        start = event_range[0]
        end = event_range[1] 
    
    #calculating adc values per event
    events = []
    adc = []
    for i in range(start,end):
        for j in range(len(data_adc[i])):
            events.append(i)
            adc.append(data_adc[i][j])
    
    #calculating #bins
    adc_bins = int(np.max(adc) - np.min(adc))
    
    #plotting hist
    plt.figure(figsize=(15,12))
    plt.title(f'ADC:iEvent (gain=={gain} & febChannel=={chan})', fontsize=20)
    plt.xlabel('iEvent', fontsize=14)
    plt.ylabel('ADC', fontsize=14)
    cmap = plt.cm.viridis  # Use the 'viridis' colormap
    hist = plt.hist2d(events,adc,cmap=cmap,bins=[30,adc_bins], cmin=1)
    plt.colorbar()
    
    #calcualting avg and stdev per bin
    avg_adc = []
    bin_lst = []
    std_up_adc = []
    std_down_adc = []
    for i in range(len(hist[1]) - 1):
        data_bin = data_feb[(data_feb['iEvent'] <= hist[1][i+1]) & (data_feb['iEvent'] > hist[1][i])]
        avg_adc.append(np.mean(data_bin['ADC']))
        std_up_adc.append(avg_adc[-1] + np.std(data_bin['ADC']))
        std_down_adc.append(avg_adc[-1] - np.std(data_bin['ADC']))
        bin_lst.append(hist[1][i+1])

    #plotting avg and stdev
    plt.plot(bin_lst,avg_adc,color='red', label='Mean/bin')
    plt.plot(bin_lst,std_up_adc,color='red', linestyle='--', label='Stdev/bin')
    plt.plot(bin_lst,std_down_adc,color='red', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../plots/ADC_iEvent_{chan}_{gain}.png')

def auto_correlation(file, gain, chan, auto_range=True):
    '''
    '''
    data = get_root_data(file) #extracting data
    data_chan = data[(data['febChannel'] == chan) & (data['gain'] == gain)]
    N_event = np.max(data_chan['iEvent'])
    matrix = np.zeros((N_event,N_event)) #creating empty matrix
    data_array = data_chan['ADC'].to_numpy()
    
    #calculating correlation coefficient for all channel combinations
    for i in range(N_event):
        for j in range(i + 1):
            data_event_i = data_array[i]
            data_event_j = data_array[j]

            #creating arrays of data
            A = np.array(data_event_i, dtype=np.float64)
            B = np.array(data_event_j, dtype=np.float64)

            if len(A) != len(B): 
                continue
            
            #corr_coefficient = (np.mean(A*B) - np.mean(A)*np.mean(B))/(np.std(A)*np.std(B)) #calculating Pearson correlation coefficient
            corr_coefficient = pearsonr(A,B)[0]
            
            #populating matrix depending on auto_range settings
            if auto_range:
                if i != j:
                    matrix[i][j] = corr_coefficient
                    matrix[j][i] = corr_coefficient
            else:
                matrix[i][j] = corr_coefficient
                matrix[j][i] = corr_coefficient

        print(f'{round((i/N_event)*100, 1)}%') #progress tracker
    
    #defining color range if auto_range == True
    if auto_range:
        max_v = max([abs(np.max(matrix)), abs(np.min(matrix))])
    else:
        max_v = 1
    
    #plotting
    plt.figure(figsize=(15,15))
    plt.title(f'FebChannel Pearson Auto Correlation Coefficient Matrix, gain: {gain}', fontsize=20)
    plt.ylim(0, 127)
    plt.xlabel('febChannel', fontsize=14)
    plt.ylabel('febChannel', fontsize=14)
    plt.imshow(matrix, vmin=-max_v,vmax=max_v, cmap='bwr')
    plt.colorbar(shrink=0.785)
    plt.tight_layout()
    plt.savefig(f'../plots/{auto_range}_{file.split('/')[-1]}_channel_gain{gain}_autocorrelation.png')

    return matrix    

def check_bcid(file, samples):
    '''
    Checks bcid order to ensure it increases as expected:
    samples == # of time samples
    ''' 
    data = get_root_data(file) #extracts data
    
    data['bcid_sorted'] = data['bcid'].apply(lambda x: check_list(x,samples)) #checks bcid order
    print(data['bcid_sorted'])
    
    #checks bcid starting values (may not be usefull)
    starts = []
    for i in range(len(data['bcid'])):
        if data['bcid'][i][0] not in starts:
            starts.append(data['bcid'][i][0])
    print(starts)        
    
def check_list(lst, samples):
    '''
    Used to check the bcid order 
    '''
    modulo = 3565 #defining modulo
    mod_list = lst #bcid list
    
    #altering list if it containts the jump from modulo -> 0
    if lst[0] >= modulo-samples: 
        mod_list = []
        for bcid in lst:
            if bcid < samples:
                mod_list.append(modulo+bcid)
            else:
                mod_list.append(bcid)
    #checking order of bcids
    if mod_list[-1] - mod_list[0] != samples-1 or len(mod_list) != samples:
        return False
    
    #returning bool of sorting status
    return all(mod_list[i] < mod_list[i + 1] for i in range(len(mod_list) - 1))

def check_matrix(file,chan1,chan2, gain):
    '''
    check for the correlation matrix. Plots a 2d hist of specified channels to check validity of correlation coefficient
    '''
    data = get_root_data(file) #extract data
    data_i = data[(data['febChannel']==chan1) & (data['gain']==gain)]['ADC'].to_numpy().flatten() #taking specified channel and gain
    data_j = data[(data['febChannel']==chan2) & (data['gain']==gain)]['ADC'].to_numpy().flatten()
    
    #making arrays
    A = np.array(data_i, dtype=np.float64)
    B = np.array(data_j, dtype=np.float64)
    
    #calculating correlation coefficient 
    corr_coefficient = pearsonr(A,B)[0]
    
    # #plotting
    A_bins = int(np.max(A) - np.min(A)) #calcualting bins
    B_bins = int(np.max(B) - np.min(B))
    plt.figure(figsize=(12,12))
    hist = plt.hist2d(A,B, bins=[A_bins,B_bins],cmin=1)
    plt.colorbar()
    plt.title(f'Channel {chan1} vs Channel {chan2}, gain: {gain}, Pearson correlation: {round(100*corr_coefficient,1)}%')
    plt.xlabel(f'Channel {chan1}')
    plt.ylabel(f'Channel {chan2}')
    plt.tight_layout()
    plt.savefig(f'../plots/{chan1}_{chan2}_{gain}_scatter.png')
    print(hist)
    
def gaussian(x, amp, mean, stddev):
    """Gaussian function for fitting."""
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def coherent_noise(file, gain):
    '''
    Plots a coherent noise plot for the specified file and gain
    '''
    data = get_root_data(file) #getting data

    #defining dictionaries, lists and starting sums
    data_dic = {}
    rms = {}
    d_rms = {}
    ch_noise = 0
    ch_noise_it = 0
    d_ch_noise = 0
    dataSum = []
    
    #looping over all channels
    N_channels = np.max(data['febChannel']) + 1
    for chan in range(N_channels):
        data_chan = data[(data['febChannel']==chan) & (data['gain']==gain)]['ADC'].to_numpy().flatten()
        data_dic[chan] = data_chan - np.mean(data_chan)
        rms[chan] = np.std(data_chan-np.mean(data_chan))
        d_rms[chan] = rms[chan]/np.sqrt(len(data_chan))
        ch_noise += rms[chan] ** 2
        d_ch_noise += d_rms[chan] ** 2
        ch_noise_it += rms[chan]
        
    
    #calculating noise
    ch_noise = np.sqrt(ch_noise)
    d_ch_noise = np.sqrt(d_ch_noise)
    avg_noise_nev = ch_noise / np.sqrt(N_channels)
    avg_noise_it = ch_noise_it/N_channels
    d_avg = d_ch_noise / np.sqrt(N_channels)
    
    #calculating hist data
    dataSum = np.sum([(data_dic[ch]) for ch in range(N_channels)], axis=0)
    
    tot_noise = np.std(dataSum)
    mu = np.mean(dataSum)
    std = np.std(dataSum)
    coh_noise = np.sqrt(tot_noise**2 - ch_noise**2) /N_channels
    pct_coh_nev = coh_noise / avg_noise_nev * 100
    pct_coh_it = coh_noise / avg_noise_it * 100
    
    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi'
    #plotting
    plt.figure(figsize=(15,11))
    y, x, _ = plt.hist(
        dataSum,
        bins=np.arange(min(dataSum), max(dataSum) + 2, 1),
        color="black",
        edgecolor="black",
        density=False)
    
    lnspc = np.linspace(min(dataSum), max(dataSum), 1000)
    p = norm.pdf(lnspc, mu, std)
    
    bin_width = x[1] - x[0]
    p *= (y.sum() * bin_width) 
    plt.plot(lnspc, p, color='red', linewidth=2, label=f'Gauss fit, $\mu$ = {round(mu,2)}, $\sigma$ = {round(std,1)}')
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.text(0.05, 0.95,f'Entries = {round(len(dataSum)/1000,1)}k\nN ch = {N_channels}\nMean = {round(mu,1)}\nRMS = {round(tot_noise,1)}\n$\sigma$ inco = {round(ch_noise,1)}\nCohe noise/ch = {round(coh_noise,4)}\nAvg noise NEVIS/ch = {round(avg_noise_nev,4)}\nAvg noise Milano/ch = {round(avg_noise_it,4)}\n[%]Cohe noise NEVIS = {round(pct_coh_nev,4)}\n[%]Cohe noise Milano = {round(pct_coh_it,4)}', transform=plt.gca().transAxes,fontsize=20, verticalalignment='top', horizontalalignment='left')
    plt.legend(fontsize=20)
    plt.title(f'EMF system test, coherent, {gain_title} gain', fontsize=30)
    
    plt.xlabel('ADC counts', fontsize=30, loc='right')
    plt.ylabel('Entries', fontsize=30, loc='top')
    plt.tight_layout()
    plt.savefig(f'../plots/coherence_hist_{gain}.png')
    
    
def coh_corr(file,gain):
    data = get_root_data(file) #extracting data
    matrix = np.zeros((128,128)) #creating empty matrix
    
    #calculating correlation coefficient for all channel combinations
    corr_sum = 0
    for i in range(128):
        for j in range(i):
            data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain)]
            data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain)]
            #Used if only one time entrie is desired
            
            '''
            #taking first time entry for each event
            for n in data_chan_i:
                A.append(n[0])
            for m in data_chan_j:
                B.append(m[0])
            
            '''
            #creating arrays of data
            A = np.array(data_chan_i['ADC'].to_numpy().flatten(), dtype=np.float64)
            B = np.array(data_chan_j['ADC'].to_numpy().flatten(), dtype=np.float64)

            if len(A) != len(B) or not np.array_equal(data_chan_i['iEvent'].to_numpy(),data_chan_j['iEvent'].to_numpy()): #skipping if arrays are different lenghts 
                continue
            
            corr_coefficient = (np.mean(A*B) - np.mean(A)*np.mean(B))
            
            #populating matrix depending on auto_range settings
            corr_sum += corr_coefficient

        print(f'{round((i/127)*100, 1)}%') #progress tracke
    
    return np.sqrt(2*corr_sum)/128

def ADC_vs_chan_permeasure(file, stdev=True):
    data = get_root_data(file)
    
    for measure in range(32):
        data_meas = data[(data['Measurement'] == measure) & (data['gain'] == 1)]
        chans = []
        means = []
        rms = []
        for chan in range(np.max(data_meas['febChannel']+1)):
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
        plt.savefig(f'../plots/ADC_chan_{measure}.png')
        plt.clf()

def ADC_vs_measurment_perchan(file,chan, stdev=True):
    data = get_root_data(file)
    
    data_chan = data[(data['febChannel'] == chan) & (data['gain'] == 1)]
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
    plt.savefig(f'../plots/ADC_vs_measure_{chan}.png')
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


def covariance(file):
    '''
    '''
    data = get_root_data(file) #extracting data
    matrix_hi = np.zeros((128,128)) #creating empty matrix
    matrix_lo = np.zeros((128,128))
    
    #calculating correlation coefficient for all channel combinations
    for i in range(128):
        for j in range(i + 1):
            data_chan_i_hi = data[(data['febChannel']==i) & (data['gain']==1)]
            data_chan_j_hi = data[(data['febChannel']==j) & (data['gain']==1)]
            data_chan_i_lo = data[(data['febChannel']==i) & (data['gain']==0)]
            data_chan_j_lo = data[(data['febChannel']==j) & (data['gain']==0)]
            
            #Used if only one time entrie is desired
            
            '''
            #taking first time entry for each event
            for n in data_chan_i:
                A.append(n[0])
            for m in data_chan_j:
                B.append(m[0])
            
            '''
            #creating arrays of data
            A_hi = np.array(data_chan_i_hi['ADC'].to_numpy().flatten(), dtype=np.float64)
            B_hi = np.array(data_chan_j_hi['ADC'].to_numpy().flatten(), dtype=np.float64)
            A_lo = np.array(data_chan_i_lo['ADC'].to_numpy().flatten(), dtype=np.float64)
            B_lo = np.array(data_chan_j_lo['ADC'].to_numpy().flatten(), dtype=np.float64)

            #if len(A) != len(B) or not np.array_equal(data_chan_i_['iEvent'].to_numpy(),data_chan_j['iEvent'].to_numpy()): #skipping if arrays are different lenghts 
                #continue
            
            #corr_coefficient = (np.mean(A*B) - np.mean(A)*np.mean(B))/(np.std(A)*np.std(B)) #calculating Pearson correlation coefficient
            if i != j:
                matrix_hi[i][j] = np.mean(A_hi*B_hi) - np.mean(A_hi)*np.mean(B_hi)
                matrix_hi[j][i] = np.mean(A_hi*B_hi) - np.mean(A_hi)*np.mean(B_hi)
                matrix_lo[i][j] = np.mean(A_lo*B_lo) - np.mean(A_lo)*np.mean(B_lo)
                matrix_lo[j][i] = np.mean(A_lo*B_lo) - np.mean(A_lo)*np.mean(B_lo)
                
        print(f'{round((i/127)*100, 1)}%') #progress tracker
    
    #defining color range if auto_range == True
    max_v = max([abs(np.max(matrix_hi)), abs(np.min(matrix_hi))])
    plt.figure(figsize=(15,15))
    plt.title(f'FebChannel Covariance Matrix, gain: 1', fontsize=20)
    plt.ylim(0, 127)
    plt.xlabel('febChannel', fontsize=14)
    plt.ylabel('febChannel', fontsize=14)
    plt.imshow(matrix_hi, vmin=-max_v,vmax=max_v, cmap='bwr')
    plt.colorbar(shrink=0.785)
    plt.tight_layout()
    plt.savefig(f'../plots/{file.split('/')[-1]}_channel_gain1_cov.png')
    plt.clf()
    
    max_v = max([abs(np.max(matrix_lo)), abs(np.min(matrix_lo))])
    plt.figure(figsize=(15,15))
    plt.title(f'FebChannel Covariance Matrix, gain: 0', fontsize=20)
    plt.ylim(0, 127)
    plt.xlabel('febChannel', fontsize=14)
    plt.ylabel('febChannel', fontsize=14)
    plt.imshow(matrix_lo, vmin=-max_v,vmax=max_v, cmap='bwr')
    plt.colorbar(shrink=0.785)
    plt.tight_layout()
    plt.savefig(f'../plots/{file.split('/')[-1]}_channel_gain0_cov.png')


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
    plt.savefig(f'../plots/2dhis_measure_vs_chan_rms.png')
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


def FFT_avg(file,gain,sample):
    data = get_root_data(file)
    fs = 200
    
    fft_res_list = []
    for chan in range(128):
        data_chan = data[(data['febChannel']==chan) & (data['gain']==gain)]
        
        signal = []
        for entry in data_chan['ADC']:
            signal.append(entry[sample])
        
        adc_values = []
        for i in signal:
            adc_values.append(i - np.mean(signal))

        # Step 3: Compute FFT
        fft_result = np.fft.fft(adc_values)

        fft_res_list.append(fft_result)
        
        print(f'{round(100*chan/128,1)}%')
    
    fft_res_arr = np.array(fft_res_list)
    
    avg_res = [np.mean(fft_res_arr[:,i]) for i in range(len(fft_res_arr[0]))]
    
    fs = 200  # Sampling frequency in Hz
    N = len(adc_values)  # Number of samples

    # Step 2: Create a time vector
    t = np.arange(N) / fs  # Time vector based on the number of samples
    # Get the frequency bins
    frequencies = np.fft.fftfreq(N, 1/fs)
    
        
        
    # Step 4: Plotting the original signal
    plt.figure(figsize=(12, 12))

    plt.plot(frequencies[:N//2], np.abs(avg_res)[:N//2])  # Plot only positive frequencies
    plt.title(f'Avg FFT of ADC Signal across all channels, time sample: {sample}, gain: {gain}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim(0, fs / 2)  # Limit x-axis to half the sampling frequency
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'../plots/fft_avg_{gain}_{sample}.png')
    
    
def FFT(file,chan,gain,sample):
    data = get_root_data(file)
    
    data_chan = data[(data['febChannel']==chan) & (data['gain']==gain)]

    signal = []
    for entry in data_chan['ADC']:
        signal.append(entry[sample])
    
    adc_values = []
    # for i in signal:
    #     adc_values.append(i - np.mean(signal))
    adc_values = signal
     
    fs = 200  # Sampling frequency in Hz
    N = len(adc_values)  # Number of samples

    # Step 2: Create a time vector
    t = np.arange(N) / fs  # Time vector based on the number of samples

    # Step 3: Compute FFT
    fft_result = np.fft.fft(adc_values)
    # Get the frequency bins
    frequencies = np.fft.fftfreq(N, 1/fs)
    ln_freq = np.log(frequencies[:N//2])

    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi'    
    # Step 4: Plotting the original signal
    plt.figure(figsize=(15, 10))

    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.plot(t, adc_values)
    plt.title(f'ADC Signal for channel {chan}, {gain_title} gain, time sample {sample}',fontsize=25)
    plt.xlabel('Time [s]',fontsize=25)
    plt.ylabel('ADC Value',fontsize=25)
    plt.xlim(0,t[-1])
    plt.grid()

    # Step 5: Plot FFT results
    plt.subplot(2, 1, 2)
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.plot(frequencies[1:N//2], np.abs(fft_result)[1:N//2])
    #plt.plot(frequencies[:N//2], np.abs(fft_result)[:50])  # Plot only positive frequencies
    plt.title('FFT of ADC Signal',fontsize=25)
    
    plt.xlabel('Frequency [Hz]',fontsize=25)
    #plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude',fontsize=25)
    plt.xlim(0, 100)  # Limit x-axis to half the sampling frequency
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'../plots/fft_{chan}_{gain}_{sample}.png')
    
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
        
    plt.figure(figsize=(13,13))
    plt.step(measure, meas_bar)
    plt.title('Mean ADC vs Measurment')
    plt.xlabel('Measurement')
    plt.ylabel('Mean ADC')
    plt.savefig(f'../plots/1d_mean_meas_hist.png')
    plt.clf()
        
    matrix[matrix == 0] = np.nan
    plt.figure(figsize=(13, 12))
    plt.title('2D Histogram of Mean ADC for all Channels and Measurements', fontsize=22)
    plt.xlabel('febChannel', fontsize=14)
    plt.ylabel('Measurement', fontsize=14)
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
    plt.colorbar(shrink=0.63, label='RMS')
    plt.tight_layout()
    plt.savefig(f'../plots/2d_mean_meas_hist.png')
    
    
def MDAC_const(file, json_file=False):
    data = get_root_data(file)
    ws_dict = {}
    if json_file != False:
        with open(json_file, 'r') as f:
            json_data = json.load(f)['protoBoardCalibs']['186974']

    j = 1
    coluta = 1
    for chan in range(128):
        if j == 9 and coluta == 1:
            j = 1
            coluta = 10
        
        elif j == 9:
            j = 1
            coluta += 1
        
        json_chan = json_data[f'coluta{coluta}'][f'channel{j}']['mdacVals']
        data_chan = data[data['febChannel'] == chan]
        ws = []
        for i in range(8):
            json_mdac = float(json_chan[f'MDACCorrectionCode{i}'])
            w_lo = np.mean(data_chan[data_chan['Measurement'] == 2*i]['ADC']) - np.mean(data_chan[data_chan['Measurement'] == 2*i + 1]['ADC'])
            w_hi = np.mean(data_chan[data_chan['Measurement'] == 2*i + 16]['ADC']) - np.mean(data_chan[data_chan['Measurement'] == 2*i + 17]['ADC'])
            ws.append(float((w_lo + w_hi)/2) - json_mdac)
        ws_dict[chan] = ws
        j += 1
        
    return ws_dict

def chi_2(file, file2, gain, channel=False):
    data_1 = get_root_data(file)
    data_2 = get_root_data(file2)
    
    if not channel:
        chis_1 = []
        chis_2 = []
        chans = []
        for chan in range(128):
            data_chan = data_1[(data_1['febChannel'] == chan) & (data_1['gain'] == gain)]
            ADC = data_chan['ADC'].to_numpy().flatten()
            bins = np.max(ADC) - np.min(ADC)
            hist, bin_edges = np.histogram(ADC, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            initial_guess = [10000, np.mean(ADC), 1]
            params, covariance = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
            
            fitted_values = []
            for x in bin_centers:
                fitted_values.append(gaussian(x,*params))
            chi_squared = np.sum(((hist - fitted_values) ** 2) / fitted_values)
            chis_1.append(chi_squared)
            chans.append(chan)
            
            data_chan = data_2[(data_2['febChannel'] == chan) & (data_2['gain'] == gain)]
            ADC = data_chan['ADC'].to_numpy().flatten()
            bins = np.max(ADC) - np.min(ADC)
            hist, bin_edges = np.histogram(ADC, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            initial_guess = [10000, np.mean(ADC), 1]
            params, covariance = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
            
            fitted_values = []
            for x in bin_centers:
                fitted_values.append(gaussian(x,*params))
            chi_squared = np.sum(((hist - fitted_values) ** 2) / fitted_values)
            chis_2.append(chi_squared)
        if gain == 0:
            gain_title = 'Lo'
            
        else:
            gain_title = 'Hi'
        
        plt.figure(figsize=(12,9))
        if np.max(chis_1) > 1 or np.max(chis_2) > 1:
            plt.yscale('log')
        plt.title(f'$\\chi^2_{{EMF}}$ and $\\chi^2_{{NEVIS}}$ vs febChannel, {gain_title} gain', fontsize=30)
        plt.xlabel('febChannel', fontsize=30, loc='right')
        plt.ylabel('$\\chi^2$', fontsize=30, loc='top')
        plt.xlim(0,127)
        plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
        plt.tick_params(axis='both', which='minor', direction='in', length=14)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
        # Add major ticks on all sides
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.step(chans, chis_1, label=f'NEVIS calibration, mean $\\chi^2$: {round(np.mean(chis_1),3)}')
        plt.step(chans, chis_2, label=f'EMF calibration, mean $\\chi^2$: {round(np.mean(chis_2),3)}')
        plt.legend(fontsize=25)
        plt.tight_layout()
        plt.savefig(f'../plots/chi_2_{gain}.png')
        
        diff = np.array(chis_2) - np.array(chis_1)

        plt.figure(figsize=(12,9))
        if np.max(diff) > 1 or np.min(diff) < -1:
            plt.ylim(-0.5, 0.5)
        plt.title(f'$\\chi^2_{{EMF}}$ - $\\chi^2_{{NEVIS}}$ vs febChannel, {gain_title} gain', fontsize=30)
        plt.xlabel('febChannel', fontsize=30, loc='right')
        plt.ylabel('$\\chi^2_{{EMF}}$ - $\\chi^2_{{NEVIS}}$', fontsize=30, loc='top')
        plt.xlim(0,127)
        #plt.yscale('log')
        plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
        plt.tick_params(axis='both', which='minor', direction='in', length=14)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
        # Add major ticks on all sides
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.step(chans, diff)
        #plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../plots/chi_2_diff_{gain}.png')
        
        return
        
    data_chan = data_1[(data_1['febChannel'] == channel) & (data_1['gain'] == gain)]
    ADC1 = data_chan['ADC'].to_numpy().flatten()
    bins = np.max(ADC1) - np.min(ADC1)
    hist_1, bin_edges_1 = np.histogram(ADC1, bins=bins, density=False)
    bin_centers_1 = 0.5 * (bin_edges_1[1:] + bin_edges_1[:-1])
    initial_guess = [1, np.mean(ADC1), 1]
    params, covariance = curve_fit(gaussian, bin_centers_1, hist_1, p0=initial_guess)
    fitted_values = []
    for x in bin_centers_1:
        fitted_values.append(gaussian(x,*params))
    chi_squared_1 = np.sum(((hist_1 - fitted_values) ** 2) / fitted_values)

    data_chan = data_2[(data_2['febChannel'] == channel) & (data_2['gain'] == gain)]
    ADC2 = data_chan['ADC'].to_numpy().flatten()
    bins = np.max(ADC2) - np.min(ADC2)
    hist_2, bin_edges_2 = np.histogram(ADC2, bins=bins, density=False)
    bin_centers_2 = 0.5 * (bin_edges_2[1:] + bin_edges_2[:-1])
    initial_guess = [1, np.mean(ADC2), 1]
    params, covariance = curve_fit(gaussian, bin_centers_2, hist_2, p0=initial_guess)
    fitted_values = []
    for x in bin_centers_2:
        fitted_values.append(gaussian(x,*params))
    chi_squared_2 = np.sum(((hist_2 - fitted_values) ** 2) / fitted_values)
    
    print(np.mean(ADC2),np.median(ADC2))
    plt.figure(figsize=(13,13))
    plt.title(f'Channel {channel} Normalized ADC, gain: {gain}', fontsize=20)
    plt.xlabel('ADC', fontsize=15)
    plt.ylabel('# Entries', fontsize=15)
    plt.tick_params(labelsize=14)
    plt.step(bin_centers_1, hist_1, label=f'NEVIS calibration, $\\chi^2$: {round(chi_squared_1,2)}')
    plt.step(bin_centers_2, hist_2, label=f'EMF calibration, $\\chi^2$: {round(chi_squared_2,2)}')
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'../plots/chi_2_hist_{channel}_{gain}.png')
    
    return

def check_calibration(file1,file2,gain):
    data_1 = get_root_data(file1)
    data_2 = get_root_data(file2)
    
    rms_1 = []
    rms_2 = []
    mean_1 = []
    mean_2 = []
    N = []
    for chan in range(128):
        data_chan_1 = data_1[(data_1['febChannel'] == chan) & (data_1['gain'] == gain)]
        data_chan_2 = data_2[(data_2['febChannel'] == chan) & (data_2['gain'] == gain)]
        
        rms_1.append(np.std(data_chan_1['ADC']))
        rms_2.append(np.std(data_chan_2['ADC']))
        mean_1.append(np.mean(data_chan_1['ADC']))
        mean_2.append(np.mean(data_chan_2['ADC']))
        N.append(len(data_chan_2['ADC'].to_numpy().flatten()))
    
    rms_1 = np.array(rms_1)
    rms_2 = np.array(rms_2)
    mean_1 = np.array(mean_1)
    mean_2 = np.array(mean_2)
    N = np.array(N)
    
    rms_diff = rms_1 - rms_2
    rms_err = np.sqrt((rms_1**2 + rms_2**2)/(2*N))
    mean_diff = np.array(mean_1) - np.array(mean_2)
    mean_err = np.sqrt((rms_1**2 + rms_2**2)/(N))
    
    if gain == 0:
        gain_title = 'Lo'
        
    else:
        gain_title = 'Hi'
    
    output_file(f'../plots/calib_rms_check_{gain}.html')
    chans = list(range(128))
    p1 = figure(title=f"Delta ADC RMS per febChannel, {gain_title} gain",
           x_axis_label=r"febChannel",
           y_axis_label=r"Delta ADC RMS",
           width=800, 
           height=700)

    for i in range(len(chans)):
        p1.segment(x0=chans[i], y0=rms_diff[i] - rms_err[i], x1=chans[i], y1=rms_diff[i] + rms_err[i], line_width=2, line_color='red')
    
    avg_rms = [np.mean(rms_diff) for i in range(128)]
    
    zeros = [0 for i in range(128)]
    
    
    
    p1.line(chans,avg_rms,  legend_label=f'Average Delta ADC RMS: {round(avg_rms[0],2)}', line_dash='dashed', line_color='red')
    p1.line(chans,zeros, legend_label='ADC RMS = 0', line_dash='dashed')
    
    # Step 4: Add scatter points
    p1.scatter(chans, rms_diff, size=5, color='navy', alpha=0.5, legend_label='Delta ADC RMS')
    show(p1)
    
    output_file(f'../plots/calib_mean_check_{gain}.html')
    p2 = figure(title=f"Delta Mean ADC per febChannel, {gain_title} gain",
           x_axis_label=r"febChannel",
           y_axis_label=r"Delta Mean ADC",
           width=800, 
           height=700)

    for i in range(len(chans)):
        p2.segment(x0=chans[i], y0=mean_diff[i] - mean_err[i], x1=chans[i], y1=mean_diff[i] + mean_err[i], line_width=2, line_color='red')
    
    
    zeros = [0 for i in range(128)]
    
    p2.line(chans,zeros, legend_label='ADC RMS = 0', line_dash='dashed')
    
    # Step 4: Add scatter points
    p2.scatter(chans, mean_diff, size=5, color='navy', alpha=0.5, legend_label='Delta Mean ADC')

    show(p2)

    
    # plt.figure(figsize=(12,9))
    # plt.title(f'$\\Delta$ ADC RMS per febChannel, {gain_title} gain', fontsize=30)
    # plt.xlabel('febChannel', fontsize=30, loc='right')
    # plt.ylabel('$\\Delta$ ADC RMS', fontsize=30, loc='top')
    # plt.xlim(0,127)
    # plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    # plt.tick_params(axis='both', which='minor', direction='in', length=14)
    # ax = plt.gca()
    # ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    # ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # # Add major ticks on all sides
    # ax.xaxis.set_ticks_position('both')
    # ax.yaxis.set_ticks_position('both')
    # plt.step(list(range(128)), rms_diff, label='$\\Delta$ ADC RMS')
    # plt.step(list(range(128)), rms_err, label='$\\sigma_{ADC RMS}$', color='orange')
    # plt.step(list(range(128)), -rms_err, color='orange')
    # plt.legend(fontsize=20, loc='upper right')
    # plt.text(x=0.01,y=0.95,s='$\\Delta ADC = NEVIS_{ADC} - EMF_{ADC}$', fontsize=17, transform=ax.transAxes)
    # plt.tight_layout()
    # plt.savefig(f'../plots/calib_rms_check_{gain}.png')
    
    # plt.figure(figsize=(12,9))
    # plt.title(f'$\\Delta$ ADC Mean per febChannel, {gain_title} gain', fontsize=30)
    # plt.xlabel('febChannel', fontsize=30, loc='right')
    # plt.ylabel('$\\Delta$ Mean ADC', fontsize=30, loc='top')
    # plt.xlim(0,127)
    # plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    # plt.tick_params(axis='both', which='minor', direction='in', length=14)
    # ax = plt.gca()
    # ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    # ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # # Add major ticks on all sides
    # ax.xaxis.set_ticks_position('both')
    # ax.yaxis.set_ticks_position('both')
    # plt.step(list(range(128)), mean_diff, label='$\\Delta$ Mean ADC')
    # plt.step(list(range(128)), mean_err, label='$\\sigma_{Mean ADC}$', color='orange')
    # plt.step(list(range(128)), -mean_err, color='orange')
    # plt.legend(fontsize=20, loc='upper right')
    # plt.text(x=0.01,y=0.95,s='$\\Delta ADC = NEVIS_{ADC} - EMF_{ADC}$', fontsize=17, transform=ax.transAxes)
    # plt.tight_layout()
    # plt.savefig(f'../plots/calib_mean_check_{gain}.png')
    
    
def quantiles(file1,file2,gain):
    data1 = get_root_data(file1)
    data2 = get_root_data(file2)
    
    percent_sigma1 = []
    percent_sigma2 = []
    diff_med1 = []
    diff_med2 = []
    
    for chan in range(128):
        data_chan1 = data1[(data1['febChannel'] == chan) & (data1['gain'] == gain)]['ADC'].to_numpy()
        med1 = np.median(data_chan1)
        rms1 = np.std(data_chan1)
        mean1 = np.mean(data_chan1)
        
        top = med1+rms1
        bot = med1-rms1
        data_bot1 = len(data_chan1[data_chan1 == int(bot)])*(1+int(bot)-bot)
        data_top1 = len(data_chan1[data_chan1 == int(bot)])*(top-int(top))
        
        data_sigma1 = data_chan1[(data_chan1 > int(bot+1)) & (data_chan1 < int(top))]
        
        percent_sigma1.append(100 * (len(data_sigma1.flatten())+data_bot1+data_top1)/len(data_chan1.flatten()))
        diff_med1.append(float(med1-mean1))
        
        data_chan2 = data2[(data2['febChannel'] == chan) & (data2['gain'] == gain)]['ADC'].to_numpy()
        med2 = np.median(data_chan2)
        rms2 = np.std(data_chan2)
        mean2 = np.mean(data_chan2)
        
        top = med2+rms2
        bot = med2-rms2
        data_bot2 = len(data_chan2[data_chan2 == int(bot)])*(1+int(bot)-bot)
        data_top2 = len(data_chan2[data_chan2 == int(bot)])*(top-int(top))
        
        data_sigma2 = data_chan2[(data_chan2 > int(bot+1)) & (data_chan2 < int(top))]
        
        percent_sigma2.append(100 * (len(data_sigma2.flatten())+data_bot2+data_top2)/len(data_chan2.flatten()))
        diff_med2.append(float(med2-mean2))
        
    if gain == 0:
        gain_title = 'Lo'
        
    else:
        gain_title = 'Hi'
    plt.figure(figsize=(12,9))
    plt.title(f'Median - Mean', fontsize=30)
    plt.xlabel('febChannel', fontsize=30, loc='right')
    plt.ylabel('Median - Mean', fontsize=30, loc='top')
    plt.xlim(0,127)
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.scatter(list(range(128)), diff_med1, label=f'Median - Mean NEVIS, avg: {round(np.mean(diff_med1),3)}')
    plt.scatter(list(range(128)), diff_med2, label=f'Median - Mean EMF, avg: {round(np.mean(diff_med2),3)}', color='orange')
    plt.legend(fontsize=20, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'../plots/median_mean_{gain}.png')
    
    plt.figure(figsize=(12,9))
    plt.title(f'% Data within $\\pm \\sigma$', fontsize=30)
    plt.xlabel('febChannel', fontsize=30, loc='right')
    plt.ylabel('% Data', fontsize=30, loc='top')
    plt.xlim(0,127)
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.scatter(list(range(128)), percent_sigma1, label=f'% Data NEVIS, avg: {round(np.mean(percent_sigma1),1)}%')
    plt.scatter(list(range(128)), percent_sigma2, label=f'% Data EMF, avg: {round(np.mean(percent_sigma2),1)}%', color='orange')
    plt.legend(fontsize=20, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'../plots/quantiles_{gain}.png')
    
    # output_file(f'../plots/median_mean_{gain}.html')
    # chans = list(range(128))
    # p1 = figure(title=f", {gain_title} gain",
    #        x_axis_label=r"febChannel",
    #        y_axis_label=r"Percent data",
    #        width=800, 
    #        height=700)

    # p1.step(chans,percent_sigma1,  legend_label=f'Percent data within +/- sigma NEVIS, Average: {round(np.mean(percent_sigma1),1)}%', line_color='red')
    # #p1.step(chans,percent_sigma2,  legend_label=f'Percent data within +/- sigma EMF, Average: {round(np.mean(percent_sigma2),1)}%', line_color='blue')
    
    # # Step 4: Add scatter points
    # show(p1)
    
def odd_even(file1,file2,gain):
    data1 = get_root_data(file1)
    data2 = get_root_data(file2)
    
    even_odd1 = []
    even_odd2 = []
    for chan in range(128):
        data_chan1 = data1[(data1['febChannel'] == chan) & (data1['gain'] == gain)]['ADC'].to_numpy()
        
        data_even1 = data_chan1[data_chan1 % 2 == 0]
        
        even_odd1.append(100*len(data_even1.flatten())/len(data_chan1.flatten()))
        
        data_chan2 = data2[(data2['febChannel'] == chan) & (data2['gain'] == gain)]['ADC'].to_numpy()

        data_even2 = data_chan2[data_chan2 % 2 == 0]
        
        even_odd2.append(100*len(data_even2.flatten())/len(data_chan2.flatten()))
        
    if gain == 0:
        gain_title = 'Lo'
        
    else:
        gain_title = 'Hi'
    
    output_file(f'../plots/odd_even_{gain}.html')
    chans = list(range(128))
    p1 = figure(title=f"Percent Even ADC, {gain_title} gain",
           x_axis_label=r"febChannel",
           y_axis_label=r"100*len(ADC_even)/len(ADC)",
           width=800, 
           height=700)

    p1.step(chans,even_odd1,  legend_label=f'% Even ADC NEVIS, Average: {round(np.mean(even_odd1),1)}', line_color='red')
    p1.step(chans,even_odd2,  legend_label=f'% Even ADC EMF, Average: {round(np.mean(even_odd2),1)}', line_color='blue')
    
    # Step 4: Add scatter points
    show(p1)