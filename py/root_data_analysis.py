#import ROOT
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
file_1 = "../data/output-data-fullfeb2v2-20240910.root:Data"

file_2 = "../data/output-data-MXSX-hct20l-2024-09-18.root:Data"

file_3 = "../data/output-data-MXSX-hct20l-2024-09-23.root:Data"

def get_root_data(file):
    '''
    Extracts data from root file into a pandas data frame
    '''
    
    with uproot.open(file) as f:
        print(f.keys())
        
        data = f.arrays(['iEvent', 'bcid', 'febId', 'buffChannel', 'febChannel', 'ADC', 'gain'], library='pd')
        
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
    plt.title('ATLAS Internal/EMF System Test')
    plt.xlabel('FebChannel')
    plt.ylabel('Mean Pedestal [ADC]')
    plt.ylim(5000,7500)
    plt.xlim(0,127)
    plt.step(febs,ADC_mean_h, color='pink', linewidth=2, label='High Gain')
    plt.step(febs,ADC_mean_l, color='blue', linewidth=2, label='Low Gain')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../plots/mean_system_test.png')
    plt.clf()
    
    #plotting stdev vs feb
    plt.figure(figsize=(12,8))
    plt.title('ATLAS Internal/EMF System Test')
    plt.xlabel('FebChannel')
    plt.ylabel('Standard Deviation')
    plt.ylim(0,20)
    plt.xlim(0,127)
    plt.step(febs,ADC_std_h, color='pink', linewidth=2, label='High Gain')
    plt.step(febs, ADC_std_l, color='blue', linewidth=2, label='Low Gain')
    plt.legend()
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
    
    #plotting
    plt.figure(figsize=(15,15))
    plt.title(f'FebChannel Pearson Correlation Coefficient Matrix, gain: {gain}', fontsize=20)
    plt.ylim(0, 127)
    plt.xlabel('febChannel', fontsize=14)
    plt.ylabel('febChannel', fontsize=14)
    plt.imshow(matrix, vmin=-max_v,vmax=max_v, cmap='bwr')
    plt.colorbar(shrink=0.785)
    plt.tight_layout()
    plt.savefig(f'../plots/{auto_range}_{file.split('/')[-1]}_channel_gain{gain}_correlation_fullavg.png')

    return matrix

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
    d_ch_noise = 0
    dataSum = []
    
    #looping over all channels
    N_channels = np.max(data['febChannel']) + 1
    for chan in range(N_channels):
        data_chan = data[(data['febChannel']==chan) & (data['gain']==gain)]['ADC'].to_numpy().flatten()
        data_dic[chan] = data_chan
        rms[chan] = np.std(data_chan-np.mean(data_chan))
        d_rms[chan] = rms[chan]/np.sqrt(len(data_chan))
        ch_noise += rms[chan] ** 2
        d_ch_noise += d_rms[chan] ** 2
    #calculating noise
    ch_noise = np.sqrt(ch_noise)
    d_ch_noise = np.sqrt(d_ch_noise)
    avg_noise = ch_noise / np.sqrt(N_channels)
    d_avg = d_ch_noise / np.sqrt(N_channels)
    
    #calculating hist data
    gain_df_channels = list(range(N_channels))
    data_by_channel = np.array([data_dic[ch] for ch in gain_df_channels])
    channel_means = np.mean(data_by_channel, axis=1)
    dataSum = np.sum([(data_dic[ch] - channel_means[i]) for i, ch in enumerate(gain_df_channels)], axis=0)
    
    tot_noise = np.std(dataSum)
    mu = round(np.mean(dataSum), 3)
    std = round(np.std(dataSum), 3)
    coh_noise = np.sqrt(tot_noise**2 - ch_noise**2) /N_channels
    pct_coh = coh_noise / avg_noise * 100
    
    #plotting
    plt.figure(figsize=(13,13))
    y, x, _ = plt.hist(
        dataSum,
        bins=np.arange(min(dataSum), max(dataSum) + 2, 1),
        color="black",
        edgecolor="black",
        density=False,
        label=rf"RMS = {np.round(tot_noise,3)}, $\mu$ = {np.round(mu,3)}, $\sigma$ = {np.round(std, 3)}",
    )
    
    plt.text(0.05, 0.95,f'Entries = {round(len(dataSum)/1000,1)}k\nN ch = {N_channels}\nMean = {round(mu,1)}\nRMS = {round(tot_noise,1)}\nAvg noise/ch = {round(avg_noise,1)}\nCohe noise/ch = {round(coh_noise,1)}\n[%]Cohe noise = {round(pct_coh,1)}', transform=plt.gca().transAxes,fontsize=15, verticalalignment='top', horizontalalignment='left')
    plt.title(f'EMF system test, coherence, gain: {gain}', fontsize=20)
    
    plt.xlabel('ADC counts', fontsize=15)
    plt.ylabel('Entries', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'../plots/coherence_hist_{gain}.png')