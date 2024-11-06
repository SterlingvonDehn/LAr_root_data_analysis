#import ROOT
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import norm
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from bokeh.plotting import figure, show
from bokeh.io import output_file
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
    
    if not os.path.exists(f'plots_{file.split('/')[-1]}'):
         os.mkdir(f'plots_{file.split('/')[-1]}')
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
    plt.savefig(f'plots_{file.split('/')[-1]}/mean_system_test.png')
    plt.clf()

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
    plt.savefig(f'plots_{file.split('/')[-1]}/std_system_test.png')
    plt.clf()
    
    #plotting num entries vs feb 
    #USED TO CHECK FIRST DATA SETS
    # plt.figure(figsize=(12,8))
    # plt.title('ATLAS Internal/EMF System Test')
    # plt.xlabel('FebChannel')
    # plt.ylabel('# of entries')
    # plt.xlim(0,127)
    # plt.step(febs,num_ent_h, color='pink', linewidth=2, label=f'High Gain, total: {len(data_high)}')
    # plt.step(febs,num_ent_l, color='blue', linewidth=2, label=f'Low Gain, total: {len(data_low)}')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('../plots/num_entries_system_test.png')
    # plt.clf()
    
def correlation(file, gain, auto_range):
    '''
    Produces a Pearson Correlation matrix plot for febChannels in the data set.
    file == root file path
    gain == 0 or 1 -> 0 (lo) 1 (hi)
    auto_range == float -> if 0 will automatically generate range
    '''
    data = get_root_data(file) #extracting data
    matrix = np.zeros((128,128)) #creating empty matrix
    
    #calculating correlation coefficient for all channel combinations
    for i in range(128):
        for j in range(i):
            data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain)]
            data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain)]
            #Used if only one time entrie is desired


            #creating arrays of data
            A = np.array(data_chan_i['ADC'].to_numpy().flatten(), dtype=np.float64)
            B = np.array(data_chan_j['ADC'].to_numpy().flatten(), dtype=np.float64)

            if len(A) != len(B) or not np.array_equal(data_chan_i['iEvent'].to_numpy(),data_chan_j['iEvent'].to_numpy()): #skipping if lenght A != lenght B or if iEvent order is not the same
                continue
            
            #calculating Pearson correlation coefficient
            corr_coefficient = pearsonr(A,B)[0]
            
            #populating matrix depending on auto_range settings
            matrix[i][j] = corr_coefficient
            matrix[j][i] = corr_coefficient
    
        print(f'{round((i/127)*100, 1)}%') #progress tracker
    
    #defining color range if auto_range == True
    if auto_range == 0:
        max_v = max([abs(np.max(matrix)), abs(np.min(matrix))])
    else:
        max_v = auto_range
    
    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi'
    
    
    if not os.path.exists(f'plots_{file.split('/')[-1]}'):
        os.mkdir(f'plots_{file.split('/')[-1]}')
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
    plt.savefig(f'plots_{file.split('/')[-1]}/pears_correlation_gain{gain}.png', bbox_inches='tight')

    return matrix    
    
def correlation_diff_event(file, gain, auto_range, event, plot=True):
    '''
    Plots a correlation matrix for unrelated events
    file == root file path
    gain == 0 or 1 -> 0 (lo) 1 (hi)
    auto_range == float -> if 0 will automatically generate range
    event == "even_odd" (comapares even entries to odd), integer (compares event i with event i+integer), "half" compares event i with event i+1/2max_event, or "n_event" used to calculate RMS for different number of entries
    '''
    data = get_root_data(file) #extracting data
    matrix = np.zeros((128,128)) #creating empty matrix
    
    #calculating correlation coefficient for all channel combinations
    data = data[data['gain']==gain]
    max_event = np.max(data['iEvent'])
    min_event = np.min(data['iEvent'])
    half = (max_event - min_event - 1)/2
    mid = min_event + half

    if type(event) == int:
        event_list_1 = []
        event_list_2 = []
        i = 0
        base = 0
        for i in range((max_event-min_event-2)//(2*event) - 1):
            base = i * event*2 + 1
            if i == 0:
                event_list_1.extend(range(base+min_event, base + event+min_event))
                event_list_2.extend(range(base+event+min_event, base + event*2+min_event)) 
            event_list_1.extend(range(base+event*2+min_event, base+event*3+min_event))
            event_list_2.extend(range(base+event*3+min_event, base+event*4+min_event))
            i += 1


    event_nums = half
        
    corr_hist = []
    for i in range(128):
        for j in range(128):
            #compares even events to odd events
            if event == "even_odd":
                data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain)].iloc[1::2]
                data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain)].iloc[::2]
            
            #compares event i to event i+event
            elif type(event) == int:
                data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain) & (data['iEvent'].isin(event_list_1))]
                data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain) & (data['iEvent'].isin(event_list_2))]

            #compares event i to event i+1/2max_event
            elif event == 'half':
                data_chan_i = data[(data['febChannel']==i) & (min_event <= data['iEvent']) & (data['iEvent'] < min_event + event_nums)]
                data_chan_j = data[(data['febChannel']==j) & (mid < data['iEvent']) & (data['iEvent'] <= mid + event_nums)]

            #Used to check statistical properties of correlations
            elif event[0] == 'n_event':
                data_chan_i_half = data[(data['febChannel']==i) & (min_event <= data['iEvent']) & (data['iEvent'] < min_event + event_nums)]
                data_chan_j_half = data[(data['febChannel']==j) & (mid < data['iEvent']) & (data['iEvent'] <= mid + event_nums)]
                
                data_chan_i = data_chan_i_half.head(event[1])
                data_chan_j = data_chan_j_half.head(event[1])
            
            else:
                print(r'Invalid event, must be one of: "even_odd", an integer, or "half"')
                return

            #creating arrays of data
            A = np.array(data_chan_i['ADC'].to_numpy().flatten(), dtype=np.float64)
            B = np.array(data_chan_j['ADC'].to_numpy().flatten(), dtype=np.float64)
            
            if len(A) != len(B): #skipping if arrays are different lenghts 
                continue
            
            #calculating Pearson correlation coefficient
            corr_coefficient = pearsonr(A,B)[0]
            
            #populating matrix 
            matrix[i][j] = corr_coefficient
            
            corr_hist.append(corr_coefficient)

        print(f'{round((i/127)*100, 1)}%') #progress tracker
    
    N = len(A)
    #defining color range if auto_range == 0
    if auto_range == 0:
        max_v = max([abs(np.max(matrix)), abs(np.min(matrix))])
    else:
        max_v = auto_range
    
    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi'
    
    if plot:
        if not os.path.exists(f'plots_{file.split('/')[-1]}'):
            os.mkdir(f'plots_{file.split('/')[-1]}')
        #plotting
        plt.figure(figsize=(15,15))
        plt.title(f'Unrelated Correlation, {gain_title} gain', fontsize=30)
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
        cbar = plt.colorbar(shrink=0.72)
        cbar.ax.tick_params(labelsize=20)  # Set the fontsize for the colorbar ticks
        cbar.set_label('Correlation Coeff', fontsize=30)
        plt.tick_params(labelsize=25)
        plt.tight_layout()
        plt.savefig(f'plots_{file.split('/')[-1]}/unrelated_correlation_gain{gain}_event_{event}.png', bbox_inches='tight')
        
        
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
        plt.hist(corr_hist, bins=70, edgecolor='black',label=f'N = {N}, $\\sigma$ = {round(np.std(corr_hist),3)}, Mean = {round(np.mean(corr_hist),3)}')
        plt.legend(fontsize=25)
        plt.tight_layout()
        plt.savefig(f'plots_{file.split('/')[-1]}/unrelated_correlation_histo_gain{gain}_event_{event}.png', bbox_inches='tight')

    return [N, np.std(corr_hist)]

def corr_vs_n(file, events, gain):
    '''
    Calculates and plots the correlation coefficient RMS vs the number of entries used to calculate them.
    file == path to data file
    events == list of number of events to be used, max must be less then half total entries
    gain == 1 or 0
    '''
    
    #calculating RMS for each number of entries
    ns = []
    stds = []
    for i in events:
        data = correlation_diff_event(file, gain, 0, ('n_event', int(i)), False)
        ns.append(data[0])
        stds.append(data[1])
        print(i)
    
    #getting theoretical line (slope 1/2)
    a = ((np.log(stds[0]) + np.log(ns[0])/2) + (np.log(stds[-1]) + np.log(ns[-1])/2))/2
    y = lambda x: -x/2 + a
    
    #Plotting
    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi'
    plt.figure(figsize=(15,12))
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.xlim(np.min(np.log(ns)), np.max(np.log(ns)))
    plt.plot(np.log(ns), np.log(stds), label='ln(RMS)')
    plt.plot(np.log(ns),y(np.log(ns)),linestyle=':', label='$y=\\frac{-x}{2}$ ' + f'+ {round(a,2)}')
    plt.xlabel('ln(N)', fontsize=30, loc='right')
    plt.ylabel('ln(RMS)', fontsize=30, loc='top')
    plt.title(f'ln(RMS) vs ln(N) for Unrelated Events, {gain_title} gain', fontsize=30)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'../plots/{gain}_lnrms_lnN.png')
    
    return(ns,stds)

def ADC_event(file, gain, event_range=False):
    '''
    Plots a 2d histogram of ADC vs events for specified channel and gain
    chan = channel #
    gain = 1 or 0
    event_range = (start_entry,end_entry), automatic if not specified
    '''
    data = get_root_data(file) #extracts data
    
    for chan in range(128):
        data_feb = data[(data['febChannel']==chan) & (data['gain']==gain)] #selects channel and gain
        data_adc = data_feb['ADC'].to_numpy()

        #selecting range
        if not event_range:
            start_event = np.min(data_feb['iEvent'])
            end = np.max(data_feb['iEvent']) - start_event
            start = 0
        else:
            start = event_range[0]
            end = event_range[1] 
        
        events = []
        adc = []
        for i in range(start,end):
            for j in range(len(data_adc[i])):
                events.append(i)
                adc.append(data_adc[i][j])
        
        #calculating #bins
        adc_bins = int(np.max(adc) - np.min(adc))
        if gain == 0:
            gain_title = 'Lo'
        else:
            gain_title = 'Hi'
        
        #plotting hist
        plt.figure(figsize=(15,12))
        plt.tick_params(labelsize=30,axis='both', which='major', direction='in', length=20)
        # Set minor ticks
        plt.tick_params(axis='both', which='minor', direction='in', length=15)
        # Create minor ticks
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
        # Add major ticks on all sides
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.title(f'ADC:iEvent, {gain_title} gain & febChannel {chan}', fontsize=30)
        plt.xlabel('iEvent', fontsize=30, loc='right')
        plt.ylabel('ADC', fontsize=30, loc='top')
        hist = plt.hist2d(events,adc,bins=[30,adc_bins], cmin=1)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)  # Set the fontsize for the colorbar ticks
        cbar.set_label('# Entries', fontsize=30)
        
        #calcualting avg and stdev per bin
        data_bin = data_feb[(data_feb['iEvent'] <= hist[1][1]) & (data_feb['iEvent'] > hist[1][0])]
        avg_adc_0 = np.mean(data_bin['ADC'])
        avg_adc = [avg_adc_0]
        bin_lst = [-1]
        if not event_range:
            bin_width = np.max(data_feb['iEvent'])/(len(hist[1]) - 1)
        else:
            bin_width = (event_range[1] - event_range[0])/(len(hist[1]) - 1)
            
        for i in range(len(hist[1]) - 1):
            data_bin = data_feb[(data_feb['iEvent'] <= hist[1][i+1]) & (data_feb['iEvent'] > hist[1][i])]
            avg_adc.append(np.mean(data_bin['ADC']))
            std_up_adc = avg_adc[-1] + np.std(data_bin['ADC'])/np.sqrt(len(data_bin['ADC'].to_numpy().flatten()))
            std_down_adc = avg_adc[-1] - np.std(data_bin['ADC'])/np.sqrt(len(data_bin['ADC'].to_numpy().flatten()))
            bin_lst.append(hist[1][i+1])
            plt.plot([bin_lst[-1]-bin_width/2,bin_lst[-1]-bin_width/2],[std_up_adc,std_down_adc],color='orange')

        #plotting avg and stdev
        plt.step(bin_lst,avg_adc,color='red', label='Mean per bin')
        plt.step(np.nan,np.nan,color='orange', label='Mean error per bin')
        plt.legend(fontsize=20)
        plt.tight_layout()
        if not os.path.exists(f'../plots_{file.split('/')[-1]}/ADC_iEvent_2d'):
            os.mkdir(f'../plots_{file.split('/')[-1]}/ADC_iEvent_2d')
        plt.savefig(f'../plots_{file.split('/')[-1]}/ADC_iEvent_2d/channel_{chan}_gain{gain}.png', bbox_inches='tight')
        plt.clf()

def auto_correlation(file, gain, chan, auto_range=True):
    '''
    NOT FUNCTIONAL, ROUGH DRAFT
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
    file == path to root file
    gain == 1 or 0
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
    plt.plot(lnspc, p, color='red', linewidth=2, label=f'Gauss fit, $\\mu$ = {round(mu,2)}, $\\sigma$ = {round(std,1)}')
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.text(0.05, 0.95,f'Entries = {round(len(dataSum)/1000,1)}k\nN ch = {N_channels}\nMean = {round(mu,1)}\nRMS = {round(tot_noise,1)}\n$\\sigma$ inco = {round(ch_noise,1)}\nCohe noise/ch = {round(coh_noise,4)}\nAvg noise/ch = {round(avg_noise_nev,4)}\n[%]Cohe noise = {round(pct_coh_nev,4)}', transform=plt.gca().transAxes,fontsize=20, verticalalignment='top', horizontalalignment='left')
    plt.legend(fontsize=20)
    plt.title(f'EMF system test, coherent, {gain_title} gain', fontsize=30)
    
    plt.xlabel('ADC counts', fontsize=30, loc='right')
    plt.ylabel('Entries', fontsize=30, loc='top')
    plt.tight_layout()
    plt.savefig(f'plots_{file.split('/')[-1]}/coherence_hist_{gain}.png', bbox_inches='tight')
    
def coh_corr(file,gain):
    '''
    Calculates coherent noise using measured covariances, good check for function above
    file == path to root file
    gain == 1 or 0
    '''
    data = get_root_data(file) #extracting data
    matrix = np.zeros((128,128)) #creating empty matrix
    
    #calculating correlation coefficient for all channel combinations
    corr_sum = 0
    for i in range(128):
        for j in range(i):
            data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain)]
            data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain)]
            #Used if only one time entrie is desired
            
            #creating arrays of data
            A = np.array(data_chan_i['ADC'].to_numpy().flatten(), dtype=np.float64)
            B = np.array(data_chan_j['ADC'].to_numpy().flatten(), dtype=np.float64)

            if len(A) != len(B) or not np.array_equal(data_chan_i['iEvent'].to_numpy(),data_chan_j['iEvent'].to_numpy()): #skipping if arrays are different lenghts 
                continue
            
            corr_coefficient = (np.mean(A*B) - np.mean(A)*np.mean(B))
            
            #populating matrix depending on auto_range settings
            corr_sum += corr_coefficient

        print(f'{round((i/127)*100, 1)}%') #progress tracker
    
    return np.sqrt(2*corr_sum)/128 #returnes coherent noise per chan calculated using measured covariences

def covariance(file):
    '''
    Plots covariance matrixes for both hi and lo gain
    file == path to root file
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

            #creating arrays of data
            A_hi = np.array(data_chan_i_hi['ADC'].to_numpy().flatten(), dtype=np.float64)
            B_hi = np.array(data_chan_j_hi['ADC'].to_numpy().flatten(), dtype=np.float64)
            A_lo = np.array(data_chan_i_lo['ADC'].to_numpy().flatten(), dtype=np.float64)
            B_lo = np.array(data_chan_j_lo['ADC'].to_numpy().flatten(), dtype=np.float64)

            if len(A_hi) != len(B_hi) or not np.array_equal(data_chan_i_hi['iEvent'].to_numpy(),data_chan_j_hi['iEvent'].to_numpy()): #skipping if arrays are different lenghts 
                continue
            
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

def FFT_avg(file,gain,sample):
    '''
    Takes the average FFT value over all febChannels
    file == path to root file
    gain == 1 or 0
    sample == desired time sample to take for each event (0 to 24)
    '''
    #extract data
    data = get_root_data(file)
    
    fs = 200 #defining sample rate 
    
    #calculating FFT for all febChannels
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
    
def FFT(file,gain,sample):
    '''
    Plots FFTs for all febChannels
    file == path to root file
    gain == 1 or 0
    sample == desired time sample to take for each event (0 to 24)
    '''
    
    data = get_root_data(file)
    
    for chan in range(128):
        data_chan = data[(data['febChannel']==chan) & (data['gain']==gain)]

        signal = []
        for entry in data_chan['ADC']:
            signal.append(entry[sample])
        
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
        plt.savefig(f'plots_{file.split('/')[-1]}/FFTs/channel_{chan}_gain{gain}_sample{sample}.png')

def FFT_hist(file,gain,sample, bin_num):
    '''
    Plot the max power spectrum in specified freq bins for all febChannels. Used to get broad understading of all channels FFTs
    file == path to root file
    gain == 1 or 0
    sample == desired time sample to take for each event (0 to 24)
    bin_num == number of frequency bins
    '''
    
    data = get_root_data(file)
    matrix = np.zeros((bin_num,128))
    fs = 200  # Sampling frequency in Hz
    
    #getting all FFTs
    for chan in range(128):
        data_chan = data[(data['febChannel']==chan) & (data['gain']==gain)]

        adc_values= []
        for entry in data_chan['ADC']:
            adc_values.append(entry[sample])
        N = len(adc_values)  # Number of samples
        # Step 2: Create a time vector
        t = np.arange(N) / fs  # Time vector based on the number of samples
        # Step 3: Compute FFT
        fft_result =  np.abs(np.fft.fft(adc_values))[1:N//2]
        # Get the frequency bins
        frequencies = np.fft.fftfreq(N, 1/fs)[1:N//2]
        num_elements = int(len(frequencies)/bin_num)
        j = 0
        for i in range(bin_num):
            bin_avg = np.max(fft_result[j:j+num_elements])
            matrix[i][chan] = bin_avg
            j += num_elements
    
    #setting 0s to nan and plotting
    matrix[matrix == 0] = np.nan
    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi' 
    plt.figure(figsize=(15,12))
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.title(f'Max FFT Magnitude for febChannels, {gain_title} gain', fontsize=30)
    plt.xlabel('febChanel', fontsize=30,loc='right')
    plt.ylabel('Frequency [Hz]', fontsize=30,loc='top')
    plt.ylim(0,bin_num)
    plt.imshow(matrix, extent=[0, matrix.shape[1], matrix.shape[0],0],aspect=4)
    cbar = plt.colorbar(shrink=0.6)
    cbar.ax.tick_params(labelsize=20)  # Set the fontsize for the colorbar ticks
    cbar.set_label('Max FFT Magnitude', fontsize=30)

    #Fxing ticks
    old_tick = []
    new_tick = []
    for i in range(6):
        old_tick.append(i*(bin_num/5))
        new_tick.append(int(i*20))
    plt.yticks(ticks=old_tick, labels=new_tick)
    plt.tight_layout()
    plt.savefig(f'plots_{file.split('/')[-1]}/FFT_hist_{gain}.png', bbox_inches='tight')
    return matrix
    
def chi_2(file, file2, gain, channel=False):
    '''
    Plots the difference in a chi^2 gaussian fit of ADC distribution of two different root files to check if data has improved
    file == path to root file
    file2 == path to root file
    gain == 1 or 0
    channel == Used to plot ADC hist of both files for specified channel
    '''
    #getting data
    data_1 = get_root_data(file)
    data_2 = get_root_data(file2)
    
    #plotting chi^2 diff if channel not specified
    if not channel:
        chis_1 = []
        chis_2 = []
        chans = []
        #fitting all channels with a gaussian
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
        
        #plotting
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
    #if channel arg given then plotting ADC hist for both files with gaussian fits    
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
    '''
    Checks effectiveness of calibration between two root files
    file1 == path to first root file
    file2 == path to second root file
    gain == 1 or 0
    '''
    data_1 = get_root_data(file1)
    data_2 = get_root_data(file2)
    
    #getting means and rms for both data sets for all febChannels
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
    
    #getting diff and errors
    rms_diff = rms_1 - rms_2
    rms_err = np.sqrt((rms_1**2 + rms_2**2)/(2*N))
    mean_diff = np.array(mean_1) - np.array(mean_2)
    mean_err = np.sqrt((rms_1**2 + rms_2**2)/(N))
    
    #plotting html
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

    
    #plotting matplotlib
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
    '''
    NOT FUNCTIONAL
    Checking quantiles
    '''
    
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
    '''
    NOT FUNCTIONAL
    Used to study odd even effect
    '''
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

if __name__ == "__main__":
    file_name = args[1]
    if not os.path.exists(f'plots_{file_name.split('/')[-1]}'):
        os.mkdir(f'plots_{file_name.split('/')[-1]}')
    EMF_system_test(file_name)
    for gain in range(2):
        correlation(file_name,gain,0)
        correlation_diff_event(file_name,gain,0, 'even_odd')
        correlation_diff_event(file_name,gain,0, 'half')
        coherent_noise(file_name,gain)
        FFT_hist(file_name,gain,0,20)

        if not os.path.exists(f'plots_{file_name.split('/')[-1]}/ADC_iEvent_2d'):
            os.mkdir(f'plots_{file_name.split('/')[-1]}/ADC_iEvent_2d')
        if not os.path.exists(f'plots_{file_name.split('/')[-1]}/FFTs'):
            os.mkdir(f'plots_{file_name.split('/')[-1]}/FFTs')
        ADC_event(file_name, gain)
        FFT(file_name,gain,0)
    
        