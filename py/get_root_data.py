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
        
    return  data

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
        for j in range(128):
            data_chan_i = data[(data['febChannel']==i) & (data['gain']==gain)]['ADC'].to_numpy().flatten() #writing adc to an array
            data_chan_j = data[(data['febChannel']==j) & (data['gain']==gain)]['ADC'].to_numpy().flatten()
            
            # A = []
            # B = []
            
            # #taking first time entry for each event
            # for n in data_chan_i:
            #     A.append(n[q])
            # for m in data_chan_j:
            #     B.append(m[q])
            
            #creating arrays of data
            A = np.array(data_chan_i, dtype=np.float64)
            B = np.array(data_chan_j, dtype=np.float64)
            if len(A) != len(B): #skipping if arrays are different lenghts 
                continue
            
            #corr_coefficient = (np.mean(A*B) - np.mean(A)*np.mean(B))
            #/(np.std(A)*np.std(B)) #calculating Pearson correlation coefficient
            corr_coefficient = pearsonr(A,B)[0]
            
            #populating matrix depending on auto_range settings
            if auto_range:
                if i != j:
                    matrix[i][j] = corr_coefficient
            else:
                matrix[i][j] = corr_coefficient

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

def ADC_event(file, chan, gain):
    '''
    Plots a 2d histogram of ADC vs events for specified channel and gain
    STILL WORKING, NOT FUNCTIONAL 
    '''
    data = get_root_data(file) #extracts data
    data_feb = data[(data['febChannel']==chan) & (data['gain']==gain)] #selects channel and gain
    data_event = data_feb['iEvent'].to_numpy() 
    data_adc = data_feb['ADC'].to_numpy()
    
    events = []
    adc = []
    for i in range(900,1100,1):
        for j in range(len(data_adc[i])):
            events.append(i)
            adc.append(data_adc[i][j])
    
    adc_bins = int(np.max(adc) - np.min(adc))
    plt.figure(figsize=(15,12))
    plt.title(f'ADC:iEvent (gain=={gain} & febChannel=={chan})', fontsize=20)
    plt.xlabel('iEvent', fontsize=14)
    plt.ylabel('ADC', fontsize=14)
    cmap = plt.cm.viridis  # Use the 'viridis' colormap
    hist = plt.hist2d(events,adc,cmap=cmap,bins=[30,adc_bins], cmin=1)
    plt.colorbar()
    
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

    plt.plot(bin_lst,avg_adc,color='red', label='Mean/bin')
    plt.plot(bin_lst,std_up_adc,color='red', linestyle='--', label='Stdev/bin')
    plt.plot(bin_lst,std_down_adc,color='red', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../plots/ADC_iEvent_{chan}_{gain}.png')
        
    # # Set up the plot
    # plt.figure(figsize=(13, 10))
    # # Create a custom colormap
    # cmap = plt.cm.viridis  # Use the 'viridis' colormap
    # cmap.set_under(color='white', alpha=0)  # Set color for under zero counts to white and transparent

    # # Plot the 2D histogram
    # plt.imshow(hist.T, origin='lower', cmap=cmap, extent=[0, 10000, 6000, 6100], vmin=1,aspect='auto')
    # plt.step(bin_lst,avg_adc, color='red')
    # # Add color bar and labels
    # plt.colorbar(label='Counts')
    # plt.xlabel('iEevent')
    # plt.ylabel('ADC')
    # plt.title('ADC:iEvent{gain=1 & febChannel=20}')

    # # Show the plot
    # plt.show()
    #print(data_feb['ADC'].to_numpy())
    #plt.hist2d(data_feb['iEvent'].to_numpy(),data_feb['ADC'].to_numpy())
    # ievents = []
    # adc_h = []
    # adc_l = []
    # data_high = data[data['gain'] == 1]
    # data_low = data[data['gain'] == 0]
    # for i in range(5038):
    #     data_event_h = data_high[data_high['iEvent'] == i]
    #     data_event_l = data_low[data_low['iEvent'] == i]
    #     ievents.append(i)
    #     adc_h.append(np.mean(data_event_h['ADC']))
    #     adc_l.append(np.mean(data_event_l['ADC']))
    
    # plt.figure(figsize=(12,8))
    # plt.title('ATLAS Internal/EMF System Test')
    # plt.xlabel('iEvent')
    # plt.ylabel('Mean Pedestal [ADC]')
    # plt.xlim(0,5038)
    # plt.step(ievents,adc_h, color='pink', linewidth=2, label='High Gain')
    # plt.step(ievents,adc_l, color='blue', linewidth=2, label='Low Gain')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('../plots/adc_ievents_test.png')
    # plt.clf()
        

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
    
    #getting data for first time entry for each event
    # adc_i = []
    # adc_j = []
    # for adc in data_i['ADC']:
    #     adc_i.append(adc[0])
    # for adc in data_j['ADC']:
    #     adc_j.append(adc[0])
    
    A = np.array(data_i, dtype=np.float64)
    B = np.array(data_j, dtype=np.float64)
    
    corr_coefficient = (np.mean(A*B) - np.mean(A)*np.mean(B))/(np.std(A)*np.std(B))
    
    # data = np.vstack((A,B)).T

    # # Calculate the mean and covariance matrix
    # mean = np.mean(data, axis=0)
    # cov_matrix = np.cov(data, rowvar=False)

    # # Calculate eigenvalues and eigenvectors
    # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # # Calculate the angle of rotation for the ellipse
    # angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * (180 / np.pi)

    # # Calculate the width and height of the ellipse
    # # The factor 2 is used to represent the full width and height
    # width = 2 * np.sqrt(eigenvalues[1])  # For the larger eigenvalue
    # height = 2 * np.sqrt(eigenvalues[0])  # For the smaller eigenvalue

    # # Create a figure and axis
    # # Create and add the ellipse
    # ellipse = Ellipse(mean, width, height, angle=45, edgecolor='blue', facecolor='none', linewidth=2)
    # ecent = np.sqrt(-((height/2 - width/2)**2 - 1))
    # ecent_2 = np.sqrt(1-corr_coefficient**2)
    # print(ecent, ecent_2)
    # # eccentricity = np.sqrt(1-corr_coefficient)
    # semi_major_axis = np.std(A) 
    # semi_minor_axis = semi_major_axis * np.sqrt(1 - eccentricity**2)
    
    # ellipse = Ellipse((np.mean(adc_i),np.mean(adc_j)), 2*semi_major_axis, 2*semi_minor_axis,angle=45)
    
    # #plotting
    A_bins = int(np.max(A) - np.min(A))
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
    '''
    data = get_root_data(file)
    hist_data = []

    # Collect data for each channel
    for chan in range(128):
        data_chan = data[(data['febChannel'] == chan) & (data['gain'] == gain)]['ADC'].to_numpy().flatten()
        hist_data.extend(data_chan - np.mean(data_chan))

    # Create histogram 
    hist, bin_edges = np.histogram(hist_data, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit the histogram with a Gaussian
    initial_guess = [np.max(hist), np.mean(hist_data), np.std(hist_data)]
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)

    # Plot the histogram and the fitted Gaussian
    plt.figure(figsize=(13,13))
    plt.hist(hist_data, bins=100, alpha=0.6, color='black')
    plt.plot(bin_centers, gaussian(bin_centers, *popt), color='red', label=f'Gaussian Fit, $\mu$={round(popt[2],3)}, $\sigma$={round(popt[1],1)}', linewidth=2)
    plt.title(f'Channels 0-127, gain: {gain}', fontsize=20)
    plt.xlabel('ADC counts', fontsize=12)
    plt.ylabel('Entries', fontsize=12)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'../plots/coherent_noise_{gain}.png')
    