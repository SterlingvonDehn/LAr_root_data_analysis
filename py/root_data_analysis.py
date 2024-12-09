import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import norm
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
import sys
import os

args = sys.argv

file_name = '' #Enter path to root file here

sample_rate = None #Enter event sample rate in Hz here

gains = [] #Enter gain titles here in list for. Ie if only hi gain enter [1]

if not os.path.exists(file_name):
    print('ENTER VALID FILE NAME')
    sys.exit()
    
if sample_rate == None:
    print('ENTER A SAMPLE RATE')
    sys.exit()

if len(gains) == 0:
    print('ENTER VALID GAIN VALUES')
    sys.exit()

plot_dir = f'plots_{file_name.split("/")[-1]}'
os.makedirs(plot_dir, exist_ok=True)

file_size = os.path.getsize(file_name)

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

def mean_rms_plotter(file):
    '''
    Plots mean pedestal ADC, stdev ADC, and number of entries vs febChannel for hi and lo gain. 
    '''
    
    chan_means_hi = {key: [] for key in range(128)}
    chan_means_lo = {key: [] for key in range(128)}
    
    if file.split(':')[-1] != 'Data':
        file = f'{file}:Data' 

    with uproot.open(file) as f:
        print(f.keys())
    
        tree = f
        # Define the chunk size (number of entries per chunk)
    for chan in range(128):
        for data in tree.iterate(step_size=640000):
            data_chan_hi = data[(data['febChannel'] == chan) & (data['gain'] == 1)]['ADC'].to_numpy()
            data_chan_lo = data[(data['febChannel'] == chan) & (data['gain'] == 0)]['ADC'].to_numpy()
            
            chan_means_hi[chan].extend(data_chan_hi)
            chan_means_lo[chan].extend(data_chan_lo)
        print(chan)

    hi_means = [np.mean(chan_means_hi[chan]) for chan in range(128)]
    lo_means = [np.mean(chan_means_lo[chan]) for chan in range(128)]
    hi_std = [np.std(chan_means_hi[chan]) for chan in range(128)]
    lo_std = [np.std(chan_means_lo[chan]) for chan in range(128)]
    
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
    plt.step(list(range(128)), hi_means, label='Hi gain', linewidth=2)
    plt.step(list(range(128)), lo_means, label='Lo gain', linewidth=2)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/mean_system_test.png')
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
    plt.step(list(range(128)), hi_std, label='Hi gain', linewidth=2)
    plt.step(list(range(128)), lo_std, label='Lo gain', linewidth=2)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/std_system_test.png')
    plt.clf()
    
def mean_rms_plotter_small(file):
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
    plt.savefig(f'{plot_dir}/pears_correlation_gain{gain}.png', bbox_inches='tight')
    
def ADC_event(file, gain, event_range=False):
    '''
    Plots a 2d histogram of ADC vs events for specified channel and gain
    chan = channel #
    gain = 1 or 0
    event_range = (start_entry,end_entry), automatic if not specified
    '''
    sub_dir = f'{plot_dir}/ADC_iEvent_2d'
    os.makedirs(sub_dir, exist_ok=True)
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
        plt.savefig(f'{sub_dir}/channel_{chan}_gain{gain}.png', bbox_inches='tight')
        plt.clf()
        
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
    plt.savefig(f'{plot_dir}/coherence_hist_{gain}.png', bbox_inches='tight')
    
def FFT(file, gain, sample, sampling_rate):
    '''
    Plots FFTs for all febChannels
    file == path to root file
    gain == 1 or 0
    sample == desired time sample to take for each event (0 to 24)
    '''
    fft_dir = f'{plot_dir}/FFTs'
    os.makedirs(fft_dir, exist_ok=True)
    
    data = get_root_data(file)
    bin_num = 20
    matrix = np.zeros((bin_num, 128))  # To store the max FFT bin value for each channel
    fs = sampling_rate  # Sampling frequency in Hz
    
    # Precompute time vector and frequency bins (avoid redundant calculation)
    N = len(data)  # Use data length for time vector and FFT length
    t = np.arange(N) / fs  # Time vector based on the number of samples
    frequencies = np.fft.fftfreq(N, 1/fs)  # Frequency bins
    
    # Loop over each channel
    for chan in range(128):
        data_chan = data[(data['febChannel'] == chan) & (data['gain'] == gain)]
        
        # Extract only the desired sample from the ADC data for each entry
        adc_values = [entry[sample] for entry in data_chan['ADC']]
        
        N = len(adc_values)  # Number of samples
        # Step 2: Create a time vector
        t = np.arange(N) / fs  # Time vector based on the number of samples
        # Step 3: Compute FFT
        # Get the frequency bins
        frequencies = np.fft.fftfreq(N, 1/fs)
        
        # Perform FFT
        fft_result = np.fft.fft(adc_values)
        fft_plot = np.abs(fft_result)[1:N//2]  # FFT magnitude (ignore DC and Nyquist)
        freq_plot = frequencies[1:N//2]  # Positive frequency bins

        # Bin the FFT results into frequency bins
        num_elements = int(len(freq_plot)/bin_num)
        j = 0
        if np.std(adc_values) != 0:
            for i in range(bin_num):
                bin_avg = np.max(fft_plot[j:j+num_elements])
                matrix[i][chan] = bin_avg
                j += num_elements

        # Plot the ADC signal, FFT, and log(FFT)
        fig, axs = plt.subplots(3, 1, figsize=(20, 13))
        for ax in axs:
            ax.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
            ax.tick_params(axis='both', which='minor', direction='in', length=14)
            
            # Set minor locator for both axes in all subplots
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            
            # Add major ticks on all sides
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
        # Plot the original ADC signal
        axs[0].plot(t, adc_values)
        axs[0].set_title(f'ADC Signal for channel {chan}, {["Lo gain", "Hi gain"][gain]}, time sample {sample}', fontsize=25)
        axs[0].set_xlabel('Time [s]', fontsize=25)
        axs[0].set_ylabel('ADC Value', fontsize=25)
        axs[0].set_xlim(0,np.max(t))
        axs[0].grid(True)

        # Plot FFT magnitude
        axs[1].plot(freq_plot, fft_plot)
        axs[1].set_title('FFT of ADC Signal', fontsize=25)
        axs[1].set_xlabel('Frequency [Hz]', fontsize=25)
        axs[1].set_ylabel('Magnitude', fontsize=25)
        axs[1].set_xlim(0, sampling_rate / 2)
        axs[1].grid(True)

        # Plot log(FFT)
        axs[2].plot(freq_plot, fft_plot / np.max(fft_plot))
        axs[2].set_title('log(FFT) of ADC Signal', fontsize=25)
        axs[2].set_xlabel('Frequency [Hz]', fontsize=25)
        axs[2].set_ylabel('log(Magnitude)', fontsize=25)
        axs[2].set_yscale('log')
        axs[2].set_xlim(0, sampling_rate / 2)
        axs[2].set_ylim(bottom=10**-2)
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig(f'{fft_dir}/channel_{chan}_gain{gain}_sample{sample}.png')
        plt.clf()  # Clear the figure to free memory

    # Final plot showing max FFT magnitudes across channels
    matrix[matrix == 0] = np.nan

    plt.figure(figsize=(15,12))
    plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
    plt.tick_params(axis='both', which='minor', direction='in', length=14)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Adjust this to control minor ticks on x-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Adjust this to control minor ticks on y-axis
    # Add major ticks on all sides
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.title(f'Max FFT Magnitude for febChannels, {["Lo gain", "Hi gain"][gain]}', fontsize=30)
    plt.xlabel('febChanel', fontsize=30,loc='right')
    plt.ylabel('Frequency [Hz]', fontsize=30,loc='top')
    plt.ylim(0,bin_num)
    plt.imshow(matrix,norm=LogNorm(), extent=[0, matrix.shape[1], matrix.shape[0],0],aspect=4)
    cbar = plt.colorbar(shrink=0.6)
    cbar.ax.tick_params(labelsize=20)  # Set the fontsize for the colorbar ticks
    cbar.set_label('Max FFT Magnitude', fontsize=30)

    #Fxing ticks
    old_tick = []
    new_tick = []
    for i in range(6):
        old_tick.append(i*(bin_num/5))
        new_tick.append(int(i*sampling_rate/10))
    plt.yticks(ticks=old_tick, labels=new_tick)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/FFT_hist_{gain}.png', bbox_inches='tight')
    

def adc_vs_bcid(file, gain):
    '''
    Plots the mean ADC per BCID for all channels
    '''
    sub_dir = f'{plot_dir}/mean_ADC_bcid'
    os.makedirs(sub_dir, exist_ok=True)

    gain_title = 'Lo' if gain == 0 else 'Hi'

    # Define the fixed length for the y-axis
    fixed_length = 3  # For example, a length of 2 units

    if file.split(':')[-1] != 'Data':
        file = f'{file}:Data' 

    with uproot.open(file) as f:
        print(f.keys())
    
        tree = f
        # Define the chunk size (number of entries per chunk)
    for chan in range(128):
    
        chan_mean = {key: [] for key in range(3564)}
        # Iterate through the tree in chunks
        for chunk in tree.iterate(step_size=348000):
            # Process the chunk of data (as a pandas DataFrame)
            data = chunk
        
            # Filter data for the current channel and gain using Dask
            data_chan = data[(data['febChannel'] == chan) & (data['gain'] == gain)]
        
            #Extract relevant data
            adc = data_chan['ADC'].to_numpy()
            bcid = data_chan['bcid'].to_numpy()

            unique_bcid = np.unique(bcid)
        
            # Loop through unique BCIDs
            for b in unique_bcid:
                # Get all ADC values for the current BCID
                bcid_adc = adc[bcid == b]
            
                # Compute mean and RMS
                chan_mean[b].extend(bcid_adc)
        
            
        ADC_means = [np.mean(chan_mean[bcid]) for bcid in range(3564)]
        ADC_rms = [np.std(chan_mean[bcid])/np.sqrt(len(chan_mean[bcid])) for bcid in range(3564)]
        # Get the current y limits based on the data
        ymin, ymax = min(ADC_means), max(ADC_means)

        # Calculate the midpoint of the data to center the y-axis
        midpoint = (ymin + ymax) / 2

        # Set the new y limits, ensuring the total length is fixed
        y_margin = fixed_length / 2
    
        # Plotting
        plt.figure(figsize=(15, 11))
        plt.title(f'Mean ADC vs BCID, channel {chan}, {gain_title} gain', fontsize=30)
        plt.xlabel('BCID', fontsize=30, loc='right')
        plt.ylabel('Mean ADC', fontsize=30, loc='top')
        plt.xlim(0, 3563)
        plt.ylim(midpoint - y_margin, midpoint + y_margin)
        plt.tick_params(labelsize=30, axis='both', which='major', direction='in', length=20)
        plt.tick_params(axis='both', which='minor', direction='in', length=14)
    
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
    
        # Error bar plot
        plt.errorbar(list(chan_mean.keys()), ADC_means, yerr=ADC_rms, fmt='o', color='black',ecolor='blue', label='Mean ADC', capsize=2, markersize=2)
        plt.legend(fontsize=20)
        plt.tight_layout()
    
        # Save the plot
        plt.savefig(f'{sub_dir}/chan{chan}_gain{gain}.png')
        plt.close()


if file_size < 600*10**6:
    mean_rms_plotter_small(file_name)
    for gain in gains:
        correlation(file_name,gain,0)
        ADC_event(file_name,gain)
        coherent_noise(file_name,gain)
        FFT(file_name,gain,0,sample_rate)

if file_size >= 600*10**6:
    mean_rms_plotter(file_name)
    for gain in gains:
        adc_vs_bcid(file_name,gain)