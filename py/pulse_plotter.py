'''
Finds and plots pulses in root data file and gain given
To run from terminal:
python3 pulse_plotter.py file_path
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import sys
import os
import uproot

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

def ADC_event(file):
    '''
    Optimized version of the function for better performance.
    '''
    # Extracts data
    data = get_root_data(file) 
    for chan in range(128):
        data_feb = data[(data['febChannel'] == chan)] 
        data_adc = data_feb['ADC'].to_numpy()
        iEvent = data_feb['iEvent'].to_numpy()

        start_event = np.min(iEvent)
        end_event = np.max(iEvent)
        
        # Efficient initialization of lists
        sus_events = []
        j=-1
        # Iterate over the event range
        for i in range(start_event, end_event):
            if i > 2 and abs(np.max(data_adc[iEvent == i]) - np.mean(data_adc[iEvent == i])) > 200:
                if len(sus_events)>0 and i == sus_events[-1][-1]+1:
                    sus_events[j].append(i)
                else:
                    sus_events.append([i])
                    j += 1
                
                

        print(chan, sus_events)
        for events in sus_events:
            pulse_plot(file,data,chan,[events[0]-1,events[-1]+2])

def pulse_plot(file,data,febchannel,event_range):
    '''
    Plots the pulses
    '''
    chan_data = data[(data['febChannel']==febchannel)]
    
    event_adc = []
    for event in range(event_range[0],event_range[1]):
        event_adc.extend(chan_data[chan_data['iEvent']==event]['ADC'].to_list()[0])
    xs = 25*np.array(list(range(len(event_adc))))
    
    gain = chan_data['gain'].to_list()[0]
    if gain == 0:
        gain_title = 'Lo'
    else:
        gain_title = 'Hi'
        
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
    plt.plot(xs,event_adc)
    plt.title(f'ADC vs sample, {gain_title} gain & febChannel {febchannel}', fontsize=30)
    plt.text(0.1, 0.9, f'Events {event_range[0]+1}-{event_range[1]}', fontsize=20, color='black', ha='left', va='top', transform=plt.gca().transAxes)
    plt.xlabel('Time (ns)', fontsize=30, loc='right')
    plt.ylabel('ADC', fontsize=30, loc='top')
    plt.xlim(0,xs[-1])
    os.makedirs(f'plots_{file.split('/')[-1]}/ADC_sample_{febchannel}',exist_ok=True)
    plt.savefig(f'plots_{file.split('/')[-1]}/ADC_sample_{febchannel}/channel_{febchannel}_gain{gain}_event{event_range[0]}.png', bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    file_name = args[1]
    ADC_event(file_name)
    