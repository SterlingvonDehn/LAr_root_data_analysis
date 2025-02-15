# LAr phase II data checks
Used to check the integrity of data produced by the new circuit boards for the LAr detectors on ATLAS.

# root_data_analysis.py
- Analyzes pedestal data root files and produces useful plots
- Open the py file and define `file_name`, `sample_rate`, and `gains`
- Run from the terminal, if in the LAr_root_data_analysis directory: `python3 py/root_data_analysis.py`
- Produces a subdirectory labeled plot_ROOT_FILE_NAME, there will be many plots including more directories containing FFTs and ADC vs iEvent for all channels

# calibration_analysis.py
- Analyzes calibration data sets and produces useful plots
- Open the py file and define `file_name`
- Run from the terminal, if in the LAr_root_data_analysis directory: `python3 py/calibration_anaysis.py`
- Produces a subdirectory labeled plot_ROOT_FILE_NAME, inside there will be many plots including more directories containing ADC_vs_measurement plots for all channels and ADC_vs_chan plots for all measurements

# pulse_plotter.py
- Checks all febChannels for pulses
- Plots pulses found
- Run from terminal, if in the LAr_root_data_analysis directory: `python3 py/calibration_anaysis.py file_name`
