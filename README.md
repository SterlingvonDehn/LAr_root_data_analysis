# LAr phase II data checks
Used to check the integrity of data produced by the new circuit boards for the LAr detectors on ATLAS.

# root_data_analysis.py
- Analizes pedestal data root files and produces usful plots
- Run from terminal, if in LAr_root_data_analysis directory: `python3 py/root_data_analysis.py PATH_TO_ROOT_FILE`
- Produces a subdirectory labeled plot_ROOT_FILE_NAME, inside there will be many plots including more directories containing FFTs and ADC vs iEvent for all channels

# calibration_analysis.py
- Analizes calibration data sets and produces usful plots
- Run from terminal, if in LAr_root_data_analysis directory: `python3 py/calibration_anaysis.py PATH_TO_ROOT_FILE`
- Produces a subdirectory labeled plot_ROOT_FILE_NAME, inside there will be many plots including more directories containing ADC_vs_measurement plots for all channels and ADC_vs_chan plots for all measurements
