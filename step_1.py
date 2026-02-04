import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.signal import savgol_filter
from find_solution import getFiniteSolution
import os
#from solver import getFiniteSolution - commented out to prevent solver running here

Llim=0.035
Rlim=0.110
rad2=0.0002 #plasma radius in meters
kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin
qe=1.60217662e-19 #elementary charge in coulombs

#%%
'''
def process_and_plot(filename):
    """""
    Processes the given CSV file to convert u8 excitation values to voltages
    Inputs: filepath (str) - path to the CSV file
    Returns: converted_voltages (np.array), sipm_data (np.array)
    """
    try:
        df = pd.read_csv(filename, header=None)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    sipm_data = df[0].values  #sipm (~escape rate)
    u8_data = df[1].values    #u8 excitations
    
    #Mapping the min/max of the u8 data to the target voltage range (-120V to -63V)
    u8_min = np.min(u8_data)
    u8_max = np.max(u8_data)
    v_min_target = -63.0  # Corresponds to the most negative u8 value
    v_max_target = 120.0   # Corresponds to the least negative u8 value
    
    slope = (v_max_target - v_min_target) / (u8_max - u8_min) #Calculate slope of linear map
    intercept = v_max_target - slope * u8_max
    
    converted_voltages = slope * u8_data + intercept  #transformed u8 exciation data

    #Plotting Section
    plt.figure(figsize=(10, 6))
    plt.plot(u8_data, sipm_data, '.', markersize=2, alpha=0.5, color='blue')
    plt.title(f"SiPM Signal vs Voltage\n(Mapped range: {v_min_target} V to {v_max_target} V)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("SiPM Signal (~Escape Rate)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return converted_voltages, sipm_data

voltages, sipm = process_and_plot('/Users/rupgango/Downloads/Dec13/134414.747.csv')
'''

#%%
def getElectrodeVoltageDrop(electrodeConfig, rpoints, zpoints, left, right, mur2, rfact): #used in final_SiPM_vs_VoltagePlot
    ''' Gets the combined Electrode "Excitement/Voltage Drop" Profile 
        without having to run the solver.
        
        Input: electrodeConfig: (voltages, borders)
        Returns: free_space_solution (2D array of "Voltage Drop" values)
    '''
    Mmax=1
    Nmax=20000
    zeros=[special.jn_zeros(m,Nmax) for m in range(Mmax)]

    nr=rpoints
    nz=zpoints
    leftbound=left
    rightbound=right
    position_map_r=np.zeros((nr,nz))
    position_map_z=np.zeros((nr,nz))
    rbound=mur2*rfact
    for rind in range(nr):
        for zind in range(nz):
            position_map_z[rind,zind]=leftbound+(rightbound-leftbound)*(zind+.5)/(nz)
            position_map_r[rind,zind]=rbound*rind/nr
            
    def getElectrodeSolution(left,right,voltage):
        cs=2*voltage/(zeros[0]*special.j1(zeros[0]))
        symmetricsolution=getFiniteSolution([cs],right-left,lambda r:voltage)
        return lambda r,z: symmetricsolution(z-(left+right)/2,r)

    electrode_voltages,electrode_borders=electrodeConfig
    free_space_solution=np.zeros((nr,nz))
    electrodes=[getElectrodeSolution(electrode_borders[i-1],electrode_borders[i],electrode_voltages[i]) 
                for i in range(1,len(electrode_voltages)-1)]   

    for electrode in electrodes:
        free_space_solution+=electrode(position_map_r,position_map_z)
    
    axial_profile = np.max(free_space_solution, axis=0)
    peak_idx = np.argmax(axial_profile)
    peak = axial_profile[peak_idx]
    barrier = np.min(axial_profile[:peak_idx])
    VoltageDrop = peak - barrier
    print(f"Calculated Voltage Drop: {VoltageDrop} V")
    return VoltageDrop

#%%
def auto_roi_from_dip(
    x, y,
    smooth_window=101,
    polyorder=3,
    baseline_quantile=0.90,
    sigma_low=3.5,     
    sigma_high=2.0,    
    prepad=500,        # how many points BEFORE the dip edge to include (time)
    ): #used in fit_and_convert_u8
    x = np.asarray(x)
    y = np.asarray(y)

    n = len(y)
    if n < 10:
        raise ValueError("Not enough points for ROI detection.")

    w = min(smooth_window, n - (1 - (n % 2))) #make sure window is odd and <= n - sort of failsafe
    if w < 5:
        y_s = y.copy()
    else:
        if w % 2 == 0:
            w -= 1
        y_s = savgol_filter(y, window_length=w, polyorder=min(polyorder, w-1))

    baseline = np.quantile(y_s, baseline_quantile)

    #NEW DEFINITION OF SIGMA
    hi = y_s[y_s >= baseline]
    if len(hi) > 20:
        med = np.median(hi)
        mad = np.median(np.abs(hi - med))
        sigma = 1.4826 * mad if mad > 0 else (np.std(hi) + 1e-12)
    else:
        med = np.median(y_s)
        mad = np.median(np.abs(y_s - med))
        sigma = 1.4826 * mad if mad > 0 else (np.std(y_s) + 1e-12)

    i0 = int(np.argmin(y_s))

    thr_low  = baseline - sigma_low  * sigma   # must cross this to count as dip
    thr_high = baseline - sigma_high * sigma   # hysteresis to find edges

    #If no dip detected then fail - failsafe 3
    if y_s[i0] > thr_low:
        mask = np.zeros(n, dtype=bool)
        return mask, None, None, (None, None), baseline, sigma, thr_high, thr_low

    #Expand from minimum to find dip edges
    i_left = i0
    while i_left > 0 and y_s[i_left] < thr_high:
        i_left -= 1

    #Pad in TIME
    i_left = max(0, i_left - prepad)

    mask = np.zeros(n, dtype=bool)
    mask[i_left:i0+1] = True     

    return mask, x[i_left], x[i0], (i_left, i0), baseline, sigma, thr_high, thr_low

#%%
def fit_and_convert_u8(filename):
    '''
    Fit function to fit for converted u8 excitation data and get a functional
    representation of voltages. Need to return this fit form to use as electrodeConfig 
    in getElectrodeVoltageDrop to get the total "drop" - x-axis on the sipm data vs drop plot.
    
    Inputs: filepath (str) - path to the CSV file
    Returns: converted_voltages (np.array), sipm_data (np.array)
    '''
    try:
        df = pd.read_csv(filename, header=None)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    sipm_data = df[0].values  #sipm (~escape rate)
    u8_data = df[1].values    #u8 excitations

    mask, xL, xR, (iL, iR), baseline, sig, thr_high, thr_low = auto_roi_from_dip(u8_data, sipm_data) #Auto-detect ROI around dip

    t = np.arange(len(u8_data))
    
    plt.figure(figsize=(10,6))
    if iL is not None:
        plt.axvspan(iL, iR, alpha=0.2) #labels the ROI
    plt.plot(t, sipm_data, '.', markersize=2, alpha=0.6)
    plt.title("SiPM vs time index")
    plt.xlabel("time index")
    plt.ylabel("SiPM")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,4))
    if iL is not None:
        plt.axvspan(iL, iR, alpha=0.2) #labels the ROI
    plt.plot(t, u8_data, '.', markersize=2, alpha=0.6)
    plt.title("u8 vs time index")
    plt.xlabel("time index")
    plt.ylabel("u8")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    #Extract ROI data for fitting
    u8_roi_for_fit = u8_data[mask]
    t_roi_for_fit = t[mask]
    sipm_roi_for_fit = sipm_data[mask]
    
    if len(t_roi_for_fit) < 2:
        raise RuntimeError("ROI too small for fitting.")
    if not np.all(np.diff(t_roi_for_fit) == 1):
        print("Warning: ROI mask is not continuous in time.")

    #IMPORTANT: center time so polyfit isn't ill-conditioned (time indices are huge)
    t0 = t_roi_for_fit[0]
    tau = t_roi_for_fit - t0

    deg = 4  #degree of polynomial fit
    p = np.poly1d(np.polyfit(tau, u8_roi_for_fit, deg))
    u8_fit = p(tau)
    #Plot: u8 vs time with fit overlay
    plt.figure(figsize=(10,4))
    plt.plot(t, u8_data, '.', markersize=2, alpha=0.3, label="all data")
    plt.plot(t_roi_for_fit, u8_roi_for_fit, '.', markersize=2, alpha=0.8, label="ROI")
    plt.plot(t_roi_for_fit, u8_fit, 'r-', linewidth=2, label=f"poly deg {deg} fit")
    plt.axvspan(iL, iR, alpha=0.15)
    plt.title("u8 vs time (ROI fit)")
    plt.xlabel("time index")
    plt.ylabel("u8")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(u8_fit)
    converted_voltages = u8_fit * 15 #convert to volatges
    return converted_voltages, u8_roi_for_fit, t_roi_for_fit, sipm_roi_for_fit #pre-lim return for test run
    
    #Final step for this function: substitute the volatges in as one of the electrodeConfig arg along with borders as the other.
#%%
def extract_measured_temp(converted_voltages, sipm_roi_for_fit):
    '''
    FINAL STEP: PLOT SIPM vs VOLTAGE DROP
    Inputs: converted_voltages (np.array) - voltages from fit, sipm_roi_for_fit (np.array) - corresponding SiPM data in the ROI
    Returns: Measured_Temp (float) - calculated temperature from the fit slope
    '''
    
    converted_voltages = np.asarray(converted_voltages)  #get the voltages as a numpy array for linspace and indexing
    v_start = float(converted_voltages[0])
    v_end   = float(converted_voltages[-1])
    n_steps = len(converted_voltages)   
    v_sweep = np.linspace(v_start, v_end, n_steps) #sets the sweep for voltage drop calculation

    #2.create for loop wuth getElectrodeVoltageDrop called inside to get voltage drop for each converted voltage
    #3.list electrode voltages and borders with converted voltages as one of them
    #4.put them as the initial argument in electrodeConfig along with borders:
    electrode_voltages_list = [[0, v, 50, -130, 0] for v in v_sweep]
    electrode_borders=[0.025,0.050,0.100,0.125]
    drops = []
    for electrode_voltages in electrode_voltages_list:
        electrodeConfig = (electrode_voltages, electrode_borders)
        drop = getElectrodeVoltageDrop(electrodeConfig, rpoints=20, zpoints=40, left=Llim, right=Rlim, mur2=rad2, rfact=3.0)
        drops.append(drop)
        
    x = -np.asarray(drops, dtype=float)
    y = np.asarray(sipm_roi_for_fit, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    ds = np.abs(np.diff(ys)) #difference between adjacent y values 
    
    n_base = max(10, int(0.15 * len(ds))) #baseline reference points (first 15%)
    baseline_noise = np.median(ds[:n_base]) + 1e-12

    factor = 9.5
    run_length = 12

    #find start of flank: sustained departure from baseline
    start_idx = None
    for i in range(len(ds) - run_length):
        if np.all(ds[i:i+run_length] > factor * baseline_noise):
            start_idx = i + 1   # +1 because ds is diff-indexed
            break

    if start_idx is None:
        raise RuntimeError("Could not detect start of flank (try factor=1.5 or run_length=8).")

    '''#find where it becomes flat again (plateau)
    end_idx = len(xs)
    flat_factor = 2.0

    for j in range(start_idx, len(ds) - run_length):
        if np.all(ds[j:j+run_length] < flat_factor * baseline_noise):
            end_idx = j + 1
            break
'''
    #fit only the flank segment
    x_fit = xs[start_idx:]
    y_fit = ys[start_idx:]

    if len(x_fit) < 2:
        raise RuntimeError("Not enough points for linear fit after trimming plateau.")

    m, b = np.polyfit(x_fit, y_fit, 1)
    Measured_Temp=qe*1.05/(kb*m) #in Kelvin

    #Plotting section
    plt.figure(figsize=(10,4))
    plt.plot(xs, ys, 'o-', markersize=3, alpha=0.5, label="data (x flipped, sorted)")
    plt.plot(x_fit, y_fit, 'o', markersize=4, label="fit region")
    xline = np.linspace(x_fit.min(), x_fit.max(), 200)
    plt.plot(xline, m*xline + b, 'r-', linewidth=2, label=f"fit: y={m:.3g}x+{b:.3g}")
    plt.xlabel("Flipped Voltage Drop (-V)")
    plt.ylabel("SiPM Signal (~Escape Rate)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return Measured_Temp

#%% - TEST RUN
converted_voltages, _, _, sipm_roi_for_fit = fit_and_convert_u8('/Users/rupgango/Downloads/Dec13/135903.810.csv')
temp = extract_measured_temp(converted_voltages, sipm_roi_for_fit)
print(f"Extracted temperature: {temp} K")

#general cross-platform code for processing all files in a directory - but commneted out for testing atm.
'''
folder = "/Users/rupgango/Downloads/Dec13"

for fname in os.listdir(folder):
    if fname.lower().endswith(".csv"):
        filepath = os.path.join(folder, fname)
        print(f"\nProcessing: {filepath}")
        converted_voltages, _, _, sipm_roi_for_fit = fit_and_convert_u8(filepath)
        temp = extract_measured_temp(converted_voltages, sipm_roi_for_fit)
        print(f"Extracted temperature: {temp} K")
        '''