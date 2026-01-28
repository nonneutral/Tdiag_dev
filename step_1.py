import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.signal import savgol_filter
from find_solution import getFiniteSolution
import os
#from solver import getFiniteSolution - commented out to prevent solver running here

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
def getElectrodeVoltageDrop(electrodeConfig, rpoints, zpoints, left, right, mur2, rfact):
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
    smooth_window=101,    #must be odd; will be adjusted if too big
    polyorder=3,
    baseline_quantile=0.90,  #baseline from top 10% of (smoothed) values
    sigma_low=6.0,        #how far below baseline counts as “in dip”
    sigma_high=2.5,       #hysteresis: when we say we've “returned” to baseline
    pad_points=10         #expand ROI a bit after detection
):
    """
    This is the Automatic ROI detection that we'll use in the next funciton.
    Inputs: x (np.array), y (np.array), various parameters for smoothing and thresholding
    Returns: mask (bool array), x_left, x_right, (i_left, i_right), baseline, robust_sigma, thr_high, thr_low
    """
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x) #Sort by x (important if the file isn't ordered) - sort of failsafe.
    x = x[order]
    y = y[order]

    n = len(y)
    if n < 10:
        raise ValueError("Not enough points for ROI detection.") #Fail-safe 1 - just file check

    w = min(smooth_window, n - (1 - (n % 2)))  #Fail safe 2 and housekeeping - window must be odd and <= n for savgol to function
    if w < 5:
        y_s = y.copy()
    else:
        if w % 2 == 0:
            w -= 1
        y_s = savgol_filter(y, window_length=w, polyorder=min(polyorder, w-1)) #this function fits a polynomial over windowed segment

    baseline = np.quantile(y_s, baseline_quantile) #Baseline estimate from high quantile

    med = np.median(y_s)
    mad = np.median(np.abs(y_s - med)) #Median Absolute Deviation
    robust_sigma = 1.4826 * mad if mad > 0 else (np.std(y_s) + 1e-12) #Noise estimation; Fallback to std if MAD=0

    #Thresholds
    thr_low  = baseline - sigma_low  * robust_sigma #baseline - 6(sigma)
    thr_high = baseline - sigma_high * robust_sigma #baseline - 2.5(sigma)

    i0 = int(np.argmin(y_s)) #Index of global minimum

    # If there's basically no dip, then fail - fail-safe 3
    if y_s[i0] > thr_low:
        # nothing significantly below baseline
        mask = np.zeros_like(y, dtype=bool)
        return mask, None, None, (None, None), baseline, robust_sigma

    #Expand left/right from minimum with hysteresis
    i_left = i0
    while i_left > 0 and y_s[i_left] < thr_high:
        i_left -= 1

    i_right = i0
    while i_right < n - 1 and y_s[i_right] < thr_high:
        i_right += 1

    # Pad a bit so ROI isn't too tight
    i_left = max(0, i_left - pad_points - 500) # some extra region for roi
    i_right = min(n - 1, i_right + pad_points)

    mask = np.zeros(n, dtype=bool)
    mask[i_left:i0+1] = True
    
    return mask, x[i_left], x[i0], (i_left, i_right), baseline, robust_sigma, thr_high, thr_low

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

    #Plotting Section
    plt.figure(figsize=(10, 6))
    plt.plot(u8_data, sipm_data, '.', markersize=2, alpha=0.5)
    if xL is not None:
        plt.axvspan(xL, xR, alpha=0.15)
        plt.title(f"SiPM vs u8 (ROI auto-detected: [{xL:.3g}, {xR:.3g}])")
    else:
        plt.title("SiPM vs u8 (No dip/ROI detected)")
    plt.xlabel("u8 Excitation Value")
    plt.ylabel("SiPM Signal (~Escape Rate)")
    plt.grid(True, linestyle='--', alpha=0.7)
    #plt.xlim(xL,xR)
    plt.tight_layout()
    plt.show()

    #Fitting Section
    u8_roi = u8_data[mask]
    sipm_roi = sipm_data[mask]
    order = np.argsort(u8_roi) #for isolating linear section from padded ROI
    u = u8_roi[order]
    s = sipm_roi[order]
    ds = np.abs(np.diff(s)) #look at CONSECUTIVE differences to find baseline end
    n_base = max(5, int(0.1 * len(ds)))   #first 10% or at least 5 points have baseline, so use that to estimate noise in this region
    baseline_noise = np.median(ds[:n_base])
    factor=5.0        #how much larger than baseline counts as "real"
    run_length=7      #how many consecutive points above threshold to confirm end of baseline

    start_idx = None
    for i in range(len(ds) - run_length):
        if np.all(ds[i:i+run_length] > factor * baseline_noise):
            start_idx = i + 1   # +1 because diff shifts index
            break

    if start_idx is None:
        raise RuntimeError("Could not detect end of baseline.")

    u8_roi_for_fit = u[start_idx:]
    sipm_roi_for_fit = s[start_idx:]

    if len(u8_roi_for_fit) < 2:
        raise RuntimeError("Not enough points for linear fit.") #Just failsafe

    m, b = np.polyfit(u8_roi_for_fit, sipm_roi_for_fit, 1) 

    #Fit Plotting Section
    plt.figure(figsize=(10, 6))
    plt.plot(u, s, '.', alpha=0.3, label="ROI (all)")
    plt.plot(u8_roi_for_fit, sipm_roi_for_fit, 'o',
            color='orange', markersize=4, label="Linear region")

    u_fit = np.linspace(u8_roi_for_fit.min(), u8_roi_for_fit.max(), 200)
    plt.plot(u_fit, m * u_fit + b, 'r-', linewidth=2,
            label=f"Fit: y = {m:.3g} x + {b:.3g}")

    plt.title("SiPM vs u8 — Linear Region After Baseline")
    plt.xlabel("u8 Excitation Value (ROI)")
    plt.ylabel("SiPM Signal (~Escape Rate)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    #convert to volatges
    converted_volatges = u_fit * 15
    return converted_volatges, u8_roi_for_fit, sipm_roi_for_fit, m, b  #pre-lim return for test run

    #Next Next Step: fit the sipm vs u8 data to get a functional form for u8 - use the roi to fit a line to the linear map,
    #then use that as input for electrodeConfig in getElectrodeVoltageDrop to get the voltage drop. <--- should be done now.
    
    #Final step for this function: substitute the volatges in as one of the electrodeConfig arg along with borders as the other.

#%% - TEST RUN
fit_and_convert_u8('/Users/rupgango/Downloads/Dec13/133632.813.csv')

#general cross-platform code for processing all files in a directory - but commneted out for testing atm.
'''
folder = "/Users/rupgango/Downloads/Dec13"

for fname in os.listdir(folder):
    if fname.lower().endswith(".csv"):
        filepath = os.path.join(folder, fname)
        print(f"\nProcessing: {filepath}")
        fit_and_convert_u8(filepath)
        '''