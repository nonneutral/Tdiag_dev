import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special
from scipy.signal import savgol_filter
from find_solution import getFiniteSolution
from datetime import datetime
#from solver import getFiniteSolution - commented out to prevent solver running here

now=str(datetime.now())
print(f'{now}\tExecution start')

Llim=0.035
Rlim=0.110
rad2=0.0002 #plasma radius in meters
kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin
qe=1.60217662e-19 #elementary charge in coulombs

#directory search function
def iter_all(substring, path):
    return list(
        os.path.join(root, entry)
        for root, dirs, files in os.walk(path)
        for entry in dirs + files
        if substring in entry
    )

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
    #print(f"Calculated Voltage Drop: {VoltageDrop} V")
    return VoltageDrop

#%%
def auto_roi_from_dip(x, y, smooth_window=101, polyorder=3, baseline_quantile=0.90,
                      sigma_low=3.5, sigma_high=2.0, prepad=500): 
    '''
    Automatically detects a dip in the data and defines a region of interest (ROI) around it.
    
    Inputs: x (array-like): x-axis data (e.g., time indices)
            y (array-like): y-axis data (e.g., SiPM signal)
            smooth_window (int): window size for Savitzky-Golay filter (must be odd)
            polyorder (int): polynomial order for Savitzky-Golay filter
            baseline_quantile (float): quantile to estimate baseline level
            sigma_low (float): number of sigmas below baseline to define dip threshold
            sigma_high (float): number of sigmas for hysteresis threshold to find edges
            prepad (int): number of points to pad on the left side of the detected dip for the ROI
            
    Returns: mask (boolean array): True for points in the ROI, False otherwise
            x_left (float): x-value of the left edge of the ROI
            x_right (float): x-value of the right edge of the ROI
            (i_left, i_right): indices of the left and right edges of the ROI
            baseline (float): estimated baseline level
            sigma (float): estimated noise level (sigma)
            thr_high (float): high threshold for edge detection
            thr_low (float): low threshold for dip detection 
    '''
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
    #print(u8_fit)
    converted_voltages = u8_fit * 15 #convert to volatges
    return converted_voltages, u8_roi_for_fit, t_roi_for_fit, sipm_roi_for_fit #pre-lim return for test run
    
    #Final step for this function: substitute the volatges in as one of the electrodeConfig arg along with borders as the other.
#%%
def getTotalVoltageDropProfile(converted_voltages):
    '''
    Gets the total voltage drop profile for the given converted voltages by calling 
    getElectrodeVoltageDrop for each voltage and putting them in a list. 
    This will be used as the x-axis for the final plot of SiPM vs Voltage Drop.
    
    Inputs: converted_voltages (np.array) - voltages from fit
    Returns: drops (list of voltage drops corresponding to each converted voltage).
    '''
    converted_voltages = np.asarray(converted_voltages)  #get the voltages as a numpy array for linspace and indexing
    v_start = float(converted_voltages[0])
    v_end   = float(converted_voltages[-1])
    n_steps = len(converted_voltages)   
    v_sweep = np.linspace(v_start, v_end, n_steps) #sets the sweep for voltage drop calculation

    #1.create for loop wuth getElectrodeVoltageDrop called inside to get voltage drop for each converted voltage
    #2.list electrode voltages and borders with converted voltages as one of them
    #3.put them as the initial argument in electrodeConfig along with borders:
    electrode_voltages_list = [[0, v, 50, -130, 0] for v in v_sweep]
    electrode_borders=[0.025,0.050,0.100,0.125]
    drops = []
    for electrode_voltages in electrode_voltages_list:
        electrodeConfig = (electrode_voltages, electrode_borders)
        drop = getElectrodeVoltageDrop(electrodeConfig, rpoints=20, zpoints=40, left=Llim, right=Rlim, mur2=rad2, rfact=3.0)
        drops.append(drop)
        
    return drops
'''
def extract_measured_temp(drops, sipm_roi_for_fit):
    
    
    x = -np.asarray(drops, dtype=float)
    y = np.log(-(np.asarray(sipm_roi_for_fit, dtype=float)[np.asarray(sipm_roi_for_fit, dtype=float) < 0]))
    plt.plot(x, y, 'o-', markersize=3, alpha=0.5)
    plt.show()
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    ds = np.abs(np.diff(ys)) #difference between adjacent y values 
    
    n_base = max(10, int(0.20 * len(ds))) #baseline reference points (first 15%)
    baseline_noise = np.median(ds[:n_base]) + 1e-12

    factor = -5
    run_length = 12

    #find start of flank: sustained departure from baseline
    start_idx = None
    for i in range(len(ds) - run_length):
        if np.all(ds[i:i+run_length] > factor * baseline_noise):
            start_idx = i + 1   # +1 because ds is diff-indexed
            break

    if start_idx is None:
        raise RuntimeError("Could not detect start of flank (try factor=1.5 or run_length=8).")
    
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
'''
def longest_true_run(mask):
    """Return (start, end) indices of the longest contiguous True run in mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return None, None
    idx = np.flatnonzero(mask)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends   = np.r_[idx[breaks], idx[-1]]
    lengths = ends - starts + 1
    j = np.argmax(lengths)
    return int(starts[j]), int(ends[j])

def pick_flank_region(xs, ys, min_pts=80):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    w = min(101, len(ys) - (1 - (len(ys) % 2)))
    if w >= 9:
        if w % 2 == 0: w -= 1
        ys_s = savgol_filter(ys, w, 2)
    else:
        ys_s = ys

    dy = np.gradient(ys_s, xs)
    dy_max = np.nanmax(dy)

    # relax threshold until we get a decent chunk of points
    for frac in [0.8, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12]:
        mask = np.isfinite(dy) & (dy > frac * dy_max)
        iL, iR = longest_true_run(mask)
        if iL is not None and (iR - iL + 1) >= min_pts:
            return iL, iR, ys_s

    # fallback: take mid 20–80% of smoothed y (usually the "linear-ish" part)
    ylo = np.quantile(ys_s, 0.2)
    yhi = np.quantile(ys_s, 0.8)
    mask = (ys_s >= ylo) & (ys_s <= yhi)
    iL, iR = longest_true_run(mask)
    if iL is None or (iR - iL + 1) < 10:
        raise RuntimeError("Could not find a stable flank region.")
    return iL, iR, ys_s

def extract_measured_temp(drops, sipm_roi_for_fit):
    x = -np.asarray(drops, dtype=float)
    sipm = np.asarray(sipm_roi_for_fit, dtype=float)

    # 1) Pairing-safe mask (same mask applied to x and sipm)
    good = np.isfinite(x) & np.isfinite(sipm) & (sipm < 0)

    print("len(drops):", len(drops), "len(sipm):", len(sipm), "len(good):", good.sum())
    print("drops: nan", np.isnan(x).sum(), "inf", np.isinf(x).sum(),
          "min", np.nanmin(x), "max", np.nanmax(x))
    print("sipm:  nan", np.isnan(sipm).sum(), "inf", np.isinf(sipm).sum(),
          "min", np.nanmin(sipm), "max", np.nanmax(sipm))

    x = x[good]
    rate = -sipm[good]

    # 2) prevent log(0) -> -inf
    eps = max(1e-12, np.percentile(rate, 1) * 1e-3)
    y = np.log(np.maximum(rate, eps))

    # 3) sort
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    # 4) final finite cleanup before polyfit
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]

    iL, iR, ys_s = pick_flank_region(xs, ys, min_pts=80)

    x_fit = xs[iL:iR+1]
    y_fit = ys[iL:iR+1]

    # (optional but nice) center x so intercept isn't huge
    x0 = np.mean(x_fit)
    m, c = np.polyfit(x_fit - x0, y_fit, 1)
    b = c - m*x0   # back to y = m*x + b

    print("fit pts:", len(x_fit), "slope m:", m)
    Measured_Temp = qe*1.05/(kb*abs(m))
    
    plt.figure(figsize=(10,4))
    plt.plot(xs, ys, 'o-', markersize=3, alpha=0.5, label="log(-SiPM)")
    plt.plot(x_fit, y_fit, 'o', markersize=4, label="fit region")
    xline = np.linspace(x_fit.min(), x_fit.max(), 200)
    plt.plot(xline, m*xline + b, 'r-', linewidth=2, label=f"fit: y={m:.3g}x+{b:.3g}")
    plt.xlabel("Flipped Voltage Drop (-V)")
    plt.ylabel("log(-SiPM)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return Measured_Temp

#%% - TEST RUN
Recompute_Drops=True #trun to False to skip voltage drop recomputation 
filepath1=iter_all('csv','../')[1]
converted_voltages, _, _, sipm_roi_for_fit = fit_and_convert_u8(filepath1)
if Recompute_Drops: 
    drops = getTotalVoltageDropProfile(converted_voltages)   
temp = extract_measured_temp(drops, sipm_roi_for_fit)
print(f"Extracted temperature: {temp} K")

now=str(datetime.now())
print(f'{now}\tExecution complete')

#general cross-platform code for processing all files in a directory - but commneted out for testing atm.
'''
folder = "/Users/rupgango/Downloads/Dec13"

for fname in os.listdir(folder):
    if fname.lower().endswith(".csv"):
        filepath = os.path.join(folder, fname)
        print(f"\nProcessing: {filepath}")
        converted_voltages, _, _, sipm_roi_for_fit = fit_and_convert_u8(filepath)
        drops = getTotalVoltageDropProfile(converted_voltages)
        temp = extract_measured_temp(drops, sipm_roi_for_fit)
        print(f"Extracted temperature: {temp} K")
        '''
        
'''
    FINAL STEP: PLOT SIPM vs VOLTAGE DROP
    Inputs: converted_voltages (np.array) - voltages from fit, sipm_roi_for_fit (np.array) - corresponding SiPM data in the ROI
    Returns: Measured_Temp (float) - calculated temperature from the fit slope
'''