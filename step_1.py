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

#%%
'''
def longest_true_run(mask):
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
'''
def auto_flank_from_slope(xs, ys,
                          smooth_window=101, polyorder=2,
                          baseline_quantile=0.25,   # <-- IMPORTANT: use LOW quantile for "flat" slopes
                          sigma_low=4.0, sigma_high=2.0,
                          pad_points=10, min_pts=50,
                          relax=True, abs_slope=False, debug=False):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xs_u, idx = np.unique(xs, return_index=True) #REMOVE duplicate xs (critical for gradient stability)
    ys = ys[idx]
    xs = xs_u

    n = len(xs)
    if n < 30:
        raise ValueError(f"Not enough points for flank detection (n={n}).")

    # --- smooth y ---
    w = min(smooth_window, n - (1 - (n % 2)))
    if w >= 9:
        if w % 2 == 0: w -= 1
        ys_s = savgol_filter(ys, w, min(polyorder, w-1))
    else:
        ys_s = ys.copy()

    dy_raw = np.gradient(ys_s, xs) #Slope

    if abs_slope:
        dy_use = np.abs(dy_raw)
    else:
        # IMPORTANT: rising edge only (kills baseline & negative-slope junk)
        dy_use = np.clip(dy_raw, 0, None)

    #smooth the slope itself so single-point dips don't chop the region
    w_dy = min(151, n - (1 - (n % 2)))
    if w_dy >= 9:
        if w_dy % 2 == 0: w_dy -= 1
        dy_use = savgol_filter(dy_use, w_dy, 2)


    #baseline slope: use the FLATTEST portion (lowest quantile)
    q = np.quantile(dy_use, baseline_quantile)
    lo = dy_use[dy_use <= q]

    baseline = np.median(lo)
    mad = np.median(np.abs(lo - baseline))
    sigma = 1.4826 * mad if mad > 0 else (np.std(lo) + 1e-12)

    i0 = int(np.argmax(dy_use))
    peak = dy_use[i0]

    thr_low  = baseline + sigma_low  * sigma
    thr_high = baseline + sigma_high * sigma

    def attempt(sl, sh, mp):
        tl = baseline + sl * sigma
        th = baseline + sh * sigma
        #IMPORTANT: don’t let "th" be tiny; keep only the steep part near the peak
        th = max(th, 0.04 * peak)   #0.10–0.25 are typical; bigger = narrower


        if peak < tl:
            return None, None, th, tl

        gap_max = 8  #allow up to 8 consecutive points below threshold while expanding

        iL = i0
        gap = 0
        while iL > 0:
            if dy_use[iL-1] > th:
                gap = 0
                iL -= 1
            else:
                gap += 1
                iL -= 1
                if gap > gap_max:
                    break

        iR = i0
        gap = 0
        while iR < n-1:
            if dy_use[iR+1] > th:
                gap = 0
                iR += 1
            else:
                gap += 1
                iR += 1
                if gap > gap_max:
                    break


        iL = max(0, iL - pad_points)
        iR = min(n-1, iR + pad_points)

        if (iR - iL + 1) < mp:
            return None, None, th, tl

        return iL, iR, th, tl

    iL, iR, thr_high, thr_low = attempt(sigma_low, sigma_high, min_pts)

    if relax and iL is None:
        #progressively easier thresholds + smaller min_pts
        for (sl, sh, mp) in [
            (3.0, 1.5, min_pts),
            (2.5, 1.2, min_pts),
            (2.0, 1.0, max(30, min_pts//2)),
            (1.5, 0.8, max(25, min_pts//3)),
            (1.2, 0.6, max(20, min_pts//4)),
        ]:
            iL, iR, thr_high, thr_low = attempt(sl, sh, mp)
            if iL is not None:
                break

    if debug:
        print(f"[slope] n={n} baseline={baseline:.3g} sigma={sigma:.3g} peak={peak:.3g} "
              f"thr_low={thr_low:.3g} thr_high={thr_high:.3g} "
              f"above_thr_high={np.sum(dy_use > thr_high)}")

    if iL is None:
        raise RuntimeError("Could not detect flank region from slope hysteresis.")

    return (iL, iR), baseline, sigma, thr_high, thr_low, dy_use, ys_s, xs, ys

def best_linear_subwindow(xs, ys, iL, iR, min_pts=100, r2_min=0.97,
                          y_floor=None, y_cap=None):
    if iL is None or iR is None:
        return None, None, np.nan

    L = iR - iL + 1
    if L < min_pts:
        return iL, iR, np.nan

    best = None
    best_len = -1
    best_mean_y = np.inf
    best_r2 = -np.inf

    for a in range(iL, iR - min_pts + 1):
        for b in range(a + min_pts - 1, iR + 1):
            xw = xs[a:b+1]
            yw = ys[a:b+1]

            if np.ptp(xw) < 1e-12:
                continue

            #y-band gate (prevents choosing the very top saturation part)
            if y_floor is not None and np.min(yw) < y_floor:
                continue
            if y_cap is not None and np.max(yw) > y_cap:
                continue

            #reject flat windows (R^2 becomes meaningless)
            if np.ptp(yw) < 0.4:
                continue

            m, c = np.polyfit(xw, yw, 1)

            #rising only
            if m <= 0:
                continue

            yhat = m*xw + c
            ss_res = np.sum((yw - yhat)**2)
            ss_tot = np.sum((yw - np.mean(yw))**2)
            if ss_tot < 1e-8:
                continue

            r2 = 1 - ss_res/(ss_tot + 1e-12)
            if not np.isfinite(r2):
                continue

            win_len = (b - a + 1)
            mean_y = float(np.mean(yw))

            # choose LONGEST window that passes r2_min;
            # tie-break: pick LOWER mean_y (earlier part of rise) 
            if r2 >= r2_min:
                if (win_len > best_len) or (win_len == best_len and mean_y < best_mean_y):
                    best_len = win_len
                    best_mean_y = mean_y
                    best = (a, b, r2)

            #fallback if nothing meets r2_min
            if best is None and r2 > best_r2:
                best_r2 = r2
                best = (a, b, r2)

    if best is None:
        return iL, iR, np.nan

    return best[0], best[1], best[2]

def extract_measured_temp(drops, sipm_roi_for_fit):
    x = -np.asarray(drops, dtype=float)
    sipm = np.asarray(sipm_roi_for_fit, dtype=float)

    #Pairing-safe mask (same mask applied to x and sipm)
    good = np.isfinite(x) & np.isfinite(sipm) & (sipm < 0)

    print("len(drops):", len(drops), "len(sipm):", len(sipm), "len(good):", good.sum())
    print("drops: nan", np.isnan(x).sum(), "inf", np.isinf(x).sum(),
          "min", np.nanmin(x), "max", np.nanmax(x))
    print("sipm:  nan", np.isnan(sipm).sum(), "inf", np.isinf(sipm).sum(),
          "min", np.nanmin(sipm), "max", np.nanmax(sipm))

    x = x[good]
    rate = -sipm[good]
    eps = max(1e-12, np.percentile(rate, 1) * 1e-3)
    y = np.log(np.maximum(rate, eps))
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]

    # Find flank region based on slope 
    (iL, iR), base_s, sig_s, thrH, thrL, slope, ys_s, xs2, ys2 = auto_flank_from_slope(
        xs, ys,
        baseline_quantile=0.25,
        sigma_low=2.5,
        sigma_high=0.8,
        pad_points=25,
        min_pts=100,
        debug=True
    )

    #IMPORTANT: from this point onward, use xs2/ys2 (the cleaned arrays the flank indices refer to)
    xs = xs2
    ys = ys2
    '''
    # Gate to the rising branch: throw away the bottom (flat baseline) part of the flank
    seg = slice(iL, iR+1)
    y_cut = np.quantile(ys[seg], 0.65)          # keep top 35% of y values inside flank
    keep = np.where(ys[seg] >= y_cut)[0]

    if keep.size > 0:
        iL = iL + keep[0]                       # shift left edge to where rise has started

    L = iR - iL + 1
    min_pts_sub = min(100, L)
    # This is the key: force fit to the LOWER/MID part of the rise, not the top
    y_floor = np.quantile(ys[seg], 0.35)   # excludes baseline-ish bottom
    y_cap   = np.quantile(ys[seg], 0.80)   # excludes top 20% (saturation-ish)

    # --- Choose a “best” linear subwindow inside the flank ---
    iL2, iR2, r2 = best_linear_subwindow(xs, ys, iL, iR, min_pts=min_pts_sub, r2_min=0.970, y_floor=y_floor, y_cap=y_cap)

    # fallback safety
    if iL2 is None or iR2 is None:
        iL2, iR2 = iL, iR

    x_fit = xs[iL2:iR2+1]
    y_fit = ys[iL2:iR2+1]

    if len(x_fit) < 2:
        raise RuntimeError("Fit window ended up too small.")

    print("flank length:", L, "min_pts_sub:", min_pts_sub, "r2:", r2)
    print("x_fit range:", x_fit.min(), "to", x_fit.max(), "fit pts:", len(x_fit))
    print("candidate flank length:", iR - iL + 1)
    '''
    
    #choose LOWER part of the rise as a CONTINUOUS interval using smoothed y.
    # ys_s returned from auto_flank_from_slope aligns with xs2/ys2
    y_sm = ys_s

    seg = slice(iL, iR+1)
    xs_seg = xs[seg]
    y_raw  = ys[seg]
    y_seg  = y_sm[seg]

    #levels of the rise computed on SMOOTHED y
    y_lo = np.quantile(y_seg, 0.117)
    y_hi = np.quantile(y_seg, 0.95)
    dy_rise = y_hi - y_lo

    #We’ll try a few “lower-rise” bands and take the first that gives a sane fit
    #(lower band => earlier part of rise)
    bands = [
        (0.117, 0.30),
        (0.12, 0.35),
        (0.14, 0.40),
        (0.15, 0.45),
        (0.17, 0.50),
    ]

    min_pts_needed = 160  #widen/narrow this as you like (e.g., 80–200)

    best = None

    for fa, fb in bands:
        yA = y_lo + fa * dy_rise
        yB = y_lo + fb * dy_rise

        idxA = np.where(y_seg >= yA)[0]
        idxB = np.where(y_seg >= yB)[0]
        if idxA.size == 0 or idxB.size == 0:
            continue

        a = int(idxA[0])
        b0 = int(idxB[0])

        #ensure minimum points (same as before)
        b = b0
        if (b - a + 1) < min_pts_needed:
            b = min(len(xs_seg) - 1, a + min_pts_needed - 1)

        #Extend b in chunks and stop when R² drops too much or slope changes too much.
        step = 25
        max_b = len(xs_seg) - 1

        #base fit on the "good region" we already have
        x0 = xs_seg[a:b+1]
        y0 = y_raw[a:b+1]
        m0, c0 = np.polyfit(x0, y0, 1)
        yhat0 = m0*x0 + c0
        ss_res0 = np.sum((y0 - yhat0)**2)
        ss_tot0 = np.sum((y0 - np.mean(y0))**2) + 1e-12
        r2_0 = 1 - ss_res0/ss_tot0

        #thresholds: tune these gently
        r2_drop_allow = 0.09      #allow some degradation as we extend
        slope_change_allow = 0.27 #allow 20% slope change

        best_b = b
        best_r2 = r2_0

        while best_b + step <= max_b:
            cand_b = min(max_b, best_b + step)

            xw = xs_seg[a:cand_b+1]
            yw = y_raw[a:cand_b+1]

            #must still rise overall (avoid plateau)
            if (np.median(yw[-10:]) - np.median(yw[:10])) < 0.3:
                break

            m1, c1 = np.polyfit(xw, yw, 1)
            if m1 <= 0:
                break

            yhat = m1*xw + c1
            ss_res = np.sum((yw - yhat)**2)
            ss_tot = np.sum((yw - np.mean(yw))**2) + 1e-12
            r2_1 = 1 - ss_res/ss_tot

            #stop if it starts bending/plateauing (linearity deteriorates)
            if r2_1 < (best_r2 - r2_drop_allow):
                break

            #stop if slope changes too much (curvature)
            if abs(m1 - m0) / (abs(m0) + 1e-12) > slope_change_allow:
                break

            best_b = cand_b
            best_r2 = r2_1

        #final fit window (same a, extended b)
        x_fit = xs_seg[a:best_b+1]
        y_fit = y_raw[a:best_b+1]

        #sanity: x span
        if np.ptp(x_fit) < 0.03:
            continue

        #quick check for positive slope
        m_tmp, c_tmp = np.polyfit(x_fit, y_fit, 1)
        if m_tmp <= 0:
            continue

        #compute r2 for reporting
        yhat = m_tmp*x_fit + c_tmp
        ss_res = np.sum((y_fit - yhat)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2) + 1e-12
        r2_tmp = 1 - ss_res/ss_tot

        best = (x_fit, y_fit, fa, fb, r2_tmp, m_tmp)
        break

    if best is None:
        raise RuntimeError("Could not find a stable lower-rise fit interval (bands too strict).")

    x_fit, y_fit, fa_used, fb_used, r2_used, m_used = best
    print(f"Lower-rise band used: {fa_used:.2f}–{fb_used:.2f} of rise; pts={len(x_fit)}; r2={r2_used:.3f}; slope={m_used:.3g}")
    print("x_fit range:", x_fit.min(), "to", x_fit.max())
    
    #center x for nicer intercept / numerics
    x0 = np.mean(x_fit)
    m, c = np.polyfit(x_fit - x0, y_fit, 1)
    b = c - m*x0
    Measured_Temp = qe * 1.05 / (kb * abs(m))

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
filepath1=iter_all('csv','../')[8]
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