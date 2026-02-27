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

Mmax=1
Nmax=20000
zeros=[special.jn_zeros(m,Nmax) for m in range(Mmax)]

#directory search function
def iter_all(substring, path):
    return list(
        os.path.join(root, entry)
        for root, dirs, files in os.walk(path)
        for entry in dirs + files
        if substring in entry
    )

#%%
def getElectrodeVoltageDrop(electrodeConfig, rpoints, zpoints, left, right, mur2, rfact,
                            debug=False, return_debug=False, debug_title=None):
    ''' Gets the combined Electrode "Excitement/Voltage Drop" Profile without running solver.
        If debug=True: plots axial_profile and highlights chosen barrier/peak points.
        If return_debug=True: returns (VoltageDrop, debug_dict)
    '''
    nr = rpoints
    nz = zpoints
    leftbound = left
    rightbound = right
    rbound = mur2 * rfact

    position_map_r = np.zeros((nr, nz))
    position_map_z = np.zeros((nr, nz))
    for rind in range(nr):
        for zind in range(nz):
            position_map_z[rind, zind] = leftbound + (rightbound - leftbound) * (zind + 0.5) / nz
            position_map_r[rind, zind] = rbound * rind / nr

    z_axis = position_map_z[0, :]  # cell centers along z
    dz = (rightbound - leftbound) / nz

    def getElectrodeSolution(left_e, right_e, voltage):
        cs = 2 * voltage / (zeros[0] * special.j1(zeros[0]))
        sym = getFiniteSolution([cs], right_e - left_e, lambda r: voltage)
        return lambda r, z: sym(z - (left_e + right_e) / 2, r)

    electrode_voltages, electrode_borders = electrodeConfig

    free_space_solution = np.zeros((nr, nz))
    electrodes = [
        getElectrodeSolution(electrode_borders[i-1], electrode_borders[i], electrode_voltages[i])
        for i in range(1, len(electrode_voltages) - 1)
    ]
    for electrode in electrodes:
        free_space_solution += electrode(position_map_r, position_map_z)

    axial_profile = free_space_solution[0, :]  # r=0 axial cut

    # ---- slopes / thresholds ----
    dVdz = np.gradient(axial_profile, dz)
    abs_dVdz = np.abs(dVdz)

    # make the quantile slice safe for small nz
    lo = min(10, max(1, nz // 10))
    hi = max(lo + 1, nz - 1)
    thr_slope = np.quantile(abs_dVdz[lo:hi], 0.05)

    is_flat_slope = abs_dVdz <= thr_slope
    d2Vdz2 = np.gradient(dVdz, dz)

    # ---- local MAXIMA (peak candidates) ----
    is_local_max = np.zeros(nz, dtype=bool)
    is_local_max[1:-1] = (axial_profile[1:-1] > axial_profile[:-2]) & (axial_profile[1:-1] >= axial_profile[2:])
    valid_max = is_local_max & is_flat_slope
    valid_max[:10] = False
    valid_max &= (d2Vdz2 < 0)  # concave down => maximum

    maxs = np.where(valid_max)[0]
    if maxs.size > 0:
        peak_idx = int(maxs[np.argmax(axial_profile[maxs])])
    else:
        peak_idx = int(np.argmax(axial_profile))  # fallback

    # ---- local MINIMA (barrier candidates) ----
    is_local_min = np.zeros(nz, dtype=bool)
    is_local_min[1:-1] = (axial_profile[1:-1] < axial_profile[:-2]) & (axial_profile[1:-1] <= axial_profile[2:])
    valid_min = is_local_min & is_flat_slope
    valid_min[:10] = False
    valid_min &= (d2Vdz2 > 0)  # concave up => minimum

    # barrier: minimum BEFORE the peak (prefer valid minima if they exist)
    if peak_idx <= 0:
        barrier_idx = 0
    else:
        mins_left = np.where(valid_min & (np.arange(nz) < peak_idx))[0]
        if mins_left.size > 0:
            barrier_idx = int(mins_left[np.argmin(axial_profile[mins_left])])
        else:
            barrier_idx = int(np.argmin(axial_profile[:peak_idx]))  # fallback

    V_peak = float(axial_profile[peak_idx])
    V_barrier = float(axial_profile[barrier_idx])
    VoltageDrop = V_peak - V_barrier  # your sign convention

    # ---- DEBUG PLOTS ----
    if debug:
        title = debug_title or "Drop debug"

        # Plot 1: axial profile + candidates + chosen points
        plt.figure(figsize=(10, 4))
        plt.plot(z_axis, axial_profile, 'k-', lw=2, label="axial_profile (r=0)")

        plt.plot(z_axis[valid_min], axial_profile[valid_min], 'o', ms=5, alpha=0.6, label="valid minima candidates")
        plt.plot(z_axis[valid_max], axial_profile[valid_max], 'o', ms=5, alpha=0.6, label="valid maxima candidates")

        plt.plot(z_axis[peak_idx], V_peak, 'o', ms=10, label=f"PEAK idx={peak_idx} V={V_peak:.3g}")
        plt.plot(z_axis[barrier_idx], V_barrier, 'o', ms=10, label=f"BARRIER idx={barrier_idx} V={V_barrier:.3g}")

        plt.axvline(z_axis[peak_idx], ls="--", alpha=0.5)
        plt.axvline(z_axis[barrier_idx], ls="--", alpha=0.5)

        for b in electrode_borders:
            if leftbound <= b <= rightbound:
                plt.axvline(b, ls=":", alpha=0.4)

        plt.title(f"{title} | Drop = {VoltageDrop:.4g} V")
        plt.xlabel("z (m)")
        plt.ylabel("V (arb)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 2: |dV/dz| and thr_slope + chosen points
        plt.figure(figsize=(10, 3.5))
        plt.plot(z_axis, abs_dVdz, 'k-', lw=2, label="|dV/dz|")
        plt.axhline(thr_slope, ls="--", lw=2, alpha=0.7, label=f"thr_slope={thr_slope:.3g}")
        plt.plot(z_axis[peak_idx], abs_dVdz[peak_idx], 'o', ms=8, label="at peak")
        plt.plot(z_axis[barrier_idx], abs_dVdz[barrier_idx], 'o', ms=8, label="at barrier")
        plt.xlabel("z (m)")
        plt.ylabel("|dV/dz|")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.imshow(free_space_solution, aspect='auto', origin='lower')
        plt.title(f"{title} | free_space_solution (r vs z index)")
        plt.xlabel("z index")
        plt.ylabel("r index")
        plt.colorbar(label="V (arb)")
        plt.tight_layout()
        plt.show()

    debug_dict = {
        "z_axis": z_axis,
        "axial_profile": axial_profile,
        "abs_dVdz": abs_dVdz,
        "thr_slope": thr_slope,
        "valid_min": valid_min,
        "valid_max": valid_max,
        "is_local_min": is_local_min,
        "is_local_max": is_local_max,
        "peak_idx": peak_idx,
        "barrier_idx": barrier_idx,
        "V_peak": V_peak,
        "V_barrier": V_barrier,
        "VoltageDrop": VoltageDrop,
        "free_space_solution": free_space_solution,
    }

    if return_debug:
        return VoltageDrop, debug_dict

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
def getTotalVoltageDropProfile(converted_voltages, debug_steps=(0, -1)):
    converted_voltages = np.asarray(converted_voltages)
    v_start = float(converted_voltages[0])
    v_end   = float(converted_voltages[-1])
    n_steps = len(converted_voltages)
    v_sweep = np.linspace(v_start, v_end, n_steps)

    electrode_borders = [0.025, 0.050, 0.100, 0.125]
    drops = []

    # allow negative indexing in debug_steps (e.g. -1)
    debug_steps_set = set([(s if s >= 0 else n_steps + s) for s in debug_steps])

    for i, v in enumerate(v_sweep):
        electrode_voltages = [0, v, -50, -130, 0]
        electrodeConfig = (electrode_voltages, electrode_borders)

        debug_now = (i in debug_steps_set)
        drop = getElectrodeVoltageDrop(
            electrodeConfig,
            rpoints=20, zpoints=40,
            left=Llim, right=Rlim,
            mur2=rad2, rfact=3.0,
            debug=debug_now,
            debug_title=f"Step {i}/{n_steps-1} | v={v:.3g}"
        )
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
    
    # --- Pick a robust "lower-rise" fit window inside the detected flank ---
    seg = slice(iL, iR+1)
    xs_seg = xs[seg]
    y_raw  = ys[seg]
    y_sm   = ys_s[seg]          # smoothed y aligned to xs/ys
    s_seg  = slope[seg]         # smoothed slope aligned to xs/ys

    Lseg = len(xs_seg)
    if Lseg < 20:
        raise RuntimeError(f"Flank segment too small (Lseg={Lseg}).")

    # adaptive minimum points
    min_pts_needed = min(160, max(30, int(0.35 * Lseg)))

    # define rise levels on SMOOTHED y
    y10 = np.quantile(y_sm, 0.10)
    y90 = np.quantile(y_sm, 0.90)
    dy  = y90 - y10

    # pick start 'a': where slope becomes meaningfully positive
    smax = float(np.max(s_seg))
    s_thr = 0.25 * smax if smax > 0 else 0.0
    a_candidates = np.where(s_seg >= s_thr)[0]
    a = int(a_candidates[0]) if a_candidates.size else int(np.argmax(s_seg))

    # nudge a so we’re not in the flat baseline
    a2 = np.where(y_sm >= y10)[0]
    if a2.size:
        a = max(a, int(a2[0]))

    # pick end 'b': first time we reach some fraction of the rise
    targets = [0.30, 0.40, 0.50, 0.60, 0.70]
    b = None
    if dy > 1e-6:
        for frac in targets:
            yT = y10 + frac * dy
            idx = np.where(y_sm >= yT)[0]
            if idx.size:
                b = int(idx[0])
                break

    # fallback if we couldn’t find a y-threshold end
    if b is None:
        b = min(Lseg - 1, a + min_pts_needed - 1)

    # enforce min points
    b = max(b, min(Lseg - 1, a + min_pts_needed - 1))

    # cap away from saturation/top: don’t extend beyond ~90% level if possible
    sat = np.where(y_sm >= y90)[0]
    if sat.size:
        b = min(b, int(sat[0]))

    # final safety: if window is still too small, just grab first min_pts_needed from a
    if (b - a + 1) < max(10, min_pts_needed // 2):
        b = min(Lseg - 1, a + min_pts_needed - 1)

    x_fit = xs_seg[a:b+1]
    y_fit = y_raw[a:b+1]

    # last-resort fallback if x-span is tiny relative to the available flank span
    if np.ptp(x_fit) < max(1e-4, 0.05 * np.ptp(xs_seg)):
        a = 0
        b = min(Lseg - 1, min_pts_needed - 1)
        x_fit = xs_seg[a:b+1]
        y_fit = y_raw[a:b+1]

    # --- quick sanity checks + debug ---
    if len(x_fit) < 2:
        raise RuntimeError("Fit window ended up too small.")

    m_tmp, c_tmp = np.polyfit(x_fit, y_fit, 1)
    if m_tmp <= 0:
        raise RuntimeError("Fit slope is not positive; window selection likely wrong.")

    print(f"[fitwin] Lseg={Lseg}, min_pts={min_pts_needed}, a={a}, b={b}, "
        f"xspan={np.ptp(x_fit):.4g}, yspan={np.ptp(y_fit):.4g}, slope={m_tmp:.4g}")
    print("x_fit range:", float(x_fit.min()), "to", float(x_fit.max()))
        
    #center x for nicer intercept / numerics
    x0 = np.mean(x_fit)
    m, c = np.polyfit(x_fit - x0, y_fit, 1)
    b = c - m*x0
    Measured_Temp = qe * 1.05 / (kb * abs(m))

    plt.figure(figsize=(10,4))
    plt.gca().set_facecolor('none')
    plt.gcf().patch.set_alpha(0)

    # Deep blue full data
    plt.plot(xs, ys, 'o-', 
            color="#05696B",     #1418E2   
            markersize=3,
            alpha=0.7,
            label="log(-SiPM)")

    # Purple fit region
    plt.plot(x_fit, y_fit, 'o', 
            color="#ff1493",   #F3CC0B   #800080
            markersize=4,
            label="fit region")

    # Pink fit line
    xline = np.linspace(x_fit.min(), x_fit.max(), 200)
    plt.plot(xline, m*xline + b, '-', 
            color="#1418E2",        
            linewidth=2,
            label=f"fit: y={m:.3g}x+{b:.3g}")

    plt.xlabel("Arb. time units", size=18)
    plt.ylabel("log(|SiPM|)", size=18)
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return Measured_Temp

#%% - TEST RUN
Recompute_Drops=True #trun to False to skip voltage drop recomputation 
filepath1=iter_all('csv','../')[8]
converted_voltages, _, _, sipm_roi_for_fit = fit_and_convert_u8(filepath1)
if Recompute_Drops: 
    drops = getTotalVoltageDropProfile(converted_voltages, debug_steps=(0, len(converted_voltages)//2, -1))   
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