#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter


Mmax=1
Nmax=20000
zeros=[special.jn_zeros(m,Nmax) for m in range(Mmax)]

# Physical Constants
e0=8.854187817e-12 #farads per meter


def iter_all(substring, path):
    return list(
        os.path.join(root, entry)
        for root, dirs, files in os.walk(path)
        for entry in dirs + files
        if substring in entry
    )

def auto_roi_from_dip(x, y, smooth_window=71, polyorder=3, baseline_quantile=0.90,
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
    mask[i_left:i0] = True     

    return mask, x[i_left], x[i0], (i_left, i0), baseline, sigma, thr_high, thr_low

def u8Correction(filename,offset):

    try:
        df = pd.read_csv(filename, header=None)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    sipm_data = df[0].values  #sipm (~escape rate)
    u8_data = df[1].values    #u8 excitations

    sipm_data = sipm_data[offset:len(sipm_data)]
    u8_data = u8_data[0:-offset]
    
    mask, xL, xR, (iL, iR), baseline, sig, thr_high, thr_low = auto_roi_from_dip(u8_data, sipm_data) #Auto-detect ROI around dip


    print(np.shape(sipm_data))
    print(np.shape(u8_data))

    u8_data = u8_data[mask]
    sipm_data = sipm_data[mask]

    order = np.argsort(u8_data)

    sipm_ordered = sipm_data[order]
    u8_ordered = u8_data[order]
    t_data = np.arange(0,len(u8_data))

    u8vsT_fit = np.polyfit(t_data,u8_ordered,4)
    #print(u8vsT_fit)
    u8_ordered_fit = np.polyval(u8vsT_fit,t_data)
    u8_corrected = u8_ordered_fit * 15

    #plt.plot(t_data,u8_ordered)
    #plt.plot(t_data,u8_ordered_fit, color="g",ms=0.4)
    #plt.show()
    #plt.plot(abs(u8_ordered_fit-u8_ordered))
    #plt.show()
    #plt.scatter(t_data,u8_ordered*15, label="original", marker=".", color="orange", s=2)
    #plt.scatter(t_data,u8_corrected, label="corrected", marker=".", color="blue", s=2)
    #plt.legend()
    #plt.title("u8 correction")
    #plt.xlabel("time (arb)")
    #plt.ylabel("u8 (arb)")
    #plt.show()

    return u8_corrected, sipm_ordered
#%%
def getFiniteSolution(cnms,length,inf,rw,preclimit=1e-13,N=Nmax,M=Mmax):
    def solution(z,r):        
        pre=-np.sign(z-length/2)/2-np.sign(-length/2-z)/2
        c1=np.sign(-length/2-z)
        c2=np.sign(z-length/2)
        c4=c2
        c3=c1           
        total=pre*inf(r)
        if cnms[0][0]==0:
            return total
                
        for m in range(M):
            for n in range(N):
                if m==0:
                    new=( 
                        special.j0(zeros[m][n]*r/rw)*cnms[m][n]
                        *(.5*(c1*np.exp(c3*zeros[m][n]*(length/2+z)/rw)+c2*np.exp(c4*zeros[m][n]*(length/2-z)/rw)))
                        )
                else:
                    new=( 
                        special.jn(m,zeros[m][n]*r/rw)*cnms[m][n]
                        *(pre+.5*(c1*np.exp(c3*zeros[m][n]*(length/2+z)/rw)+c2*np.exp(c4*zeros[m][n]*(length/2-z)/rw)))
                        )
                total+=new
                if np.all(abs(new)/(abs(new)+abs(total))<preclimit):
                    return total
        return total
    return solution

def finiteChargeCylinder(cpl,rp,rw,length):
    coeffs=(
        (cpl/(np.pi*e0*(rw*special.j1(zeros[0]))**2))
        *(
        ((np.log(rp/rw)-.5)*(rp*rw*special.j1(zeros[0]*rp/rw)/zeros[0]))
        +((rw*rw*2*special.jn(2,zeros[0]*rp/rw)-rp*rw*zeros[0]*special.jn(3,zeros[0]*rp/rw))/(2*zeros[0]*zeros[0]))
        +(rw*(-rp*zeros[0]*special.j1(zeros[0]*rp/rw)*np.log(rp/rw)-rw*special.j0(zeros[0]*rp/rw))/(zeros[0]*zeros[0]))
        )
        )
    def inf(r):
        r_l_rp=(np.sign(rp-r)+1)/2
        return ( 
            r_l_rp       *(cpl/(2*np.pi*e0)) *(np.log(rp/rw)+(r*r-rp*rp) /(2*rp*rp))
            + (1-r_l_rp) *(cpl/(2*np.pi*e0)) *(np.log(np.maximum(r,rp)   /rw))
            )
    return getFiniteSolution([coeffs],length,inf,rw=rw)

def unitFreeSpaceSolutionsCalculation(electrode_input,nr,nz,rad2):
    initial_voltages,final_voltages,electrode_borders,Llim,Rlim,rw = electrode_input

    position_map_r=np.zeros((nr,nz))
    position_map_z=np.zeros((nr,nz))

    rbound = rad2*3.0
    for rind in range(nr):
        for zind in range(nz):
            position_map_z[rind,zind]=Llim+(Rlim-Llim)*(zind+.5)/(nz)
            position_map_r[rind,zind]=rbound*rind/nr
    
    def getElectrodeSolution(left,right,voltage):
        cs=2*voltage/(zeros[0]*special.j1(zeros[0]))
        symmetricsolution=getFiniteSolution([cs],right-left,lambda r:voltage,rw=rw)
        return lambda r,z: symmetricsolution(z-(left+right)/2,r)

    free_space_soln_list = [] #result of adding 1 excitation to i'th electrode
    free_1d_list = []
    for i in np.arange(0,len(initial_voltages)):
        voltages = np.zeros(len(initial_voltages))
        voltages[i] = 1

        #print(f"voltages: {voltages}")

        

        electrodes=[getElectrodeSolution(electrode_borders[j-1],electrode_borders[j],voltages[j]) for j in range(1,len(voltages)-1)]
            #probably zero everywhere other than i
    
        free_space_solution=np.zeros((nr,nz))

        for electrode in electrodes:
            free_space_solution+=electrode(position_map_r,position_map_z)
        
        free_1d = free_space_solution[0]
        free_1d_list.append(free_1d)
        free_space_soln_list.append(free_space_solution)


    return free_space_soln_list,free_1d_list,position_map_r,position_map_z

def superpose_potential(unit_solutions, electrode_voltages):
    total = np.zeros_like(unit_solutions[0])
    
    for i, V in enumerate(electrode_voltages):
        total += V * unit_solutions[i]
    
    return total

def u8_to_drop(unit_solutions, u8_input, initial_voltages, nz, Rlim, Llim):
    peak_idx = -1
    barrier_idx = -1

    if (u8_input - initial_voltages[2]) > 0.005:
        print(u8_input - initial_voltages[2])
        return float("nan"), peak_idx, barrier_idx

    dz = (Rlim - Llim) / nz
    
    voltages = np.array(initial_voltages, copy=True)
    voltages[1] = u8_input

    free_1d = superpose_potential(unit_solutions, voltages)[0]

    # ---- slopes / thresholds ----
    dVdz = np.gradient(free_1d, dz)
    abs_dVdz = np.abs(dVdz)
    lo = min(10, max(1, nz // 10))
    hi = max(lo + 1, nz - 1)
    thr_slope = np.quantile(abs_dVdz[lo:hi], 0.07)

    is_flat_slope = abs_dVdz <= thr_slope
    d2Vdz2 = np.gradient(dVdz, dz)

    # ---- local MAXIMA (peak candidates) ----
    is_local_max = np.zeros(nz, dtype=bool)
    is_local_max[1:-1] = (free_1d[1:-1] > free_1d[:-2]) & (free_1d[1:-1] >= free_1d[2:])
    valid_max = is_local_max & is_flat_slope
    valid_max[:10] = False
    valid_max &= (d2Vdz2 < 0)
    maxs = np.where(valid_max)[0]
    if maxs.size > 0:
        peak_idx = int(maxs[np.argmax(free_1d[maxs])])
    else:
        peak_idx = int(np.argmax(free_1d))  # fallback

    # ---- local MINIMA (barrier candidates) ----
    is_local_min = np.zeros(nz, dtype=bool)
    is_local_min[1:-1] = (free_1d[1:-1] < free_1d[:-2]) & (free_1d[1:-1] <= free_1d[2:])
    valid_min = is_local_min & is_flat_slope
    valid_min[:10] = False
    valid_min &= (d2Vdz2 > 0)
    
    if peak_idx <= 0:
        barrier_idx = 0
    else:
        mins_left = np.where(valid_min & (np.arange(nz) < peak_idx))[0]
        if mins_left.size > 0:
            barrier_idx = int(mins_left[np.argmin(free_1d[mins_left])])
        else:
            barrier_idx = int(np.argmin(free_1d[:peak_idx]))  # fallback

    # Ensure indices are in range
    peak_idx = max(0, min(nz-1, peak_idx))
    barrier_idx = max(0, min(nz-1, barrier_idx))

    # Compute voltage drop
    if peak_idx == barrier_idx:
        VoltageDrop = float("nan")
    else:
        V_peak = float(free_1d[peak_idx])
        V_barrier = float(free_1d[barrier_idx])
        VoltageDrop = V_peak - V_barrier

    return VoltageDrop, peak_idx, barrier_idx

def convert_u8_array_to_vacdrop_array(filepath, electrode_input, offset, nr, nz, rad2):
    initial_voltages, final_voltages, electrode_borders, Llim, Rlim, rw = electrode_input
    u8_corrected, sipm_ordered = u8Correction(filepath,offset)
    unit_solutions, unit_free_1ds, position_map_r, position_map_z = unitFreeSpaceSolutionsCalculation(electrode_input, nr=nr, nz=nz, rad2=rad2)
    #plt.scatter(u8_corrected,sipm_ordered,marker=".")
    #plt.title("sipm vs u8")
    #plt.xlabel("u8 (arb)")
    #plt.ylabel("sipm (arb)")
    #for voltages in unit_solutions:
    #    plt.imshow(voltages, aspect='auto', origin='lower')
    #    plt.colorbar(label="Potential")
    #    plt.xlabel("z index")
    #    plt.ylabel("r index")
    #    plt.title("2D Potential Map")
    #plt.show()


    #fig, axes = plt.subplots(len(unit_solutions), 1, figsize=(10, 8), sharex=True)

    #for i, voltages in enumerate(unit_solutions):
    #    im = axes[i].imshow(voltages, aspect='auto', origin='lower')
    #    axes[i].set_ylabel("r index")
    #    axes[i].set_title(f"2D Potential Map {i+1}")
    #    
    #    # Attach colorbar to each subplot
    #    fig.colorbar(im, ax=axes[i], label="Potential")

    #axes[-1].set_xlabel("z index")

    #plt.tight_layout()
    #plt.show()


    drops_data = np.zeros_like(u8_corrected)


    red_blue = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

    # Generate colors for each point
    n_points = len(u8_corrected)
    colors = red_blue(np.linspace(0, 1, n_points))  # first point red, last blue

    for i, u8 in enumerate(u8_corrected):
        voltages = np.array([0,u8,-50,-130,0])
        drop,peak_idx,barrier_idx = u8_to_drop(unit_solutions,u8,voltages,nz,Rlim,Llim)
        #print(f"i: {i}/{len(u8_corrected)}, u8: {u8}, drop: {drop:.10E}")
        drops_data[i]=drop
        #if i%250==0:
        #    plt.plot(position_map_z[0],superpose_potential(unit_solutions,voltages)[0],zorder=1, color=colors[i])
        #    if not np.isnan(drop):
        #        plt.scatter(position_map_z[0,peak_idx],
        #                    superpose_potential(unit_solutions,voltages)[0,peak_idx],
        #                    color="black",marker="^",zorder=2,s=4)
        #        plt.scatter(position_map_z[0,barrier_idx],
        #                    superpose_potential(unit_solutions,voltages)[0,barrier_idx],
        #                    color="black",marker="^",zorder=2,s=4)
    #plt.ylabel(f"Potential (V)")
    #plt.xlabel(f"Axial position (m)")
    #plt.show()

    mask = ~np.isnan(drops_data)

    drops_data = drops_data[mask]
    u8_corrected = u8_corrected[mask]
    sipm_corrected = sipm_ordered[mask]

    #plt.plot(u8_corrected, drops_data, marker=".", color="green")
    #plt.title("Vacuum drop vs u8")
    #plt.xlabel("u8 (arb)")
    #plt.ylabel("Vacuum drop (V)")
    #plt.show()


    #plt.scatter(u8_corrected, sipm_corrected, marker=".", color="green")
    #plt.title("u8 vs sipm")
    #plt.xlabel("u8 (arb)")
    #plt.ylabel("Vacuum drop (V)")
    #plt.show()

    #plt.scatter(drops_data, np.log10(-sipm_corrected), marker=".", color="purple")
    #plt.title("Vacuum drop vs sipm")
    #plt.xlabel("Vacuum drop (V)")
    #plt.ylabel("sipm (arb)")
    #plt.xlim(0,2)
    #plt.gca().invert_xaxis()
    #plt.show()
    return drops_data, u8_corrected, sipm_corrected

#convert_u8_array_to_vacdrop_array(filepath1, electrode_input, nr, nz, rad2)
#returns drops_data, u8_corrected, sipm_corrected
