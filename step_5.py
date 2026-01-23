from solver_copy import *
def drop_for_rampfrac(rf, omega_r_use, sol): # finds "drop", i.e. potential difference between plasma and barrier, for a given rampfrac
    
    volts = np.array(initial_voltages) + (np.array(final_voltages) - np.array(initial_voltages)) * rf # volts along the ramp
    
    sol_tmp = sol
    if sol_tmp is None:
        return np.inf, volts, None

    vfree_tmp = sol_tmp[4]          # free_space_solution
    vsc_tmp   = sol_tmp[3]          # voltageGuess (space-charge corrected)
    vfree_on  = vfree_tmp[0, :]
    vsc_on    = vsc_tmp[0, :]
    print(f'rf: {rf:.4f}')

    peak_idx   = int(np.argmax(vfree_on))
    barrier_idx = int(np.argmin(vsc_on[0:peak_idx]))  # left barrier only may need to change if plasma shifts right
    drop = float(vsc_on[peak_idx] - vfree_on[barrier_idx])  # magnitude in volts

    if vsc_on[peak_idx] - vfree_on[barrier_idx] < 0:
        print(f"  Warning: negative drop ({drop:.3f} V) for rampfrac={rf:.4f}; plasma may be escaping.")
        

    return drop, volts, sol_tmp
def coarse_scan(scan_points=21, omega_r= None, sol= None):
    grid = np.linspace(0.0, 1, scan_points) # 21 is number of points to scan arbitrarily chosen
    drops = []

    if omega_r is None or sol is None:
        raise RuntimeError("coarse_scan requires omega_r and sol parameters.")

    for rf in grid:
        d, _, _ = drop_for_rampfrac(rf, omega_r, sol)
        drops.append(d)

    drops = np.array(drops)
    x_arange = np.linspace(0,1,10000)
    drops_interp = np.interp(x_arange, grid, drops)


    plt.title("drop vs rampfracs coarse scan with interpolation")
    plt.plot(grid, drops,"o", label="coarse scan")
    plt.plot(x_arange, drops_interp, "-", label="interp_points = 10000")
    plt.legend()
    plt.show()
    return np.array([grid, drops])
def find_rf_for_target_drop(target_drop,interp_points=10000, grid=None, drops=None):
    """
    finds rampfrac such that drop_for_rampfrac(rampfrac*) == target_drop using linear interpolation method
    
    returns: (rampfrac where target_drop is reached, achieved_drop)
    """
    print("find_rf_for_target_drop")
    if grid is None or drops is None:
        raise RuntimeError("coarse scan data not available; run coarse_scan() first.")

    x_arange = np.linspace(0,1,interp_points)
    drops_interp = np.interp(x_arange, grid, drops)
    rf_star = x_arange[np.argmin(abs(drops_interp - target_drop))]
    achieved_drop = drops_interp[np.argmin(drops_interp - target_drop)]

    return np.array([rf_star, achieved_drop])
print("--- Performing coarse scan to map rampfrac to drop ---")
grid, drops = coarse_scan(scan_points=21)
print("--- Coarse scan complete ---")

def step_5(target_drop, sol):

    #sol = find_solution(NVal=N_e,T_e=T_e,fE=omega_r/(2*np.pi),mur2=rad2,B=B2,
    #                    electrodeConfig=(initial_voltages,electrode_borders),
    #                    left=Llim,right=Rlim,zpoints=40,rpoints=20,rfact=3.0,plotting=True, coarse_sol_divisor=50, InitializeWithPlasmaLength = True)

    grid, drops = coarse_scan(scan_points=21, sol=sol, omega_r=omega_r)
    rf_star, achieved_drop = find_rf_for_target_drop(target_drop, interp_points=10000, grid=grid, drops=drops)
    return np.array([rf_star, achieved_drop])