from find_solution import find_solution
import numpy as np
import matplotlib.pyplot as plt

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

def step_5(target_drop, sol, initial_voltages, final_voltages, N_e, T_e, rad2, B, electrodeConfig,Llim, Rlim, zpoints=40, rpoints=20, rfact=3.0, plotting=False, coarse_sol_divisor=100):
    """
    input: target_drop in volts, find_solution output sol
    output: NDarray: [rf_star, achieved_drop]
    """
    omega_r = sol[6]
    def drop_for_rampfrac(rf): # finds "drop", i.e. potential difference between plasma and barrier, for a given rampfrac
        electrode_borders = electrodeConfig[1]
        volts = np.array(initial_voltages) + (np.array(final_voltages) - np.array(initial_voltages)) * rf # volts along the ramp
        
        sol_tmp = find_solution(N_e,
                            T_e,
                            omega_r/(2*np.pi),
                            mur2=rad2,
                            B=B,
                            electrodeConfig=(volts, electrode_borders),
                            left=Llim,
                            right=Rlim,
                            zpoints=zpoints,
                            rpoints=rpoints,
                            rfact=rfact,
                            plotting=plotting,
                            coarse_sol_divisor=coarse_sol_divisor,
                            InitializeWithPlasmaLength = False,
                            fail_action='raise', debug_tag='')
        if sol_tmp is None:
            return np.inf, volts, None

        vfree_tmp = sol_tmp[4]          # free_space_solution
        vsc_tmp   = sol_tmp[3]          # voltageGuess (space-charge corrected)
        vfree_on  = vfree_tmp[0, :]
        vsc_on    = vsc_tmp[0, :]
        print(f'rf: {rf:.4f}')
        peak_idx   = int(np.argmax(vfree_on))
        barrier_idx = int(np.argmin(vfree_on[0:peak_idx]))  # left barrier only may need to change if plasma shifts right
        drop = float(vsc_on[peak_idx] - vfree_on[barrier_idx])  # magnitude in volts

        if vsc_on[peak_idx] - vfree_on[barrier_idx] < 0:
            print(f"  Warning: negative drop ({drop:.3f} V) for rampfrac={rf:.4f}; plasma may be escaping.")
            

        return drop, volts, sol_tmp

    def coarse_scan(scan_points=21, sol= None):
        grid = np.linspace(0.0, 1.0, scan_points) # 21 is number of points to scan arbitrarily chosen
        drops = []

        if sol is None:
            raise RuntimeError("coarse_scan requires find_solution parameters.")

        for rf in grid:
            d, _, _ = drop_for_rampfrac(rf)
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



    
    
    print("--- Performing coarse scan to map rampfrac to drop ---")
    grid, drops = coarse_scan(scan_points=21, sol=sol)
    print("--- Coarse scan complete ---")
    rf_star, achieved_drop = find_rf_for_target_drop(target_drop, interp_points=10000, grid=grid, drops=drops)
    return rf_star, np.array([grid, drops])