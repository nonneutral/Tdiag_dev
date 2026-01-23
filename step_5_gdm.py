#Find rampfrac that gives target drop using gradient descent method

from find_solution import find_solution
import numpy as np
import matplotlib.pyplot as plt



def gradient_descent_method(initial_voltages, final_voltages, N_e, T_e, omega_r, rad2, B, electrodeConfig, Llim, Rlim, zpoints=40, rpoints=20, rfact=3.0, plotting=False, coarse_sol_divisor=100):
    print("gradient_descent_method")

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
        #peak_idx   = int(np.argmax(vfree_on))
        #barrier_idx = int(np.argmin(vfree_on[0:peak_idx]))  # left barrier only may need to change if plasma shifts right
        
        peak_idx = 17
        barrier_idx = 5
        print(f"peak_idx, {peak_idx}")
        print(f"barrier_idx, {barrier_idx}")
        drop = float(vsc_on[peak_idx] - vfree_on[barrier_idx])  # magnitude in volts

        if vsc_on[peak_idx] - vfree_on[barrier_idx] < 0:
            print(f"  Warning: negative drop ({drop:.3f} V) for rampfrac={rf:.4f}; plasma may be escaping.")
            

        return drop, volts, sol_tmp
    
    rf = 0.0  # initial guess for rampfrac
    learning_rate = 0.00005
    tolerance = 1e-6
    max_iterations = 20

    history_drop = []
    history_rf = []
    for iteration in range(max_iterations):
        drop, _, _ = drop_for_rampfrac(rf)
        drop_plus, _, _ = drop_for_rampfrac(rf + tolerance)
        gradient = (drop_plus**2 - drop**2) / tolerance
        history_drop.append(drop)
        history_rf.append(rf)

        rf_old = rf
        rf -= learning_rate * gradient
        if rf_old - rf < tolerance*0.1:
            break
        
        plt.plot(history_rf, history_drop, marker='o')
        plt.xlabel('Ramp Fraction')
        plt.ylabel('Drop (V)')
        plt.title('Gradient Descent History')
        plt.grid()
        plt.show()
    achieved_drop = drop_for_rampfrac(rf)[0]
    print(f"Converged in {iteration} iterations. Rampfrac: {rf}, Achieved Drop: {achieved_drop}")
    return np.array([rf, achieved_drop])