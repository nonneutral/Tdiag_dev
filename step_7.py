from solver_copy import *

def compute_esc_electrons(fine_sol, T_e): #, N_now, lastescapeE
    ngrid = fine_sol[0]
    full_scc_solution = fine_sol[3] #Full Space-Charge-Corrected (SCC) Solution, i.e. voltageGuess
    position_map_z = fine_sol[1]
    volume_elements = fine_sol[7]
    N_cell = ngrid * volume_elements
    #N_cell = N_cell*N_now/np.sum(N_cell) #normalize grid of number of e- per cell so it sums to N_now
    escape_sum = np.zeros(len(N_cell)) #len apparently returns the number of r values     escapeE = np.zeros(len(N_cell))
    print(f"sum(ngrid*volume_elements) = {np.sum(N_cell):.3f} ---")
    for r, x in enumerate(N_cell): 
        oneD_solution = full_scc_solution[r, :] #Solution across z-axis per radial point, r
        axial_well_idx = np.argmax(oneD_solution)
        barrier_idx = np.argmin(oneD_solution[0:axial_well_idx])  # left barrier only
        escapeE = q_e * abs(oneD_solution[axial_well_idx] - oneD_solution[barrier_idx])
        E_int = erfc(np.sqrt(escapeE / (kb * T_e)))
        escape_sum[r] = E_int * np.sum(N_cell[r, :]) #these are the ones that leave the well from that r
        onaxis_drop = abs(oneD_solution[axial_well_idx] - oneD_solution[barrier_idx])
    return np.sum(escape_sum),onaxis_drop

def escape_curve_scan(start_drop, end_drop, data_points = 25, sol=None):
    """
    Docstring for escape_curve_scan
    we use the term 'drop' to refer to the potential difference between plasma center and barrier
    :param start_drop: in volts
    :param end_drop: in volts
    :param data_points: number of data points
    """
    if sol is None:
        raise RuntimeError("escape_curve_scan requires a fine solution 'sol' parameter.")
    
    start_drop = start_drop # volts
    end_drop = end_drop # volts

    rampfrac_start = find_rf_for_target_drop(start_drop,interp_points=10000)[0]
    rampfrac_end = find_rf_for_target_drop(end_drop)[0]
    
    # ===== ESCAPE CURVE LOOP USING KEEP_SUM METHOD ===== #
    #to be updated for consistency with new definition of compute_kept_electrons
    ramp_values = np.linspace(rampfrac_start, rampfrac_end, data_points) # find rampfrac_end 
    #!!! instead of hard-coding this linspace, make it go from rampfrac equiv of 20 kT/e to 0 kT/e
    #you can do this by finding the points for 20 kT/e and 10 kT/e and extrapolating

    escaped_list = []
    remaining_list = []
    frac_escaped_list = []
    drop_list = []

    N_current = N_e  # total electrons at ramp start

    for i, rampfrac in enumerate(ramp_values):
        print(f"\n--- Rampfrac = {rampfrac:.3f} ---")
        print(f"Electrons entering this step: {N_current:.3e}")

        # Update electrode voltages based on ramp fraction
        current_voltages = initial_voltages + (final_voltages - initial_voltages) * rampfrac

        # Solve for plasma configuration at this ramp
        fine_sol = sol

        # Compute number of electrons that stay trapped
        N_entering = N_current
        N_erfc,onaxis_drop = compute_esc_electrons(fine_sol, T_e)
        escaped_list.append(N_erfc)

        if i == 0:
            N_escaped = 0
        else:
            N_escaped = escaped_list[i] - escaped_list[i - 1]
        
        N_current = N_current - N_escaped

        #escaped_list.append(N_escaped)
        remaining_list.append(N_current)
        frac_escaped = N_escaped / N_entering
        
        frac_escaped_list.append(frac_escaped)
        drop_list.append(onaxis_drop)
        
        print(f"Ramp {rampfrac:.3f}: frac escaped = {frac_escaped:.3e}, on-axis confinement ('drop') = {onaxis_drop:.3e}")
        print(f"Escaped this step: {N_escaped:.3e}")
        print(f"Remaining after step: {N_current:.3e}")

        if N_current < 1:
            print("Plasma fully escaped — stopping early.")
            break
    return ramp_values, escaped_list, remaining_list, frac_escaped_list, drop_list



ramp_values, escaped_list, remaining_list, frac_escaped_list, drop_list = escape_curve_scan(rampfrac_start, rampfrac_end, data_points=25)

#===== PLOT ESCAPE CURVE ===== #
def plot_escape_curve(ramp_values, escaped_list, remaining_list, frac_escaped_list, drop_list):

    plt.figure(figsize=(7, 5))
    #plt.plot(ramp_values[:len(remaining_list)], remaining_list, '-o', label="Remaining electrons")
    plt.plot(drop_list[:len(escaped_list)], escaped_list, '-o', label="Escaped per step")
    plt.xlabel("Confinement ('drop') in volts")
    plt.ylabel("Electrons")
    plt.yscale("log")
    plt.title("Plasma escape during ramp-down")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===== PLOT FRACTION ESCAPED PER STEP ===== #

    plt.figure(figsize=(7, 5))
    plt.plot(ramp_values[:len(frac_escaped_list)], np.log(frac_escaped_list), '-o', label="Fraction escaped per step")
    plt.xlabel("Ramp fraction (0 = strong confinement, 1 = weak)")
    plt.ylabel("Fraction escaped")
    plt.title("Fraction of plasma escaped at each ramp step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

