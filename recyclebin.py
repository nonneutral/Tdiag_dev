#%%
"""
# if reachable, bracket around target
idx_sorted = np.argsort(np.abs(drops - target_drop))
best_idx = int(idx_sorted[0])
best_rf  = float(grid[best_idx])
best_drop = float(drops[best_idx])

# try to find a sign-change bracket in the grid (assumes roughly monotonic)
bracket = None
for i in range(len(grid) - 1):
    if (drops[i] - target_drop) * (drops[i+1] - target_drop) <= 0:
        bracket = (float(grid[i]), float(grid[i+1]))
        break

if bracket is None:
    # Not bracketed -> just take closest scan point
    rampfrac_star = best_rf
    achieved_drop = best_drop
else:

    lo, hi = bracket
    # bisection refine (no extra imports)
    for _ in range(15):
        mid = 0.5 * (lo + hi)
        dmid, _, _ = drop_for_rampfrac(mid, omega_r)
        if np.abs(dmid - target_drop) <= eps50:
            lo = hi = mid
            break
        # choose side that contains the target
        dlo, _, _ = drop_for_rampfrac(lo, omega_r)
        if (dlo - target_drop) * (dmid - target_drop) <= 0:
            hi = mid
        else:
            lo = mid
    rampfrac_star = 0.5 * (lo + hi)
    achieved_drop, _, _ = drop_for_rampfrac(rampfrac_star, omega_r)



    achieved_drop, _, _ = drop_for_rampfrac(rampfrac_star, omega_r)
rampfrac_history.append(rampfrac_star)
"""


#def compute_kept_electrons(fine_sol, T_e): #, N_now, lastescapeE
#    ngrid = fine_sol[0]
#    full_scc_solution = fine_sol[3] #Full Space-Charge-Corrected (SCC) Solution, i.e. voltageGuess
#    #position_map_z = fine_sol[1]
#    volume_elements = fine_sol[7]
#    N_cell = ngrid * volume_elements
#    N_cell = N_cell*N_now/np.sum(N_cell) #normalize grid of number of e- per cell so it sums to N_now
#    escape_sum = np.zeros(len(N_cell)) #len apparently returns the number of r values
#    escapeE = np.zeros(len(N_cell))
#    print(f"sum(ngrid*volume_elements) = {np.sum(N_cell):.3f} ---")
#    for r, x in enumerate(N_cell): 
#        oneD_solution = full_scc_solution[r, :] #Solution across z-axis per radial point, r
#        axial_well_idx = np.argmax(oneD_solution)
#        barrier_idx = np.argmin(oneD_solution)
#        escapeE[r] = q_e * abs(oneD_solution[axial_well_idx] - oneD_solution[barrier_idx])
#        E_int = erf(np.sqrt(lastescapeE[r] / (kb * T_e))) - erf(np.sqrt(escapeE[r] / (kb * T_e)))
#        escape_sum[r] = E_int * np.sum(N_cell[r, :]) #these are the ones that leave the well from that r
#    return np.sum(escape_sum),escapeE
#%%    



# 1) Index where free-space potential peaks (on-axis)
#peak_idx = int(np.argmax(vfree_on))
#peak_z = float(z_axis[peak_idx])
#vfree_peak = float(vfree_on[peak_idx])

# 2) Space-charge-corrected potential at that (on-axis) index
#Peak_Space_Charge_Pot = float(voltageGuess[0, peak_idx])

# 3) Space-charge-corrected potential at electrode border (on-axis)
#idx_elec   = int(np.argmin(np.abs(z_axis - electrode_borders[1]))) 
#z_nearest  = float(z_axis[idx_elec])
#v_sc_near  = float(v_sc_on[idx_elec])

#!!! barrier voltage is not simply the voltage at the electrode border
#instead use the same thing as for the escape loop <--- addressed within the drop_for_rampfrac function below


#Adjust initial_voltages so that potential drop ≈ 10 kT/e
#!!! the voltage ramp applied to the electrodes is given: the data was taken using a known voltage ramp
#therefore it is not meaningful to change those
#instead, adjust the ramp frac
#if current_drop/thermal_voltage < 10 then ramp frac is too great
#may need to iterate (solve again)

# --- STEP 5 RED0: Find rampfrac* such that (Peak_Space_Charge_Pot - v_sc_near) ~= 10*kB*T/q_e --- #


# --- Step 6 ---
#rf_star = find_rf_for_target_drop(target_drop_eg,interp_points=10000)[0]
#achieved_drop = find_rf_for_target_drop(target_drop_eg,interp_points=10000)[1]


#print(f"\n[Rampfrac Tuning] target drop = {target_drop_eg:.3f} V")
#print(f"[Rampfrac Tuning] rampfrac start = {rf_star:.4f}, achieved drop = {achieved_drop:.3f} V")
