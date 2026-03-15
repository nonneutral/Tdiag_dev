# %%
import numpy as np
import matplotlib.pyplot as plt
from solver2 import *
from step_1 import analyse_experimental_results, iter_all


def full_protocol(electrode_borders, T_current):
    
    # Input 
    N_e=3e5 #number of electrons
    #T_e=1960 #plasma temperature in kelvin
    rad2=0.0008 #plasma radius in meters
    B2=2 #magnetic field in tesla
    #T_current = 200

    # initial guess for rotation frequency. 
    # obsolete anyway since it can now be calculated based on
    # 1. initially plasma length from infinte plasma's potential and electrodes.
    # 2. later by iteration of find_solution omega_r gives rad that equals input rad
    # Left in because find_soltuion requires this variable to not be None.
    # In the future require find_solution to allow None input. 
    freq_guess = 2.0e6  
    omega_r  = 2*np.pi*freq_guess
    # Initial Input Values
    initial_voltages=np.array([0,-63,-50,-130,0]) #in volts
    final_voltages=np.array([0,0,-50,-130,0], dtype=float) #in volts

    #electrode_borders=[0.025,0.050,0.100,0.125] #in meters
    Llim=0.035
    Rlim=0.115
    rw=.017 #radius of inner wall of cylindrical electrodes, in meters
    rampfrac=0.9
    current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages) * rampfrac

    #print(f"Extracted temperature: {measured_temp} K")
    #T_current = measured_temp
    
    # full scan for T_diag vs T_actual (step 4-8), now run as explicit pipeline steps
    print(f"T = {T_current}")

    plasma_config = [float(N_e), float(T_current), float(omega_r), float(rad2), float(B2)]
    electrode_input = [
        np.array(initial_voltages),
        np.array(final_voltages),
        np.array(electrode_borders),
        float(Llim),
        float(Rlim),
        float(rw)
    ]

    #user-chosen scan window (VOLTS)
    start_drop = 10*kb*T_current/q_e
    end_drop   = -1
    d_points = 100
    initial_scan_points = 41

    # Step 1: retune omega_r

    omega_r = protocol_step_1_find_omega_r(plasma_config, electrode_input)
    # plasma_config[2] updated in-place

    # -------------------------
    # Step 2: coarse scan -> (grid_interp, drops_interp)
    # -------------------------
    print("--- Performing coarse scan to map rampfrac to drop ---")
    grid, drops, grid_interp, drops_interp = protocol_step_2_coarse_scan(
        plasma_config, electrode_input, initial_scan_points
    )

    # -------------------------
    # Step 3: find rampfrac correpsonding to target drop of 10 kT/e 
    # -------------------------
    target_drop_eg = 10 * kb * T_current / q_e
    rf_eg, achieved_drop_eg, current_voltages_eg = protocol_step_3_find_rf_for_target_drop(
        plasma_config, electrode_input, grid_interp, drops_interp, target_drop_eg
    )
    print(f"rf for 10 kT/e target: rf={rf_eg:.6f}, achieved_drop={achieved_drop_eg:.6f} V")

    # -------------------------
    # Step 4: fine solution at set target 
    # -------------------------
    print(f"--- Finding fine solution for target drop of {target_drop_eg:.3f} V ---")
    fine_sol = protocol_step_4_find_solution(
        plasma_config, electrode_input, current_voltages_eg,
        zpoints=80, rpoints=40, rfact=3.0,
        plotting=True, coarse_sol_divisor=100
    )
    plot_density(fine_sol)

    # -------------------------
    #Intermediate step to find relevant rampfrac range for escape curve 
    # scan (between start_drop and end_drop)
    # -------------------------
    rampfrac_start, achieved_start_drop, _ = protocol_step_3_find_rf_for_target_drop(
        plasma_config, electrode_input, grid_interp, drops_interp, start_drop
    )
    rampfrac_end, achieved_end_drop, _ = protocol_step_3_find_rf_for_target_drop(
        plasma_config, electrode_input, grid_interp, drops_interp, end_drop
    )
    print(f"scan window: start rf={rampfrac_start:.6f} (achieved {achieved_start_drop:.6f} V) "
            f"-> end rf={rampfrac_end:.6f} (achieved {achieved_end_drop:.6f} V)")

    # -------------------------
    # Step 5: escape curve scan
    # -------------------------
    ramp_values, escaped_list, remaining_list, frac_escaped_list, drop_list, vacdrop_list, history_full_solutions_list = protocol_step_5_escape_curve_scan(
        plasma_config, electrode_input, rampfrac_start, rampfrac_end, d_points
    )

    plot_escape_curve(ramp_values, escaped_list, frac_escaped_list, drop_list, yscale='linear')

    # Save
    np.savetxt(
        f"T{T_current}_N{plasma_config[0]:.2e}_omega_r{plasma_config[2]:.2e}_rad{plasma_config[3]}_B{plasma_config[4]}.csv",
        np.array([ramp_values, escaped_list, frac_escaped_list, drop_list, vacdrop_list]),
        delimiter=","
    )

    # -------------------------
    # Step 6/7: temperature inference (keep your exact calls)
    # -------------------------
    T_actual = T_current

    Tvac, errvac = linear_model_T_diag(
        escaped_list, vacdrop_list,
        "Log(Escaped electrons) vs Confinement with Linear Fit, vacdrop",
        xlabel_str="confinement voltage / V",
        saveplotttitle="Escape_plot_vac",
        crop_factor_input=0.591
    )
    print(f"Actual Temperature: {T_actual:.2f} K")
    print(f"Percentage Error: {abs(Tvac - T_actual) / T_actual * 100:.2f}%")

    Tdrop, errdrop = linear_model_T_diag(
        escaped_list, drop_list,
        "Log(Escaped electrons) vs Confinement with Linear Fit, vacdrop",
        xlabel_str=r"confinement voltage ('drop') / V",
        saveplotttitle="Escape_plot_drop",
        crop_factor_input=0.591
    )
    print(f"Actual Temperature: {T_actual:.2f} K")
    print(f"Percentage Error: {abs(Tdrop - T_actual) / T_actual * 100:.2f}%")

    #

    y_plot = np.array(escaped_list)
    x_plot_vac = np.array(vacdrop_list)
    x_plot_drop = np.array(drop_list)

    plt.figure(figsize=(6,4))
    plt.scatter(x_plot_vac, np.log(y_plot+1))
    #plt.scatter(x_plot_vac, np.log(y_plot))
    plt.xlabel("vacuum drop (V)")
    plt.ylabel("number of escaped electrons")
    plt.title("Escaped electrons vs Vacuum Drop")
    #plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.show()
    #

    plt.plot(x_plot_vac, x_plot_drop, label="vacuum drop vs drop", marker='o', linestyle='-', color='blue', markersize=5, linewidth=1)
    plt.xlabel("vacuum confinement (V)")
    plt.ylabel("drop (V)")
    plt.title("Vacuum confinement vs Drop")
    plt.legend()
    plt.show()
    return escaped_list, vacdrop_list, drop_list


# %%
"""
for d in np.arange(0.075, 0.125, 0.010):
    print(f"Running full protocol with electrode border at {d:.3f} m")
    print(f"Electrode borders: {[0.025, 0.050, 0.050+d, 0.075+d]}")
    escaped_list_now, vacdrop_list_now, drop_list_now = full_protocol(electrode_borders=[0.025, 0.050, 0.050+d, 0.075+d])
    np.savetxt(
        f"full_protocol_scan_d{d:.3f}.csv",
        np.array([escaped_list_now, vacdrop_list_now, drop_list_now]),
        delimiter=","
    )"""

for T in [1600,1700,1800,1900,2000]:
    print(f"Running full protocol with T={T} K")
    electrode_borders = [0.025, 0.050, 0.100, 0.125]
    print(f"Electrode borders: {electrode_borders}")
    escaped_list_now, vacdrop_list_now, drop_list_now = full_protocol(electrode_borders=[0.025, 0.050, 0.100, 0.125], T_current=T)
    np.savetxt(
        f"full_protocol_scan_T{T:.0f}K_trial2.csv",
        np.array([escaped_list_now, vacdrop_list_now, drop_list_now]),
        delimiter=",")
# %%
