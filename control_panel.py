'''
import numpy as np
import matplotlib.pyplot as plt
from solver2 import *

# Input 
N_e=3e6 #number of electrons
#T_e=1960 #plasma temperature in kelvin
rad2=0.0008 #plasma radius in meters
B2=2 #magnetic field in tesla
freq_guess = 2.0e6  # initial guess for rotation frequency in Hz
omega_r  = 2*np.pi*freq_guess

# Initial Input Values
initial_voltages=np.array([0,-63,-50,-130,0]) #in volts
final_voltages=np.array([0,0,-50,-130,0], dtype=float) #in volts
electrode_borders=[0.025,0.050,0.100,0.125] #in meters
Llim=0.035
Rlim=0.115
rw=.017 #radius of inner wall of cylindrical electrodes, in meters
rampfrac=0.9
current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages) * rampfrac

T_current = [375.66]

#%%full scan for T_diag vs T_actual (step 4-8)
for T_current in T_current:
    print(f"T = {T_current}")
    
    plasma_config = [float(N_e),float(T_current),float(omega_r),float(rad2),float(B2)]
    electrode_input = [np.array(initial_voltages),np.array(final_voltages),np.array(electrode_borders),float(Llim),float(Rlim),float(rw)]

    
    #plasma_config[1] = T_current

    start_drop = 0.5 # volts (this is 10 kT/e)
    end_drop = 0 # volts
    d_points = 20  # number of data points in escape curve scan
    initial_scan_points = 41
    ramp_values, escaped_list, frac_escaped_list, drop_list, vacdrop_list = evaporative_protocol(plasma_config,electrode_input,start_drop,end_drop,d_points,initial_scan_points)

    np.savetxt(f"T{T_current}_N{plasma_config[0]:.2e}_omega_r{plasma_config[2]:.2e}_rad{plasma_config[3]}_B{plasma_config[4]}.csv",
               np.array([ramp_values, escaped_list, frac_escaped_list, drop_list, vacdrop_list]),delimiter=",")
    
    T_actual = T_current



    Tvac, errvac = linear_model_T_diag(escaped_list, vacdrop_list,
                            "Log(Escaped electrons) vs Confinement with Linear Fit, vacdrop", 
                            xlabel_str="confinement voltage / V",
                            saveplotttitle="Escape_plot_vac",
                            crop_factor_input=0.591)
    print(f"Actual Temperature: {T_actual:.2f} K")
    print(f"Percentage Error: {abs(Tvac - T_actual) / T_actual * 100:.2f}%")


    Tdrop, errdrop = linear_model_T_diag(escaped_list, drop_list,"Log(Escaped electrons) vs Confinement with Linear Fit, vacdrop",
                                xlabel_str=r"confinement voltage ('drop') / V",
                                saveplotttitle="Escape_plot_drop",
                                crop_factor_input=0.591)
    print(f"Actual Temperature: {T_actual:.2f} K")
    print(f"Percentage Error: {abs(Tdrop -T_actual) / T_actual * 100:.2f}%")
'''
# %%
import numpy as np
import matplotlib.pyplot as plt
from solver2 import *
from step_1 import analyse_experimental_results, iter_all

# Input 
N_e=3e6 #number of electrons
#T_e=1960 #plasma temperature in kelvin
rad2=0.0008 #plasma radius in meters
B2=2 #magnetic field in tesla
freq_guess = 2.0e6  # initial guess for rotation frequency in Hz
omega_r  = 2*np.pi*freq_guess

# Initial Input Values
initial_voltages=np.array([0,-63,-50,-130,0]) #in volts
final_voltages=np.array([0,0,-50,-130,0], dtype=float) #in volts
electrode_borders=[0.025,0.050,0.100,0.125] #in meters
Llim=0.035
Rlim=0.115
rw=.017 #radius of inner wall of cylindrical electrodes, in meters
rampfrac=0.9
current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages) * rampfrac



# Experimental results analysis (uncomment to run)
#%%
Recompute_Drops=True #trun to False to skip voltage drop recomputation 
filepath1=iter_all('csv','../')[8] #load data
measured_temp = analyse_experimental_results(filepath1, Recompute_Drops=Recompute_Drops)

print(f"Extracted temperature: {measured_temp} K")
T_current = measured_temp

# %% full scan for T_diag vs T_actual (step 4-8), now run as explicit pipeline steps
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
start_drop = 0.1
end_drop   = -0.5
d_points = 50
initial_scan_points = 41

#%% Step 1: retune omega_r

omega_r = protocol_step_1_find_omega_r(plasma_config, electrode_input)
# plasma_config[2] updated in-place

#%% -------------------------
# Step 2: coarse scan -> (grid_interp, drops_interp)
# -------------------------
print("--- Performing coarse scan to map rampfrac to drop ---")
grid, drops, grid_interp, drops_interp = protocol_step_2_coarse_scan(
    plasma_config, electrode_input, initial_scan_points
)

#%% -------------------------
# Step 3: find rampfrac correpsonding to target drop of 10 kT/e 
# -------------------------
target_drop_eg = 10 * kb * T_current / q_e
rf_eg, achieved_drop_eg, current_voltages_eg = protocol_step_3_find_rf_for_target_drop(
    plasma_config, electrode_input, grid_interp, drops_interp, target_drop_eg
)
print(f"rf for 10 kT/e target: rf={rf_eg:.6f}, achieved_drop={achieved_drop_eg:.6f} V")

#%% -------------------------
# Step 4: fine solution at set target 
# -------------------------
print(f"--- Finding fine solution for target drop of {target_drop_eg:.3f} V ---")
fine_sol = protocol_step_4_find_solution(
    plasma_config, electrode_input, current_voltages_eg,
    zpoints=80, rpoints=40, rfact=3.0,
    plotting=True, coarse_sol_divisor=100
)
plot_density(fine_sol)

#%% -------------------------
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

#%% -------------------------
# Step 5: escape curve scan
# -------------------------
ramp_values, escaped_list, remaining_list, frac_escaped_list, drop_list, vacdrop_list = protocol_step_5_escape_curve_scan(
    plasma_config, electrode_input, rampfrac_start, rampfrac_end, d_points
)

plot_escape_curve(ramp_values, escaped_list, frac_escaped_list, drop_list, yscale='linear')

# Save
np.savetxt(
    f"T{T_current}_N{plasma_config[0]:.2e}_omega_r{plasma_config[2]:.2e}_rad{plasma_config[3]}_B{plasma_config[4]}.csv",
    np.array([ramp_values, escaped_list, frac_escaped_list, drop_list, vacdrop_list]),
    delimiter=","
)

#%% -------------------------
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

# %%

