#%%initialisation 

import numpy as np
from find_solution import find_solution
from step_4 import step_4
from step_5 import step_5
from step_7 import escape_curve_scan, plot_escape_curve
from find_solution import plot_density
from step_8 import linear_model_T_diag
# physical constants
kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin
e0=8.854187817e-12 #farads per meter


# plasma parameters
T_e=174 #plasma temperature in kelvin
N_e=8e6 #number of electrons
rad2=0.0002 #plasma radius in meters
B=1.6 #magnetic field in tesla
freq_guess = 2.0e6  # obsolete
omega_r  = 2*np.pi*freq_guess
q_e=1.60217662e-19 #elementary charge in coulombs
m_e=9.1093837e-31 #electron mass in kilograms

# electrode parameters 0.025,0.025,0.050,0.025,0.025 
electrode_borders=[0.025,0.050,0.100,0.125]
initial_voltages=np.array([0,-50,-10,-50,0]) #excitations initial in volts
final_voltages=np.array([0,-15,-10,-50,0], dtype=float) #excitations final in volts
electrode_voltages=initial_voltages
rw=.017 #radius of inner wall of cylindrical electrodes, in meters
Llim=0.035 #axial window limit, in meters
Rlim=0.110


# tolerances for bessel function expansions
Mmax=1
Nmax=20000

#%% rampfrac
rampfrac=0.0
current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages)*rampfrac
electrodeConfig=(current_voltages,electrode_borders)

#%%Step 4: retune omega_r to hit target radius
sol = step_4(N_e, 
            T_e, 
            omega_r, 
            rad2, 
            B, 
            electrodeConfig, 
            Llim, 
            Rlim, 
            zpoints=40,
            rpoints=20,
            rfact=3.0,
            plotting=True, 
            coarse_sol_divisor=20)

plot_density(sol)

#%% Step 4.5: gradient descent method
"""
from step_5_gdm import gradient_descent_method

step_5_gdm_results = gradient_descent_method(initial_voltages,
                                              final_voltages,
                                              N_e,
                                              T_e,
                                              sol[6],
                                              rad2,
                                              B,
                                              electrodeConfig,
                                              Llim,
                                              Rlim,
                                              zpoints=40,
                                              rpoints=20,
                                              rfact=3.0,
                                              plotting=False,
                                              coarse_sol_divisor=100
                                              )
print(f"Step 4.5 GDM results: rampfrac = {step_5_gdm_results[0]:.6f}, achieved drop = {step_5_gdm_results[1]:.6f} V")

"""

#%%Step 5: Find rampfrac to hit target drop
target_drop = 10 * kb * T_e / q_e  # volts (this is 10 kT/e)

step_5_results = step_5(target_drop, 
                        sol, 
                        initial_voltages, 
                        final_voltages,
                        N_e,
                        T_e,
                        rad2,
                        B,
                        electrodeConfig,
                        Llim,
                        Rlim,
                        zpoints=40,
                        rpoints=20,
                        rfact=3.0,
                        plotting=False,
                        coarse_sol_divisor=50)

electrodeConfig_step_5 = (np.array(initial_voltages) + (np.array(final_voltages) - np.array(initial_voltages)) * step_5_results[0], electrode_borders)
grid, drops = step_5_results[1]

#%%Step 6: Fine solution with retuned omega_r and rampfrac

step_6 = find_solution(N_e,
                        T_e,
                        sol[6]/(2*np.pi),
                        rad2,
                        B,
                        electrodeConfig_step_5,
                        Llim,
                        Rlim,
                        zpoints=80,
                        rpoints=40,
                        rfact=3.0,
                        plotting=True,
                        coarse_sol_divisor=100)
#%%Step 7: Escape curve scan

start_drop = 10 * kb * T_e / q_e  # volts (this is 20 kT/e)
end_drop = 0 # volts


ramp_values, escaped_list, remaining_list, frac_escaped_list, drop_list = escape_curve_scan(start_drop, 
                              end_drop, 
                              data_points = 25, 
                              coarse_scan_result=np.array([grid, drops]), 
                              initial_voltages=initial_voltages, 
                              final_voltages=final_voltages,
                              N_e=N_e,
                              T_e=T_e,
                              rad2=rad2,
                              B2=B,
                              electrode_borders=electrode_borders,
                              Llim=Llim,
                              Rlim=Rlim,
                              omega_r=sol[6])

plot_escape_curve(ramp_values, escaped_list, remaining_list, frac_escaped_list, drop_list)
#%%Step 8:linear fit



T_inferred = linear_model_T_diag(escaped_list, drop_list)


print(f"\nInferred Temperature from Escape Curve: {T_inferred:.2f} K")
print(f"Actual Temperature: {T_e:.2f} K")
print(f"Percentage Error: {abs(T_inferred - T_e) / T_e * 100:.2f}%")