import numpy as np

from solver_copy import find_solution

# physical constants
kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin
e0=8.854187817e-12 #farads per meter

# particle properties
m_e=9.1093837e-31 #electron mass in kilograms
q_e=1.60217662e-19 #elementary charge in coulombs


# plasma parameters
N_e=8e6 #number of electrons
T_e=1960 #plasma temperature in kelvin
freq_guess = 2.0e6  # obsolete
omega_r  = 2*np.pi*freq_guess
rad2=0.0002 #plasma radius in meters
B=1.6 #magnetic field in tesla

# electrode parameters
electrode_borders=np.array([0.025,0.050,0.100,0.125])
initial_voltages=np.array([0,-50,-10,-50,0]) #excitations initial in volts
final_voltages=np.array([0,-15,-10,-50,0], dtype=float) #excitations final in volts
electrode_voltages=initial_voltages 
rw=.017 #radius of inner wall of cylindrical electrodes, in meters
Llim=0.035 #axial window limit, in meters
Rlim=0.110

"""np.array([N_e,
    T_e,
    freq_guess,
    rad2,
    B,
    np.array([initial_voltages,electrode_borders]),
    Llim,
    Rlim,
    40,
    20,
    3.0,
    True,
    20,
    False,
    'raise',
    ''])"""


# tolerances for bessel function expansions
Mmax=1
Nmax=20000


rampfrac=0.9
current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages)*rampfrac


#Step 3: Coarse solution with guessed frequency


sol = find_solution(NVal=N_e,
                    T_e=T_e,
                    fE=freq_guess,
                    mur2=rad2,
                    B=B,
                    electrodeConfig=np.array([initial_voltages,electrode_borders]),
                    left=Llim,
                    right=Rlim,
                    zpoints=40,
                    rpoints=20,
                    rfact=3.0,
                    plotting=True,
                    coarse_sol_divisor=20,
                    InitializeWithPlasmaLength = False,
                    fail_action='raise',
                    debug_tag='')
    

#Step 7: scan parameters
target_drop = 10 * kb * T_e / q_e  # volts (this is 10 kT/e)
end_drop = 0 # volts

