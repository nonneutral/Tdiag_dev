from solver_copy import *

def step_4(N_e, T_e, omega_r, rad2, B2, initial_voltages, electrode_borders, Llim, Rlim):
    """
    (2)  human (eventually ML) guesses the rotation rate ω_r and 2D density profile n(r,z) for initial excitations
    (3)  coarse plasma solve (δN/N ~ 10%)
    repeat (2)-(3) until r_p from solver is within 10% of measured r_p

    :param N_e: Number of particles
    :param T_e: Temperature
    :param omega_r: initial omega_r guess
    :param rad2: target radius
    :param B2: axial magnetic field
    :param initial_voltages: initial voltages on electrodes
    :param electrode_borders: electrode border positions
    :param Llim: left boundary
    :param Rlim: right boundary
    :return: final sol with omega_r that achieves target radius within 1% tolerance or None if not reachable
    """
    print("---------omega_r retuning--------------------")

    for _ in range(8):  #COARSE LOOP - range(number) is just number of iterations to try
        if _ == 0:
            initialse_using_pl = True
        else: 
            initialse_using_pl = False
        sol = find_solution(NVal=N_e,T_e=T_e,fE=omega_r/(2*np.pi),mur2=rad2,B=B2,
                            electrodeConfig=(initial_voltages,electrode_borders),
                            left=Llim,right=Rlim,zpoints=40,rpoints=20,rfact=3.0,plotting=True, coarse_sol_divisor=50, InitializeWithPlasmaLength = initialse_using_pl)
        omega_r = sol[6]
        r_mean = sol[5]     #returned rmean
        vfree = sol[4]   #returned free_space_solution
        print(f'potential-to-kT ratio: {np.max(-q_e*vfree)/(kb*T_e):0.2f}')
        if abs(r_mean - rad2) <= 0.01 * rad2:
            print("Desired radius achieved within 1% tolerance.")
            break
        omega_new = retune_omega_iteration(omega_r, r_mean, rad2) #using funciton to retune omega_r and hit traget radius.
        if omega_new is None:
            print("Target radius not reachable with current parameters.")
            break
        omega_r = omega_new
    
    print("--- omega_r retuning complete ---")
    return sol