from solver_copy import *
from step_4 import step_4
from step_5 import step_5

def find_fine_solution(NVal,T_e,fE,mur2,B,target_drop,left,right,zpoints,
                  rpoints,rfact=3.0,plotting=True, coarse_sol_divisor=100, grid=None, drops=None):
    """
    Plasma solution with retuned omega_r based on target radius (rad2) 
    and rampfrac based on target potential drop.
    """
    print("--- STEP 4 ---")
    sol_step4 = step_4(NVal, T_e, fE, mur2, B, initial_voltages, electrode_borders, left, right)[6]
    omega_r = sol_step4
    print("--- STEP 5 ---")
    rf_star = step_5(target_drop=target_drop,sol=sol_step4)[0]
    
    current_voltages = np.array(initial_voltages) + (np.array(final_voltages) - np.array(initial_voltages)) * rf_star
    
    print("--- Finding fine solution---")
    fine_sol = find_solution(
        NVal, T_e, fE=omega_r/(2*np.pi), mur2=mur2, B=B, 
        electrodeConfig=(current_voltages, electrode_borders),
        left=left, right=right,
        zpoints=zpoints, rpoints=rpoints, rfact=rfact,
        plotting=plotting, coarse_sol_divisor=coarse_sol_divisor,
        InitializeWithPlasmaLength=False,
        fail_action='raise', debug_tag='fine_solution'
    )
    print("---Fine solution found---")
    return fine_sol