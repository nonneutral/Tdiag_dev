
import numpy as np
import matplotlib.pyplot as plt

q_e=1.60217662e-19 #elementary charge in coulombs
kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin



def linear_model_T_diag(escaped_list, drop_list):
    """
    Linear fit model for temperature estimation from escape curve data.
    
    :param escape_list: List of escaped electrons at each drop
    :param drop_list: List of confinement ('drop') values in volts
    :return: Estimated temperature in Kelvin
    """
    #Linear temperature estimate from escape curve using
    #equation 8 from Eggleston 1992 paper
    dnep = np.log(escaped_list[-1]/escaped_list[0]) 
    dV = drop_list[-1]-drop_list[0]
    slope = dnep/dV
    T_estimate = -q_e*1.05/(kb*slope)
    print(f"Estimated temperature from escape curve: {T_estimate:.2f} K")
    #Quick Check: Linear fit - but with fitting
    curvefit = np.polyfit(drop_list, np.log(escaped_list), 1)
    slope = curvefit[0]
    T_estimate2 = -q_e*1.05/(kb*slope)
    print(f"Estimated temperature from escape curve (polyfit): {T_estimate2} K")

    plt.figure(figsize=(7, 5))
    plt.plot(drop_list, np.log(escaped_list), 'o', label="Data points")
    plt.plot(drop_list, np.polyval(curvefit, drop_list), '-', label="Linear fit")
    plt.xlabel("Confinement ('drop') in volts")
    plt.ylabel("Log(Escaped electrons)")
    plt.title("Log(Escaped electrons) vs Confinement with Linear Fit")
    plt.grid(True)
    return T_estimate2


