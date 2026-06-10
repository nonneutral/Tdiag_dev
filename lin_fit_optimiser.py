import numpy as np
import matplotlib.pyplot as plt
from solver2 import *



def linear_model_T_diag(escaped_list, drop_list, title, xlabel_str, saveplotttitle, crop_factor_input = 0.5, plotting = True):
    """
    Linear fit model for temperature estimation from escape curve data.
    
    :param escape_list: List of escaped electrons at each drop
    :param drop_list: List of confinement ('drop') values in volts
    :return: Estimated temperature in Kelvin
    """
    crop_factor=crop_factor_input
    escape_sublist=escaped_list[0:int(len(escaped_list)*crop_factor)]
    drop_sublist=drop_list[0:int(len(drop_list)*crop_factor)]
    #Linear temperature estimate from escape curve using
    #equation 8 from Eggleston 1992 paper
    dnep = np.log(escape_sublist[-1]/escape_sublist[0]) 
    dV = drop_sublist[-1]-drop_sublist[0]
    slope = dnep/dV
    T_estimate = -q_e*1.05/(kb*slope)
    print(f"Estimated temperature from escape curve (last-first): {T_estimate:.2f} K")
    #Quick Check: Linear fit - but with fitting
    curvefit,cov = np.polyfit(drop_sublist, np.log(escape_sublist), 1, cov=True)
    slope = curvefit[0]
    T_estimate2 = -q_e*1.05/(kb*slope)
    print(f"Estimated temperature from escape curve (polyfit): {T_estimate2} K")
    print(f"cov: {cov}")
    errors = np.sqrt(np.diag(cov))
    print(f"err of slope: {errors[0]}")

    if plotting == True:
        plt.figure(figsize=(7, 5))
        plt.gca().invert_xaxis()
        plt.plot(drop_list, np.log10(escaped_list), '.', label="Solver data", ms=8.0,color="#ff1493")
        plt.plot(drop_list, np.polyval(curvefit/np.log(10), drop_list), '-', label="Linear fit", color="#1418E2",ms=9.0)
        plt.xlabel(xlabel_str, fontsize=18)
        plt.ylabel(r"$\log(N_{\text{esc}}) ~ \text{/} ~ \text{A.U.}$", fontsize=18)    
        plt.legend(fontsize=18)
        plt.grid(True, "both")
        plt.savefig(f'Escape_plot {saveplotttitle}.png', transparent=True)
        plt.show()
    return T_estimate2,errors[0]


def find_crop_factor(escaped_list, vacdrop_list, drop_list):
    errvac_list = []
    errdrop_list = []
    cf_list = np.linspace(0.5,1,1000)
    for cf in cf_list:
        Tvac, errvac = linear_model_T_diag(escaped_list, vacdrop_list,
                           "Log(Escaped electrons) vs Confinement with Linear Fit, vacdrop", 
                           xlabel_str="Confinement / V",
                           saveplotttitle="Escape_plot_vac",
                           crop_factor_input=cf,
                           plotting=False)

        Tdrop, errdrop = linear_model_T_diag(escaped_list, drop_list,"Log(Escaped electrons) vs Confinement with Linear Fit, vacdrop",
                                xlabel_str="Confinement ('drop') / V",
                                saveplotttitle="Escape_plot_drop",
                                crop_factor_input=cf,
                                plotting=False)
        #
        # 
        errvac_list.append(errvac)
        errdrop_list.append(errdrop)
    return cf_list,errvac_list,errdrop_list


min_cf_list = []
min_cf_drop_list = []

for d in np.arange(0.040,0.105,0.005):
    data = np.loadtxt(f"useful_data_2/trial6_full_protocol_scan_d{d:.3f}.csv",delimiter=",")
    data_label = str(f"d = {round(d*1000)} mm")

    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]
    l_p_list = data[3,:]



    cf_list,errvac_list,errdrop_list = find_crop_factor(escaped_list, vacdrop_list, drop_list)
    min_cf=cf_list[np.argmin(errvac_list)]
    min_cf_drop=cf_list[np.argmin(errdrop_list)]
    min_cf_list.append(min_cf)
    min_cf_drop_list.append(min_cf_drop)

    plt.plot(cf_list,errvac_list,label=f"d = {d:.3f}")
    #plt.plot(cf_list,errdrop_list,label="drop_fit")
    plt.legend()
plt.show()

for d in np.arange(0.040,0.105,0.005):
    data = np.loadtxt(f"useful_data_2/trial6_full_protocol_scan_d{d:.3f}.csv",delimiter=",")
    data_label = str(f"d = {round(d*1000)} mm")

    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]
    l_p_list = data[3,:]



    plt.plot(np.linspace(0, 100, len(escaped_list)), np.log10(escaped_list), label=f"d = {d:.3f}")
    #plt.plot(cf_list,errdrop_list,label="drop_fit")
    plt.title("Escaped electrons vs Vacuum Drop")
    #plt.gca().invert_xaxis()
    plt.xlabel("Vacuum drop (V)")
    plt.ylabel("Number of escaped electrons")
    plt.legend()
plt.show()



print(f"cf for best lin fit {min_cf}")
print(f"cf for best lin fit (drop) {min_cf_drop}")
print(f"idx: {np.argmin(errvac_list)}")

print(f"min cf list: {min_cf_list}")
plt.plot(np.arange(0.040,0.105,0.005),min_cf_list)