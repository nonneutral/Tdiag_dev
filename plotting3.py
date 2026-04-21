import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from step_1 import iter_all,extract_measured_temp

q_e = 1.602176634e-19 #elementary charge in coulombs
kb = 1.380649e-23 #Boltzmann constant in J/K

T_list = np.arange(50, 401, 10)
#T_list = np.arange(37.5, 326, 12.5)
#T_list = np.append(T_list, np.array([350, 375, 400]))
err_list = []
array_size = []
T_estimate_list = []

for i in range(len(T_list)):
    T = T_list[i]
    filename = iter_all('csv','T_analysis_April_21')[i]
    print(filename)
    df = pd.read_csv(filename, header=None)
    df = df.T

    rampfrac_list = df[0].values
    escaped_list = df[1].values
    vacdrop_list = df[2].values
    drop_list = df[3].values
    l_p_list = df[4].values

    #dy = np.gradient(np.log10(escaped_list), vacdrop_list)
    #mask = (dy < np.average(dy))
    fit,cov = np.polyfit(vacdrop_list, np.log10(escaped_list), deg = 1, cov=True)
    values = np.polyval(fit, vacdrop_list)
    err = np.sqrt(np.diag(cov))[0]
    err_list.append(err)

    linear_regime = int(0.7*len(vacdrop_list))
    lin_vacdrop = vacdrop_list[:linear_regime]
    lin_escaped = escaped_list[:linear_regime]

    fit_lin,cov_lin = np.polyfit(lin_vacdrop, np.log(lin_escaped), deg = 1, cov=True)
    values_lin = np.polyval(fit_lin, lin_vacdrop)
    slope = fit_lin[0]
    T_estimate = -q_e*1.05/(kb*slope)
    T_estimate_list.append(T_estimate)


    array_size.append(len(escaped_list))
    print(f"T={T} K: slope={fit[0]:.3f} +- {err:.3f} K")
    print(f"T_estimate = {T_estimate}")
    if T >300:
        plt.plot(vacdrop_list, np.log10(escaped_list), marker="o", linestyle="-", label=f"T={T} K")
        plt.plot(vacdrop_list, values, linestyle="--")
        #plt.plot(lin_vacdrop,np.log10(lin_escaped),label=f"Linear regime for T={T} K", linestyle="-", color="red")
        #plt.plot(lin_vacdrop, values_lin/np.log(10), linestyle="--", color="orange")
        plt.xlabel("vacuum drop (V)")
        plt.ylabel("number of escaped electrons (log10)")
        plt.title("Escaped electrons vs Vacuum Drop")
        plt.gca().invert_xaxis()
        plt.legend()
plt.show()


plt.plot(T_list, T_estimate_list, marker="o", linestyle="-")
plt.plot(T_list, T_list, linestyle="--", color="red")
plt.xlabel("Actual Temperature (K)")
plt.ylabel("Estimated Temperature (K)")
plt.title("Estimated vs Actual Temperature")
#plt.ylim(0, 120)
plt.show()

plt.plot(T_list, 100*(T_estimate_list-T_list)/T_list, marker="o", linestyle="-")
plt.xlabel("Actual Temperature (K)")
plt.ylabel("Temperature Estimate percentage difference (%)")
plt.title("Temperature Estimate Deviation vs Actual Temperature")
plt.show()




plt.plot(T_list, err_list, marker="o", linestyle="-")
plt.xlabel("Actual Temperature (K)")
plt.ylabel("Error in Slope Estimate")
plt.title("Error in Slope Estimate vs Actual Temperature")
plt.show()
    # %%

