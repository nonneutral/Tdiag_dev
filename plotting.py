#%%
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import lin_fit_optimiser as lfo
import pandas as pd

d50mm = np.loadtxt("./useful_data/full_protocol_scan_d0.050.csv", delimiter=",")
d55mm = np.loadtxt("./useful_data/full_protocol_scan_d0.055.csv", delimiter=",")
d60mm = np.loadtxt("./useful_data/full_protocol_scan_d0.060.csv", delimiter=",")
d65mm = np.loadtxt("./useful_data/full_protocol_scan_d0.065.csv", delimiter=",")
d70mm = np.loadtxt("./useful_data/full_protocol_scan_d0.070.csv", delimiter=",")
d75mm = np.loadtxt("./useful_data/full_protocol_scan_d0.075.csv", delimiter=",")
d85mm = np.loadtxt("./useful_data/full_protocol_scan_d0.085.csv", delimiter=",")
d95mm = np.loadtxt("./useful_data/full_protocol_scan_d0.095.csv", delimiter=",")
d105mm = np.loadtxt("./useful_data/full_protocol_scan_d0.105.csv", delimiter=",")
d115mm = np.loadtxt("./useful_data/full_protocol_scan_d0.115.csv", delimiter=",")
T100K = np.loadtxt("./useful_data/full_protocol_scan_T100K_trial2.csv", delimiter=",")
T200K = np.loadtxt("./useful_data/full_protocol_scan_T200K_trial2.csv", delimiter=",")
T300K = np.loadtxt("./useful_data/full_protocol_scan_T300K_trial2.csv", delimiter=",")
T400K = np.loadtxt("./useful_data/full_protocol_scan_T400K_trial2.csv", delimiter=",")
T500K = np.loadtxt("./useful_data/full_protocol_scan_T500K_trial2.csv", delimiter=",")
T600K = np.loadtxt("./useful_data/full_protocol_scan_T600K_trial2.csv", delimiter=",")
T700K = np.loadtxt("./useful_data/full_protocol_scan_T700K_trial2.csv", delimiter=",")
T800K = np.loadtxt("./useful_data/full_protocol_scan_T800K_trial2.csv", delimiter=",")
T900K = np.loadtxt("./useful_data/full_protocol_scan_T900K_trial2.csv", delimiter=",")
T1000K = np.loadtxt("./useful_data/full_protocol_scan_T1000K_trial2.csv", delimiter=",")
T1100K = np.loadtxt("./useful_data/full_protocol_scan_T1100K_trial2.csv", delimiter=",")
T1200K = np.loadtxt("./useful_data/full_protocol_scan_T1200K_trial2.csv", delimiter=",")
T1300K = np.loadtxt("./useful_data/full_protocol_scan_T1300K_trial2.csv", delimiter=",")
T1400K = np.loadtxt("./useful_data/full_protocol_scan_T1400K_trial2.csv", delimiter=",")
T1500K = np.loadtxt("./useful_data/full_protocol_scan_T1500K_trial2.csv", delimiter=",")
T1600K = np.loadtxt("./useful_data/full_protocol_scan_T1600K_trial2.csv", delimiter=",")
T1700K = np.loadtxt("./useful_data/full_protocol_scan_T1700K_trial2.csv", delimiter=",")
T1800K = np.loadtxt("./useful_data/full_protocol_scan_T1800K_trial2.csv", delimiter=",")
T1900K = np.loadtxt("./useful_data/full_protocol_scan_T1900K_trial2.csv", delimiter=",")
T2000K = np.loadtxt("./useful_data/full_protocol_scan_T2000K_trial2.csv", delimiter=",")

temperature_list = [T100K, T200K, T300K, T400K, T500K, T600K, T700K, T800K, T900K, T1000K, T1100K, T1200K, T1300K, T1400K, T1500K, T1600K, T1700K, T1800K, T1900K, T2000K]
distance_list =[d50mm,
                d55mm,
                d60mm,
                d65mm,
                d70mm,
                d75mm,
                d85mm,
                d95mm,
                d105mm]

distance_list_str =["50 mm",
                "55 mm",
                "60 mm",
                "65 mm",
                "70 mm",
                "75 mm",
                "85 mm",
                "95 mm",
                "105 mm",]

Temperature_list_str =["T 100 K",
                "T 200 K",
                "T 300 K",
                "T 400 K",
                "T 500 K",
                "T 600 K",
                "T 700 K",
                "T 800 K",
                "T 900 K",
                "T 1000 K",
                "T 1100 K",
                "T 1200 K",
                "T 1300 K",
                "T 1400 K",
                "T 1500 K",
                "T 1600 K",
                "T 1700 K",
                "T 1800 K",
                "T 1900 K",
                "T 2000 K"
                ]
#fig2,axs2 = plt.figure()

# ---------------- Length investigation ----------------
"""
for i in range(len(distance_list)):
    data = distance_list[i]
    distance = distance_list_str[i]

    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]

    fit, cov = np.polyfit(vacdrop_list, np.log(escaped_list), deg=1, cov=True)
    polyval = np.polyval(fit, vacdrop_list)

    err_list_d.append(cov[0,0])
    if (i+1)%2:
        axs[0,0].plot(vacdrop_list, np.log10(escaped_list),
                    label=f"{distance}",
                    marker='o', linestyle='-', markersize=5, linewidth=1)

        axs[0,0].plot(vacdrop_list, polyval/np.log(10))

axs[0,0].set_xlabel("Confinement (V)",fontsize=18)
axs[0,0].set_ylabel(r"$N_\text{esc}$ L investigation ",fontsize=18)
axs[0,0].legend()


# ---------------- Error vs length ----------------
axs[0,1].plot([50,55,60,65,70,75,85,95,105],
              err_list_d,
              marker="o",
              linestyle="-")

axs[0,1].set_xlabel("Length between inner electrodes (mm)",fontsize=18)
axs[0,1].set_yscale("log")
axs[0,1].set_ylabel(r"Square-fit error: $\sigma \, (V^{-1})$",fontsize=18)


# ---------------- Temperature investigation ----------------
"""

q_e=1.60217662e-19 #electron charge in coulombs
kb = kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin

#fig1, axs1 = plt.subplots(1, 2, figsize=(20,10))
fig2, axs2 = plt.subplots(1, 2, figsize=(20,10))


err_list = []
T_fit_list = []

for j in range(len(temperature_list)):
    data = temperature_list[j]
    temperature = Temperature_list_str[j]

    crop_factor = 1
    
    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]
    xaxis = vacdrop_list

    fit, cov = np.polyfit(xaxis, np.log(escaped_list), deg=1, cov=True)
    polyval = np.polyval(fit, xaxis)

    T_estimate = -q_e*1.05/(kb*fit[0])
    T_fit_list.append(T_estimate)

    err_list.append(cov[0,0])
    axs2[0].plot(xaxis, np.log10(escaped_list),
                label=f"{temperature}",
                marker='o', linestyle='-', markersize=5, linewidth=1)

    axs2[0].plot(xaxis, polyval/np.log(10))

axs2[0].set_xlabel("Confinement (V)",fontsize=18)
axs2[0].set_ylabel(r"$N_\text{esc}$ T investigation",fontsize=18)
axs2[0].legend()

#fig.text(0.5, 0.96, "Finite length investigation", ha='center', fontsize=16)
#fig.text(0.5, 0.48, "Temperature investigation", ha='center', fontsize=16)

# ---------------- Error vs temperature ----------------
axs2[1].plot(np.arange(100,2100,100),
              err_list,
              marker='o',
              linestyle='-',
              markersize=5,
              linewidth=1)

axs2[1].set_xlabel(r"Temperature (K)",fontsize=18)
axs2[1].set_yscale("log")
axs2[1].set_ylabel(r"Square-fit error: $\sigma \, (V^{-1})$",fontsize=18)
plt.show()

plt.plot(np.arange(100,2100,100),100*np.abs(np.arange(100,2100,100)-T_fit_list)/np.arange(100,2100,100), marker="o", linestyle="-",color="b")
#plt.plot(np.arange(100,2100,100),np.arange(100,2100,100))
plt.title("Percentage deviation of linear fit T from input T", fontsize=20)
plt.xlabel("Input Temperature (K)", fontsize=20)
plt.ylabel("Deviation of linear fit T from input T", fontsize=20)
plt.show()


fig, ax = plt.subplots(figsize=(10,6))  # create one figure and axis

for k in np.arange(0, len(temperature_list), step=3):
    data = temperature_list[k]
    temperature = Temperature_list_str[k]

    crop_factor = 1
    
    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]

    # Plot all datasets on the same axis
    ax.plot(vacdrop_list, drop_list,
            marker="o", linestyle="-",
            label=f"T = {temperature} K")
ax.plot(vacdrop_list, vacdrop_list, linestyle="--", color="black", label="y=x reference")  # Add a reference line y=x
# Labels, legend, title
ax.set_xlabel("Vacuum drop (V)", fontsize=18)
ax.set_ylabel("Drop (V)", fontsize=18)
ax.legend()
ax.set_title("Drop vs Vacuum Drop", fontsize=20)
plt.tight_layout()
plt.show()

"""
err_list_d = []
l_p_s = []
crop_factor_list = []



d_range=np.arange(0.040,0.105,0.005)
for d in d_range:
    data = np.loadtxt(f"useful_data_2/trial6_full_protocol_scan_d{d:.3f}.csv",delimiter=",")
    data_label = str(f"d = {round(d*1000)} mm")

    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]
    l_p_list = data[3,:]

    crop_factor,_,_ = lfo.find_crop_factor(escaped_list, vacdrop_list, drop_list)
    print(f"Crop factor for {data_label}: {crop_factor}")
    crop_factor_list.append(crop_factor)
    fit, cov = np.polyfit(vacdrop_list, np.log(escaped_list), deg=1, cov=True)
    polyval = np.polyval(fit, vacdrop_list)

    err_list_d.append(cov[0,0])

    plasma_length = np.average(l_p_list[0])
    #plt.plot(l_p_list)
    l_p_s.append(plasma_length)
    
    if int(round(d * 1000)) % 10 == 0:
        axs[0].plot(vacdrop_list, np.log10(escaped_list),
                    label=data_label,
                    marker='o', linestyle='-', markersize=7, linewidth=1)
        axs[0].plot(vacdrop_list, polyval/np.log(10),color="black",linewidth = 3)

axs[0].set_xlabel("Confinement (V)",fontsize=25)
axs[0].set_ylabel(r"$N_\text{esc}$ L investigation",fontsize=25)
axs[0].tick_params(axis='both', labelsize=20)
axs[0].legend(fontsize=25)
axs[0].grid(True)

#fig.text(0.5, 0.96, "Finite length investigation", ha='center', fontsize=16)
#fig.text(0.5, 0.48, "Temperature investigation", ha='center', fontsize=16)

# ---------------- Error vs temperature ----------------
axs[1].plot(np.array(d_range)*1000,
              err_list_d,
              marker='o',
              linestyle='-',
              markersize=7,
              linewidth=3,color="b")
axs[1].grid(True)
axs[1].set_xlabel(r"electrodes distance (mm)",fontsize=25)
axs[1].set_yscale("log")
axs[1].set_ylabel(r"Square-fit error: $\sigma \, (V^{-1})$",fontsize=25)
axs[1].tick_params(axis='both', labelsize=20)

plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

plt.scatter(d_range, l_p_s)

print(crop_factor_list)"""

#%%

import pandas as pd
from scipy.interpolate import interp1d
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import lin_fit_optimiser as lfo


def iter_all(substring, path):
    return list(
        os.path.join(root, entry)
        for root, dirs, files in os.walk(path)
        for entry in dirs + files
        if substring in entry
    )
def u8Correction(filename):

    try:
        df = pd.read_csv(filename, header=None)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    print(f"file name: {filename}")
    sipm_data = df[0].values  #sipm (~escape rate)
    u8_data = df[1].values    #u8 excitations
    

    sipm_data = sipm_data
    u8_data = u8_data
    
    print(np.shape(sipm_data))
    print(np.shape(u8_data))

    order = np.argsort(u8_data)

    sipm_ordered = sipm_data[order]
    u8_ordered = u8_data[order]
    t_data = np.arange(0,len(u8_data))

    u8vsT_fit = np.polyfit(t_data,u8_ordered,4)
    #print(u8vsT_fit)
    u8_ordered_fit = np.polyval(u8vsT_fit,t_data)
    u8_corrected = u8_ordered_fit * 15

    #plt.plot(t_data,u8_ordered)
    #plt.plot(t_data,u8_ordered_fit, color="g",ms=0.4)
    #plt.plot(abs(u8_ordered_fit-u8_ordered))
    #plt.show()
    #plt.scatter(t_data,u8_ordered*15, label="original", marker=".", color="orange", s=2)
    #plt.scatter(t_data,u8_corrected, label="corrected", marker=".", color="blue", s=2)
    #plt.legend()
    #plt.title("u8 correction")
    #plt.xlabel("time (arb)")
    #plt.ylabel("u8 (arb)")
    #plt.show()

    return u8_corrected, sipm_ordered
#%%

def find_offset(file_number):
    solver_vs_experimental_fit = []

    filepath1=iter_all('csv','../')[file_number] #load data
    #df = pd.read_csv(filepath1, header=None)

    #sipm_data = np.abs(df[0].values)  #sipm (~escape rate)
    #u8_data = df[1].values*15   #u8 excitations

    u8_data_init, sipm_data_init = u8Correction(filepath1)
    sipm_data_init = np.abs(sipm_data_init)
    plt.scatter(u8_data_init, (sipm_data_init), label="original", marker="o", color="red", s=4)
    plt.xlabel("u8 (V)")
    plt.ylabel("sipm (a.u.)")
    plt.title("Original u8 vs sipm")
    plt.legend()
    plt.show()
    x_range = np.arange(150,1000,50)
    for x in x_range:

        offset=x #offset to trim data for better fit - adjust as needed based on data length and quality
        
        u8_data = u8_data_init
        sipm_data = sipm_data_init
        sipm_data = sipm_data[offset:len(sipm_data)]
        u8_data = u8_data[0:-offset]


        order = np.argsort(u8_data)
        u8_data = u8_data[order]
        sipm_data = sipm_data[order]

        u8_solver, sipm_solver,_,_,_ = np.loadtxt("useful_data/T200_N3.00e+05_omega_r8.02e+04_rad0.0008_B2.0.csv",delimiter=",")
        u8_solver = -63 * (1-u8_solver)
        sipm_solver = sipm_solver+1

        u8_solver = u8_solver[len(u8_solver)//2:]
        sipm_solver = sipm_solver[len(sipm_solver)//2:]    


        mask = u8_data > u8_solver[0]  # Only consider data points where u8_data is greater than the first point of u8_solver
        mask = mask & (u8_data < u8_solver[-1])  # Also ensure u8_data is less than the last point of u8_solver


        u8_data = u8_data[mask]
        sipm_data = sipm_data[mask]

        f = interp1d(u8_solver, sipm_solver, kind='linear', fill_value="interpolate")
        sipm_interp = f(u8_data)



        diff = np.abs(np.log10(sipm_interp)-np.log10(1000*sipm_data))
        print(f"Offset: {offset}, Sum of differences: {sum(diff)}")
        solver_vs_experimental_fit.append(sum(diff))
        
        #plt.plot(u8_solver, np.log10(sipm_solver), marker="o", color="blue", label=f"offset = {offset}")
        plt.plot(u8_data, np.log10(sipm_interp), linestyle="-", color="blue", label=f"offset = {offset}")
        plt.plot(u8_data, np.log10(1000*sipm_data))
        
        plt.plot(u8_data, diff, linestyle="-", color="darkred", label=f"offset = {offset}")
        plt.legend()
        plt.xlabel("u8 (V)")
        plt.ylabel("sipm (a.u.)")
        plt.title("Solver vs Experimental Data Fit")
        plt.show()
    plt.plot(x_range, solver_vs_experimental_fit, marker="o", linestyle="-", color="blue")
    plt.title("Fit of solver to experimental data vs offset", fontsize=20)
    plt.yscale("log")

    min_offset = x_range[np.argmin(solver_vs_experimental_fit)]
    return min_offset

#min_offset = find_offset(30)


#%%

offset_list = []

for file_number in np.arange(1,502,50):
    filepath1=iter_all('csv','Dec13')[file_number] #load data
    #df = pd.read_csv(filepath1, header=None)
    #u8_data = df[1].values*15   #u8 excitations
    #sipm_data = df[0].values  #sipm (~escape rate)
    offset = find_offset(file_number)
    offset_list.append(offset)
    u8_data, sipm_data = u8Correction(filepath1)
    plt.scatter(u8_data,sipm_data, label = file_number, s = 4)
plt.legend()
plt.xlabel("u8 (V)")
plt.ylabel("sipm (a.u.)")
plt.title("u8 vs sipm for all files")
plt.xlim(-54,-52)
plt.show()
    #list_offsets.append(min_offset)
    #print(f"Best offset for file {file_number}: {min_offset}")

# %%
