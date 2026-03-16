import os
import re
import numpy as np
import matplotlib.pyplot as plt

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
fig, axs = plt.subplots(2, 2, figsize=(12,10))

# ---------------- Length investigation ----------------
err_list_d = []

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
axs[0,1].set_ylabel(r"$\sigma \, (V^{-1})$",fontsize=18)


# ---------------- Temperature investigation ----------------
err_list = []

for j in range(len(temperature_list)):
    data = temperature_list[j]
    temperature = Temperature_list_str[j]

    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]

    fit, cov = np.polyfit(vacdrop_list, np.log(escaped_list), deg=1, cov=True)
    polyval = np.polyval(fit, vacdrop_list)

    err_list.append(cov[0,0])
    if (j+1)%2:
        axs[1,0].plot(vacdrop_list, np.log10(escaped_list),
                    label=f"{temperature}",
                    marker='o', linestyle='-', markersize=5, linewidth=1)

        axs[1,0].plot(vacdrop_list, polyval/np.log(10))

axs[1,0].set_xlabel("Confinement (V)",fontsize=18)
axs[1,0].set_ylabel(r"$N_\text{esc}$ T investigation",fontsize=18)
axs[1,0].legend()

#fig.text(0.5, 0.96, "Finite length investigation", ha='center', fontsize=16)
#fig.text(0.5, 0.48, "Temperature investigation", ha='center', fontsize=16)

# ---------------- Error vs temperature ----------------
axs[1,1].plot(np.arange(100,2100,100),
              err_list,
              marker='o',
              linestyle='-',
              markersize=5,
              linewidth=1)

axs[1,1].set_xlabel(r"Temperature (K)",fontsize=18)
axs[1,1].set_ylabel(r"$\sigma \, (V^{-1})$",fontsize=18)


plt.tight_layout(rect=[0,0,1,0.95])
plt.show()