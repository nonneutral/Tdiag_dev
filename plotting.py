import os
import re
import numpy as np
import matplotlib.pyplot as plt

d50mm = np.loadtxt("full_protocol_scan_d0.050.csv", delimiter=",")
d55mm = np.loadtxt("full_protocol_scan_d0.055.csv", delimiter=",")
d60mm = np.loadtxt("full_protocol_scan_d0.060.csv", delimiter=",")
d65mm = np.loadtxt("full_protocol_scan_d0.065.csv", delimiter=",")
d70mm = np.loadtxt("full_protocol_scan_d0.070.csv", delimiter=",")
d75mm = np.loadtxt("full_protocol_scan_d0.075.csv", delimiter=",")
d85mm = np.loadtxt("full_protocol_scan_d0.085.csv", delimiter=",")
d95mm = np.loadtxt("full_protocol_scan_d0.095.csv", delimiter=",")
d105mm = np.loadtxt("full_protocol_scan_d0.105.csv", delimiter=",")
d115mm = np.loadtxt("full_protocol_scan_d0.115.csv", delimiter=",")
T100K = np.loadtxt("full_protocol_scan_T100K_trial2.csv", delimiter=",")
T200K = np.loadtxt("full_protocol_scan_T200K_trial2.csv", delimiter=",")
T400K = np.loadtxt("full_protocol_scan_T400K_trial2.csv", delimiter=",")
T600K = np.loadtxt("full_protocol_scan_T600K_trial2.csv", delimiter=",")
T800K = np.loadtxt("full_protocol_scan_T800K_trial2.csv", delimiter=",")
T1000K = np.loadtxt("full_protocol_scan_T1000K_trial2.csv", delimiter=",")
T1500K = np.loadtxt("full_protocol_scan_T1500K_trial2.csv", delimiter=",")
#T2000K = np.loadtxt("full_protocol_scan_T2000_trial2.csv", delimiter=",")


distance_list =[d50mm,
                d55mm,
                d60mm,
                d65mm,
                d70mm,
                d75mm,
                d85mm,
                d95mm,
                d105mm]

distance_list_str =["d50mm",
                "d55mm",
                "d60mm",
                "d65mm",
                "d70mm",
                "d75mm",
                "d85mm",
                "d95mm",
                "d105mm",]
Temperature_list =[T100K,
                T200K,
                T400K,
                T600K,
                T800K,
                T1000K,
                T1500K,
                ]
Temperature_list_str =["T100K",
                "T200K",
                "T400K",
                "T600K",
                "T800K",
                "T1000K",
                "T1500K",
                ]

for i in range(len(distance_list)):
    data = distance_list[i]
    distance = distance_list_str[i]
    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]
    plt.plot(vacdrop_list, np.log10(escaped_list), label=f"{distance}", marker='o', linestyle='-', markersize=5, linewidth=1)
plt.xlabel("vacuum drop (V)")
plt.ylabel("number of escaped electrons")
plt.title("length investigation")
plt.legend()
plt.show()

err_list = []

for j in range(len(Temperature_list)):
    data = Temperature_list[j]
    temperature = Temperature_list_str[j]
    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]
    fit, cov = np.polyfit(vacdrop_list, np.log(escaped_list), deg=1, cov=True)
    polyval = np.polyval(fit, vacdrop_list)
    err_list.append(cov[0,0])
    plt.plot(vacdrop_list, np.log10(escaped_list), label=f"{temperature}", marker='o', linestyle='-', markersize=5, linewidth=1)
    plt.plot(vacdrop_list, polyval/np.log(10), label=f"{temperature}", linestyle='-', markersize=5, linewidth=1)
plt.xlabel("vacuum drop (V)")
plt.ylabel("number of escaped electrons")
#plt.ylim(0, 5)
plt.title("temperature investigation")
plt.legend()
plt.show()

plt.plot([100,200,400,600,800,1000,1500], err_list, marker='o', linestyle='-', markersize=5, linewidth=1)