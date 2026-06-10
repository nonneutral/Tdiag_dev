#%%
#This file is for plotting the results of the finite length and temperature investigations. It is not used for any data processing, just for visualisation.

import os
import re
import numpy as np
import matplotlib.pyplot as plt
#import lin_fit_optimiser as lfo
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
"""""
# ---------------- Temperature investigation ----------------


q_e=1.60217662e-19 #electron charge in coulombs
kb = kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin

#fig1, axs1 = plt.subplots(1, 2, figsize=(20,10))
fig2, axs2 = plt.subplots(1, 2, figsize=(20,10))


err_list = []
T_fit_list = []

for j in range(len(temperature_list)):
    data = temperature_list[j]
    temperature = Temperature_list_str[j]

    crop_factor = 0.6
    
    escaped_list = data[0,:]
    vacdrop_list = data[1,:]
    drop_list = data[2,:]
    xaxis = vacdrop_list

    linear = int(crop_factor*len(escaped_list))
    lin_esc = escaped_list[:linear]
    lin_vac = vacdrop_list[:linear]
    fit, cov = np.polyfit(lin_vac, np.log(lin_esc), deg=1, cov=True)
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

plt.plot(np.arange(100,2100,100),T_fit_list-np.arange(100,2100,100), marker="o", linestyle="-",color="b")
#plt.plot(np.arange(100,2100,100),np.arange(100,2100,100))
plt.title("linear fit T vs input T", fontsize=20)
plt.xlabel("Input Temperature (K)", fontsize=20)
plt.ylabel("Deviation of linear fit T from input T", fontsize=20)
plt.show()
""

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
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Storage ----------------
err_list_d = []
l_p_s = []
crop_factor_list = []

# ---------------- Figure setup ----------------
fig, axs = plt.subplots(2, 1, figsize=(8.27, 9))  # A4 vertical

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
})

# ---------------- Parameter sweep ----------------
d_range = np.arange(0.040, 0.105, 0.005)

linestyles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "d", "v"]
sum_escape_for_T = []
where_ncyl_list = []
V_ncyl_list = []
last_vacdrop_list = []
T_list = []

# ---------------- Main loop ----------------
for i, d in enumerate(d_range):

    data = np.loadtxt(
        f"useful_data_2/trial6_full_protocol_scan_d{d:.3f}.csv",
        delimiter=","
    )

    data_label = f"d = {int(round(d*1000))} mm"

    escaped_cumulative = data[0, :]
    vacdrop_list = data[1, :]
    l_p_list = data[3, :]


    sum_escape_for_T.append(escaped_cumulative[-1])

    escaped_list = np.diff(escaped_cumulative, prepend=escaped_cumulative[0])
    escaped_list = escaped_list[1:]  # Remove any non-positive values
    vacdrop_list = vacdrop_list[1:]
    last_vacdrop = vacdrop_list[-1]

    last_vacdrop_list.append(last_vacdrop)


    where_ncyl = np.argmin(np.abs(escaped_cumulative - 0.05*32906))  # Find index where escaped_cumulative is closest to 0.05Ncyl
    where_ncyl_list.append(where_ncyl)

    crop_factor = where_ncyl
    crop_factor_list.append(crop_factor)
    V_ncyl_list.append(vacdrop_list[crop_factor])  # Store the confinement voltage at 0.05Ncyl

    escaped_list_lin = escaped_list[:crop_factor]  # Remove non-positive values from escaped_list
    vacdrop_list_lin = vacdrop_list[:crop_factor]  # Remove corresponding values from vacdrop_list
    
    l_p_list = l_p_list[-65:]
    print(f"len(escaped_list){len(escaped_list)}")

    # ---------------- Plasma length ----------------
    plasma_length = np.average(l_p_list)
    l_p_s.append(plasma_length)
    print(l_p_list)

    # ---------------- Fit ----------------
    fit, cov = np.polyfit(vacdrop_list_lin, np.log(escaped_list_lin), deg=1, cov=True)
    polyval = np.polyval(fit, vacdrop_list)

    T = -1.05 * 1.60217662e-19 / (1.38064852e-23 * fit[0])  # Calculate temperature from slope
    
    fit2, cov2 = np.polyfit(vacdrop_list, np.log(escaped_list), deg=1, cov=True)
    err_list_d.append(np.sqrt(cov2[0, 0]))
    T_list.append(T)

    # ---------------- Plot curves (top panel) ----------------
    if int(round(d * 1000)) % 10 == 0:

        style = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]

        # Data
        axs[0].plot(vacdrop_list, np.log10(escaped_list),
                    linestyle="none",
                    marker=marker,
                    markersize=4,
                    alpha=0.5,
                    color="black")

        # Fit
        axs[0].plot(vacdrop_list, polyval / np.log(10),
                    linestyle=style,
                    linewidth=1.8,
                    label=data_label)

# ---------------- Top plot formatting ----------------
axs[0].set_xlabel("Confinement (V)")
axs[0].set_ylabel(r"$\log_{10}(N_\mathrm{esc})$")
axs[0].set_title("Finite Length Investigation")

axs[0].invert_xaxis()
axs[0].set_xlim(right=0)  # Set x-axis limit to just beyond the last vacuum drop
axs[0].legend(loc="lower right", ncol=1, frameon=False, fontsize=14)
axs[0].grid(True, linestyle="--", alpha=0.4)

# =====================================================
# Bottom plot: MAIN RESULT (what you asked for)
# Plasma length vs fit error
# =====================================================

l_p_s = np.array(l_p_s)
l_p_s = d_range
err_list_d = np.array(err_list_d)

axs[1].scatter(l_p_s*100, err_list_d,
               marker="o")

# Optional trend line (helps interpretation)
fit_lp = np.polyfit(l_p_s, np.log(err_list_d), 1)
x_fit = np.linspace(min(l_p_s), max(l_p_s), 100)
y_fit = np.exp(np.polyval(fit_lp, x_fit))

axs[1].plot(x_fit*100, y_fit,
            linestyle="--",
            linewidth=1.5)

# ---------------- Bottom plot formatting ----------------
axs[1].set_xlabel("Trap Length (cm)")
axs[1].set_ylabel(r"Fit standard deviation $\sigma$")
axs[1].set_title("Relationship between Trap Length and Fit Error")

axs[1].set_yscale("log")
axs[1].grid(True, linestyle="--", alpha=0.4)

# ---------------- Final layout ----------------
plt.tight_layout()
plt.savefig("finite_length_plasma_error_A4.pdf", bbox_inches="tight")
plt.show()
#%%


T_err = np.array(T_list) * 0.04  # Assuming 4% uncertainty
# Data
diff = np.array(T_list)-np.array(200)
percent_diff = 100 * diff / np.array(T_list)

# Create side-by-side subplots
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(8.27, 5))  # A4 width

# =========================
# Left: Deviation
# =========================
ax3.plot(l_p_s*100, diff, marker="o", linewidth=2)
ax3.axhline(0, linestyle="--")
ax3.errorbar(l_p_s*100,diff,  yerr=T_err, fmt='o', color='blue', ecolor='gray', elinewidth=3, capsize=3, label="Actual Temperature ± 4%")
ax3.set_xlim(l_p_s[0]*100-0.1, l_p_s[-1]*100+0.1)
ax3.set_xlabel("Trap Length(cm)")
ax3.set_ylabel(r"$\Delta T = T_\text{Solver} - T_\text{Lin-fit}$ (K)")
ax3.set_title("Absolute Deviation")

# =========================
# Right: Percentage deviation
# =========================
ax4.plot(l_p_s*100, percent_diff, marker="o", linewidth=2)
ax4.errorbar(l_p_s*100, percent_diff, yerr=4, fmt='o', color='blue', ecolor='gray', elinewidth=3, capsize=3, label="Percentage Deviation ± 4%")
ax4.axhline(0, linestyle="--")
#ax4.set_yscale("log")

ax4.set_xlim(l_p_s[0]*100-0.1, l_p_s[-1]*100+0.1)
ax4.set_xlabel("Trap Length (cm)")
ax4.set_ylabel(r"Percentage Deviation: $\Delta T / T_\text{Lin-fit}$ (%)")
ax4.set_title("Percentage Deviation")

# Layout
plt.tight_layout()

# Save
plt.savefig("Finite_length_effect_deviation.pdf", bbox_inches="tight")
