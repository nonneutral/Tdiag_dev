import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from step_1 import iter_all,extract_measured_temp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.rcParams.update({
    # Figure
    "figure.figsize": (6.3, 4.0),
    "figure.dpi": 100,

    # Fonts (clean academic look)
    "font.family": "serif",
    "font.size": 11,

    # Axis labels
    "axes.labelsize": 11,
    "axes.titlesize": 12,

    # Ticks
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # Legend
    "legend.fontsize": 10,
    "legend.frameon": False,   # cleaner, paper-like
    "legend.loc": "best",

    # Lines
    "lines.linewidth": 1.5,

    # Grid (optional but useful for data plots)
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})
# Constants
q_e = 1.602176634e-19  # elementary charge (C)
kb = 1.380649e-23     # Boltzmann constant (J/K)

# Temperature list
T_list = np.arange(50, 401, 10)
T_list = np.append(T_list, np.arange(500, 1001, 100))
T_err = 0.04 * T_list  # 4% error

# Storage
err_list = []
T_estimate_list = []
number_of_points = []

file_directory = iter_all('csv', 'T_analysis_April_21(rampfrom10kBT)')

# --- A4 styling ---


# =========================
# Main loop (analysis + plot 1)
# =========================
fig1, ax1 = plt.subplots(figsize=(6.3, 5.0))  # A4 width
sum_escaped = []
where_ncyl_list = []
crop_factor_list = []
V_ncyl_list = []

for i in range(len(T_list)):
    T = T_list[i]
    filename = file_directory[i]
    print(filename)

    df = pd.read_csv(filename, header=None).T

    if T < 500:
        escaped_cumulative = df[1].values
        vacdrop_list = df[2].values


        
    elif T <= 1000:
        escaped_cumulative = df[0].values
        vacdrop_list = df[1].values

    sum_escape_for_T = escaped_cumulative[-1]
    escaped_list = np.diff(escaped_cumulative, prepend=escaped_cumulative[0])
    escaped_list = escaped_list[1:]  # Remove any non-positive values
    vacdrop_list = vacdrop_list[1:]

    print(f"T = {T}, escaped_list {escaped_list}")
    
    sum_escape_for_T = escaped_cumulative[-1]
    sum_escaped.append(sum_escape_for_T)
    #escaped_list = escaped_cumulative[1:]

    number_of_points.append(len(escaped_list))
    where_ncyl = np.argmin(np.abs(escaped_cumulative - 0.05*0.8226519655*T*T))  # Find index where escaped_cumulative is closest to 0.05Ncyl
    where_ncyl_list.append(where_ncyl)
    if where_ncyl == len(vacdrop_list):
        where_ncyl = len(vacdrop_list)-1
    crop_factor = where_ncyl
    crop_factor_list.append(crop_factor)
    V_ncyl_list.append(vacdrop_list[crop_factor])  # Store the confinement voltage at 0.05Ncyl


    # Fit (log10)
    fit, cov = np.polyfit(vacdrop_list, np.log10(escaped_list), deg=1, cov=True)
    values = np.polyval(fit, vacdrop_list)
    err = np.sqrt(np.diag(cov))[0]
    err_list.append(err)

    # Linear regime fit (natural log)
    #linear_regime = int(0.6 * len(vacdrop_list))
    lin_vacdrop = vacdrop_list[:crop_factor]
    lin_escaped = escaped_list[:crop_factor]

    fit_lin, cov_lin = np.polyfit(lin_vacdrop, np.log(lin_escaped), deg=1, cov=True)
    values = np.polyval(fit_lin, vacdrop_list)
    slope = fit_lin[0]

    # Temperature estimate
    T_estimate = -q_e * 1.05 / (kb * slope)
    T_estimate_list.append(T_estimate)

    print(f"T={T} K: slope={fit[0]:.3f} ± {err:.3f}")
    print(f"T_estimate = {T_estimate:.2f} K")

    if i % 3 == 0:
        ax1.plot(vacdrop_list, np.log10(escaped_list),
                    marker="o", markersize=3,
                    linewidth=1, label=f"T={T} K")

        ax1.plot(vacdrop_list, values/np.log(10),
                    linestyle="--", linewidth=1,
                    color="black")  

# Axis formatting
ax1.set_xlabel("Vacuum drop (V)")
ax1.set_ylabel("Escaped electrons (log$_{10}$)")
ax1.set_title("Escaped Electrons vs Vacuum Drop")
ax1.invert_xaxis()
ax1.legend(loc="lower left",ncol=2, fontsize=11)
plt.tight_layout()
plt.savefig("escaped_vs_vacdrop_A4.pdf", bbox_inches="tight")
plt.show()



# =========================
# Plot 2: Estimated vs Actual Temperature
# =========================

fig2, ax2 = plt.subplots()  # A4 width

ax2.plot( T_estimate_list, T_list, marker="o", linewidth=2, label="Solver Temperature")
ax2.plot(T_estimate_list, T_estimate_list, linestyle="--", linewidth=2, label="Linear-fit Temperature (y = x)")
ax2.errorbar(T_estimate_list, T_list, yerr=T_err, fmt='o', color='blue', ecolor='gray', elinewidth=3, capsize=3, label="Actual Temperature ± 4%")
ax2.set_ylabel("Solver Temperature (K)")
ax2.set_xlabel("Linear-fit Temperature (K)")
ax2.set_title("Solver vs Linear-fit Temperature")
ax2.set_xlim(37, 600)
ax2.set_ylim(0, 700)
ax2.legend()

plt.tight_layout()
plt.savefig("T_estimate_vs_actual_A4.pdf", bbox_inches="tight")
plt.show()

# Data
diff = np.array(T_list)-np.array(T_estimate_list)
percent_diff = 100 * diff / np.array(T_estimate_list)

# Create side-by-side subplots
fig, (ax3, ax4) = plt.subplots(1, 2)  # A4 width

# =========================
# Left: Deviation
# =========================
ax3.plot(T_estimate_list, diff, marker="o", linewidth=2)
ax3.axhline(0, linestyle="--")
ax3.errorbar(T_estimate_list,diff,  yerr=T_err, fmt='o', color='blue', ecolor='gray', elinewidth=3, capsize=3, label="Actual Temperature ± 4%")
ax3.invert_yaxis()
ax3.set_xlabel("Linear-fit Temperature (K)")
ax3.set_ylabel(r"$\Delta T = T_\text{Solver} - T_\text{Lin-fit}$ (K)")
ax3.set_title("Absolute Deviation")

# =========================
# Right: Percentage deviation
# =========================
ax4.plot(T_estimate_list, percent_diff, marker="o", linewidth=2)
ax4.errorbar(T_estimate_list, percent_diff, yerr=4, fmt='o', color='blue', ecolor='gray', elinewidth=3, capsize=3, label="Percentage Deviation ± 4%")
ax4.axhline(0, linestyle="--")
#ax4.set_yscale("log")
ax4.invert_yaxis()
ax4.set_xlabel("Linear-fit Temperature (K)")
ax4.set_ylabel(r"Percentage Deviation: $\Delta T / T_\text{Lin-fit}$ (%)")
ax4.set_title("Percentage Deviation")

# Layout
plt.tight_layout()

# Save
plt.savefig("T_deviation_combined_A4.pdf", bbox_inches="tight")

plt.show()
plt.plot(T_list, err_list, marker="o", linestyle="-")
#plt.yscale('log')
plt.xlabel("Solver Temperature (K)")
plt.ylabel(r"Standard Deviation in linear-fit $\sigma$")
plt.title("Standard Deviation in Least Squares Fit vs Solver Temperature")
plt.savefig("Standard_Deviation_in_Least_Squares_Fit.pdf", bbox_inches="tight")
plt.show()

# Debye cylinder 
plt.plot(T_list, sum_escaped/(0.8226519655*T_list*T_list), marker="o", linestyle="-")
plt.xlabel("Solver Temperature (K)")
plt.ylabel(r"Total Debye Cylinder Escaping ($N_\text{cyl}$)")
plt.title(r"Total Electrons Escaped Normalized by $N_\text{cyl}$")
plt.savefig("Total_Debye_Cylinder_Escaping.pdf", bbox_inches="tight")
plt.show()

# %%
