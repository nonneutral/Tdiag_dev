#This file is used to investigate offset of the u8 data. It is not used for any data processing, just for visualisation.

import pandas as pd
from scipy.interpolate import interp1d
import os
import numpy as np
import matplotlib.pyplot as plt
#import lin_fit_optimiser as lfo


def iter_all(substring, path):
    return list(
        os.path.join(root, entry)
        for root, dirs, files in os.walk(path)
        for entry in dirs + files
        if substring in entry
    )


filepath1=iter_all('csv','../')[20] #load data
df = pd.read_csv(filepath1, header=None)

sipm_data = np.abs(df[0].values)  #sipm (~escape rate)
u8_data = df[1].values*15   #u8 excitations

solver_vs_experimental_fit = []


#for x in np.arange(150,1000,1):

offset=1000 #offset to trim data for better fit - adjust as needed based on data length and quality

sipm_data = sipm_data[offset:len(sipm_data)]
u8_data = u8_data[0:-offset]


order = np.argsort(u8_data)
u8_data = u8_data[order]
sipm_data = sipm_data[order]


u8_solver, sipm_solver,_,_,_ = np.loadtxt("useful_data\T300_N3.00e+05_omega_r9.02e+04_rad0.0008_B2.0.csv",delimiter=",")
u8_solver = -63 * (1-u8_solver)



mask = u8_data > u8_solver[0]  # Only consider data points where u8_data is greater than the first point of u8_solver
mask = mask & (u8_data < u8_solver[-1])  # Also ensure u8_data is less than the last point of u8_solver


u8_data = u8_data[mask]
sipm_data = sipm_data[mask]

f = interp1d(u8_solver, sipm_solver, kind='linear', fill_value="extrapolate")
sipm_interp = f(u8_data)



diff = np.abs(np.log10(sipm_interp)-np.log10(1000*sipm_data))/sipm_data

solver_vs_experimental_fit.append(sum(diff))
plt.plot(u8_data, sipm_interp, marker="o", linestyle="-", color="blue", label=f"offset = {offset}")
plt.plot(u8_data, sipm_data)    
plt.plot(np.arange(150,1000,1), solver_vs_experimental_fit, marker="o", linestyle="-", color="blue")
plt.title("Fit of solver to experimental data vs offset", fontsize=20)
plt.yscale("log")
