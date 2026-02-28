"""
import numpy as np
import matplotlib.pyplot as plt
from solver2 import *

# %%
def find_crop_factor(cf_list):
    errvac_list = []
    errdrop_list = []
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

cf_list,errvac_list,errdrop_list = find_crop_factor(np.linspace(0.59,0.60,100))

plt.plot(cf_list,errvac_list,label="vac_fit")
plt.plot(cf_list,errdrop_list,label="drop_fit")
plt.legend()
plt.show()

min_cf=cf_list[np.argmin(errvac_list)]
min_cf_drop=cf_list[np.argmin(errdrop_list)]

print(f"cf for best lin fit {min_cf}")
print(f"cf for best lin fit (drop) {min_cf_drop}")
print(f"idx: {np.argmin(errvac_list)}")
"""