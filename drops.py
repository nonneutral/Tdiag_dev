#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special

Mmax=1
Nmax=20000
zeros=[special.jn_zeros(m,Nmax) for m in range(Mmax)]

# Physical Constants
e0=8.854187817e-12 #farads per meter


def iter_all(substring, path):
    return list(
        os.path.join(root, entry)
        for root, dirs, files in os.walk(path)
        for entry in dirs + files
        if substring in entry
    )
def u8Correction(filename):
    df = pd.read_csv(filename, header=None)

    sipm_data = np.array(df[0].values)  #sipm (~escape rate)
    u8_data = np.array(df[1].values)    #u8 excitations
    

    print(np.shape(sipm_data))
    print(np.shape(u8_data))

    order = np.argsort(u8_data)

    sipm_ordered = sipm_data[order]
    u8_ordered = u8_data[order]
    t_data = np.arange(0,len(u8_data))

    u8vsT_fit = np.polyfit(t_data,u8_ordered,4)
    print(u8vsT_fit)
    u8_ordered_fit = np.polyval(u8vsT_fit,t_data)

    #plt.plot(t_data,u8_ordered)
    #plt.plot(t_data,u8_ordered_fit, color="g",ms=0.4)
    #plt.show()
    #plt.plot(abs(u8_ordered_fit-u8_ordered))
    #plt.show()
    u8_corrected = u8_ordered_fit * 15

    return u8_corrected, sipm_ordered
#%%
def getFiniteSolution(cnms,length,inf,rw,preclimit=1e-13,N=Nmax,M=Mmax):
    def solution(z,r):        
        pre=-np.sign(z-length/2)/2-np.sign(-length/2-z)/2
        c1=np.sign(-length/2-z)
        c2=np.sign(z-length/2)
        c4=c2
        c3=c1           
        total=pre*inf(r)
        if cnms[0][0]==0:
            return total
                
        for m in range(M):
            for n in range(N):
                if m==0:
                    new=( 
                        special.j0(zeros[m][n]*r/rw)*cnms[m][n]
                        *(.5*(c1*np.exp(c3*zeros[m][n]*(length/2+z)/rw)+c2*np.exp(c4*zeros[m][n]*(length/2-z)/rw)))
                        )
                else:
                    new=( 
                        special.jn(m,zeros[m][n]*r/rw)*cnms[m][n]
                        *(pre+.5*(c1*np.exp(c3*zeros[m][n]*(length/2+z)/rw)+c2*np.exp(c4*zeros[m][n]*(length/2-z)/rw)))
                        )
                total+=new
                if np.all(abs(new)/(abs(new)+abs(total))<preclimit):
                    return total
        return total
    return solution

def finiteChargeCylinder(cpl,rp,rw,length):
    coeffs=(
        (cpl/(np.pi*e0*(rw*special.j1(zeros[0]))**2))
        *(
        ((np.log(rp/rw)-.5)*(rp*rw*special.j1(zeros[0]*rp/rw)/zeros[0]))
        +((rw*rw*2*special.jn(2,zeros[0]*rp/rw)-rp*rw*zeros[0]*special.jn(3,zeros[0]*rp/rw))/(2*zeros[0]*zeros[0]))
        +(rw*(-rp*zeros[0]*special.j1(zeros[0]*rp/rw)*np.log(rp/rw)-rw*special.j0(zeros[0]*rp/rw))/(zeros[0]*zeros[0]))
        )
        )
    def inf(r):
        r_l_rp=(np.sign(rp-r)+1)/2
        return ( 
            r_l_rp       *(cpl/(2*np.pi*e0)) *(np.log(rp/rw)+(r*r-rp*rp) /(2*rp*rp))
            + (1-r_l_rp) *(cpl/(2*np.pi*e0)) *(np.log(np.maximum(r,rp)   /rw))
            )
    return getFiniteSolution([coeffs],length,inf,rw=rw)

def unitFreeSpaceSolutionsCalculation(electrode_input,nr,nz,rad2):
    initial_voltages,final_voltages,electrode_borders,Llim,Rlim,rw = electrode_input

    position_map_r=np.zeros((nr,nz))
    position_map_z=np.zeros((nr,nz))

    rbound = rad2*3.0
    for rind in range(nr):
        for zind in range(nz):
            position_map_z[rind,zind]=Llim+(Rlim-Llim)*(zind+.5)/(nz)
            position_map_r[rind,zind]=rbound*rind/nr
    
    def getElectrodeSolution(left,right,voltage):
        cs=2*voltage/(zeros[0]*special.j1(zeros[0]))
        symmetricsolution=getFiniteSolution([cs],right-left,lambda r:voltage,rw=rw)
        return lambda r,z: symmetricsolution(z-(left+right)/2,r)

    free_space_soln_list = [] #result of adding 1 excitation to i'th electrode
    free_1d_list = []
    for i in np.arange(0,len(initial_voltages)):
        voltages = np.zeros(len(initial_voltages))
        voltages[i] = 1

        #print(f"voltages: {voltages}")

        

        electrodes=[getElectrodeSolution(electrode_borders[j-1],electrode_borders[j],voltages[j]) for j in range(1,len(voltages)-1)]
            #probably zero everywhere other than i
    
        free_space_solution=np.zeros((nr,nz))

        for electrode in electrodes:
            free_space_solution+=electrode(position_map_r,position_map_z)
        
        free_1d = free_space_solution[0]
        free_1d_list.append(free_1d)
        free_space_soln_list.append(free_space_solution)


    return free_space_soln_list,free_1d_list,position_map_r,position_map_z

def superpose_potential(unit_solutions, electrode_voltages):
    total = np.zeros_like(unit_solutions[0])
    
    for i, V in enumerate(electrode_voltages):
        total += V * unit_solutions[i]
    
    return total

def u8_to_drop(unit_solutions, u8_input, initial_voltages, nz, Rlim, Llim):
    peak_idx = -1
    barrier_idx = -1

    if (u8_input - initial_voltages[2]) > 0.005:
        print(u8_input - initial_voltages[2])
        return float("nan"), peak_idx, barrier_idx

    dz = (Rlim - Llim) / nz
    
    voltages = np.array(initial_voltages, copy=True)
    voltages[1] = u8_input

    free_1d = superpose_potential(unit_solutions, voltages)[0]

    # ---- slopes / thresholds ----
    dVdz = np.gradient(free_1d, dz)
    abs_dVdz = np.abs(dVdz)
    lo = min(10, max(1, nz // 10))
    hi = max(lo + 1, nz - 1)
    thr_slope = np.quantile(abs_dVdz[lo:hi], 0.07)

    is_flat_slope = abs_dVdz <= thr_slope
    d2Vdz2 = np.gradient(dVdz, dz)

    # ---- local MAXIMA (peak candidates) ----
    is_local_max = np.zeros(nz, dtype=bool)
    is_local_max[1:-1] = (free_1d[1:-1] > free_1d[:-2]) & (free_1d[1:-1] >= free_1d[2:])
    valid_max = is_local_max & is_flat_slope
    valid_max[:10] = False
    valid_max &= (d2Vdz2 < 0)
    maxs = np.where(valid_max)[0]
    if maxs.size > 0:
        peak_idx = int(maxs[np.argmax(free_1d[maxs])])
    else:
        peak_idx = int(np.argmax(free_1d))  # fallback

    # ---- local MINIMA (barrier candidates) ----
    is_local_min = np.zeros(nz, dtype=bool)
    is_local_min[1:-1] = (free_1d[1:-1] < free_1d[:-2]) & (free_1d[1:-1] <= free_1d[2:])
    valid_min = is_local_min & is_flat_slope
    valid_min[:10] = False
    valid_min &= (d2Vdz2 > 0)
    
    if peak_idx <= 0:
        barrier_idx = 0
    else:
        mins_left = np.where(valid_min & (np.arange(nz) < peak_idx))[0]
        if mins_left.size > 0:
            barrier_idx = int(mins_left[np.argmin(free_1d[mins_left])])
        else:
            barrier_idx = int(np.argmin(free_1d[:peak_idx]))  # fallback

    # Ensure indices are in range
    peak_idx = max(0, min(nz-1, peak_idx))
    barrier_idx = max(0, min(nz-1, barrier_idx))

    # Compute voltage drop
    if peak_idx == barrier_idx:
        VoltageDrop = float("nan")
    else:
        V_peak = float(free_1d[peak_idx])
        V_barrier = float(free_1d[barrier_idx])
        VoltageDrop = V_peak - V_barrier

    return VoltageDrop, peak_idx, barrier_idx

 


# Initial Input Values
initial_voltages=np.array([0,-63,-50,-130,0]) #in volts
final_voltages=np.array([0,0,-50,-130,0], dtype=float) #in volts
electrode_borders=[0.025,0.050,0.100,0.125] #in meters
Llim=0.035
Rlim=0.115
rw=.017 #radius of inner wall of cylindrical electrodes, in meters
rampfrac=0.9
current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages) * rampfrac

rad2 = 0.0008

electrode_input = [
    np.array(initial_voltages),
    np.array(final_voltages),
    np.array(electrode_borders),
    float(Llim),
    float(Rlim),
    float(rw)
]

nr=20
nz=40





filepath1=iter_all('csv','../')[8] #load data

u8_corrected, sipm_ordered = u8Correction(filepath1)
plt.scatter(u8_corrected,sipm_ordered,marker=".")


unit_free_solns, unit_free_1ds, position_map_r, position_map_z = unitFreeSpaceSolutionsCalculation(electrode_input,nr,nz,rad2)

print(unit_free_solns)

for voltages in unit_free_solns:
    plt.imshow(voltages, aspect='auto', origin='lower')
    plt.colorbar(label="Potential")
    plt.xlabel("z index")
    plt.ylabel("r index")
    plt.title("2D Potential Map")
    plt.show()

for voltages in unit_free_1ds:
    plt.plot(position_map_z[0],voltages)
plt.show()


"""
v8s = np.arange(-63,0,1)
drops = []
for v8 in v8s:
    print(f"v8: {v8}")
    voltages = np.array([0,v8,-50,-130,0])
    total_potential = superpose_potential(unit_free_solns, voltages)
    
    drop = u8_to_drop(unit_free_solns,v8,voltages,nz,Rlim,Llim)
    drops.append(drop)
    plt.plot(position_map_z[0],total_potential[0])

plt.show()

plt.plot(drops,v8s)
plt.show()"""

drops_data = np.zeros_like(u8_corrected)

for i, u8 in enumerate(u8_corrected):
    voltages = np.array([0,u8,-50,-130,0])
    drop,peak_idx,barrier_idx = u8_to_drop(unit_free_solns,u8,voltages,nz,Rlim,Llim)
    print(f"i: {i}/{len(u8_corrected)}, u8: {u8}, drop: {drop:.10E}")
    drops_data[i]=drop
    if i%1000==0:
        plt.plot(position_map_z[0],superpose_potential(unit_free_solns,voltages)[0],zorder=1)
        if not np.isnan(drop):
            plt.scatter(position_map_z[0,peak_idx],
                        superpose_potential(unit_free_solns,voltages)[0,peak_idx],
                        color="black",marker="^",zorder=2,s=2)
            plt.scatter(position_map_z[0,barrier_idx],
                        superpose_potential(unit_free_solns,voltages)[0,barrier_idx],
                        color="black",marker="^",zorder=2,s=2)

plt.show()

mask = ~np.isnan(drops_data)

drops_data = drops_data[mask]
u8_corrected = u8_corrected[mask]
#%%
plt.plot(u8_corrected,drops_data)
plt.title("u8 to drop conversion")
plt.xlabel("u8 (V)")
plt.ylabel("drop (V)")
#plt.xlim(-0.1,0.1)

# %%
