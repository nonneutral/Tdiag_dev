# Solver for plasma of known number of electrons (NVal), temperature (T_e), and rotation frequency (fE)
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import special
from pylab import rcParams
from scipy.special import erf,erfc #import error function

start_total_time = time.time()

rcParams['figure.figsize'] = 10, 6

rw=.017 #radius of inner wall of cylindrical electrodes, in meters
q_e=1.60217662e-19 #elementary charge in coulombs
m_e=9.1093837e-31 #electron mass in kilograms
kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin
e0=8.854187817e-12 #farads per meter
T_e=1960 #plasma temperature in kelvin
N_e=8e6 #number of electrons
rad2=0.0002 #plasma radius in meters
B2=1.6 #magnetic field in tesla

Mmax=1
Nmax=20000
zeros=[special.jn_zeros(m,Nmax) for m in range(Mmax)]

def getFiniteSolution(cnms,length,inf,preclimit=1e-13,N=Nmax,M=Mmax):
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
    return getFiniteSolution([coeffs],length,inf)

def plasma_length_guess(NVal,mur2,rw,q_e,position_map_z,free_space_solution,electrode_borders):
    potl_barrier = free_space_solution[0,:].copy()
    electrode_centre = np.average(electrode_borders)
    inElectrode = (electrode_borders[1] < position_map_z[0])&(position_map_z[0] < electrode_borders[-2]) 
    rp = mur2
    rw = rw
    cpl = NVal/(electrode_borders[2]-electrode_borders[1])
    potl_inf = (cpl*(-q_e)/(4*np.pi*e0))*(2**np.log(rw/rp)+1)
    potl_inf = potl_inf - np.min(np.abs(potl_barrier[inElectrode])) 
    print(np.min(np.abs(potl_barrier[inElectrode])))
    # need to make this more general (- sign must be removed) not dependent on charge sign
    potl_diff = np.abs(potl_inf - potl_barrier)
    plasma_left_end = position_map_z[0,np.argmin(potl_diff[position_map_z[0]<electrode_centre])]
    plasma_right_end = position_map_z[0,np.argmin(potl_diff[position_map_z[0]>electrode_centre])]+ electrode_centre
    plasma_length =  plasma_right_end - plasma_left_end
    
    plt.plot(position_map_z[0,:], potl_barrier, label='On-axis potential')
    plt.plot(position_map_z[0,:], np.full(len(position_map_z[0,:]), potl_inf), label='Infinite-length space charge potential')
    plt.show()

    return [plasma_left_end, plasma_right_end, plasma_length]

def find_solution(NVal,T_e,fE,mur2,B,electrodeConfig,left,right,zpoints,rpoints,rfact,plotting, coarse_sol_divisor, InitializeWithPlasmaLength = False):
    nr=rpoints
    nz=zpoints
    omega_c=q_e*B/m_e
    omega_r=fE*2*np.pi
    ri=np.array([[max(0,rind-.5) for zind in range(nz)] for rind in range(nr)])
    ro=np.array([[max(0,rind+.5) for zind in range(nz)] for rind in range(nr)])
    mursq=(1./3.)*(ro*ro*ro-ri*ri*ri)/(ro-ri)
    leftbound=left
    rightbound=right
    rbound=mur2*rfact
    position_map_r=np.zeros((nr,nz))
    position_map_z=np.zeros((nr,nz))
    for rind in range(nr):
        for zind in range(nz):
            position_map_z[rind,zind]=leftbound+(rightbound-leftbound)*(zind+.5)/(nz)
            position_map_r[rind,zind]=rbound*rind/nr

    def getElectrodeSolution(left,right,voltage):
        cs=2*voltage/(zeros[0]*special.j1(zeros[0]))
        symmetricsolution=getFiniteSolution([cs],right-left,lambda r:voltage)
        return lambda r,z: symmetricsolution(z-(left+right)/2,r)

    electrode_voltages,electrode_borders=electrodeConfig
    free_space_solution=np.zeros((nr,nz))
    electrodes=[getElectrodeSolution(electrode_borders[i-1],electrode_borders[i],electrode_voltages[i]) 
                for i in range(1,len(electrode_voltages)-1)]   

    for electrode in electrodes:
        free_space_solution+=electrode(position_map_r,position_map_z)
   
    inner_outer_pair_lambdas = [(lambda x,y:0,
                                 finiteChargeCylinder(np.pi*(rbound*(.5)/nr)**2,rbound*.5/nr,rw,(rightbound-leftbound)/nz))
                                ]+[
                                    (finiteChargeCylinder(np.pi*(rbound*(rind-.5)/nr)**2,rbound*(rind-.5)/nr,rw,(rightbound-leftbound)/nz),
                                     finiteChargeCylinder(np.pi*(rbound*(rind+.5)/nr)**2,rbound*(rind+.5)/nr,rw,(rightbound-leftbound)/nz)) 
                                    for rind in range(1,nr)]
    radial_finite_charge_cylinders = lambda r,z,rind: inner_outer_pair_lambdas[rind][1](r,z)-inner_outer_pair_lambdas[rind][0](r,z)
    voltage_maps=[]
    backward_voltage_maps=[]
    for rind in range(nr):
        voltage_maps.append(radial_finite_charge_cylinders(position_map_z-np.min(position_map_z),position_map_r,rind))
        backward_voltage_maps.append(np.flip(voltage_maps[-1],1))


    def get_voltage_from_charge_distr(charges,cutoff=1e-20):
        total=np.zeros((nr,nz))
        for rind in range(nr):
            for zind in range(nz):
                if abs(charges[rind,zind])>cutoff:
                    if zind!=0:
                        total[:,zind:]+=voltage_maps[rind][:,:-zind]*charges[rind,zind]
                    else:
                        total+=voltage_maps[rind]*charges[rind,zind]
                        continue
                    total[:,:zind]+=backward_voltage_maps[rind][:,-zind-1:-1]*charges[rind,zind]  
        return total

    # initial omega_r calculation based on initial guess plasma length, NVal, and mur2 - 5th Nov, 2025
    # plasma length based initialisation 
    lp_init_config = np.array([0.0, 0.0, 0.0])
    lp_init = 0
    
    if InitializeWithPlasmaLength == True:
        lp_init_config = np.array(plasma_length_guess(NVal, mur2, rw, q_e, position_map_z, free_space_solution, electrode_borders))
        lp_init = lp_init_config[2]
        print(f'Initial plasma length estimate: {lp_init_config[2]:0.3e} m (from {lp_init_config[0]:0.3e} m to {lp_init_config[1]:0.3e} m)')
        #guesses omega_r based on the initial plasma length given
        quad_eq_c = (NVal*q_e**2)/(np.pi*lp_init*mur2*mur2*(2*m_e*e0))
        omega_r = (omega_c-np.sqrt(omega_c*omega_c-4*quad_eq_c))

    print(f'Initial omega_r estimate = {omega_r}')

    phieff=np.array([[.5*m_e*omega_r*(omega_c-omega_r)*(rbound*rind/nr)**2 
                       for zind in range(nz)] for rind in range(nr)]) 
    volume_elements=np.array([[np.pi*((rbound*(rind+.5)/nr)**2-(rbound*max(0,rind-.5)/nr)**2)
                               *(rightbound-leftbound)/nz 
                                for zind in range(nz)] for rind in range(nr)])
    #----Manual initial guess for ngrid - UPDATED ON 23rd Oct, 2025----
    """
    exponential_guess = -((-q_e * free_space_solution) + phieff) / (kb * T_e)
    magic = 600 #Cutoff for exp calculation
    mx_guess = np.max(exponential_guess)
    ngrid_guess = np.zeros((nr, nz))
    valid_cells = exponential_guess > mx_guess - magic #Apply exp only where exponent is reasonably large
    ngrid_guess[valid_cells] = np.exp(exponential_guess[valid_cells] - mx_guess) 
    total_particles_guess = np.sum(ngrid_guess * volume_elements) #Normalize the initial guess to match NVal
    ngrid = ngrid_guess * (NVal / total_particles_guess) #THE NEW ngrid GUESS
    print(f"Initial guess generated with {np.sum(ngrid * volume_elements):.2e} particles.")
    #----End of updated initial guess; update replaced just the line: ngrid = np.zeros((nr, nz))----
    """
    ngrid = np.zeros((nr, nz))
    n0=2*m_e*e0*omega_r*(omega_c-omega_r)/(q_e*q_e)
    debye_length=np.sqrt((e0*kb*T_e)/(q_e*q_e*n0))
    a=rbound
    b=(rightbound-leftbound)
    lambdac=-1/(((np.pi/(2*b))**2+(zeros[0][0]/a)**2)*debye_length*debye_length)
    #print('what is lambda c',lambdac)
    epsapprox=1/(2-lambdac)
    epsilon=epsapprox
    #magic=60

    
    #find initial region of interest based on electrode borders
    roi_init = position_map_z[0,:] >= electrode_borders[1]
    roi_init &= position_map_z[0,:] <= electrode_borders[-2]

    for i in range(np.int64(1e6)):
        voltageGuess=np.copy(free_space_solution)
        voltageGuess+=get_voltage_from_charge_distr(q_e*ngrid)
        exponential=-(voltageGuess*(-q_e)+phieff)/(kb*T_e)
        
        #roi calculation 
        exponential_roi_init = exponential[0,roi_init]
        mx_exp_z = position_map_z[0,roi_init][np.argmax(exponential_roi_init)]
        #mx_exp = np.max(exponential_roi_init)
        roi_left_ind = np.argmin(exponential[0,:][position_map_z[0,:] < mx_exp_z])
        roi_left = position_map_z[0,position_map_z[0,:] < mx_exp_z][roi_left_ind]
        roi_right_ind = np.argmin(exponential[0,:][position_map_z[0,:] > mx_exp_z])
        roi_right = position_map_z[0,position_map_z[0,:] > mx_exp_z][roi_right_ind]
        plasma_length = roi_right - roi_left #plasma length estimate from current iteration
        
        mx=np.max(exponential)
        nnew=np.zeros((nr,nz))
        #nnew[exponential>mx-magic]=np.exp(exponential-mx)[exponential>mx-magic]
        nnew=np.exp(exponential-mx)

        nnew[position_map_z<roi_left]=0
        nnew[position_map_z>roi_right]=0


        total=np.sum(nnew*volume_elements)
        nnew*=NVal/total
        ngrid=ngrid*(1-epsilon)+epsilon*nnew
        err=np.sum(np.abs((ngrid-nnew)*volume_elements))

        if  i%100==0:
            if plotting:
                # print(f'iteration = {i:0.1e}',
                #       f'\t fraction of gridpoints with charge = {np.sum(ngrid!=0)/np.sum(ngrid>=-1):0.2f}',
                #       f'\t misplaced electrons = {err:0.1e}'
                #       )
                plt.title("on-axis potential")
                plt.plot(position_map_z[0,:],free_space_solution[0,:],label="solver electrode potential")
                plt.plot(position_map_z[0,:],voltageGuess[0,:],label="charge-corrected potential")
                plt.ylabel("potential (V)")
                plt.xlabel("position (m)")
                plt.legend()
                plt.show()
            #if err<np.sum(NVal):
                #epsilon=epsapprox*np.sum(ngrid>=np.max(ngrid)/2)/np.sum(ngrid>=-1)
            if err<NVal/coarse_sol_divisor:
                break

    NS=ngrid*volume_elements
    rmean=np.sqrt(np.sum(mursq*NS)/(np.sum(NS)))*rbound/nr
    phi=np.max(free_space_solution[0,:])-np.max(voltageGuess[0,:])
    print('N, T, φ, ρ, r_0, λ_D, l_p')
    print(f'{NVal:0.3e}\t{T_e:0.3e}\t{phi:0.3e}\t{n0:0.3e}\t{rmean:0.3e}\t{debye_length:0.3e} \t{plasma_length:0.3e}')

    if plotting:  
        plt.figure(figsize=(6,4))
        plt.imshow(ngrid,cmap='gnuplot')
        plt.title("density")
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(6,4))
        plt.imshow(voltageGuess,cmap='gnuplot')
        plt.title("potential")
        plt.colorbar()
        plt.show()
    return ngrid,position_map_z,position_map_r,voltageGuess,free_space_solution,rmean,omega_r,volume_elements

"""
Main function call

Parameters:
1: electrodeConfig - a tuple of two things (electrode_voltages,electrode_borders)
    a: electrode_voltages - a list; the voltages on the electrodes. 
        The first and last electrodes are assumed to be infinitely long (recommend grounding)
    b: electrode_borders - a list; the positions of the borders between electrodes. 
        Note that if electrode_voltages has length N, electrode borders has length N-1
    example: ([0,300,150,0],[-.005,0,.005])    
2: NVal - a float, the number of electrons
3: mur2 - a guess for the (rms) radius of the plasma, measured in meters
4: left - the left z bound of the solution region
5: right - the right z bound of the solution region
6: omegarguess - the initial guess for the rotation rate (can be bad)
7: zpoints - the number of gridpoints in z (optional, default 120)
8: rpoints - the number of gridpoints in r (optional, default 60)
9: T_e - the temperature (K)
10: rfact - default value 2.5. 
    It is the factor by which we multiply mur2 to get the radial extent of the solution region
    a little lower is okay for plasma, a little higher is better for hot clouds
    EDH: found the following cryptic note, maybe about rfact
    2.5281640150894664, 40K, plasma
    3.8, 100k, plasma ish
    10.5, 400k, slightly flattened
11: plotting - whether or not to print out debugging output and periodically show a picture of the plasma. 

Output:
ngrid: the density profile 
position_map_z: the z values of each grid point (a matrix)
position_map_r: the r values of each grid point (a matrix)
voltageGuess: the potential of each grid point (a matrix)
free_space_solution: the potential of each point in the absence of charge (a matrix)
rmean: average radius
"""

# Initial Input Values!!!
initial_voltages=np.array([0,-50,-10,-50,0])
final_voltages=np.array([0,-15,-10,-50,0], dtype=float)
electrode_borders=[0.025,0.050,0.100,0.125]
Llim=0.035
Rlim=0.110
rampfrac=0.9
NVal=8.0e6
#---ADDED VARIABLES FOR COARSE LOOP---
freq_guess = 2.0e6  # initial guess for rotation frequency in Hz
#EDH: generate this from NVal, rad2, and plasma length
#: guess plasma length based on the confining potentials
#: use the infinite length space charge formula for \phi_0, with N=NVal and r=1 mm
#: plasma length is distance between the points on blue curve at offset = \phi_0 from the potential minimum
omega_r  = 2*np.pi*freq_guess
#---END OF ADDED VARIABLES FOR COARSE LOOP---
current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages)*rampfrac

#---FUNCTION TO RETUNE OMEGA_R TO HIT TARGET RADIUS; STEP 4 ON SLIDES - UPDATED 25th OCT---
def retune_omega_to_hit_radius(omega_r, r_mean, r_target):
    r_ratio_sq = (r_mean / r_target)**2
    omega_r_new = r_ratio_sq * omega_r
    return omega_r_new

for _ in range(8):  #COARSE LOOP - range(number) is just number of iterations to try
    if _ == 0:
        initialse_using_pl = True
    else: 
        initialse_using_pl = False
    sol = find_solution(NVal=N_e,T_e=T_e,fE=omega_r/(2*np.pi),mur2=rad2,B=B2,
                        electrodeConfig=(initial_voltages,electrode_borders),
                        left=Llim,right=Rlim,zpoints=40,rpoints=20,rfact=3.0,plotting=True, coarse_sol_divisor=50, InitializeWithPlasmaLength = initialse_using_pl)
    omega_r = sol[6]
    r_mean = sol[5]     #returned rmean
    vfree = sol[4]   #returned free_space_solution
    print(f'potential-to-kT ratio: {np.max(-q_e*vfree)/(kb*T_e):0.2f}')
    if abs(r_mean - rad2) <= 0.01 * rad2:
        print("Desired radius achieved within 10% tolerance.")
        break
    omega_new = retune_omega_to_hit_radius(omega_r, r_mean, rad2) #using funciton to retune omega_r and hit traget radius.
    if omega_new is None:
        print("Target radius not reachable with current parameters.")
        break
    omega_r = omega_new

#Unpacking coarse solution to get factors to adjust intial excitations - STEP 5 ON SLIDES
n=sol[0]
voltageGuess=sol[3]
vfree=sol[4]
maxvolt=np.max(voltageGuess)
minvolt=np.min(voltageGuess)
maxN=np.max(n)
position_map_z=sol[1]
position_map_r=sol[2]
maxr=np.max(position_map_r)
dr=-position_map_r[0,0]+position_map_r[1,0]

z_axis = position_map_z[0, :]
vfree_on = vfree[0, :]
v_sc_on = voltageGuess[0, :]

# 1) Index where free-space potential peaks (on-axis)
peak_idx = int(np.argmax(vfree_on))
peak_z = float(z_axis[peak_idx])
vfree_peak = float(vfree_on[peak_idx])

# 2) Space-charge-corrected potential at that (on-axis) index
Peak_Space_Charge_Pot = float(voltageGuess[0, peak_idx])

# 3) Space-charge-corrected potential at electrode border (on-axis)
idx_elec   = int(np.argmin(np.abs(z_axis - electrode_borders[1]))) 
z_nearest  = float(z_axis[idx_elec])
v_sc_near  = float(v_sc_on[idx_elec])

print("\n--- Peak / Left Electrode Potentials (on-axis) ---")
print(f"Free-space peak index (on-axis): {peak_idx}")
print(f"z at free-space peak: {peak_z:.6f} m")
print(f"Free-space potential at peak: {vfree_peak:.6f} V")
print(f"Space-charge corrected potential at peak index: {Peak_Space_Charge_Pot:.6f} V")
print(f"Nearest grid index: {idx_elec}")
print(f"Nearest grid z: {z_nearest:.6f} m")
print(f"Space-charge potential (nearest cell): {v_sc_near:.6f} V")

#Adjust initial_voltages so that potential drop ≈ 10 kT/e
thermal_voltage = 10 * kb * T_e / q_e # 1960 is the current T_e
current_drop = Peak_Space_Charge_Pot - v_sc_near
scaling_factor = thermal_voltage / current_drop

#Only scale non-zero voltages (ground stays at 0)
print(f"\n[Voltage Adjustment] Scaling factor = {scaling_factor:.3f}")
initial_voltages = initial_voltages * scaling_factor
final_voltages = final_voltages * scaling_factor
print(f"Adjusted initial voltages (V): {initial_voltages}")

print('Now proceeding to fine solution where the potential drops to 10kT/e.') #SOLVING FOR FINE SOLUTION - STEP 6 ON SLIDES
fine_sol=find_solution(NVal=N_e,T_e=T_e,fE=omega_r/(2*np.pi),mur2=rad2,B=B2,
                      electrodeConfig=(initial_voltages,electrode_borders),
                      left=Llim,right=Rlim,zpoints=40,rpoints=20,rfact=3.0,plotting=True, coarse_sol_divisor=100, InitializeWithPlasmaLength=False)

n=fine_sol[0]
voltageGuess=fine_sol[3]
vfree=fine_sol[4]
maxvolt=np.max(voltageGuess)
minvolt=np.min(voltageGuess)
maxN=np.max(n)
position_map_z=fine_sol[1]
position_map_r=fine_sol[2]
maxr=np.max(position_map_r)
dr=-position_map_r[0,0]+position_map_r[1,0]

#---END OF ADDED UPDATES---

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d') 
ax.set_zlim([-.5e+2,maxN*1.2])
plt.title("plasma density")
ax.plot_surface( np.flip(position_map_z,0),  position_map_r, np.flip(n,0),cmap=cm.coolwarm, linewidth=0, antialiased=False)    
ax.set_yticks([maxr+dr,maxr+dr-.001,maxr+dr-.002])
ax.set_yticklabels(["0","1","2"])
ax.set_ylabel("r (mm)")
ax.set_xlabel("z (m)")
ax.set_zlabel("rho (N/m^3)")
plt.show()
end_total_time = time.time()
print(f"\nTotal execution time: {end_total_time - start_total_time:.2f} seconds")

print(n)
    
def compute_kept_electrons(fine_sol, T_e):
    ngrid = fine_sol[0]
    full_scc_solution = fine_sol[3] #Full Space-Charge-Corrected (SCC) Solution, i.e. voltageGuess
    #position_map_z = fine_sol[1]
    volume_elements = fine_sol[7]
    N_cell = ngrid * volume_elements
    keep_sum = np.zeros(len(N_cell)) #len apparently returns the number of r values
    for r, x in enumerate(N_cell): 
        oneD_solution = full_scc_solution[r, :] #Solution across z-axis per radial point, r
        axial_well_idx = np.argmax(oneD_solution)
        barrier_idx = np.argmin(oneD_solution)
        escapeE = q_e * abs(oneD_solution[axial_well_idx] - oneD_solution[barrier_idx])
        E_int = 1.0 - erfc(np.sqrt(escapeE / (kb * T_e)))  #!!!should be difference of erf, not erfc
        keep_sum[r] = E_int * np.sum(N_cell[r, :]) #keep: these are the ones that stay in the well
    return np.sum(keep_sum)
    
#ramp step 1: N=NVal, np.sum(keep_sum) --> keep[1]
#ramp step 2: N=NVal, np.sum(keep_sum) --> keep[2]
#ramp step 3: N=NVal - (keep[1]-keep[2]), np.sum(keep_sum) --> keep[3]
#ramp step 4: N=NVal[3] - (keep[2]-keep[3]), np.sum(keep_sum) --> keep[4]
#ramp step 5: N=NVal[4] - (keep[3]-keep[4]), np.sum(keep_sum) --> keep[5]
#ramp step 6: N=NVal[5] - (keep[4]-keep[5]), np.sum(keep_sum) --> keep[6]
#etc.

# ===== ESCAPE CURVE LOOP USING KEEP_SUM METHOD ===== #

ramp_values = np.linspace(0, 0.45, 40)
kept_list = []
escaped_list = []
remaining_list = []
frac_escaped_list = []

N_initial = N_e  # total electrons at ramp start
N_current = N_initial

for i, rampfrac in enumerate(ramp_values):
    print(f"\n--- Rampfrac = {rampfrac:.3f} ---")
    print(f"Electrons entering this step: {N_current:.3e}")

    # Update electrode voltages based on ramp fraction
    current_voltages = initial_voltages + (final_voltages - initial_voltages) * rampfrac

    # Solve for plasma configuration at this ramp
    fine_sol = find_solution(
        NVal=N_current, T_e=T_e, fE=omega_r / (2 * np.pi),
        mur2=rad2, B=B2,
        electrodeConfig=(current_voltages, electrode_borders),
        left=Llim, right=Rlim,
        zpoints=40, rpoints=20,
        rfact=3.0, plotting=False,
        coarse_sol_divisor=100,
        InitializeWithPlasmaLength=False
    )

    # Compute number of electrons that stay trapped
    N_entering = N_current
    N_kept = compute_kept_electrons(fine_sol, T_e)
    kept_list.append(N_kept)

    if i == 0:
        N_escaped = N_initial - N_kept
    else:
        N_escaped = kept_list[i - 1] - kept_list[i]
    
    N_current = N_current - N_escaped

    escaped_list.append(N_escaped)
    remaining_list.append(N_current)
    frac_escaped = N_escaped / N_entering
    
    frac_escaped_list.append(frac_escaped)

    print(f"Ramp {rampfrac:.3f}: frac escaped = {frac_escaped:.3e}")
    print(f"Escaped this step: {N_escaped:.3e}")
    print(f"Remaining after step: {N_current:.3e}")

    if N_current < 1:
        print("Plasma fully escaped — stopping early.")
        break


# ===== PLOT ESCAPE CURVE ===== #

plt.figure(figsize=(7, 5))
#plt.plot(ramp_values[:len(remaining_list)], remaining_list, '-o', label="Remaining electrons")
plt.plot(ramp_values[:len(escaped_list)], escaped_list, '-o', label="Escaped per step")
plt.xlabel("Ramp fraction (0 = strong confinement, 1 = weak)")
plt.ylabel("Electrons")
plt.yscale("log")
plt.title("Plasma escape during ramp-down")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===== PLOT FRACTION ESCAPED PER STEP ===== #

plt.figure(figsize=(7, 5))
plt.plot(ramp_values[:len(frac_escaped_list)], frac_escaped_list, '-o', label="Fraction escaped per step")
plt.xlabel("Ramp fraction (0 = strong confinement, 1 = weak)")
plt.ylabel("Fraction escaped")
plt.title("Fraction of plasma escaped at each ramp step")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
