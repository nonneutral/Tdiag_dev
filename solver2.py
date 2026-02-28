# Solver for plasma of known number of electrons (NVal), temperature (T_e), and rotation frequency (omega_r)
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import special
from pylab import rcParams
from scipy.special import erfc
from datetime import datetime

# Physical Constants
kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin
e0=8.854187817e-12 #farads per meter

# Simulation Parameters
q_e=1.60217662e-19 #electron charge in coulombs
m_e=9.1093837e-31 #electron mass in kilograms


rcParams['figure.figsize'] = 10, 6

Mmax=1
Nmax=20000
zeros=[special.jn_zeros(m,Nmax) for m in range(Mmax)]

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

def plasma_length_guess(N_e,mur2,rw,q_e,position_map_z,free_space_solution,electrode_borders):
    potl_barrier = free_space_solution[0,:].copy()
    electrode_centre = np.average(electrode_borders)
    inElectrode = (electrode_borders[1] < position_map_z[0])&(position_map_z[0] < electrode_borders[-2]) 
    rp = mur2
    rw = rw
    cpl = N_e/(electrode_borders[2]-electrode_borders[1])
    potl_inf = (cpl*(-q_e)/(4*np.pi*e0))*(2*np.log(rw/rp)+1)
    potl_inf = potl_inf - np.min(np.abs(potl_barrier[inElectrode])) 
    # need to make this more general (- sign must be removed) not dependent on charge sign
    potl_diff = np.abs(potl_inf - potl_barrier)
    plasma_left_end = position_map_z[0,np.argmin(potl_diff[position_map_z[0]<electrode_centre])]
    plasma_right_end = position_map_z[0,np.argmin(potl_diff[position_map_z[0]>electrode_centre])]+ electrode_centre
    plasma_length =  plasma_right_end - plasma_left_end
    
    plt.plot(position_map_z[0,:], potl_barrier, label='On-axis potential')
    plt.plot(position_map_z[0,:], np.full(len(position_map_z[0,:]), potl_inf), label='Infinite-length space charge potential')
    plt.show()

    return [plasma_left_end, plasma_right_end, plasma_length]

def find_solution(N_e,T_e,omega_r,mur2,B,electrodeConfig,left,right,rw,zpoints,
                  rpoints,rfact,plotting, coarse_sol_divisor, InitializeWithPlasmaLength = False,
                  fail_action='raise', debug_tag=''):
    """
    find_solution: solves for the equilibrium of a non-neutral plasma
    
    returns: 0:ngrid, 1:position_map_z, 2:position_map_r, 3:voltageGuess, 4:free_space_solution, 5:rmean, 6:omega_r, 7:volume_elements, 8:isConfined, 9:drop
    """
    nr=rpoints
    nz=zpoints
    omega_c=q_e*B/m_e
    omega_r=omega_r
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
        symmetricsolution=getFiniteSolution([cs],right-left,lambda r:voltage,rw=rw)
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
        lp_init_config = np.array(plasma_length_guess(N_e, mur2, rw, q_e, position_map_z, free_space_solution, electrode_borders))
        lp_init = lp_init_config[2]
        print(f'Initial plasma length estimate: {lp_init_config[2]:0.3e} m (from {lp_init_config[0]:0.3e} m to {lp_init_config[1]:0.3e} m)')
        #guesses omega_r based on the initial plasma length given
        quad_eq_c = (N_e*q_e**2)/(np.pi*lp_init*mur2*mur2*(2*m_e*e0))
        omega_r = .5*(omega_c-np.sqrt(omega_c*omega_c-4*quad_eq_c))

    print(f'omega_r = {omega_r}')
    #note: u_eff is an energy as defined here, not an electrostatic potential
    U_eff=np.array([[.5*m_e*omega_r*(omega_c-omega_r)*(rbound*rind/nr)**2 
                       for zind in range(nz)] for rind in range(nr)]) 
    volume_elements=np.array([[np.pi*((rbound*(rind+.5)/nr)**2-(rbound*max(0,rind-.5)/nr)**2)
                               *(rightbound-leftbound)/nz 
                                for zind in range(nz)] for rind in range(nr)])
    #----Manual initial guess for ngrid - UPDATED ON 23rd Oct, 2025----
    """
    exponential_guess = -((-q_e * free_space_solution) + U_eff) / (kb * T_e)
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
    
    # roi_init
    dz = (rightbound-leftbound)/nz
    free_on = free_space_solution[0,:]
    dVdz = np.gradient(free_on, dz)
    abs_dVdz = np.abs(dVdz)

    # make the quantile slice safe for small nz
    lo = min(10, max(1, nz // 10))
    hi = max(lo + 1, nz - 1)
    thr_slope = np.quantile(abs_dVdz[lo:hi], 0.05)

    is_flat_slope = abs_dVdz <= thr_slope
    d2Vdz2 = np.gradient(dVdz, dz)

    # ---- local MAXIMA (peak candidates) ----
    is_local_max = np.zeros(nz, dtype=bool)
    is_local_max[1:-1] = (free_on[1:-1] > free_on[:-2]) & (free_on[1:-1] >= free_on[2:])
    valid_max = is_local_max & is_flat_slope
    valid_max[:10] = False
    valid_max &= (d2Vdz2 < 0)  # concave down => maximum

    maxs = np.where(valid_max)[0]
    if maxs.size > 0:
        peak_idx_init = int(maxs[np.argmax(free_on[maxs])])
    else:
        peak_idx_init = int(np.argmax(free_on))  # fallback
  
    for i in range(np.int64(1e6)):
        voltageGuess=np.copy(free_space_solution)
        voltageGuess+=get_voltage_from_charge_distr(q_e*ngrid)
        exponential=-(voltageGuess*(-q_e)+U_eff)/(kb*T_e)
        
        # -------- FAIL-FAST GUARD #1: NaNs/Infs in exponential --------
        if not np.all(np.isfinite(exponential)):
            msg = f"[find_solution:{debug_tag}] non-finite exponential encountered (NaN/Inf)."
            if fail_action == "raise":
                raise RuntimeError(msg)
            return None
        #on axis potentials

        sc_on = voltageGuess[0,:]

        
        #roi calculation 
        
        
        #mx_exp = np.max(exponential_roi_init)
        roi_left_ind = np.argmin(exponential[0,:peak_idx_init])
        roi_left = position_map_z[0,:peak_idx_init][roi_left_ind]
        roi_right_ind = np.argmin(exponential[0,peak_idx_init:])
        roi_right = position_map_z[0,peak_idx_init:][roi_right_ind]
        plasma_length = roi_right - roi_left #plasma length estimate from current iteration
        
        mx=np.max(exponential)
        nnew=np.zeros((nr,nz))
        #nnew[exponential>mx-magic]=np.exp(exponential-mx)[exponential>mx-magic]
        nnew=np.exp(exponential-mx)
        nnew[position_map_z<roi_left]=0
        nnew[position_map_z>roi_right]=0
        

        total=np.sum(nnew*volume_elements)


        peak_idx = int(np.argmax(free_on[roi_left_ind:])+roi_left_ind) 
        #print(f"peak_idx inside find_solution= {peak_idx}")
        if peak_idx > 0:
            barrier_idx = int(np.argmin(sc_on[0:peak_idx]))  # left barrier only may need to change if plasma shifts right
            drop = sc_on[peak_idx] - sc_on[barrier_idx]
        else:
            drop = 0.0

        isConfined = drop > 1*kb*T_e/q_e

        # -------- FAIL-FAST GUARD #2: total <= 0 or non-finite -> normalization will explode --------
        if (not np.isfinite(total)) or (total <= 0):
            msg = f"[find_solution:{debug_tag}] invalid total={total} after ROI truncation (likely ROI too small/empty)."
            if fail_action == "raise":
                raise RuntimeError(msg)
            return None
        
        nnew*=N_e/total
        
        # -------- FAIL-FAST GUARD #3: nnew becomes non-finite after scaling --------
        if not np.all(np.isfinite(nnew)):
            msg = f"[find_solution:{debug_tag}] non-finite nnew after scaling by NVal/total."
            if fail_action == "raise":
                raise RuntimeError(msg)
            return None

        ngrid=ngrid*(1-epsilon)+epsilon*nnew
        
        # -------- FAIL-FAST GUARD #4: ngrid blows up --------
        if not np.all(np.isfinite(ngrid)):
            msg = f"[find_solution:{debug_tag}] non-finite ngrid after relaxation update."
            if fail_action == "raise":
                raise RuntimeError(msg)
            return None

        err=np.sum(np.abs((ngrid-nnew)*volume_elements))
        
        # -------- FAIL-FAST GUARD #5: err becomes NaN/Inf (then break condition never triggers) --------
        if not np.isfinite(err):
            msg = f"[find_solution:{debug_tag}] non-finite err; solver would never satisfy err < threshold."
            if fail_action == "raise":
                raise RuntimeError(msg)
            return None

        if  i%100==0:
            if plotting:
                # print(f'iteration = {i:0.1e}',
                #       f'\t fraction of gridpoints with charge = {np.sum(ngrid!=0)/np.sum(ngrid>=-1):0.2f}',
                #       f'\t misplaced electrons = {err:0.1e}'
                #       )
                plt.title("on-axis potential")
                plt.plot(position_map_z[0,:],free_space_solution[0,:],label="solver electrode potential")
                plt.plot(position_map_z[0,:],voltageGuess[0,:],label="charge-corrected potential")
                plt.scatter(position_map_z[0,barrier_idx],voltageGuess[0,barrier_idx], label="barrier",color="r",marker=10)
                plt.scatter(position_map_z[0,peak_idx],voltageGuess[0,peak_idx], label="peak",color="r",marker=10)
                plt.ylabel("potential (V)")
                plt.xlabel("position (m)")
                plt.legend()
                plt.show()
            #if err<np.sum(NVal):
                #epsilon=epsapprox*np.sum(ngrid>=np.max(ngrid)/2)/np.sum(ngrid>=-1)
            if err<N_e/coarse_sol_divisor:
                break
            if isConfined == False:
                break

    print(f"drop = {drop}")

    NS=ngrid*volume_elements
    rmean=np.sqrt(np.sum(mursq*NS)/(np.sum(NS)))*rbound/nr
    phi=np.max(free_on)-np.max(sc_on)




    print('N, T, φ, ρ, r_0, λ_D, l_p')
    print(f'{N_e:0.3e}\t{T_e:0.3e}\t{phi:0.3e}\t{n0:0.3e}\t{rmean:0.3e}\t{debye_length:0.3e} \t{plasma_length:0.3e}')

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
    return ngrid,position_map_z,position_map_r,voltageGuess,free_space_solution,rmean,omega_r,volume_elements,isConfined,drop,roi_left_ind

def retune_omega_iteration(omega_r, r_mean, r_target): #for step 4
    r_ratio_sq = (r_mean / r_target)**2
    omega_r_new = r_ratio_sq * omega_r
    return omega_r_new

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


def find_omega_r(plasma_config, electrode_input):
    """
    (2)  human (eventually ML) guesses the rotation rate ω_r and 2D density profile n(r,z) for initial excitations
    (3)  coarse plasma solve (δN/N ~ 10%)
    repeat (2)-(3) until r_p from solver is within 10% of measured r_p

    :param N_e: Number of particles
    :param T_e: Temperature
    :param omega_r: initial omega_r guess
    :param rad2: target radius
    :param B2: axial magnetic field
    :param initial_voltages: initial voltages on electrodes
    :param electrode_borders: electrode border positions
    :param Llim: left boundary
    :param Rlim: right boundary
    :return: final sol with omega_r that achieves target radius within 1% tolerance or None if not reachable
    """
    N_e, T_e, omega_r, rad2, B2 = plasma_config
    initial_voltages,final_voltages,electrode_borders,Llim,Rlim,rw = electrode_input
    print("---------omega_r retuning--------------------")

    for _ in range(8):  #COARSE LOOP - range(number) is just number of iterations to try
        if _ == 0:
            initialse_using_pl = True
        else: 
            initialse_using_pl = False
        sol = find_solution(N_e=N_e,T_e=T_e,omega_r=omega_r,mur2=rad2,B=B2,
                            electrodeConfig=(initial_voltages,electrode_borders),
                            left=Llim,right=Rlim,rw=rw,zpoints=40,rpoints=20,rfact=3.0,plotting=True, coarse_sol_divisor=50, InitializeWithPlasmaLength = initialse_using_pl)
        omega_r = sol[6]
        r_mean = sol[5]     #returned rmean
        vfree = sol[4]   #returned free_space_solution
        print(f'potential-to-kT ratio: {np.max(-q_e*vfree)/(kb*T_e):0.2f}')
        if abs(r_mean - rad2) <= 0.01 * rad2:
            print("Desired radius achieved within 1% tolerance.")
            break
        omega_new = retune_omega_iteration(omega_r, r_mean, rad2) #using funciton to retune omega_r and hit traget radius.
        if omega_new is None:
            print("Target radius not reachable with current parameters.")
            break
        omega_r = omega_new
    
    print("--- omega_r retuning complete ---")
    return omega_r
#--- STEP 5 ---
# We scan rampfrac to map rampfrac to drop. We use linear interpolation to speed up finding rampfrac for target drop.
def drop_for_rampfrac(plasma_config, electrode_input, rf): # finds "drop", i.e. potential difference between plasma and barrier, for a given rampfrac
    
    N_e, T_e, omega_r, rad2, B2 = plasma_config
    initial_voltages, final_voltages, electrode_borders, Llim, Rlim, rw = electrode_input
    initial_voltages = np.array(initial_voltages)
    final_voltages = np.array(final_voltages)
    electrode_borders = np.array(electrode_borders)
    Llim = float(Llim)
    Rlim = float(Rlim)
    rw = float(rw)
    volts = np.array(initial_voltages) + (np.array(final_voltages) - np.array(initial_voltages)) * rf # volts along the ramp
    sol_tmp = find_solution(
        N_e=N_e, T_e=T_e, omega_r=omega_r, mur2=rad2, B=B2,
        electrodeConfig=(volts, electrode_borders),
        left=Llim, right=Rlim, rw=rw,
        zpoints=80, rpoints=40, rfact=3.0,
        plotting=True, coarse_sol_divisor=50,
        InitializeWithPlasmaLength=False, fail_action='return_none', debug_tag=f"rampfrac={rf:.3f}"
    ) # a "rough" solve
    
    if sol_tmp is None:
        return np.inf, volts, None
    isConfined = sol_tmp[8]
    drop = sol_tmp[9]
    #vfree_tmp = sol_tmp[4]          # free_space_solution
    #vsc_tmp   = sol_tmp[3]          # voltageGuess (space-charge corrected)
    #vfree_on  = vfree_tmp[0, :]
    #vsc_on    = vsc_tmp[0, :]
    print(f'rf: {rf:.4f}')

    
    #peak_idx   = int(np.argmax(vfree_on))
    #barrier_idx = int(np.argmin(vsc_on[0:peak_idx]))  # left barrier only may need to change if plasma shifts right
    #drop = float(vsc_on[peak_idx] - vfree_on[barrier_idx])  # magnitude in volts

    #if vsc_on[peak_idx] - vfree_on[barrier_idx] < 0:
    #    print(f"  Warning: negative drop ({drop:.3f} V) for rampfrac={rf:.4f}; plasma may be escaping.")
    return drop, isConfined

def coarse_scan(plasma_config, electrode_input, scan_points=21,interp_points=10000):
    grid = np.linspace(0.0, 1.0, scan_points) # 21 is number of points to scan arbitrarily chosen
    drops = []
    for rf in grid:
        d, isConfined = drop_for_rampfrac(plasma_config, electrode_input, rf)
        if isConfined == False:
            print("Plasma not confined; ending scan.")
            break
        else:
            drops.append(d)
    grid = grid[0:len(drops)]

    drops = np.array(drops)
    grid_interp = np.linspace(0.0,1.0,interp_points)
    drops_fit = np.polyfit(grid, drops, deg=1)
    drops_interp = np.polyval(drops_fit, grid_interp)


    plt.title("drop vs rampfracs coarse scan with interpolation")
    plt.plot(grid, drops,"o", label="coarse scan")
    plt.plot(grid_interp, drops_interp, "-", label="interp_points = 10000")
    plt.legend()
    plt.show()
    return grid, drops, grid_interp, drops_interp
def find_rf_for_target_drop(grid_interp,drops_interp,target_drop,interp_points=10000):
    """
    finds rampfrac such that drop_for_rampfrac(rampfrac*) == target_drop using linear interpolation method
    
    returns: (rampfrac where target_drop is reached, achieved_drop)
    """
    if grid_interp is None or drops_interp is None:
        raise RuntimeError("coarse scan data not available; run coarse_scan() first.")

    rf_star = grid_interp[np.argmin(abs(drops_interp - target_drop))]
    achieved_drop = drops_interp[np.argmin(drops_interp - target_drop)]

    return np.array([rf_star, achieved_drop])



def find_fine_solution(plasma_config,electrode_input,zpoints,
                  rpoints,target_drop,rfact=3.0,plotting=True, coarse_sol_divisor=100, grid_interp=None, drops_interp=None):
    """
    Plasma solution with retuned omega_r based on target radius (rad2) 
    and rampfrac based on target potential drop.
    """

    N_e, T_e, omega_r, mur2, B2 = plasma_config
    initial_voltages, final_voltages, electrode_borders, Llim, Rlim, rw = electrode_input

    print("--- STEP 4 ---")
    omega_r = find_omega_r(plasma_config, electrode_input)
    
    print("--- STEP 5 ---")
    rf_star = find_rf_for_target_drop(grid_interp,drops_interp,target_drop)[0]
    
    current_voltages = np.array(initial_voltages) + (np.array(final_voltages) - np.array(initial_voltages)) * rf_star
    
    print("--- Finding fine solution---")
    fine_sol = find_solution(N_e, T_e, omega_r, mur2, B2, 
        electrodeConfig=(current_voltages, electrode_borders),
        left=Llim, right=Rlim, rw=rw,
        zpoints=zpoints, rpoints=rpoints, rfact=rfact,
        plotting=plotting, coarse_sol_divisor=coarse_sol_divisor,
        InitializeWithPlasmaLength=False,
        fail_action='raise', debug_tag='fine_solution'
    )
    print("---Fine solution found---")
    return fine_sol

def plot_density(sol):

    fine_sol = sol
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

#%%--- STEP 7 ---
#--------ESCAPE CURVE SCAN - STEP 7 ON SLIDES--------#
def compute_esc_electrons(fine_sol, T_e): #, N_now, lastescapeE
    ngrid = fine_sol[0]
    full_scc_solution = fine_sol[3] #Full Space-Charge-Corrected (SCC) Solution, i.e. voltageGuess
    #position_map_z = fine_sol[1]
    volume_elements = fine_sol[7]
    free_space_solution = fine_sol[4]
    roi_left_ind = fine_sol[10] #gets ROI from fine_sol
    N_cell = ngrid * volume_elements
    #N_cell = N_cell*N_now/np.sum(N_cell) #normalize grid of number of e- per cell so it sums to N_now
    escape_sum = np.zeros(len(N_cell[:,0])) #len apparently returns the number of r values     escapeE = np.zeros(len(N_cell))    
    print(f"sum(ngrid*volume_elements) = {np.sum(N_cell):.3f} ---")
    for r, x in enumerate(N_cell): 
        oneD_solution = full_scc_solution[r, :] #Solution across z-axis per radial point, 
        oneD_free = free_space_solution[r, :]

        axial_well_idx = np.argmax(oneD_free[roi_left_ind:])+roi_left_ind  # index of axial well peak. skips region outside of ROI 
        barrier_idx = np.argmin(oneD_solution[0:axial_well_idx])  # left barrier only
        
        dropnow = abs(oneD_solution[axial_well_idx] - oneD_solution[barrier_idx]) 
        escapeE = q_e * dropnow
        E_int = erfc(np.sqrt(escapeE / (kb * T_e)))
        escape_sum[r] = E_int * np.sum(N_cell[r, :])
         #these are the ones that leave the well from that r
        #changed to oneD_drops 02/02/2026
        if r==0:
            onaxis_drop = dropnow
            barrier2_idx = np.argmin(oneD_free[0:axial_well_idx])  # left barrier only
            vacuum_drop = abs(oneD_free[axial_well_idx] - oneD_free[barrier2_idx]) 
    #print(f"escape_sum: {escape_sum}")
    return np.sum(escape_sum),onaxis_drop,vacuum_drop

def escape_curve_scan(plasma_config, electrode_input, rampfrac_start, rampfrac_end, data_points):

    # ===== ESCAPE CURVE LOOP USING KEEP_SUM METHOD ===== #
    #to be updated for consistency with new definition of compute_kept_electrons
    ramp_values = np.linspace(rampfrac_start, rampfrac_end, data_points) # find rampfrac_end 
    #!!! instead of hard-coding this linspace, make it go from rampfrac equiv of 20 kT/e to 0 kT/e
    #you can do this by finding the points for 20 kT/e and 10 kT/e and extrapolating
    N_e, T_e, omega_r, rad2, B2 = plasma_config
    initial_voltages,final_voltages,electrode_borders,Llim,Rlim,rw = electrode_input
    escaped_list = []
    remaining_list = []
    frac_escaped_list = []
    drop_list = []
    vacdrop_list = []

    N_current = N_e  # total electrons at ramp start

    for i, rampfrac in enumerate(ramp_values):
        nowtime=str(datetime.now())
        
        print(f'{nowtime}\t Step {i}')
        print(f"\n--- Rampfrac = {rampfrac:.3f} ---")
        print(f"Electrons entering this step: {N_current:.3e}")
        # Update electrode voltages based on ramp fraction
        current_voltages = initial_voltages + (final_voltages - initial_voltages) * rampfrac

        # Solve for plasma configuration at this ramp
        fine_sol = find_solution(
            N_e=N_current, T_e=T_e, omega_r=omega_r,
            mur2=rad2, B=B2,
            electrodeConfig=(current_voltages, electrode_borders),
            left=Llim, right=Rlim, rw=rw,
            zpoints=80, rpoints=40,
            rfact=3.0, plotting=False,
            coarse_sol_divisor=100,
            InitializeWithPlasmaLength=False
        ) #now more precise

        # Compute number of electrons that stay trapped
        N_entering = N_current
        N_erfc,onaxis_drop,vacuum_drop = compute_esc_electrons(fine_sol, T_e)
        
        if not fine_sol[8]: #breaks when isConfined==False
            print("Plasma not confined -- stopping early.")
            break
        
        escaped_list.append(N_erfc)

        if i == 0:
            N_escaped = 0
        else:
            N_escaped = escaped_list[i] - escaped_list[i - 1]
        
        N_current = N_current - N_escaped

        #escaped_list.append(N_escaped)
        remaining_list.append(N_current)
        frac_escaped = N_escaped / N_entering
        
        frac_escaped_list.append(frac_escaped)
        drop_list.append(onaxis_drop)
        vacdrop_list.append(vacuum_drop)
        

        print(f"Ramp {rampfrac:.3f}: frac escaped = {frac_escaped:.3e}, on-axis confinement ('drop') = {onaxis_drop:.3e}")
        print(f"Escaped this step: {N_escaped:.3e}")
        print(f"Remaining after step: {N_current:.3e}")

        if N_current < 1:
            print("Plasma fully escaped — stopping early.")
            break
    ramp_values = ramp_values[:len(frac_escaped_list)]
    return ramp_values, escaped_list, remaining_list, frac_escaped_list, drop_list, vacdrop_list

#%%===== PLOT ESCAPE CURVE ===== #
def plot_escape_curve(ramp_values, escaped_list, frac_escaped_list, drop_list, yscale='log'):

    plt.figure(figsize=(7, 5))
    #plt.plot(ramp_values[:len(remaining_list)], remaining_list, '-o', label="Remaining electrons")
    plt.plot(drop_list[:len(escaped_list)+1], escaped_list, '-o', label="Escaped per step")
    plt.xlabel("Confinement ('drop') in volts")
    plt.ylabel("Electrons")
    plt.yscale(yscale)
    plt.title("Plasma escape during ramp-down")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===== PLOT FRACTION ESCAPED PER STEP ===== #

    plt.figure(figsize=(7, 5))
    plt.plot(ramp_values[:len(frac_escaped_list)], np.log10(frac_escaped_list), '-o', label="Fraction escaped per step")
    plt.xlabel("Ramp fraction (0 = strong confinement, 1 = weak)")
    plt.ylabel("Fraction escaped (log10)")
    plt.title("Fraction of plasma escaped at each ramp step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- STEP 8 ---
#%% Estimate temperature from escape curve

def linear_model_T_diag(escaped_list, drop_list, title, xlabel_str, saveplotttitle, crop_factor_input = 0.5, plotting = True):
    """
    Linear fit model for temperature estimation from escape curve data.
    
    :param escape_list: List of escaped electrons at each drop
    :param drop_list: List of confinement ('drop') values in volts
    :return: Estimated temperature in Kelvin
    """
    crop_factor=crop_factor_input
    escape_sublist=escaped_list[0:int(len(escaped_list)*crop_factor)]
    drop_sublist=drop_list[0:int(len(drop_list)*crop_factor)]
    #Linear temperature estimate from escape curve using
    #equation 8 from Eggleston 1992 paper
    dnep = np.log(escape_sublist[-1]/escape_sublist[0]) 
    dV = drop_sublist[-1]-drop_sublist[0]
    slope = dnep/dV
    T_estimate = -q_e*1.05/(kb*slope)
    print(f"Estimated temperature from escape curve (last-first): {T_estimate:.2f} K")
    #Quick Check: Linear fit - but with fitting
    curvefit,cov = np.polyfit(drop_sublist, np.log(escape_sublist), 1, cov=True)
    slope = curvefit[0]
    T_estimate2 = -q_e*1.05/(kb*slope)
    print(f"Estimated temperature from escape curve (polyfit): {T_estimate2} K")
    print(f"cov: {cov}")
    errors = np.sqrt(np.diag(cov))
    print(f"err of slope: {errors[0]}")

    if plotting == True:
        plt.figure(figsize=(7, 5))
        plt.gca().invert_xaxis()
        plt.plot(drop_list, np.log10(escaped_list), '.', label="Solver data", ms=8.0,color="#ff1493")
        plt.plot(drop_list, np.polyval(curvefit/np.log(10), drop_list), '-', label="Linear fit", color="#1418E2",ms=9.0)
        plt.xlabel(xlabel_str, fontsize=18)
        plt.ylabel(r"$\log(N_{\text{esc}}) ~ \text{/} ~ \text{A.U.}$", fontsize=18)    
        plt.legend(fontsize=18)
        plt.grid(True, "both")
        plt.savefig(f'Escape_plot {saveplotttitle}.png', transparent=True)
        plt.show()
    return T_estimate2,errors[0]

# %% loop function

def evaporative_protocol(plasma_config,electrode_input,start_drop,end_drop,d_points,initial_scan_points):
    starttime=str(datetime.now())
    start_stopwatch = time.time()

    print("--- solver_copy.py loaded ---")


    # Input 
    N_e, T_e, omega_r, rad2, B2 = plasma_config #number of electrons
    initial_voltages, final_voltages, electrode_borders, Llim, Rlim, rw = electrode_input

    # We iterate to find omega_r that achieves target radius within 1% tolerance

    omega_r = find_omega_r(plasma_config, electrode_input)
    plasma_config[2] = omega_r

    print("--- Performing coarse scan to map rampfrac to drop ---")
    grid, drops, grid_interp, drops_interp = coarse_scan(plasma_config, electrode_input, scan_points=initial_scan_points)
    target_drop_eg = 10 * kb * T_e / q_e  # volts (this is 10 kT/e)
    rf, achieved_drop = find_rf_for_target_drop(grid_interp,drops_interp,target_drop_eg, interp_points=10000)
    current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages) * rf

    print(f"--- Finding fine solution for target drop of {target_drop_eg:.3f} V ---")
    fine_sol = find_solution(*plasma_config,electrodeConfig = [current_voltages,electrode_borders],
                            left=Llim,right=Rlim,rw=rw,zpoints=80,rpoints=40,
                            rfact=3.0,plotting=True,coarse_sol_divisor=100)


    plot_density(fine_sol)


    rampfrac_start = find_rf_for_target_drop(grid_interp,drops_interp,start_drop)[0]
    rampfrac_end = find_rf_for_target_drop(grid_interp,drops_interp,end_drop)[0]

    print(f"scanning from rampfrac {rampfrac_start:.3f} (drop {start_drop:.2f} V) to {rampfrac_end:.3f} (drop {end_drop:.2f} V)")
    escape_curve_data = escape_curve_scan(plasma_config,
                                        electrode_input,
                                        rampfrac_start, 
                                        rampfrac_end, 
                                        data_points=d_points)

    ramp_values = escape_curve_data[0]
    escaped_list = escape_curve_data[1]
    remaining_list = escape_curve_data[2]
    frac_escaped_list = escape_curve_data[3]
    drop_list = escape_curve_data[4]
    vacdrop_list = escape_curve_data[5]

    plot_escape_curve(ramp_values, escaped_list, frac_escaped_list, drop_list, yscale='linear')

    # --- STEP 8 ---
    # Estimate temperature from escape curve
    print(f"inferred using drop list: {drop_list}")
    T_inferred,_ = linear_model_T_diag(escaped_list, drop_list,"Log(Escaped electrons) vs Confinement with Linear Fit", 
                                     xlabel_str=r"confinement voltage ('drop') / V",
                            saveplotttitle="Escape_plot_drop",
                            crop_factor_input=0.591)
    #print(f"\nInferred Temperature from Escape Curve: {T_inferred:.2f} K")
    print(f"Actual Temperature: {T_e:.2f} K")
    print(f"Percentage Error: {abs(T_inferred - T_e) / T_e * 100:.2f}%")

    print(f"inferred using vacdrop list: {vacdrop_list}")
    Tvac,_ = linear_model_T_diag(escaped_list, vacdrop_list,"Log(Escaped electrons) vs Confinement with Linear Fit",
                               xlabel_str="confinement voltage / V",
                               saveplotttitle="Escape_plot_vac",
                               crop_factor_input=0.591)
    print(f"Actual Temperature: {T_e:.2f} K")
    print(f"Percentage Error: {abs(Tvac - T_e) / T_e * 100:.2f}%")

    end_stopwatch = time.time()
    print(f"\nTotal execution time: {end_stopwatch - start_stopwatch:.2f} seconds")
    stoptime=str(datetime.now())
    print(f'{starttime}\tExecution start')
    print(f'{stoptime}\tExecution finish')

    return ramp_values, escaped_list, frac_escaped_list, drop_list, vacdrop_list

# %%