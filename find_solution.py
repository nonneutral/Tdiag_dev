import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import special
from pylab import rcParams
from scipy.special import erf #import error function
from scipy.special import erfc

start_total_time = time.time()

rcParams['figure.figsize'] = 10, 6

# Physical Constants
kb=1.38064852e-23 #Boltzmann's constant in joules per kelvin
e0=8.854187817e-12 #farads per meter

# Simulation Parameters
q_e=1.60217662e-19 #electron charge in coulombs
m_e=9.1093837e-31 #electron mass in kilograms
rw=.017 #radius of inner wall of cylindrical electrodes, in meters

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
    potl_inf = (cpl*(-q_e)/(4*np.pi*e0))*(2*np.log(rw/rp)+1)
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

def find_solution(NVal,T_e,fE,mur2,B,electrodeConfig,left,right,zpoints,
                  rpoints,rfact,plotting, coarse_sol_divisor, InitializeWithPlasmaLength = False,
                  fail_action='raise', debug_tag=''):
    """
    find_solution: solves for the equilibrium of a non-neutral plasma
    
    returns: 0:ngrid, 1:position_map_z, 2:position_map_r, 3:voltageGuess, 4:free_space_solution, 5:rmean, 6:omega_r, 7:volume_elements
    """
    
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
        omega_r = .5*(omega_c-np.sqrt(omega_c*omega_c-4*quad_eq_c))

    print(f'omega_r = {omega_r}')

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
        
        # -------- FAIL-FAST GUARD #1: NaNs/Infs in exponential --------
        if not np.all(np.isfinite(exponential)):
            msg = f"[find_solution:{debug_tag}] non-finite exponential encountered (NaN/Inf)."
            if fail_action == "raise":
                raise RuntimeError(msg)
            return None

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
        
        vfree_on = free_space_solution[0,:]
        vsc_on = voltageGuess[0,:]
        peak_idx   = int(np.argmax(vfree_on))
        barrier_idx = int(np.argmin(vfree_on[0:peak_idx]))  # left barrier only may need to change if plasma shifts right
        drop = float(vsc_on[peak_idx] - vfree_on[barrier_idx])  # magnitude in volts
        
        isConfined = drop > 5*(kb*T_e)/q_e
        
        # -------- FAIL-FAST GUARD #2: total <= 0 or non-finite -> normalization will explode --------
        if (not np.isfinite(total)) or (total <= 0):
            msg = f"[find_solution:{debug_tag}] invalid total={total} after ROI truncation (likely ROI too small/empty)."
            if fail_action == "raise":
                raise RuntimeError(msg)
            return None
        
        nnew*=NVal/total
        
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
    return ngrid,position_map_z,position_map_r,voltageGuess,free_space_solution,rmean,omega_r,volume_elements, isConfined

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