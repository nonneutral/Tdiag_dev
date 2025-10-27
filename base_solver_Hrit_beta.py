# Solver for plasma of known number of electrons (NVal), temperature (T_e), and rotation frequency (fE)
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import special
from pylab import rcParams

start_total_time = time.time()

rcParams['figure.figsize'] = 10, 6

rw=.017 #radius of inner wall of cylindrical electrodes, in meters
q_e=1.60217662e-19
m_e=9.1093837e-31
kb=1.38064852e-23
e0=8.854187817e-12#farads per meter
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

def find_solution(NVal,T_e,fE,mur2,B,electrodeConfig,left,right,zpoints,rpoints,rfact,plotting, coarse_sol_divisor):
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

    phieff=np.array([[.5*m_e*omega_r*(omega_c-omega_r)*(rbound*rind/nr)**2 
                       for zind in range(nz)] for rind in range(nr)]) 
    volume_elements=np.array([[np.pi*((rbound*(rind+.5)/nr)**2-(rbound*max(0,rind-.5)/nr)**2)
                               *(rightbound-leftbound)/nz 
                                for zind in range(nz)] for rind in range(nr)])
    #----Manual initial guess for ngrid - UPDATED ON 23rd Oct, 2025----
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
    n0=2*m_e*e0*omega_r*(omega_c-omega_r)/(q_e*q_e)
    debye_length=np.sqrt((e0*kb*T_e)/(q_e*q_e*n0))
    a=rbound
    b=(rightbound-leftbound)
    lambdac=-1/(((np.pi/(2*b))**2+(zeros[0][0]/a)**2)*debye_length*debye_length)
    #print('what is lambda c',lambdac)
    epsapprox=1/(2-lambdac)
    epsilon=epsapprox
    magic=60
    
    for i in range(np.int64(1e6)):
        voltageGuess=np.copy(free_space_solution)
        voltageGuess+=get_voltage_from_charge_distr(q_e*ngrid)
        exponential=-(voltageGuess*(-q_e)+phieff)/(kb*T_e)
        mx=np.max(exponential)
        nnew=np.zeros((nr,nz))
        nnew[exponential>mx-magic]=np.exp(exponential-mx)[exponential>mx-magic]
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
    print('N, φ, ρ, r_0, λ_D')
    print(f'{NVal:0.3e}\t{T_e:0.3e}\t{phi:0.3e}\t{n0:0.3e}\t{rmean:0.3e}\t{debye_length:0.3e}')

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

    return ngrid,position_map_z,position_map_r,voltageGuess,free_space_solution,rmean

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
initial_voltages=np.array([0,-50,-10,-50,0])
final_voltages=np.array([0,-15,-10,-50,0])
electrode_borders=[0.025,0.050,0.100,0.125]
Llim=0.035
Rlim=0.100
rampfrac=0.9
#---ADDED VARIABLES FOR COARSE LOOP---
B=1.6
mur2=0.00165
freq_guess = 1.0e5  # initial guess for rotation frequency in Hz
omega_c = q_e*B/m_e
omega_r  = 2*np.pi*freq_guess
#---END OF ADDED VARIABLES FOR COARSE LOOP---
current_voltages=np.array(initial_voltages) + (final_voltages-initial_voltages)*rampfrac

#---FUNCTION TO RETUNE OMEGA_R TO HIT TARGET RADIUS; STEP 4 ON SLIDES - UPDATED 25th OCT---
def retune_omega_to_hit_radius(omega_r, omega_c, r_mean, r_target, m=m_e):
    '''Defining effective solution as kappa and using that it scales as r^-2 with 
       a constant T_e, we can tune omega_r to hit the target radius'''
    kappa_current = 0.5 * m * omega_r * (omega_c - omega_r)
    kappa_target = kappa_current * (r_mean / r_target)**2
    disc = omega_c**2 - 8.0 * kappa_target / m
    if disc <= 0:
        # target not reachable with current N, Te, electrodes - worst case scenario
        return None
    return 0.5 * (omega_c - np.sqrt(disc))

for _ in range(8):  #COARSE LOOP - range(number) is just number of iterations to try
    sol = find_solution(NVal=8.0e6,T_e=1960,fE=omega_r/(2*np.pi),mur2=0.00165,B=1.6,
                        electrodeConfig=(initial_voltages,electrode_borders),
                        left=Llim,right=Rlim,zpoints=40,rpoints=20,rfact=3.0,plotting=True, coarse_sol_divisor=50)
    r_mean = sol[-1]     #returned rmean
    vfree = sol[4]   #returned free_space_solution
    print(f'potetnial-to-kT ratio: {np.max(-q_e*vfree)/(kb*1960):0.2f}')
    if abs(r_mean - mur2) <= 0.10 * mur2:
        print("Desired radius achieved within 10% tolerance.")
        break
    omega_new = retune_omega_to_hit_radius(omega_r, omega_c, r_mean, mur2) #using funciton to retune omega_r and hit traget radius.
    if omega_new is None:
        print("Target radius not reachable with current parameters.")
        break
    omega_r = omega_new

print('Now proceeding to fine solution.') #SOLVING FOR FINE SOLUTION - STEP 6 ON SLIDES
fine_sol=find_solution(NVal=8.0e6,T_e=1960,fE=1.0e5,mur2=0.00165,B=1.6,
                      electrodeConfig=(initial_voltages,electrode_borders),
                      left=Llim,right=Rlim,zpoints=40,rpoints=20,rfact=3.0,plotting=True, coarse_sol_divisor=100)
#---END OF ADDED UPDATES---

n=fine_sol[0]
voltageGuess=fine_sol[3]
maxvolt=np.max(voltageGuess)
minvolt=np.min(voltageGuess)
maxN=np.max(n)
position_map_z=fine_sol[1]
position_map_r=fine_sol[2]
maxr=np.max(position_map_r)
dr=-position_map_r[0,0]+position_map_r[1,0]

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