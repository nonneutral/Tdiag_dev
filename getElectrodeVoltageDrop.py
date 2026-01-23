import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.signal import savgol_filter
from find_solution import getFiniteSolution


def getElectrodeVoltageDrop(electrodeConfig, rpoints, zpoints, left, right, mur2, rfact):
    ''' Gets the combined Electrode "Excitement/Voltage Drop" Profile 
        without having to run the solver.
        
        Input: electrodeConfig: (voltages, borders)
        Returns: free_space_solution (2D array of "Voltage Drop" values)
    '''
    Mmax=1
    Nmax=20000
    zeros=[special.jn_zeros(m,Nmax) for m in range(Mmax)]

    nr=rpoints
    nz=zpoints
    leftbound=left
    rightbound=right
    position_map_r=np.zeros((nr,nz))
    position_map_z=np.zeros((nr,nz))
    rbound=mur2*rfact
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
    return free_space_solution