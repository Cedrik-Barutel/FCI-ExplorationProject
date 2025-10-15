#%% IMPORT
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import datetime
from matplotlib.animation import FuncAnimation
from scipy import integrate
from scipy.interpolate import UnivariateSpline
import pandas as pd
import scipy as scipy
import logging
import os
import csv

import sys
# local library
dir_local = os.path.dirname(__file__)
sys.path.append(dir_local)
lib_path = "/home/cedrik/Documents/filaments-crosslinkers-projects/lib"
sys.path.append(lib_path)

logger = logging.getLogger(__name__)

import lib_simulation as LS

#%% VARIABLES
folder_name = "1510_DimensionlessNumber" # name of the simulation 
working_path = os.path.dirname(__file__)
folder = working_path  + "/"+ folder_name

N_save = 200 # number of save
timestep = 1e-3 #timestep
stop_time = 5*60 # max simulating time 
Lx = 20 #length of the simulation box
Nx = 2**8


'Dimensions of the filaments'
# position of the filaments
xl_A = -7
xr_A = 3
xl_B = -3
xr_B = 7

LA = xr_A-xl_A
LB = xr_B-xl_B


Ly = LS.Coefficient("Ly",0.02) # Distance between the filaments
Lz = LS.Coefficient("Lz",0.02) # Witdh of the filaments
n_s = LS.Coefficient("n_s",200) # Lineic binsind site concentration
Eb_Pa = LS.Coefficient("Eb_Pa",0) # motors bound to (a)
Eb_Pb = LS.Coefficient("Eb_Pb",0) # motors bound to (b)
Eb_Pab = LS.Coefficient("Eb_Pab",0) # passive bound to (ab) 

l_0 = LS.Coefficient("l_0",1) # Witdh of the filaments
t_0 = LS.Coefficient("t_0",1) # Witdh of the filaments
D_0 = LS.Coefficient("D_0",l_0.v**2/t_0.v)
k_0 = LS.Coefficient("k_0",1/t_0.v)
V_0 = LS.Coefficient("V_0",l_0.v/t_0.v)


Dam1 = LS.Coefficient("Dam1",0) # always zero in our case
Dam2 = LS.Coefficient("Dam2",40) # DAMKOHLER NUMBER # we choose

Pe1 = LS.Coefficient("Pe1",0.1) # PECLET NUMBER # we choose
Pe2 = LS.Coefficient("Pe2",0) # always zero



D_Pa = LS.Coefficient("D_Pa",0.05) # motors bound to (a) The value we fix
D_Pb = LS.Coefficient("D_Pb",D_Pa.v) # motors bound to (b)

D_Pab = LS.Coefficient("D_Pab",D_Pa.v/2) # motors bound to (ab) The value we fix

v_A = LS.Coefficient("v_A", -1*Pe1.v*D_Pa.v*V_0.v/D_0.v) # Sliding velocity of filament A
v_B = LS.Coefficient("v_B", -v_A.v) # Sliding velocity of filament B



Koff_5_Pa = LS.Coefficient("Koff_5_Pa",Dam1.v*k_0.v*v_A.v/V_0.v/Pe1.v) # Rates of reaction 1 around Ma_eq
Koff_6_Pb = LS.Coefficient("Koff_6_Pb",Dam1.v*k_0.v*v_A.v/V_0.v/Pe1.v) # Rates of reaction 2 around Mb_eq
Koff_7_Pab = LS.Coefficient("Koff_7_Pab",Dam2.v*k_0.v/D_0.v*D_Pab.v) # Rates of reaction 3 around Mab_eq
Koff_8_Pab = LS.Coefficient("Koff_8_Pab",Dam2.v*k_0.v/D_0.v*D_Pab.v) # Rates of reaction 4 around Mab_eq


stop_time = -4/(v_A.v)
if stop_time>= 5*60:
    stop_time = 5*60


print("Dam1 = "+str(Dam1.v))
print("Pe1 = "+str(Pe1.v))
print("Dam2 = "+str(Dam2.v))
print("Pe2 = "+str(Pe2.v))

print("V = "+str(-v_A.v))
print("D_Pa= "+str(D_Pa.v))
print("D_Pab= "+str(D_Pab.v))

print("Koff_Pa = "+str(Koff_5_Pa.v))
print("Koff_Pab = "+str(Koff_8_Pab.v))


print()
print("Stop_time = "+str(stop_time))

#%%
h = 1e-3 # numerical help to avoid division by 0
h_log = 2*h # numerical help to log(0)

r = 0  # Stabilization with r*diffusion coefficient

# Phase field model 
li = 0.5 # length of the filaments interface
D_f = 0.2 # diffusion coefficient of the phase field
G_f = 1/18*li**2 # actual coefficient used


timestepper = d3.RK443 #time iteration scheme
dealias = 3/2 # anti aliasing factor




#%% DEDALUS BASIS, FUNCTION DEFINITION AND DEDALUS FIELD
# BUILDING DEDALUS COORDINATES AND BASIS
coords = d3.CartesianCoordinates('x')

dtype = np.float64
dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
x = dist.local_grids(xbasis)
ex = coords.unit_vector_fields(dist)

# DEFINING FUNCTIONS AND SUBTITUTION
dx = lambda A: d3.Differentiate(A, coords['x'])
ddx = lambda A: dx(dx(A))

# SETTING DEDALUS FIELDS

f_A = dist.Field(name='f_A',bases=(xbasis)) # filament A
f_B = dist.Field(name='f_B',bases=(xbasis)) # filament B
f_D = dist.Field(name='f_D',bases=(xbasis)) # Overlap 

Pa = dist.Field(name='Pa',bases=(xbasis)) # Concentration of 
Pb = dist.Field(name='Pb',bases=(xbasis)) # Concentration of 
Pab = dist.Field(name='Pab',bases=(xbasis)) # Concentration of 


# Equilibirum densities
Pa_eq_5 = dist.Field(name='Pa_eq_5',bases=(xbasis)) # Equilibrium Concentration of Ma 
Pb_eq_6 = dist.Field(name='Pb_eq_6',bases=(xbasis)) # Concentration of 
Pab_eq_7 = dist.Field(name='Pab_eq_7',bases=(xbasis)) # Concentration of
Pab_eq_8 = dist.Field(name='Pab_eq_8',bases=(xbasis)) # Concentration of  
 
# Velocities
V_A = dist.Field(name='V_A') # Velocity of A
V_B = dist.Field(name='V_B') # Velocity of B


#%% INITIAL CONDITIONS
'FILAMENTS PHASE FIELD'
f_A['g'] = LS.function_filament(f_A['g'], xl_A, xr_A, x[0], li)
f_B['g'] = LS.function_filament(f_B['g'], xl_B, xr_B, x[0], li)
f_D['g'] = np.minimum(f_A['g'],f_B['g'])

'NUMBER OF PARTICLE'
# 1st guess on the number of particle definition

h_log_d = 1e-5

n_ma = 0.05
n_mb = 0.05
n_mab = 0.1

n_pa = 0.1
n_pb = 0.1
n_pab = 0.3



Pab['g'] = n_pab*f_D['g']

Pa['g'] = n_pa*(f_A['g']-Pab['g'])
Pb['g'] = n_pb*(f_B['g']-Pab['g'])

Eb_Pa.v = -np.log(h_log_d+n_pa/(1-n_pa))
Eb_Pb.v = -np.log(h_log_d+n_pb/(1-n_pb))

Eb_Pab.v = -np.log(h_log_d+n_pab/(1-n_pab-n_pa*(1-n_pab))*(1-n_pab)/(1-n_pab-n_pa*(1-n_pab)))

print(Eb_Pa.v)
print(Eb_Pb.v)
print(Eb_Pab.v)


#%%

Pa_eq_5['g'] = (f_A['g']-Pab['g'])/(1+np.exp(Eb_Pa.v))
Pb_eq_6['g'] = (f_B['g']-Pab['g'])/(1+np.exp(Eb_Pb.v))
Pab_eq_7['g'] = 0.5*(1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B['g']-Pb['g'])))
Pab_eq_8['g'] = 0.5*(1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A['g']-Pa['g'])))

#%%
for i in range(200):    
    
    Pab_eq_7['g'] = 0.5*(1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B['g']-Pb['g'])))
    Pab_eq_8['g'] = 0.5*(1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A['g']-Pa['g'])))

    Pab_eq_7['g'][Pab_eq_7['g']<=0] =0
    Pab_eq_8['g'][Pab_eq_7['g']<=0] =0

    Pab['g'] = 0.5*(Pab_eq_7['g']+Pab_eq_8['g'])




#%%
# Pab['g']=0.3*f_D['g']-0.05*(x[0]-(xl_B+xr_B)/2)*f_D['g']
# Mab['g']=0.0*f_D['g']
# Ma['g']=0.0*f_D['g']
# Mb['g']=0.0*f_D['g']



#%%

# %%
plt.figure(dpi=200)
plt.title("Passive")
plt.plot(x[0],f_A['g'],label = "$f^{A}$")
plt.plot(x[0],f_B['g'],label = "$f^{B}$")
plt.plot(x[0],Pa['g'],label = "$P^{a}$")
plt.plot(x[0],Pb['g'],label = "$P^{b}$")
plt.plot(x[0],Pab['g'],label = "$P^{ab}$")
plt.plot(x[0],0.3*f_D['g'],label = "$test$")

plt.plot(x[0],Pa_eq_5['g'],label = "$P^{a}$",alpha = 0.3)
plt.plot(x[0],Pb_eq_6['g'],label = "$P^{b}$",alpha = 0.3)
plt.plot(x[0],Pab_eq_7['g'],label = "$P^{ab}$",alpha = 0.3)
plt.plot(x[0],Pab_eq_8['g'],label = "$P^{ab}$",alpha = 0.3)

plt.ylim(-0.1,1.1)
plt.legend()
plt.show()





# %% EQUATIONS OF THE PROBLEM
# it's better to write the full equations at once instead of using variables

problem = d3.IVP([ f_A,f_B,f_D,
                    Pa, Pb, Pab,
                    Pa_eq_5, Pb_eq_6, Pab_eq_7, Pab_eq_8,
                    V_A,V_B,
                    ],
                  namespace=locals()) # Declaration of the problem variables

# - Cahn Hillard equation for the filaments - #
problem.add_equation("dt(f_A) +D_f*ddx(-2*f_A +G_f*ddx(f_A)) = D_f*ddx(4*(f_A)**3-6*(f_A)**2) -dx(f_A*V_A)")
problem.add_equation("dt(f_B) +D_f*ddx(-2*f_B +G_f*ddx(f_B)) = D_f*ddx(4*(f_B)**3-6*(f_B)**2) -dx(f_B*V_B)")
problem.add_equation("f_D = f_A*f_B")

# - Equation of the particles - #

problem.add_equation("dt(Pa)"
                     "-D_Pa.v/D_0.v*r*ddx(Pa)" # stabilization
                     "-D_Pa.v/D_0.v*ddx(Pa)" # diffusion 
                     "+D_Pa.v/D_0.v*Dam1.v*(Pa -Pab)" # chemical reactions
                     "="
                     "-D_Pa.v/D_0.v*r*ddx(Pa)" # stabilization
                     "-D_Pa.v/D_0.v*dx(Pa/(h+f_A-Pab)*dx(f_A))" # 
                     "+D_Pa.v/D_0.v*dx(Pa/(h+f_A-Pab)*dx(Pab))" #
                     "+D_Pa.v/D_0.v*Pe1.v*dx(Pa)" # convected flux
                     "+D_Pa.v/D_0.v*Dam1.v*(Pa_eq_5 -Pab_eq_7)") # chemical reactions

problem.add_equation("dt(Pb)"
                     "-D_Pb.v/D_0.v*r*ddx(Pb)" # stabilization
                     "-D_Pb.v/D_0.v*ddx(Pb)" # diffusion 
                     "+D_Pb.v/D_0.v*Dam1.v*(Pb -Pab)" # chemical reactions
                     "="
                     "-D_Pb.v/D_0.v*r*ddx(Pb)" # stabilization
                     "-D_Pb.v/D_0.v*dx(Pb/(h+f_B-Pab)*dx(f_B))" # 
                     "+D_Pb.v/D_0.v*dx(Pb/(h+f_B-Pab)*dx(Pab))" #
                     "+D_Pb.v/D_0.v*Pe1.v*dx(-Pb)" # minus because v_B
                     "+D_Pb.v/D_0.v*Dam1.v*(Pb_eq_6 -Pab_eq_8)") # chemical reactions


problem.add_equation("dt(Pab)"
                      "-D_Pab.v/D_0.v*r*ddx(Pab)" # stabilization
                      "-D_Pab.v/D_0.v*ddx(Pab)" # diffusion 
                      "+D_Pab.v/D_0.v*Dam2.v*(Pab +Pab)" # chemical reactions
                      "="
                      "-D_Pab.v/D_0.v*r*ddx(Pab)" # stabilization
                      "-D_Pab.v/D_0.v*dx( (f_B/(h+f_D-Pab) +1/(h+f_A-Pab)-1/(h+f_A-Pab-Pa))/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(f_A))" # 
                      "-D_Pab.v/D_0.v*dx( (f_A/(h+f_D-Pab) +1/(h+f_B-Pab)-1/(h+f_B-Pab-Pb))/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(f_B))" # 
                      "-D_Pab.v/D_0.v*dx( 1/(h+f_A-Pab-Pa)/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(Pa))" # 
                      "-D_Pab.v/D_0.v*dx( 1/(h+f_B-Pab-Pb)/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(Pb))" # 
                      "" # convected flux
                      "+D_Pab.v/D_0.v*Dam2.v*(Pab_eq_7 +Pab_eq_8)") # chemical reactions


problem.add_equation("Pa_eq_5 = (f_A-Pab)/(1+np.exp(Eb_Pa.v))")
problem.add_equation("Pb_eq_6 = (f_B-Pab)/(1+np.exp(Eb_Pb.v))")

problem.add_equation("Pab_eq_7 = 0.5*(1+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B-Pb)))")
problem.add_equation("Pab_eq_8 = 0.5*(1+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A-Pa)))")



# - Velocities - #
problem.add_equation("V_A =  v_A.v")
problem.add_equation("V_B =  v_B.v")





#%%


#%% BUILDING SOLVER
solver = problem.build_solver(timestepper,ncc_cutoff=1e-4)
solver.stop_sim_time = stop_time


#%%
'Setting the paramters used and setting the save of the simulation'
date = datetime.datetime.now()
name = str(folder)

analysis = solver.evaluator.add_file_handler(folder, sim_dt=stop_time/N_save, max_writes=N_save)
analysis.add_tasks(solver.state, layout='g') # Save all variables of the problem
# analysis.add_task(n_A,layout = 'g',name = 'n_A') #Save a specific variable


ListCoeffSave = {obj for name, obj in globals().items() if isinstance(obj, LS.Coefficient)}
with open( folder+ "/" +"sparameters_"+folder_name+'.csv', 'w', newline='') as filecsv:
    fieldnames = ['name','value']
    writer = csv.DictWriter(filecsv, fieldnames=fieldnames) 
    writer.writeheader()
    for Coeff in ListCoeffSave:   
        LS.function_save_parameters(writer, fieldnames, Coeff)


# %% Starting the main loop
print("Start")
j=0
t=0
T_N0 = datetime.datetime.now()
while solver.proceed:
    t=t+1   
    # V_A['g'] = 0 
    # V_B['g'] = 0 

    solver.step(timestep) # solving the equations   

    if solver.iteration % int(stop_time/(N_save*timestep)) == 0 :
        j=j+1
        T_N1 = datetime.datetime.now()
        T_LEFT = (T_N1-T_N0)*(N_save-j)
        logger.info('%i/%i, T=%0.2e, t_left = %s' %(j,N_save,solver.sim_time,str(T_LEFT)))
        T_N0 = datetime.datetime.now()
        #print(V_B['g'][20])
        if j%1  == 0 and j<10 or j%10 == 0:
                f_A.change_scales(1)
                f_B.change_scales(1)
                f_D.change_scales(1)

                Pa.change_scales(1)
                Pb.change_scales(1)

                Pab.change_scales(1)


                plt.plot()
                plt.plot(x[0],f_A['g'],color = 'blue',alpha = 0.5)
                plt.plot(x[0],f_B['g'],color = 'red',alpha = 0.5)
                plt.plot(x[0],Pa['g'],color = 'blue',label = "$P^{a}$")
                plt.plot(x[0],Pb['g'],color = 'red',label = "$P^{b}$")
                plt.plot(x[0],Pab['g'],color = 'violet',label = "$P^{ab}$")
                
                plt.show()

#%%

# %% Getting the saved files
tasks = d3.load_tasks_to_xarray(folder +"/"+folder_name+"_s1.h5") # Downloadig the files
x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])
print(folder +"/"+folder_name+"_s1.h5")
print("\nduration:")
# print( T_N1-date)


#%%