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

logger = logging.getLogger(__name__)

import lib_simulation as LS

#%% VARIABLES
folder_name = "Test_2208" # name of the simulation 
working_path = os.path.dirname(__file__)
folder = working_path  + "/"+ folder_name

N_save = 1000 # number of save
timestep = 2e-3 #timestep
stop_time = 25 # max simulating time 
Lx = 27 #length of the simulation box
Nx = 2**13


'Dimensions of the filaments'
# position of the filaments
xl_A = -20
# xr_A = 0.77
# xl_B = -0.77
xr_A = 0
xl_B = -17
xr_B = xl_B+6

LA = xr_A-xl_A
LB = xr_B-xl_B


Ly = LS.Coefficient("Ly",0.02) # Distance between the filaments
Lz = LS.Coefficient("Lz",0.02) # Witdh of the filaments

n_s = LS.Coefficient("n_s",200) # Lineic binsind site concentration

Eb_D = LS.Coefficient("Eb_D",-1) # Binding energy 

D_D = LS.Coefficient("D_D",0.01) # diffusion coefficient 

k_off_D = LS.Coefficient("k_off_D",0.001) # OFF rate of chemical reaction 1
eta = LS.Coefficient("eta",250) # Shear viscosity
gamma = LS.Coefficient("gamma",1) # Viscous friction
G = LS.Coefficient("G",1) # Elastic relaxation time
E = LS.Coefficient("E",1) # Young modulus
act = LS.Coefficient("act",-400) # Activity coefficient




#%%
h = 1e-3 # numerical help to avoid division by 0
h_log = 1e-3 # numerical help to log(0)

r = 0  # Stabilization with r*diffusion coefficient

# Phase field model 
li = 0.1 # length of the filaments interface
D_f = 0.2 # diffusion coefficient of the phase field
G_f = 1/18*li**2 # actual coefficient used


timestepper = d3.RK443 #time iteration scheme
dealias = 3/2 # anti aliasing factor


#%% DEDALUS BASIS, FUNCTION DEFINITION AND DEDALUS FIELD
# BUILDING DEDALUS COORDINATES AND BASIS
coords = d3.CartesianCoordinates('x')

dtype = np.float64
dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-21, -21+Lx), dealias=dealias)
x = dist.local_grids(xbasis)
ex = coords.unit_vector_fields(dist)

# DEFINING FUNCTIONS AND SUBTITUTION
dx = lambda A: d3.Differentiate(A, coords['x'])
ddx = lambda A: dx(dx(A))

# SETTING DEDALUS FIELDS

f_A = dist.Field(name='f_A',bases=(xbasis)) # filament A
f_B = dist.Field(name='f_B',bases=(xbasis)) # filament B
f_D = dist.Field(name='f_D',bases=(xbasis)) # Overlap 


n_D = dist.Field(name='n_D',bases=(xbasis)) # Concentration of D

# Equilibirum densities
n_D_eq = dist.Field(name='n_D_eq',bases=(xbasis)) # Equilibrium concentration

# Velocities
V_A = dist.Field(name='V_A',bases=(xbasis)) # Velocity of A
V_B = dist.Field(name='V_B',bases=(xbasis)) # Velocity of B

grad_mu_fA = dist.Field(name='grad_mu_fA',bases=(xbasis))
grad_mu_fB = dist.Field(name='grad_mu_fB',bases=(xbasis))

f_el_A = dist.Field(name='f_el_A',bases=(xbasis))
f_el_B = dist.Field(name='f_el_B',bases=(xbasis))
f_el_D = dist.Field(name='f_el_D',bases=(xbasis))

F_fA_ent = dist.Field(name='F_fA_ent',bases=(xbasis))
F_fA_visc = dist.Field(name='F_fA_visc',bases=(xbasis))
F_fA_el = dist.Field(name='F_fA_el',bases=(xbasis))

F_fB_ent = dist.Field(name='F_fB_ent',bases=(xbasis))
F_fB_visc = dist.Field(name='F_fB_visc',bases=(xbasis))
F_fB_el = dist.Field(name='F_fB_el',bases=(xbasis))

F_fA_act = dist.Field(name='F_fA_act',bases=(xbasis))
F_fB_act = dist.Field(name='F_fB_act',bases=(xbasis))


F_A = dist.Field(name='F_A',bases=(xbasis))
F_B = dist.Field(name='F_B',bases=(xbasis))



S_el_xx = dist.Field(name='S_el_xx',bases=(xbasis))
S_el_xy = dist.Field(name='S_el_xy',bases=(xbasis))
S_el_yx = dist.Field(name='S_el_yx',bases=(xbasis))
S_el_yy = dist.Field(name='S_el_yy',bases=(xbasis))

S_visc = dist.Field(name='S_visc',bases=(xbasis))


# %% EQUATIONS OF THE PROBLEM
problem = d3.IVP([ f_A,f_B,f_D, n_D, n_D_eq, V_A,V_B,F_A,F_B,
                 grad_mu_fA,grad_mu_fB, F_fA_ent,
                  F_fB_ent,F_fA_act,F_fB_act], namespace=locals()) # Declaration of the problem variables

# - Cahn Hillard equation for the filaments - #
problem.add_equation("dt(f_A) +D_f*ddx(-2*f_A + G_f*ddx(f_A)) = D_f*ddx(4*(f_A)**3-6*(f_A)**2) -dx(f_A*V_A) ")
problem.add_equation("dt(f_B) +D_f*ddx(-2*f_B + G_f*ddx(f_B)) = D_f*ddx(4*(f_B)**3-6*(f_B)**2) -dx(f_B*V_B)")
problem.add_equation("f_D = f_A*f_B")

# - Equation of the particles - #
problem.add_equation("dt(n_D) -D_D.v*ddx(n_D) -r*D_D.v*ddx(n_D) +k_off_D.v*n_D = -D_D.v*dx(n_D/(h+f_D)*dx(f_D)) -dx(n_D*0.5*(V_A+V_B))  -r*D_D.v*ddx(n_D) +k_off_D.v*n_D_eq") 

# - Equation of the mobilities - #
# problem.add_equation("C_D =  D_D/(kBT)*n_D*(h+f_D-n_D)/(h+f_D)")

# - Equation of the equilibrium concentrations - #
problem.add_equation("n_D_eq = f_D*(1/(1+np.exp(Eb_D.v)))")
# problem.add_equation("n_D_eq = 0.6*f_D")#*(1/(1+np.exp(Eb_D)))")


# - Equation of the viscous stress and elastic stress - #

# - Gradient of the chemical potential of the filaments - #
problem.add_equation("grad_mu_fA = -( np.log((h_log+f_D)/(h_log+f_D-n_D))*dx(f_B) +f_B*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-n_D)*dx(f_D-n_D)  ) )")
problem.add_equation("grad_mu_fB = -( np.log((h_log+f_D)/(h_log+f_D-n_D))*dx(f_A) +f_A*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-n_D)*dx(f_D-n_D)  ) )")

# - Integration of the forces - #
problem.add_equation("F_fA_ent = d3.Integrate(f_A*( n_s.v*grad_mu_fA)  ,('x'))")
problem.add_equation("F_fA_act = -d3.Integrate(f_A*(f_D*act.v)  ,('x'))")

problem.add_equation("F_fB_ent = d3.Integrate(f_B*( n_s.v*grad_mu_fB)  ,('x'))")
problem.add_equation("F_fB_act = d3.Integrate(f_B*(act.v)  ,('x'))")


problem.add_equation("F_A = F_fA_ent +F_fA_act")
problem.add_equation("F_B = F_fB_ent +F_fB_act")
        
# gamma = 0.0001
# problem.add_equation("V_A=-Ly/LA/m*F_A")
problem.add_equation("V_A=0") # Filament A is fixed

problem.add_equation("V_B=-Ly.v/LB/eta.v*F_B")

#%% INITIAL CONDITIONS
'FILAMENTS PHASE FIELD'
f_A['g'] = LS.function_filament(f_A['g'], xl_A, xr_A, x[0], li)
f_B['g'] = LS.function_filament(f_B['g'], xl_B, xr_B, x[0], li)
f_D['g'] = np.minimum(f_A['g'],f_B['g'])

'NUMBER OF PARTICLE'
# 1st guess on the number of particle definition

D_eq = 0.2
Eb_D.v = -np.log(D_eq/(1-D_eq))

# n_D['g'] = 0.0*f_D['g']#*f_D['g']/(1+np.exp(Eb_D))
n_D['g'] = f_D['g']/(1+np.exp(Eb_D.v))

#%%
S_el_xx['g'] = 0
S_el_xy['g'] = 0

S_el_yx['g'] = 0
S_el_yy['g'] = 0

#%%
plt.figure(dpi=200)
plt.plot(x[0],f_A['g'])
plt.plot(x[0],f_B['g'])
plt.plot(x[0],n_D['g'])
plt.show()





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
    solver.step(timestep) # solving the equations   

    if solver.iteration % int(stop_time/(N_save*timestep)) == 0 :
        j=j+1
        T_N1 = datetime.datetime.now()
        T_LEFT = (T_N1-T_N0)*(N_save-j)
        logger.info('%i/%i, T=%0.2e, t_left = %s' %(j,N_save,solver.sim_time,str(T_LEFT)))
        T_N0 = datetime.datetime.now()

        if j%1  == 0 and j<10 or j%10 == 0:
                f_A.change_scales(1)
                f_B.change_scales(1)
                n_D.change_scales(1)
                f_D.change_scales(1)
                n_D_eq.change_scales(1)

                plt.plot(x[0],f_A['g'],color = 'blue',alpha = 0.5)
                plt.plot(x[0],f_B['g'],color = 'red',alpha = 0.5)
                plt.plot(x[0],n_D['g'],color = 'purple',label = "n_D")
                plt.plot(x[0],f_D['g'],color = 'black',label = "f_D",alpha = 0.5)
                plt.plot(x[0],n_D_eq['g'],color = 'purple',label = "n_D_eq",alpha = 0.5)
                plt.legend()
                plt.show()

#%%

# %% Getting the saved files
# tasks = d3.load_tasks_to_xarray(folder +"/"+folder_name+"_s1.h5") # Downloadig the files
# x_tasks = np.array(tasks['n_D']['x'])
# t_tasks = np.array(tasks['n_D']['t'])
print(folder +"/"+folder_name+"_s1.h5")
print(T_N1-date)


#%%