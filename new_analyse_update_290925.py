#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:18:43 2025

@author: cedrik
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import datetime
from matplotlib.animation import FuncAnimation
import scipy
from scipy.optimize import curve_fit
# from scipy.differentiate import derivative
from scipy import integrate
import pandas as pd
import scipy as scipy
import logging
import os
import sys

# local library
dir_local = os.path.dirname(__file__)
sys.path.append(dir_local)
lib_path = "/home/cedrik/Documents/filaments-crosslinkers-projects/lib"
sys.path.append(lib_path)


logger = logging.getLogger(__name__)
import lib_simulation as LS





#%%
dir_local = os.path.dirname(__file__)

name =    "Braun2011_diff_0,5_chem_15,0"
dir_input_file = '/home/cedrik/Documents/FCI-ExplorationProject/__pycache__/FCI-ExplorationProject-main_ben/'+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = "output_test1"


bool_anim = 0

# print(name_save)




#%% loading data
tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files

x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])/60

# x_tasks = np.array(tasks['n_Mab']['x'])
# t_tasks = np.array(tasks['n_Mab']['t'])/60

#%%
n_img = 2
N_end = len(t_tasks)
extension_animation = ".gif"
frame_per_second = 20


# %% ANIMATION
if bool_anim:
    fig = plt.figure(figsize=(5, 2), dpi=200)
    def animate(i):
        if i%(N_end/100) == 0:
            print(i)
        plt.clf()
        plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$\phi^A$")
        plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
        
        plt.plot(x_tasks,tasks['Pab'][i],color = 'violet',linestyle="-",label = r"$P^{ab}$")
        plt.plot(x_tasks,tasks['Pa'][i],color = 'blue',linestyle="-",label = r"$P^{a}$")
        plt.plot(x_tasks,tasks['Pb'][i],color = 'red',linestyle="-",label = r"$P^{b}$")

        plt.plot(x_tasks,tasks['Mab'][i],color = 'violet',linestyle="--",label = r"$P^{ab}$")
        plt.plot(x_tasks,tasks['Ma'][i],color = 'blue',linestyle="--",label = r"$P^{a}$")
        plt.plot(x_tasks,tasks['Mb'][i],color = 'red',linestyle="--",label = r"$P^{b}$")

        # plt.plot(x_tasks,tasks['f_D'][i]-tasks['n_D'][i], color='red',alpha = 0.5, label = r"$\phi^B$")

        plt.legend(loc='upper left',fontsize=5)
    
        
    t=np.arange(0,N_end,n_img) # New time array with only n images  
    ani = FuncAnimation(fig, animate, frames=t,
                        interval=1, repeat=False)
    #name = "D"+str(D)+"a"+str(alpha)+".gif"
    ani.save(dir_output_file+"/"+name_output_file+"_density"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)



# %% ANIMATION
# if bool_anim:
#     fig = plt.figure(figsize=(5, 2), dpi=200)
#     def animate(i):
#         if i%(N_end/100) == 0:
#             print(i)
#         plt.clf()    
#         plt.plot(x_tasks,tasks['S_el_xx'][i],linestyle="-",label = r"$S^{el}_{xx}$")
#         plt.plot(x_tasks,tasks['S_el_xy'][i],linestyle="-",label = r"$S^{el}_{xy}$")
#         plt.plot(x_tasks,tasks['S_el_yx'][i],linestyle="-",label = r"$S^{el}_{yx}$")
#         plt.plot(x_tasks,tasks['S_el_yy'][i],linestyle="-",label = r"$S^{el}_{yy}$")
#         # plt.plot(x_tasks,tasks['S_el_xx'][i]-tasks['S_el_yy'][i],linestyle="-",label = r"$S^{el}_{yy}$")

        
        
#         # plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$\phi^A$")
#         # plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
#         # plt.plot(x_tasks,tasks['f_D'][i]-tasks['n_D'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
#         plt.ylim(-5000,5000)
#         plt.legend(loc='upper left')
    
        
#     t=np.arange(0,N_end,n_img) # New time array with only n images  
#     ani = FuncAnimation(fig, animate, frames=t,
#                         interval=1, repeat=False)
#     #name = "D"+str(D)+"a"+str(alpha)+".gif"
#     ani.save(dir_output_file+"/"+name_output_file+"_stresses"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)

#%% Force calculation



# FA_vis = np.zeros(len(t_tasks))
# FA_fri = np.zeros(len(t_tasks))
# FA_ent = np.zeros(len(t_tasks))
# FA_ela = np.zeros(len(t_tasks))
# FA_act = np.zeros(len(t_tasks))

# FB_vis = np.zeros(len(t_tasks))
# FB_fri = np.zeros(len(t_tasks))
# FB_ent = np.zeros(len(t_tasks))
# FB_ela = np.zeros(len(t_tasks))
# FB_act = np.zeros(len(t_tasks))

# VA = np.zeros(len(t_tasks))
# VB = np.zeros(len(t_tasks))

Overlap = np.zeros(len(t_tasks))

N_Pab = np.zeros(len(t_tasks))


for i in range(len(t_tasks)):
    # FA_vis[i] = tasks['F_fA_vis'][i][10]
    # FA_fri[i] = tasks['F_fA_fri'][i][10]
    # FA_ent[i] = tasks['F_fA_ent'][i][10]
    # FA_ela[i] = tasks['F_fA_ela'][i][10]
    # FA_act[i] = tasks['F_fA_act'][i][10]
    
    # FB_vis[i] = tasks['F_fB_vis'][i][10]
    # FB_fri[i] = tasks['F_fB_fri'][i][10]
    # FB_ent[i] = tasks['F_fB_ent'][i][10]
    # FB_ela[i] = tasks['F_fB_ela'][i][10]
    # FB_act[i] = tasks['F_fB_act'][i][10]
    
    N_Pab[i] = integrate.simpson(tasks['Pab'][i],x_tasks)
    # NB[i] = integrate.simpson(tasks['Pb'][i])
    # NAB[i] = integrate.simpson(tasks['Pab'][i]) 
 
    # VA[i] = tasks['V_A'][i][10]
    # VB[i] = tasks['V_B'][i][10]
    
    Overlap[i] = integrate.simpson(tasks['f_D'][i],x_tasks)

#%%


#%%
for t in range(len(tasks['Pab'])):
    tasks['Pab'][t][tasks['Pab'][t]<=0] = 0
    tasks['f_B'][t][tasks['f_B'][t]<=0] = 0
    tasks['f_A'][t][tasks['f_A'][t]<=0] = 0


#%%
A = np.zeros((len(t_tasks),len(x_tasks),3))
# for t in range(len(t_tasks)):
#     print(t)
#     A[t] = tasks['n_D'][t]
#     for i in range(len(x_tasks)):
#         if A[t][i]<=0.021:
#             A[t][i] = -0.1*tasks['f_B'][t][i]

A[:,:,1]= 0+0.8*tasks['Pab']/np.max(tasks['Pab'])     
A[:,:,0]= 0+0.5*tasks['f_B']/np.max(tasks['f_B']) + 0.3*tasks['f_A']/np.max(tasks['f_A'])    

#
#%%
plt.figure(dpi = 200,figsize= (4,6))
# plt.pcolormesh(x_tasks,-t_tasks, tasks['n_D'])
plt.pcolormesh(x_tasks,t_tasks,A)
plt.gca().invert_yaxis()
plt.ylabel("time (min)")
plt.xlabel("distance to template filament end ($\mu$m)")
plt.show()



#%%
"""
FROM HERE I AM TESTING STUFF
"""

def func_linear(x,E):
    return E*x
    
    
def func_power(x,E):
    return np.power(x,E)   
    


#%%

ib = 1 # first point
ie = 50 # last point

x = Overlap[ib]/Overlap[ib:ie]
y = (N_Pab[ib:ie]/Overlap[ib:ie])/(N_Pab[ib]/Overlap[ib])
popt2, pcov2 =scipy.optimize.curve_fit(func_power, x, y)

plt.figure(dpi=200,figsize=(8,8))
plt.title("Sweeping efficiency = "+str(popt2[0]))
plt.plot(Overlap[ib]/Overlap[ib:ie],(N_Pab[ib:ie]/Overlap[ib:ie])/(N_Pab[ib]/Overlap[ib]),linewidth=3)
# plt.plot(Overlap[ib]/Overlap[ib:ie],np.power(Overlap[ib]/Overlap[ib:ie],E))

# plt.plot(x,func_linear(x,E))
plt.plot(x,func_power(x,popt2[0]))



# plt.ylim(0,1)
# plt.xlim(0,5.1)
plt.xscale('log')
plt.yscale('log')
plt.show()

#%%
#%%