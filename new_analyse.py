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

name =    "Braun2011_1609_test_k_0001_slide"
dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/Braun2011/'+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = "output_test1"


bool_anim = 1

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
# plt.figure(dpi=200,figsize=(6,3))
# plt.rcParams['font.size'] = '14'
# plt.plot(x_tasks,tasks['Pab'][100],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)
# # plt.plot(x_tasks,tasks['n_D'][300],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)
# # plt.plot(x_tasks,tasks['n_D'][650],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)

# plt.fill_between(x_tasks,tasks['f_A'][100],color="red",alpha=0.15)
# plt.fill_between(x_tasks,tasks['f_B'][100],color="red",alpha=0.15)

# plt.gca().spines['left'].set_linewidth(2)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_linewidth(2)
# # plt.yticks(color='w')
# # plt.gca().spines['bottom'].set_visible(False)
# # plt.gca().get_xaxis().set_visible(False)
# plt.hlines(0.59,-27,6,color="black",linestyles="--")
# plt.ylim(-0.02,1.02)
# plt.xlim(-22,5.5)

# # plt.legend()
# plt.show()


#%%
"""
FROM HERE I AM TESTING STUFF
"""

#%%
ib = 0
ie = -1
plt.figure(dpi=200)
plt.title("$V_B(t)$")
plt.plot(t_tasks[ib:ie],tasks['V_B'][ib:ie],label = r"$V_B(t)$")
# plt.plot(t_tasks,Overlap,label = r"$Overlap$")
plt.legend()
plt.show()




#%% Sweeping coeffi
plt.figure(dpi=200)
plt.plot(t_tasks[1:],10*N_Pab[1:]/Overlap[1:],label="density")
plt.plot(t_tasks[1:],Overlap[1:],label="Overlap")
plt.plot(t_tasks[1:],10*N_Pab[1:]/Overlap[1:]**2,label="density/overlap")
plt.legend()
plt.show()

#%%
x=np.linspace(0,5,50)
plt.figure(dpi=100)
plt.plot(Overlap,N_Pab/Overlap,linewidth=3)
plt.plot(x, 0.5-0.043*x)
# plt.ylim(0,1)
# plt.xlim(0,5.1)
plt.show()

#%%
x=np.linspace(0,5,50)
plt.figure(dpi=200,figsize=(8,8))
E = 0.79 #todo 0.95 'number of particles staying in overlap'
plt.title("Sweeping efficiency = "+str(E))
plt.plot(Overlap[1]/Overlap,(N_Pab/Overlap)/(N_Pab[1]/Overlap[1]),linewidth=3)
plt.plot(Overlap[1]/Overlap,np.power(Overlap[1]/Overlap,E))
# plt.ylim(0,1)
# plt.xlim(0,5.1)
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%

N_Pab[-1]/N_Pab[1]

#%%

1+np.log(N_Pab[-1]/N_Pab[1])/np.log(Overlap[1]/Overlap[-1])

#%%
#%%