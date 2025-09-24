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

logger = logging.getLogger(__name__)
import lib_simulation as LS





#%%
dir_local = os.path.dirname(__file__)

def commacolon(x):
    if "." in str(x):
        return str(x).replace('.',',')
    else:
        return str(x)
diff=0.05
activity=0
name = f"Braun2011_diff_{commacolon(diff)}_chem_{commacolon(activity)}"
dir_input_file = "/home/benji/PycharmProjects/FCI-ExplorationProject/"+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = f"output_{name}"


bool_anim = 1

# print(name_save)




#%% loading data
tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files

x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])/60

# x_tasks = np.array(tasks['n_Mab']['x'])
# t_tasks = np.array(tasks['n_Mab']['t'])/60



#%%
n_img = 4
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
        plt.plot(x_tasks,tasks['Pab'][i],color = 'black',linestyle="-",label = r"$x_{D}$")
        # plt.plot(x_tasks,tasks['Pab'][i]/(0.001+tasks['f_D'][i]),color = 'black',linestyle="-",label = r"$coeff$")

        plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$\phi^A$")
        plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
        # plt.plot(x_tasks,tasks['f_D'][i]-tasks['Pab'][i], color='red',alpha = 0.5, label = r"$\phi^B$")

        plt.legend(loc='upper left')
    
        
    t=np.arange(0,N_end,n_img) # New time array with only n images  
    ani = FuncAnimation(fig, animate, frames=t,
                        interval=1, repeat=False)
    #name = "D"+str(D)+"a"+str(alpha)+".gif"
    ani.save(dir_input_file+"/"+name_output_file+"_density"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)



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
#         # plt.plot(x_tasks,tasks['f_D'][i]-tasks['Pab'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
#         plt.ylim(-5000,5000)
#         plt.legend(loc='upper left')
    
        
#     t=np.arange(0,N_end,n_img) # New time array with only n images  
#     ani = FuncAnimation(fig, animate, frames=t,
#                         interval=1, repeat=False)
#     #name = "D"+str(D)+"a"+str(alpha)+".gif"
#     ani.save(dir_output_file+"/"+name_output_file+"_stresses"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)

#%% Force calculation



Force_ent_A = np.zeros(len(t_tasks))
Force_visc_A = np.zeros(len(t_tasks))
Force_el_A = np.zeros(len(t_tasks))
Force_ent_B = np.zeros(len(t_tasks))

Force_calc_A = np.zeros(len(t_tasks))

Vitesse_B = np.zeros(len(t_tasks))


concentration = np.zeros(len(t_tasks))

Pab = np.zeros(len(t_tasks))
Pab_tot = np.zeros(len(t_tasks))

Overlap = np.zeros(len(t_tasks))


for i in range(len(t_tasks)):
    Force_ent_A[i] = tasks['F_fA_ent'][i]
    # Force_visc_A[i] = tasks['F_fA_visc'][i][10]
    # Force_el_A[i] = tasks['F_fA_el'][i][10]

    Force_ent_B[i] = tasks['F_fB_ent'][i]
    concentration[i] = tasks['Pab'][i][int(len(x_tasks)/2)]
    
    Vitesse_B[i] = tasks['V_B'][i]
    Pab[i] = integrate.simpson(tasks['f_D'][i]*tasks['Pab'][i],x_tasks)
    Pab_tot[i] = integrate.simpson(tasks['Pab'][i],x_tasks)

    Overlap[i] = integrate.simpson(tasks['f_D'][i],x_tasks)

#%%
rho = Pab/Overlap
rho_tot = Pab_tot/Overlap

#%%

A = np.zeros((len(t_tasks),len(x_tasks),3))
# for t in range(len(t_tasks)):
#     print(t)
#     A[t] = tasks['Pab'][t]
#     for i in range(len(x_tasks)):
#         if A[t][i]<=0.021:
#             A[t][i] = -0.1*tasks['f_B'][t][i]

#%%
for t in range(len(tasks['Pab'])):
    tasks['Pab'][t][tasks['Pab'][t]<=0] = 0
    tasks['f_B'][t][tasks['f_B'][t]<=0] = 0
    tasks['f_A'][t][tasks['f_A'][t]<=0] = 0


#%%
A[:,:,1]= 0+0.8*tasks['Pab']/np.max(tasks['Pab'])     
A[:,:,0]= 0+0.5*tasks['f_B']/np.max(tasks['f_B']) + 0.3*tasks['f_A']/np.max(tasks['f_A'])    

 
#%%
plt.figure(dpi = 200,figsize= (4,6))
# plt.pcolormesh(x_tasks,-t_tasks, tasks['Pab'])
plt.pcolormesh(x_tasks,t_tasks,A)
plt.gca().invert_yaxis()
plt.ylabel("time (min)")
plt.xlabel("distance to template filament end (um)")
plt.savefig(os.path.join(dir_input_file, f"{name_output_file}_heatmap.png"), dpi=200)
plt.show()

#%%
plt.figure(dpi=200,figsize=(6,3))
plt.rcParams['font.size'] = '14'
plt.plot(x_tasks,tasks['Pab'][100],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)
# plt.plot(x_tasks,tasks['Pab'][300],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)
# plt.plot(x_tasks,tasks['Pab'][650],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)

plt.fill_between(x_tasks,tasks['f_A'][100],color="red",alpha=0.15)
plt.fill_between(x_tasks,tasks['f_B'][100],color="red",alpha=0.15)

plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_linewidth(2)
# plt.yticks(color='w')
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().get_xaxis().set_visible(False)
plt.hlines(0.59,-27,6,color="black",linestyles="--")
plt.ylim(-0.02,1.02)
plt.xlim(-22,5.5)

# plt.legend()
plt.show()


#%%
"""
FROM HERE I AM TESTING STUFF
"""

#%%
ib = 0
ie = -1
plt.figure(dpi=200)
plt.plot(t_tasks[ib:ie],Vitesse_B[ib:ie],label = r"$V(t)$")
# plt.plot(t_tasks,Overlap,label = r"$Overlap$")

plt.legend()
plt.show()

#%%
plt.figure(dpi=200)
plt.plot(t_tasks[ib:ie],Overlap[ib:ie],label = r"$Overlap(t)$")
plt.legend()
plt.show()

#%%
plt.figure(dpi=200)
plt.plot(t_tasks,Pab,label = r"$N$")
plt.plot(t_tasks,Pab_tot,label = r"$N_{tot}$")

plt.plot(t_tasks,rho,label = r"$\rho$")
plt.plot(t_tasks,rho_tot,label = r"$\rho_{tot}$")

plt.legend()
plt.show()


#%%
plt.figure(dpi=200)
plt.plot(rho[ib:ie],Vitesse_B[ib:ie],label="rho")
plt.plot(rho_tot[ib:ie],Vitesse_B[ib:ie],label="rho_tot")
plt.legend()
plt.show()













#%%import matplotlib.pyplot as plt
try:
    img = plt.imread("fit_background.png")
    fig, ax = plt.subplots(dpi=200)
    ax.imshow(img, extent=[0, 1, 0, 1])
    ax.plot(2*rho_tot[ib:ie], 30*Vitesse_B[ib:ie])
    plt.xlim(-50,300)
    plt.ylim(-50,300)
except FileNotFoundError:
    # Skip background plotting if the image is not available
    fig, ax = plt.subplots(dpi=200)
    ax.plot(2*rho_tot[ib:ie], 30*Vitesse_B[ib:ie])
    plt.xlim(-50,300)
    plt.ylim(-50,300)

#%%
plt.figure(dpi=200)
plt.plot(Overlap[2]/Overlap[ib:ie],rho[ib:ie]/rho[2],label="rho")
plt.plot(Overlap[2]/Overlap[ib:ie],rho_tot[ib:ie]/rho_tot[2],label="rho_tot")

plt.legend()
plt.show()
#%%
plt.figure(dpi=200)
plt.plot(t_tasks[ib:ie],Overlap[ib:ie]/Overlap[2],label="rho")
plt.legend()
plt.show()
#%%


def func_fit(x, a, b):
    return a * x + b

popt, pcov = curve_fit(func_fit, rho[ib:ie], Vitesse_B[ib:ie])

#%%
plt.figure(dpi=200)
plt.plot(rho[ib:ie],Vitesse_B[ib:ie], label="rho" )
plt.plot(rho[ib:ie],func_fit(rho[ib:ie],popt[0],popt[1]), label="rho_fit" )
plt.legend()
plt.plot()

#%%
plt.figure(dpi=200)
plt.plot(Vitesse_B[ib:ie],rho[ib:ie], label="rho" )
plt.plot(Vitesse_B[ib:ie],(Vitesse_B[ib:ie]-popt[1])/popt[0], label="rho_fit" )
plt.legend()
plt.plot()

#%%
plt.figure(dpi=200)
plt.plot(t_tasks[ib:ie],rho[ib:ie], label="rho" )
plt.plot(t_tasks[ib:ie],(Vitesse_B[ib:ie]-popt[1])/popt[0], label="rho_fit" )
plt.legend()
plt.plot()


#%%
plt.figure(dpi=200)
plt.plot(Overlap[ib]/Overlap[ib:ie],rho[ib:ie]/rho[ib],label="rho")
# plt.plot(Overlap[ib]/Overlap[ib:ie],((Vitesse_B[ib:ie]-popt[1])/popt[0]),label="fit")

plt.legend()
plt.show()

#%%
plt.figure(dpi=200)
plt.plot(Overlap[ib]/Overlap[ib:ie],Vitesse_B[ib:ie],label="rho")

plt.legend()
plt.show()


#%%
plt.figure(dpi=200)
plt.plot(t_tasks[ib:ie],Overlap[ib:ie]/Overlap[ib],label="rho")

plt.legend()
plt.show()
#%%
#%%
#%%
#%%
#%%
#%%
#%%