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


def commacolon(x):
    s = str(x)
    return s.replace('.', ',') if '.' in s else s

def run_sweeping_eff(diff, activity):

    name = f"Braun2011_diff_{commacolon(diff)}_chem_{commacolon(activity)}"
    name_input_file = name+'_s1'
    extension_input_file = '.h5'
    working_path = os.path.dirname(__file__)
    folder = f"{working_path}/test/{name}"
    dir_input_file = f"{folder}/"
    dir_output_file = folder
    name_output_file = f"output_{name}"

    bool_anim = False

    # print(name_save)


    tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files

    x_tasks = np.array(tasks['f_A']['x'])
    t_tasks = np.array(tasks['f_A']['t'])/60

    n_img = 2
    N_end = len(t_tasks)
    extension_animation = ".gif"
    frame_per_second = 20

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

    Overlap = np.zeros(len(t_tasks))

    N_Pab = np.zeros(len(t_tasks))


    for i in range(len(t_tasks)):
        N_Pab[i] = integrate.simpson(tasks['Pab'][i],x_tasks)
        Overlap[i] = integrate.simpson(tasks['f_D'][i],x_tasks)

    for t in range(len(tasks['Pab'])):
        tasks['Pab'][t][tasks['Pab'][t]<=0] = 0
        tasks['f_B'][t][tasks['f_B'][t]<=0] = 0
        tasks['f_A'][t][tasks['f_A'][t]<=0] = 0

    A = np.zeros((len(t_tasks),len(x_tasks),3))

    A[:,:,1]= 0+0.8*tasks['Pab']/np.max(tasks['Pab'])
    A[:,:,0]= 0+0.5*tasks['f_B']/np.max(tasks['f_B']) + 0.3*tasks['f_A']/np.max(tasks['f_A'])

    def func_power(x,E):
        return np.power(x,E)

    ib = 1 # first point
    ie = 50 # last point

    x = Overlap[ib]/Overlap[ib:ie]
    y = (N_Pab[ib:ie]/Overlap[ib:ie])/(N_Pab[ib]/Overlap[ib])
    popt2, pcov2 =scipy.optimize.curve_fit(func_power, x, y)

    plt.figure(dpi=200,figsize=(8,8))
    sweeping_eff = popt2[0]
    plt.title(f"Sweeping efficiency = {str(sweeping_eff)}")
    plt.plot(Overlap[ib]/Overlap[ib:ie],(N_Pab[ib:ie]/Overlap[ib:ie])/(N_Pab[ib]/Overlap[ib]),linewidth=3, label='Data')

    plt.plot(x, func_power(x, popt2[0]), label=f'Fit: $x^{{{sweeping_eff:.3f}}}$')
    plt.xlabel('Normalized L_ov')
    plt.ylabel('Normalized N_Pab/L_ov')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    plt.savefig(os.path.join(dir_input_file, f"{name_output_file}_sweeping_eff.png"), dpi=200)
    # Calculate the mean derivative of the fit across all data points
    # Derivative of x^E is E * x^(E-1)
    derivative_fit = np.mean(popt2[0] * np.power(x, popt2[0] - 1))

    df = pd.DataFrame({
        'diffusion': [diff],
        'k_off': [activity],
        'sweeping efficiency': [sweeping_eff],
        'derivative_fit': [derivative_fit]
    })
    df.to_csv(os.path.join(dir_input_file, f"{name_output_file}_sweeping_eff.csv"))
    return