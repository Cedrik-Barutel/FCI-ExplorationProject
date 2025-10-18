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

# ... existing code ...

def run_analysis(diff: float, activity: float, bool_anim: bool = True, plots: bool = False):
    """
    Load results for the given (diff, activity) and produce analysis outputs.
    Expects files in ./test/Braun2011_diff_{diff}_chem_{activity}/

    bool_anim: if True, renders and saves animation (heavy)
    plots: if True, generates extra figures; if False, keeps analysis headless
    """

    def commacolon(x):
        if "." in str(x):
            return str(x).replace('.', ',')
        else:
            return str(x)

    name = f"Nondimensional_diff_{commacolon(diff)}_koff_{commacolon(activity)}"
    cwd = os.getcwd()
    dir_input_file = f"{cwd}/test_nondim/{name}/"
    name_input_file = name + '_s1'
    extension_input_file = '.h5'

    dir_output_file = dir_local
    name_output_file = f"output_{name}"

    # loading data
    tasks = d3.load_tasks_to_xarray(dir_input_file + name_input_file + extension_input_file)

    x_tasks = np.array(tasks['f_A']['x'])
    t_tasks = np.array(tasks['f_A']['t']) / 60

    n_img = 4
    N_end = len(t_tasks)
    extension_animation = ".gif"
    frame_per_second = 20

    # ANIMATION (optional and heavy)
    if bool_anim:
        fig = plt.figure(figsize=(5, 2), dpi=200)

        def animate(i):
            if i % max(1, int(N_end / 100)) == 0:
                print(i)
            plt.clf()
            plt.plot(x_tasks, tasks['Pab'][i], color='black', linestyle="-", label=r"$x_{D}$")
            plt.plot(x_tasks, tasks['f_A'][i], color='blue', alpha=0.5, label=r"$\phi^A$")
            plt.plot(x_tasks, tasks['f_B'][i], color='red', alpha=0.5, label=r"$\phi^B$")
            plt.legend(loc='upper left')

        t = np.arange(0, N_end, n_img)
        os.makedirs(dir_input_file, exist_ok=True)
        ani = FuncAnimation(fig, animate, frames=t, interval=1, repeat=False)
        ani.save(os.path.join(dir_input_file, name_output_file + "_density" + extension_animation), writer='ffmpeg',
                 fps=frame_per_second)
        plt.close(fig)

    # Force calculation
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
        Force_ent_B[i] = tasks['F_fB_ent'][i]
        concentration[i] = tasks['Pab'][i][int(len(x_tasks) / 2)]
        Vitesse_B[i] = tasks['V_B'][i]
        Pab[i] = integrate.simpson(tasks['f_D'][i] * tasks['Pab'][i], x_tasks)
        Pab_tot[i] = integrate.simpson(tasks['Pab'][i], x_tasks)
        Overlap[i] = integrate.simpson(tasks['f_D'][i], x_tasks)

    rho = Pab / Overlap
    rho_tot = Pab_tot / Overlap

    A = np.zeros((len(t_tasks), len(x_tasks), 3))
    for t in range(len(tasks['Pab'])):
        tasks['Pab'][t][tasks['Pab'][t] <= 0] = 0
        tasks['f_B'][t][tasks['f_B'][t] <= 0] = 0
        tasks['f_A'][t][tasks['f_A'][t] <= 0] = 0

    A[:, :, 1] = 0 + 0.8 * tasks['Pab'] / np.max(tasks['Pab'])
    A[:, :, 0] = 0 + 0.5 * tasks['f_B'] / np.max(tasks['f_B']) + 0.3 * tasks['f_A'] / np.max(tasks['f_A'])

    # Save only the key kymograph; avoid interactive show
    fig_h = plt.figure(dpi=200, figsize=(4, 6))
    plt.pcolormesh(x_tasks, t_tasks, A)
    plt.gca().invert_yaxis()
    plt.ylabel("time (min)")
    plt.xlabel("distance to template filament end (um)")
    os.makedirs(dir_input_file, exist_ok=True)
    plt.savefig(os.path.join(dir_input_file, f"{name_output_file}_kymograph.png"), dpi=200)

    if plots:
        # optional display only when requested
        pass
    plt.close(fig_h)

    # Optionally skip the rest of the plotting-heavy section unless plots=True
    if plots:
        # ... keep your additional figures here if desired ...
        pass

    # Cleanup to reduce memory footprint in batch runs
    try:
        import gc
        plt.close('all')
        gc.collect()
    except Exception:
        pass

    return {
        "dir_input": dir_input_file,
        "name": name,
        "rho": rho,
        "rho_tot": rho_tot,
        "Vitesse_B": Vitesse_B,
        "Overlap": Overlap,
    }

# Optional CLI for manual runs
def main(diff,activity):
    run_analysis(diff, activity, bool_anim=True)

if __name__ == "__main__":
    main()