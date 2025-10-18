import os
import csv
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import dedalus.public as d3

# Local imports (assumed available in project)
import lib_simulation as LS

logger = logging.getLogger(__name__)

def commacolon(x):
    s = str(x)
    return s.replace('.', ',') if '.' in s else s

def run_simulation(diff: float, activity: float, debug_plots: bool = False):
    """
    Run the simulation with provided parameters.
    Saves outputs into ./test/Braun2011_diff_{diff}_chem_{activity}/

    debug_plots:
      - False (default): minimize plotting/UI overhead for batch runs
      - True: show intermediate diagnostic plots
    """
    # --- parameters formerly globals, now from args ---
    folder_name = f"Braun2011_diff_{commacolon(diff)}_chem_{commacolon(activity)}"
    working_path = os.path.dirname(__file__)
    folder = f"{working_path}/test/{folder_name}"

    # Lighter IO for batch runs
    N_save = 200  # number of savesteps
    timestep = 5e-3
    stop_time = 5 * 60
    Lx = 27
    Nx = 2 ** 8

    # Geometry
    xl_A = -20
    xr_A = 0
    xl_B = -6
    xr_B = xl_B + 6

    LA = xr_A - xl_A
    LB = xr_B - xl_B

    # Coefficients
    Ly = LS.Coefficient("Ly", 0.02)
    Lz = LS.Coefficient("Lz", 0.02)

    n_s = LS.Coefficient("n_s", 200)

    Eb_Ma = LS.Coefficient("Eb_Ma", 0)
    Eb_Mb = LS.Coefficient("Eb_Mb", 0)
    Eb_Mab = LS.Coefficient("Eb_Mab", 0)
    Eb_Pa = LS.Coefficient("Eb_Pa", 0)
    Eb_Pb = LS.Coefficient("Eb_Pb", 0)
    Eb_Pab = LS.Coefficient("Eb_Pab", -1)

    D_Ma = LS.Coefficient("D_Ma", 0.01)
    D_Mb = LS.Coefficient("D_Mb", 0.01)
    D_Mab = LS.Coefficient("D_Mab", 0.001)

    D_Pa = LS.Coefficient("D_Pa", 0.5)
    D_Pb = LS.Coefficient("D_Pb", 0.5)
    D_Pab = LS.Coefficient("D_Pab", diff)

    V_Ma = LS.Coefficient("V_Ma", 0)
    V_Mb = LS.Coefficient("V_Mb", 0)
    V_Mab = LS.Coefficient("V_Mab", 0)

    Koff_1_Ma = LS.Coefficient("Koff_1_Ma", 30)
    Koff_2_Mb = LS.Coefficient("Koff_2_Mb", 30)
    Koff_3_Mab = LS.Coefficient("Koff_3_Mab", 30)
    Koff_4_Mab = LS.Coefficient("Koff_4_Mab", 30)

    Koff_5_Pa = LS.Coefficient("Koff_5_Pa", 0)
    Koff_6_Pb = LS.Coefficient("Koff_6_Pb", 0)
    Koff_7_Pab = LS.Coefficient("Koff_7_Pab", activity)
    Koff_8_Pab = LS.Coefficient("Koff_8_Pab", activity)

    eta = LS.Coefficient("eta", 0)
    gamma = LS.Coefficient("gamma", 3000)
    E = LS.Coefficient("E", 0)
    act = LS.Coefficient("act", 3)
    K = LS.Coefficient("K", 0)
    G = LS.Coefficient("G", 0)

    # Numerics
    h = 1e-3
    h_log = 2 * h
    r = 0
    li = 0.5
    D_f = 0.2
    G_f = 1 / 18 * li ** 2
    timestepper = d3.RK443
    dealias = 3 / 2

    # --- ND reference scales (keep names in equations unchanged) ---
    L_ref = Lx              # length scale
    D0_ref = D_f            # choose D0 = D_f
    tau = L_ref**2 / D0_ref # time scale

    # --- ND rescale floats used directly in PDEs
    D_f = D_f / D0_ref                 # -> 1.0
    G_f = G_f / (L_ref**2)             # -> G_f_hat

    # --- ND rescale coefficients IN PLACE (names unchanged in equations)
    # keep original gamma for scaling hats:
    _gamma0 = gamma.v

    # Diffusivities (divide by D0)
    D_Ma.v  = D_Ma.v  / D0_ref
    D_Mb.v  = D_Mb.v  / D0_ref
    D_Mab.v = D_Mab.v / D0_ref
    D_Pa.v  = D_Pa.v  / D0_ref
    D_Pb.v  = D_Pb.v  / D0_ref
    D_Pab.v = D_Pab.v / D0_ref

    # Reaction rates (multiply by tau)
    Koff_1_Ma.v  = Koff_1_Ma.v  * tau
    Koff_2_Mb.v  = Koff_2_Mb.v  * tau
    Koff_3_Mab.v = Koff_3_Mab.v * tau
    Koff_4_Mab.v = Koff_4_Mab.v * tau
    Koff_5_Pa.v  = Koff_5_Pa.v  * tau
    Koff_6_Pb.v  = Koff_6_Pb.v  * tau
    Koff_7_Pab.v = Koff_7_Pab.v * tau
    Koff_8_Pab.v = Koff_8_Pab.v * tau

    # Elastic ODE (multiply by tau; we keep /Ly in the equation, so K gets tau only)
    G.v = G.v * tau
    K.v = K.v * tau

    # Force/closure nondimensional groups (now overwrite the same names)
    # gamma becomes 1 (force scale); eta, n_s, act, E turned into their hat values
    gamma.v = 1.0
    eta.v   = eta.v   * L_ref / _gamma0
    n_s.v   = n_s.v   * L_ref**2 / (_gamma0 * D0_ref)
    act.v   = act.v   * L_ref**2 / (_gamma0 * D0_ref)
    E.v     = E.v     * L_ref**2 / (_gamma0 * D0_ref)   # note: equation multiplies by Lz later

    # Drift coefficients turned into dimensionless velocities (Péclet)
    V_Ma.v  = V_Ma.v  * L_ref / D0_ref
    V_Mb.v  = V_Mb.v  * L_ref / D0_ref
    V_Mab.v = V_Mab.v * L_ref / D0_ref

    # Make Ly, Lz nondimensional lengths
    Ly.v = Ly.v / L_ref
    Lz.v = Lz.v / L_ref

    # Dedalus basis and fields
    coords = d3.CartesianCoordinates('x')
    dtype = np.float64
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-21, -21 + Lx), dealias=dealias)
    x = dist.local_grids(xbasis)
    ex = coords.unit_vector_fields(dist)

    dx = lambda A: d3.Differentiate(A, coords['x'])
    ddx = lambda A: dx(dx(A))

    f_A = dist.Field(name='f_A', bases=(xbasis))
    f_B = dist.Field(name='f_B', bases=(xbasis))
    f_D = dist.Field(name='f_D', bases=(xbasis))

    Ma = dist.Field(name='Ma', bases=(xbasis))
    Mb = dist.Field(name='Mb', bases=(xbasis))
    Mab = dist.Field(name='Mab', bases=(xbasis))

    Pa = dist.Field(name='Pa', bases=(xbasis))
    Pb = dist.Field(name='Pb', bases=(xbasis))
    Pab = dist.Field(name='Pab', bases=(xbasis))

    Ma_eq_1 = dist.Field(name='Ma_eq_1', bases=(xbasis))
    Mb_eq_2 = dist.Field(name='Mb_eq_2', bases=(xbasis))
    Mab_eq_3 = dist.Field(name='Mab_eq_3', bases=(xbasis))
    Mab_eq_4 = dist.Field(name='Mab_eq_4', bases=(xbasis))

    Pa_eq_5 = dist.Field(name='Pa_eq_5', bases=(xbasis))
    Pb_eq_6 = dist.Field(name='Pb_eq_6', bases=(xbasis))
    Pab_eq_7 = dist.Field(name='Pab_eq_7', bases=(xbasis))
    Pab_eq_8 = dist.Field(name='Pab_eq_8', bases=(xbasis))

    C_Mab_Mab_inv = dist.Field(name='C_Mab_Mab_inv', bases=(xbasis))
    C_Mab_Pab = dist.Field(name='C_Mab_Pab', bases=(xbasis))
    C_Mab_fA = dist.Field(name='C_Mab_fA', bases=(xbasis))
    C_Mab_fB = dist.Field(name='C_Mab_fB', bases=(xbasis))

    C_Pab_Pab = dist.Field(name='C_Pab_Pab', bases=(xbasis))
    C_Pab_Mab = dist.Field(name='C_Pab_Mab', bases=(xbasis))
    C_Pab_fA = dist.Field(name='C_Pab_fA', bases=(xbasis))
    C_Pab_fB = dist.Field(name='C_Pab_fB', bases=(xbasis))

    Coeff_fri = dist.Field(name='Coeff_fri')

    V_A = dist.Field(name='V_A')
    V_B = dist.Field(name='V_B')

    grad_mu_fA = dist.Field(name='grad_mu_fA', bases=(xbasis))
    grad_mu_fB = dist.Field(name='grad_mu_fB', bases=(xbasis))

    u_el = dist.Field(name='u_el', bases=(xbasis))

    F_fA_vis = dist.Field(name='F_fA_vis')
    F_fA_fri = dist.Field(name='F_fA_fri')
    F_fA_ent = dist.Field(name='F_fA_ent')
    F_fA_act = dist.Field(name='F_fA_act')
    F_fA_ela = dist.Field(name='F_fA_ela')

    F_fB_vis = dist.Field(name='F_fB_vis')
    F_fB_fri = dist.Field(name='F_fB_fri')
    F_fB_ent = dist.Field(name='F_fB_ent')
    F_fB_act = dist.Field(name='F_fB_act')
    F_fB_ela = dist.Field(name='F_fB_ela')

    F_A = dist.Field(name='F_A')
    F_B = dist.Field(name='F_B')

    # Initial conditions
    f_A['g'] = LS.function_filament(f_A['g'], xl_A, xr_A, x[0], li)
    f_B['g'] = LS.function_filament(f_B['g'], xl_B, xr_B, x[0], li)
    f_D['g'] = np.minimum(f_A['g'], f_B['g'])

    h_log_d = 1e-5
    n_ma = 0.05
    n_mb = 0.05
    n_mab = 0.1
    n_pa = 0.1
    n_pb = 0.1
    n_pab = 0.3

    Mab['g'] = n_mab * f_D['g']
    Pab['g'] = n_pab * f_D['g']

    Ma['g'] = n_ma * (f_A['g'] - Mab['g'] - Pab['g'])
    Mb['g'] = n_mb * (f_B['g'] - Mab['g'] - Pab['g'])
    Pa['g'] = n_pa * (f_A['g'] - Mab['g'] - Pab['g'])
    Pb['g'] = n_pb * (f_B['g'] - Mab['g'] - Pab['g'])

    Eb_Ma.v = -np.log(h_log_d + n_ma / (1 - n_pa - n_ma))
    Eb_Mb.v = -np.log(h_log_d + n_mb / (1 - n_pb - n_mb))
    Eb_Pa.v = -np.log(h_log_d + n_pa / (1 - n_pa - n_ma))
    Eb_Pb.v = -np.log(h_log_d + n_pb / (1 - n_pb - n_mb))

    Eb_Mab.v = -np.log(h_log_d + n_mab / (1 - n_pab - n_mab - n_ma * (1 - n_mab - n_pab)) * (1 - n_pab - n_mab) / (1 - n_pab - n_mab - n_ma * (1 - n_mab - n_pab)))
    Eb_Pab.v = -np.log(h_log_d + n_pab / (1 - n_pab - n_mab - n_pa * (1 - n_mab - n_pab)) * (1 - n_pab - n_mab) / (1 - n_pab - n_mab - n_pa * (1 - n_mab - n_pab)))

    Pa_eq_5['g'] = (f_A['g'] - Pab['g'] - Mab['g'] - Ma['g']) / (1 + np.exp(Eb_Pa.v))
    Pb_eq_6['g'] = (f_B['g'] - Pab['g'] - Mab['g'] - Mb['g']) / (1 + np.exp(Eb_Pb.v))
    Pab_eq_7['g'] = 0.5 * (1 - Mab['g'] + Pa['g'] * np.exp(-(Eb_Pab.v - Eb_Pa.v)) - np.sqrt((1 - Mab['g'] + Pa['g'] * np.exp(-(Eb_Pab.v - Eb_Pa.v))) ** 2 - 4 * Pa['g'] * np.exp(-(Eb_Pab.v - Eb_Pa.v)) * (f_B['g'] - Mab['g'] - Pb['g'] - Mb['g'])))
    Pab_eq_8['g'] = 0.5 * (1 - Mab['g'] + Pb['g'] * np.exp(-(Eb_Pab.v - Eb_Pb.v)) - np.sqrt((1 - Mab['g'] + Pb['g'] * np.exp(-(Eb_Pab.v - Eb_Pb.v))) ** 2 - 4 * Pb['g'] * np.exp(-(Eb_Pab.v - Eb_Pb.v)) * (f_A['g'] - Mab['g'] - Pa['g'] - Ma['g'])))

    Ma_eq_1['g'] = (f_A['g'] - Pab['g'] - Mab['g'] - Pa['g']) / (1 + np.exp(Eb_Ma.v))
    Mb_eq_2['g'] = (f_B['g'] - Pab['g'] - Mab['g'] - Pb['g']) / (1 + np.exp(Eb_Mb.v))
    Mab_eq_3['g'] = 0.5 * (1 - Pab['g'] + Ma['g'] * np.exp(-(Eb_Mab.v - Eb_Ma.v)) - np.sqrt((1 - Pab['g'] + Ma['g'] * np.exp(-(Eb_Mab.v - Eb_Ma.v))) ** 2 - 4 * Ma['g'] * np.exp(-(Eb_Mab.v - Eb_Ma.v)) * (f_B['g'] - Pab['g'] - Pb['g'] - Mb['g'])))
    Mab_eq_4['g'] = 0.5 * (1 - Pab['g'] + Mb['g'] * np.exp(-(Eb_Mab.v - Eb_Mb.v)) - np.sqrt((1 - Pab['g'] + Mb['g'] * np.exp(-(Eb_Mab.v - Eb_Mb.v))) ** 2 - 4 * Mb['g'] * np.exp(-(Eb_Mab.v - Eb_Mb.v)) * (f_A['g'] - Pab['g'] - Pa['g'] - Ma['g'])))

    for _ in range(200):
        Pab_eq_7['g'] = 0.5 * (1 - Mab['g'] + Pa['g'] * np.exp(-(Eb_Pab.v - Eb_Pa.v)) - np.sqrt((1 - Mab['g'] + Pa['g'] * np.exp(-(Eb_Pab.v - Eb_Pa.v))) ** 2 - 4 * Pa['g'] * np.exp(-(Eb_Pab.v - Eb_Pa.v)) * (f_B['g'] - Mab['g'] - Pb['g'] - Mb['g'])))
        Pab_eq_8['g'] = 0.5 * (1 - Mab['g'] + Pb['g'] * np.exp(-(Eb_Pab.v - Eb_Pb.v)) - np.sqrt((1 - Mab['g'] + Pb['g'] * np.exp(-(Eb_Pab.v - Eb_Pb.v))) ** 2 - 4 * Pb['g'] * np.exp(-(Eb_Pab.v - Eb_Pb.v)) * (f_A['g'] - Mab['g'] - Pa['g'] - Ma['g'])))
        Mab_eq_3['g'] = 0.5 * (1 - Pab['g'] + Ma['g'] * np.exp(-(Eb_Mab.v - Eb_Ma.v)) - np.sqrt((1 - Pab['g'] + Ma['g'] * np.exp(-(Eb_Mab.v - Eb_Ma.v))) ** 2 - 4 * Ma['g'] * np.exp(-(Eb_Mab.v - Eb_Ma.v)) * (f_B['g'] - Pab['g'] - Pb['g'] - Mb['g'])))
        Mab_eq_4['g'] = 0.5 * (1 - Pab['g'] + Mb['g'] * np.exp(-(Eb_Mab.v - Eb_Mb.v)) - np.sqrt((1 - Pab['g'] + Mb['g'] * np.exp(-(Eb_Mab.v - Eb_Mb.v))) ** 2 - 4 * Mb['g'] * np.exp(-(Eb_Mab.v - Eb_Mb.v)) * (f_A['g'] - Pab['g'] - Pa['g'] - Ma['g'])))

        Pab_eq_7['g'][Pab_eq_7['g'] <= 0] = 0
        Pab_eq_8['g'][Pab_eq_7['g'] <= 0] = 0
        Mab_eq_3['g'][Mab_eq_3['g'] <= 0] = 0
        Mab_eq_4['g'][Mab_eq_4['g'] <= 0] = 0

        Mab['g'] = 0.5 * (Mab_eq_3['g'] + Mab_eq_4['g'])
        Pab['g'] = 0.5 * (Pab_eq_7['g'] + Pab_eq_8['g'])

    # Optional quick plots
    if debug_plots:
        _val = integrate.simpson(Mab['g'], x[0])
        print(_val)

        fig1 = plt.figure(dpi=200)
        plt.title("Motors")
        plt.plot(x[0], f_A['g'], label="$f^{A}$")
        plt.plot(x[0], f_B['g'], label="$f^{B}$")
        plt.plot(x[0], Ma['g'], label="$M^{a}$")
        plt.plot(x[0], Mb['g'], label="$M^{b}$")
        plt.plot(x[0], Mab['g'], label="$M^{ab}$")
        plt.plot(x[0], Ma_eq_1['g'], label="$M^{a}$", alpha=0.3)
        plt.plot(x[0], Mb_eq_2['g'], label="$M^{b}$", alpha=0.3)
        plt.plot(x[0], Mab_eq_3['g'], label="$M^{ab}_{eq}$", alpha=0.3)
        plt.plot(x[0], Mab_eq_4['g'], label="$M^{ab}_{eq}$", alpha=0.3)
        plt.hlines(0.6, -20, 0)
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.close(fig1)

        fig2 = plt.figure(dpi=200)
        plt.title("Passive")
        plt.plot(x[0], f_A['g'], label="$f^{A}$")
        plt.plot(x[0], f_B['g'], label="$f^{B}$")
        plt.plot(x[0], Pa['g'], label="$P^{a}$")
        plt.plot(x[0], Pb['g'], label="$P^{b}$")
        plt.plot(x[0], Pab['g'], label="$P^{ab}$")
        plt.plot(x[0], 0.3 * f_D['g'], label="$test$")
        plt.plot(x[0], Pa_eq_5['g'], label="$P^{a}$", alpha=0.3)
        plt.plot(x[0], Pb_eq_6['g'], label="$P^{b}$", alpha=0.3)
        plt.plot(x[0], Pab_eq_7['g'], label="$P^{ab}$", alpha=0.3)
        plt.plot(x[0], Pab_eq_8['g'], label="$P^{ab}$", alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.close(fig2)

    # Problem definition
    problem = d3.IVP(
        [
            f_A, f_B, f_D,
            Ma, Mb, Mab, Pa, Pb, Pab,
            Ma_eq_1, Mb_eq_2, Mab_eq_3, Mab_eq_4,
            Pa_eq_5, Pb_eq_6, Pab_eq_7, Pab_eq_8,
            C_Mab_Mab_inv,
            V_A, V_B,
            F_A, F_B,
            Coeff_fri,
            grad_mu_fA, grad_mu_fB,
            u_el,
            F_fA_vis, F_fA_fri, F_fA_ent, F_fA_act, F_fA_ela,
            F_fB_vis, F_fB_fri, F_fB_ent, F_fB_act, F_fB_ela
        ],
        namespace={**locals(), 'np': np, 'd3': d3}
    )

    # f_A, f_B (D_f and G_f already converted to ND values above)
    problem.add_equation("dt(f_A) +D_f*ddx(-2*f_A +G_f*ddx(f_A)) = D_f*ddx(4*(f_A)**3-6*(f_A)**2) -dx(f_A*V_A)")
    problem.add_equation("dt(f_B) +D_f*ddx(-2*f_B +G_f*ddx(f_B)) = D_f*ddx(4*(f_B)**3-6*(f_B)**2) -dx(f_B*V_B)")
    problem.add_equation("f_D = f_A*f_B")

    # Ma (D_Ma, Koff_* are ND via .v above; V_Ma is ND via .v above; V_A is a field, solved ND via force balance)
    problem.add_equation("dt(Ma)-r*D_Ma.v*ddx(Ma)-D_Ma.v*ddx(Ma)+Koff_1_Ma.v*Ma -Koff_3_Mab.v*Mab = -r*D_Ma.v*ddx(Ma)-D_Ma.v*dx(Ma/(h+f_A-Mab-Pab-Pa)*dx(f_A))+D_Ma.v*dx(Ma/(h+f_A-Mab-Pab-Pa)*dx(Pa))+D_Ma.v*dx(Ma/(h+f_A-Mab-Pab-Pa)*dx(Mab))+D_Ma.v*dx(Ma/(h+f_A-Mab-Pab-Pa)*dx(Pab))-dx(Ma*(h+f_A-Mab-Pab-Pa-Ma)/(h+f_A-Mab-Pab-Pa)*V_Ma.v)-dx(Ma*V_A)+Koff_1_Ma.v*Ma_eq_1  -Koff_3_Mab.v*Mab_eq_3")
    problem.add_equation("dt(Mb)-r*D_Mb.v*ddx(Mb)-D_Mb.v*ddx(Mb)+Koff_2_Mb.v*Mb -Koff_4_Mab.v*Mab = -r*D_Mb.v*ddx(Mb)-D_Mb.v*dx(Mb/(h+f_B-Mab-Pab-Pb)*dx(f_B))-D_Mb.v*dx(Mb/(h+f_B-Mab-Pab-Pb)*dx(Pb))+D_Mb.v*dx(Mb/(h+f_B-Mab-Pab-Pb)*dx(Mab))+D_Mb.v*dx(Mb/(h+f_B-Mab-Pab-Pb)*dx(Pab))-dx(Mb*(h+f_B-Mab-Pab-Pb-Mb)/(h+f_B-Mab-Pab-Pb)*V_Mb.v)-dx(Mb*V_B) +Koff_2_Mb.v*Mb_eq_2  -Koff_4_Mab.v*Mab_eq_4")
    problem.add_equation("dt(Mab)-r*D_Mab.v*ddx(Mab)-D_Mab.v*ddx(Mab)+Koff_3_Mab.v*Mab +Koff_4_Mab.v*Mab = -r*D_Mab.v*ddx(Mab)-D_Mab.v*dx( (f_B/(h+f_D-Mab-Pab) +1/(h+f_A-Mab-Pab)-1/(h+f_A-Mab-Pab-Ma-Pa))/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(f_A)) -D_Mab.v*dx( (f_A/(h+f_D-Mab-Pab) +1/(h+f_B-Mab-Pab)-1/(h+f_B-Mab-Pab-Mb-Pb))/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(f_B)) -D_Mab.v*dx( 1/(h+f_A-Mab-Pab-Ma-Pa)/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Ma)) -D_Mab.v*dx( 1/(h+f_B-Mab-Pab-Mb-Pb)/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Mb)) -D_Mab.v*dx( 1/(h+f_A-Mab-Pab-Ma-Pa)/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Pa)) -D_Mab.v*dx( 1/(h+f_B-Mab-Pab-Mb-Pb)/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Pb)) -D_Mab.v*dx( (1/(h+f_D-Mab-Pab) +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab))/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Pab)) -dx(Mab/C_Mab_Mab_inv*V_Mab.v) -dx(Mab*0.5*(V_A+V_B)) +Koff_3_Mab.v*Mab_eq_3  +Koff_4_Mab.v*Mab_eq_4")

    problem.add_equation("dt(Pa)-r*D_Ma.v*ddx(Pa)-D_Pa.v*ddx(Pa)+Koff_5_Pa.v*Pa -Koff_7_Pab.v*Pab = -r*D_Pa.v*ddx(Pa)-D_Pa.v*dx(Pa/(h+f_A-Mab-Pab-Ma)*dx(f_A))+D_Pa.v*dx(Pa/(h+f_A-Mab-Pab-Ma)*dx(Ma))+D_Pa.v*dx(Pa/(h+f_A-Mab-Pab-Ma)*dx(Mab))+D_Pa.v*dx(Pa/(h+f_A-Mab-Pab-Ma)*dx(Pab))-dx(Pa*V_A)+Koff_5_Pa.v*Pa_eq_5  -Koff_7_Pab.v*Pab_eq_7")
    problem.add_equation("dt(Pb)-r*D_Ma.v*ddx(Pb)-D_Pb.v*ddx(Pb)+Koff_6_Pb.v*Pb -Koff_8_Pab.v*Pab = -r*D_Pb.v*ddx(Pb)-D_Pb.v*dx(Pb/(h+f_B-Mab-Pab-Mb)*dx(f_B))+D_Pb.v*dx(Pb/(h+f_B-Mab-Pab-Mb)*dx(Mb))+D_Pb.v*dx(Pb/(h+f_B-Mab-Pab-Mb)*dx(Mab))+D_Pb.v*dx(Pb/(h+f_B-Mab-Pab-Mb)*dx(Pab))-dx(Pb*V_B)+Koff_6_Pb.v*Pb_eq_6  -Koff_8_Pab.v*Pab_eq_8")
    problem.add_equation("dt(Pab)-r*D_Pab.v*ddx(Pab)-D_Pab.v*ddx(Pab)+Koff_7_Pab.v*Pab +Koff_8_Pab.v*Pab = -r*D_Pab.v*ddx(Pab)-D_Pab.v*dx( (f_B/(h+f_D-Mab-Pab) +1/(h+f_A-Mab-Pab)-1/(h+f_A-Mab-Pab-Ma-Pa))/( 1/(Pab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(f_A)) -D_Pab.v*dx( (f_A/(h+f_D-Mab-Pab) +1/(h+f_B-Mab-Pab)-1/(h+f_B-Mab-Pab-Mb-Pb))/( 1/(Pab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(f_B)) -D_Pab.v*dx( 1/(h+f_A-Mab-Pab-Ma-Pa)/( 1/(Pab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Ma)) -D_Pab.v*dx( 1/(h+f_B-Mab-Pab-Mb-Pb)/( 1/(Pab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Mb)) -D_Pab.v*dx( 1/(h+f_A-Mab-Pab-Ma-Pa)/( 1/(Pab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Pa)) -D_Pab.v*dx( 1/(h+f_B-Mab-Pab-Mb-Pb)/( 1/(Pab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Pb)) -D_Pab.v*dx( (1/(h+f_D-Mab-Pab) +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab))/( 1/(Pab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )*dx(Mab)) -dx(Pab*0.5*(V_A+V_B)) +Koff_7_Pab.v*Pab_eq_7  +Koff_8_Pab.v*Pab_eq_8")

    problem.add_equation("C_Mab_Mab_inv = 1/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma-Pa)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb-Pb)-1/(h+f_B-Mab-Pab) )")

    problem.add_equation("Ma_eq_1 = (f_A-Mab-Pab-Pa)/(1+np.exp(Eb_Ma.v))")
    problem.add_equation("Mb_eq_2 = (f_B-Mab-Pab-Pb)/(1+np.exp(Eb_Mb.v))")
    problem.add_equation("Mab_eq_3 = 0.5*(1-Pab+Ma*np.exp(-(Eb_Mab.v-Eb_Ma.v))-np.sqrt( (1-Pab+Ma*np.exp(-(Eb_Mab.v-Eb_Ma.v)))**2 -4*Ma*np.exp(-(Eb_Mab.v-Eb_Ma.v))*(f_B-Pab-Mb-Pb)))")
    problem.add_equation("Mab_eq_4 = 0.5*(1-Pab+Mb*np.exp(-(Eb_Mab.v-Eb_Mb.v))-np.sqrt( (1-Pab+Mb*np.exp(-(Eb_Mab.v-Eb_Mb.v)))**2 -4*Mb*np.exp(-(Eb_Mab.v-Eb_Mb.v))*(f_A-Pab-Ma-Pa)) )")

    problem.add_equation("Pa_eq_5 = (f_A-Mab-Pab-Ma)/(1+np.exp(Eb_Pa.v))")
    problem.add_equation("Pb_eq_6 = (f_B-Mab-Pab-Mb)/(1+np.exp(Eb_Pb.v))")
    problem.add_equation("Pab_eq_7 = 0.5*(1-Mab+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1-Mab+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B-Mab-Mb-Pb)))")
    problem.add_equation("Pab_eq_8 = 0.5*(1-Mab+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1-Mab+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A-Mab-Ma-Pa)))")

    # Elastic ODE (G, K already scaled by tau; keep /Ly)
    problem.add_equation("dt(u_el) + G.v*u_el = K.v*(V_B-V_A)/Ly.v")

    # Chemical potential gradients (fixed a small typo at the end of grad_mu_fB)
    problem.add_equation("grad_mu_fA = -1*( np.log((h_log+f_D)/(h_log+f_D-Mab-Pab))*dx(f_B) +f_B*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Mab-Pab)*dx(f_D-Mab-Pab)) + 1/(h+f_A-Mab-Pab)*dx(f_A-Mab-Pab) -1/(h+f_A-Mab-Pab-Ma-Pa)*dx(f_A-Mab-Pab-Ma-Pa) )")
    problem.add_equation("grad_mu_fB = -1*( np.log((h_log+f_D)/(h_log+f_D-Mab-Pab))*dx(f_A) +f_A*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Mab-Pab)*dx(f_D-Mab-Pab)) + 1/(h+f_B-Mab-Pab)*dx(f_B-Mab-Pab) -1/(h+f_B-Mab-Pab-Mb-Pb)*dx(f_B-Mab-Pab-Mb-Pb) )")

    # Friction coefficient (eta is ND; Ly, Lz are ND; integral is over ND x)
    problem.add_equation("Coeff_fri = Lz.v/Ly.v*eta.v*d3.Integrate((f_D),('x'))")

    # Forces (gamma == 1 now, so viscous term is -V_*; other coefficients are ND)
    problem.add_equation("F_fA_vis = -gamma.v*V_A")
    problem.add_equation("F_fA_fri = Coeff_fri*(V_B-V_A)")
    problem.add_equation("F_fA_ent = -n_s.v*d3.Integrate(f_A*(grad_mu_fA),('x'))")
    problem.add_equation("F_fA_act = -act.v*d3.Integrate(f_A*(Mab),('x'))")
    problem.add_equation("F_fA_ela = -E.v*Lz.v*d3.Integrate(f_A*(u_el*f_D),('x'))")

    problem.add_equation("F_fB_vis = -gamma.v*V_B")
    problem.add_equation("F_fB_fri = -Coeff_fri*(V_B-V_A)")
    problem.add_equation("F_fB_ent = -n_s.v*d3.Integrate(f_B*(grad_mu_fB),('x'))")
    problem.add_equation("F_fB_act = act.v*n_s.v*d3.Integrate(f_B*(Mab),('x'))")
    problem.add_equation("F_fB_ela = E.v*Lz.v*d3.Integrate(f_B*(u_el*f_D),('x'))")

    problem.add_equation("F_A = F_fA_vis + F_fA_fri + F_fA_ent + F_fA_act + F_fA_ela")
    problem.add_equation("F_B = 0")
    # Velocity closure (now dimensionless: gamma == 1, Coeff_fri is ND)
    problem.add_equation("V_B = 1/(gamma.v+Coeff_fri)*(F_fB_ent + F_fB_act)")
    problem.add_equation("V_A = 0")

    u_el['g'] = 0

    # Build solver
    solver = problem.build_solver(timestepper, ncc_cutoff=1e-4)
    solver.stop_sim_time = stop_time

    # Ensure output folder exists
    os.makedirs(folder, exist_ok=True)

    # Save setup
    date = datetime.datetime.now()
    analysis = solver.evaluator.add_file_handler(folder, sim_dt=stop_time / N_save, max_writes=N_save)
    analysis.add_tasks(solver.state, layout='g')

    ListCoeffSave = {obj for name, obj in locals().items() if isinstance(obj, LS.Coefficient)}
    with open(f"{folder}/sparameters_{folder_name}.csv", 'w', newline='') as filecsv:
        fieldnames = ['name', 'value']
        writer = csv.DictWriter(filecsv, fieldnames=fieldnames)
        writer.writeheader()
        for Coeff in ListCoeffSave:
            LS.function_save_parameters(writer, fieldnames, Coeff)

    # Main loop
    logger.info("Start")
    j = 0
    t_step = 0
    T_N0 = datetime.datetime.now()
    T_N1 = T_N0  # initialize for duration

    while solver.proceed:
        t_step += 1
        solver.step(timestep)

        if solver.iteration % int(stop_time / (N_save * timestep)) == 0:
            j += 1
            T_N1 = datetime.datetime.now()
            T_LEFT = (T_N1 - T_N0) * (N_save - j)
            logger.info('%i/%i, T=%0.2e, t_left = %s', j, N_save, solver.sim_time, str(T_LEFT))
            T_N0 = datetime.datetime.now()

            if debug_plots and (j % 1 == 0 and j < 10 or j % 10 == 0):
                f_A.change_scales(1); f_B.change_scales(1); f_D.change_scales(1)
                Ma.change_scales(1); Ma_eq_1.change_scales(1)
                Mb.change_scales(1); Mb_eq_2.change_scales(1)
                Mab.change_scales(1); Mab_eq_3.change_scales(1); Mab_eq_4.change_scales(1)
                grad_mu_fB.change_scales(1); C_Mab_fA.change_scales(1)
                C_Mab_Mab_inv.change_scales(1)
                Pa.change_scales(1); Pb.change_scales(1); Pab.change_scales(1)

                fig, axs = plt.subplots(2, dpi=120)
                fig.suptitle(str(V_B['g'][0]))
                axs[0].plot(x[0], f_A['g'], color='blue', alpha=0.5)
                axs[0].plot(x[0], f_B['g'], color='red', alpha=0.5)
                axs[0].plot(x[0], Ma['g'], color='blue', label="$M^{a}$")
                axs[0].plot(x[0], Mb['g'], color='red', label="$M^{b}$")
                axs[0].plot(x[0], Mab['g'], color='purple', label="$M^{ab}$")

                axs[1].plot(x[0], f_A['g'], color='blue', alpha=0.5)
                axs[1].plot(x[0], f_B['g'], color='red', alpha=0.5)
                axs[1].plot(x[0], Pa['g'], color='blue', label="$P^{a}$")
                axs[1].plot(x[0], Pb['g'], color='red', label="$P^{b}$")
                axs[1].plot(x[0], Pab['g'], color='violet', label="$P^{ab}$")
                plt.close(fig)

    # Return path to saved HDF5
    h5_path = f"{folder}/{folder_name}_s1.h5"
    print(h5_path)
    print("\nduration:")
    print(T_N1 - date)

    # Cleanup to release memory across batch runs
    try:
        plt.close('all')
    except Exception:
        pass
    try:
        import gc
        del solver
        del problem
        gc.collect()
    except Exception:
        pass

    return h5_path
