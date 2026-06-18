"""Models 2 & 3 without hyperfine: the bare 7-level NV (ground + excited triplet + ISC)."""
# ruff: noqa: F405

import numpy as np
import qutip as qt

from ._base import *  # noqa: F403


def H_no(b, om_r):
    """
    Calculates the Hamiltonian for a given magnetic field and Rabi frequeNCy.

    Parameters:
    b (np.array): Magnetic field vector B0(B_amp,phi_B,theta_B).
    om_r (float): Rabi frequeNCy.

    Returns:
    list: A list containing the Hamiltonian Qobj terms.
    """
    Ham_0 = [
        D_GS * SZ_GS**2
        + E_GS * (SX_GS**2 - SY_GS**2)
        + MU_E * np.dot(b, S_GS)
        + D_ES * SZ_ES**2
        + E_ES * (SX_ES**2 - SY_ES**2)
        + MU_E * np.dot(b, S_ES)
    ]
    H_int = [
        [np.sqrt(2) * om_r * (SX_GS), "cos(w*t)*cos(p)"],
        [np.sqrt(2) * om_r * (SY_GS), "cos(w*t)*sin(p)"],
        [np.sqrt(2) * om_r * (SX_ES), "cos(w*t)*cos(p)"],
        [np.sqrt(2) * om_r * (SY_ES), "cos(w*t)*sin(p)"],
    ]
    return [*Ham_0, *H_int]


def L_no(w_p, k_index=K_IND, K_S=K_S):
    """
    Returns the Lindblad operators of the system, iNCluding optical transitions based on the given k_index.

    Parameters:
    - w_p (float): Laser pump rate.
    - k_index (int, optional): Index for the optical transition rates. Defaults to K_IND.

    Returns:
    - c_ops (list): List of Lindblad operators Qobj.
    """
    k41 = K_S[k_index][0]
    k52 = K_S[k_index][0]
    k63 = K_S[k_index][0]
    k57 = K_S[k_index][2]
    k67 = K_S[k_index][2]
    k47 = K_S[k_index][1]
    k71 = K_S[k_index][3]
    k72 = K_S[k_index][4]
    k73 = K_S[k_index][4]

    c_ops = []

    c_ops.append(
        np.sqrt(w_p) * (EXCITED[1] * GROUND[1].dag())
    )  # N1 to N4 #type: ignore
    c_ops.append(
        np.sqrt(w_p) * (EXCITED[2] * GROUND[2].dag())
    )  # N2 to N5 #type: ignore
    c_ops.append(
        np.sqrt(w_p) * (EXCITED[0] * GROUND[0].dag())
    )  # N3 to N6 #type: ignore

    c_ops.append(
        np.sqrt(k41) * (GROUND[1] * EXCITED[1].dag())
    )  # N4 to N1 #type: ignore
    c_ops.append(np.sqrt(k71) * (GROUND[1] * ISC.dag()))  # N7 to N1 #type: ignore

    c_ops.append(
        np.sqrt(k52) * (GROUND[2] * EXCITED[2].dag())
    )  # N5 to N2 #type: ignore
    c_ops.append(np.sqrt(k72) * (GROUND[2] * ISC.dag()))  # N7 to N2 #type: ignore

    c_ops.append(
        np.sqrt(k63) * (GROUND[0] * EXCITED[0].dag())
    )  # N6 to N3 #type: ignore
    c_ops.append(np.sqrt(k73) * (GROUND[0] * ISC.dag()))  # N7 to N3 #type: ignore

    c_ops.append(np.sqrt(k47) * (ISC * EXCITED[1].dag()))  # N4 to N7 #type: ignore
    c_ops.append(np.sqrt(k57) * (ISC * EXCITED[2].dag()))  # N5 to N7 #type: ignore
    c_ops.append(np.sqrt(k67) * (ISC * EXCITED[0].dag()))  # N6 to N7 #type: ignore
    # Add collapse operators for decohereNCe
    c_ops.append(np.sqrt(GAMMA_GS[1]) * SZ_GS)
    c_ops.append(np.sqrt(GAMMA_GS[0] / 2) * (SM_GS))
    c_ops.append(np.sqrt(GAMMA_GS[0] / 2) * (SP_GS))
    c_ops.append(np.sqrt(GAMMA_ES[1]) * SZ_ES)
    c_ops.append(np.sqrt(GAMMA_ES[0] / 2) * (SM_ES))
    c_ops.append(np.sqrt(GAMMA_ES[0] / 2) * (SP_ES))
    return c_ops


def dynamics_no(
    dt,
    init_state,
    b=None,
    om=None,
    p=None,
    om_r=None,
    w_p=None,
    k_index=K_IND,
    ti=0.0,
    mode="Free",
    progress_bar="ON",
    i=0,
):
    """
    Perform dynamics simulation based on the given parameters, iNCluding optical transition rates index.
    Where, when using k_index=2 -> K_S[k_index]=[62.7,12.97,80.0,3.45,1.08], and the full K_S list is:
        K_S=[[66.0,0.0,57.0,1.0,0.7],
             [77.0,0.0,30.0,3.3,0.0],
             [62.7,12.97,80.0,3.45,1.08],
             [63.2,10.8,60.7,0.8,0.4],
             [67.4,9.9,96.6,4.83,1.055]]
    Parameters:
    - dt (float): Time step for the calculations.
    - init_state: INITial state for the simulation.
    - b (np.array, optional): Magnetic field vector. Defaults to B0(B_amp,phi_B,theta_B).
    - om (float, optional): Angular frequeNCy of the system. Defaults to OMEGA.
    - om_r (float, optional): Angular frequeNCy for MW-ON evolution. Defaults to OM_R.
    - w_p (float, optional): FrequeNCy for laser-ON evolution. Defaults to W_P.
    - k_index=k_index (int, optional): Index for the optical transition rates. Defaults to K_IND.
    - ti (float, optional): INITial time for the simulation. Defaults to 0.0.
    - mode (str, optional): Mode of the simulation. Can be "Free", "MW", "Laser", or "Laser-MW". Defaults to "Free".
    - progress_bar (str, optional): Progress bar option. Can be "ON" or "OFF". Defaults to "ON".
    - i (int, optional): Iteration number. Defaults to 0.

    Returns:
    - tf (float): Final time of the simulation.
    - result: Result of the simulation.
    """
    # Default values
    if b is None:
        b = B
    if om is None:
        om = OMEGA
    if p is None:
        p = PHI
    if om_r is None:
        om_r = OM_R
    if w_p is None:
        w_p = W_P

    # Arguments for the Hamiltonian
    args = {"w": om, "p": p}

    # Define the time resolution
    t_bins = 1000 if dt <= 5 else 5000

    tf = ti + dt

    # Define collapse operators and Hamiltonian based on mode
    match mode:
        case "Free":
            c_ops = L_no(0.0, k_index=k_index)
            Ham = H_no(b, 0.0)
        case "MW":
            c_ops = L_no(0.0, k_index=k_index)
            Ham = H_no(b, om_r)
        case "Laser":
            c_ops = L_no(w_p, k_index=k_index)
            Ham = H_no(b, 0.0)
        case "Laser-MW":
            c_ops = L_no(w_p, k_index=k_index)
            Ham = H_no(b, om_r)
        case _:
            raise ValueError('mode must be one of "Free", "MW", "Laser", or "Laser-MW"')

    # Call the master equation solver
    match progress_bar:
        case "OFF":
            result = qt.mesolve(
                Ham,
                init_state,
                np.linspace(ti, tf, t_bins + 1),
                c_ops,
                args=args,
                options={"store_states": True},
            )
        case "ON":
            print(f"{mode} {int(i + 1)} \n ti    |    tf \n {ti:.2f} | {tf:.2f}")
            result = qt.mesolve(
                Ham,
                init_state,
                np.linspace(ti, tf, t_bins + 1),
                c_ops,
                args=args,
                options={"store_states": True, "progress_bar": "tqdm"},
            )
        case _:
            raise ValueError('progress_bar must be "ON" or "OFF"')

    return tf, result
