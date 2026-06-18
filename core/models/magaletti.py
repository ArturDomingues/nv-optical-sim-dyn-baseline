"""Model 1 (Magaletti): 7-level NV and its 14-level hyperfine extension."""
# ruff: noqa: F405

import numpy as np
import qutip as qt

from ._base import *  # noqa: F403


def H_mg(om_r):
    """Returns the Hamiltonian of the system based on whether the MW is on or off
    Parameters:
        om_r (float) - Rabi frequeNCy

    Returns:
        Ham_0 (list) - list of the Hamiltonian terms and their time dependeNCe
    """
    Ham_0 = [
        [0.5 * om_r * (GROUND[1] * GROUND[2].dag()), "exp(1j*w*t)"],  # type: ignore
        [0.5 * om_r * (GROUND[2] * GROUND[1].dag()), "exp(-1j*w*t)"],
    ]  # type: ignore
    return Ham_0


def L_mg(w_p, k_index=K_IND, K_S=K_S):
    """Returns the Lindblad operators of the system.

    Parameters:
        w_p (float) - Laser pump rate

    Returns:
        c_ops (list) - list of the Lindblad operators
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
    )  # N3 to N6  #type: ignore

    c_ops.append(
        np.sqrt(k41) * (GROUND[1] * EXCITED[1].dag())
    )  # N4 to N1 #type: ignore
    c_ops.append(np.sqrt(k71) * (GROUND[1] * ISC.dag()))  # N7 to N1    #type: ignore

    c_ops.append(
        np.sqrt(k52) * (GROUND[2] * EXCITED[2].dag())
    )  # N5 to N2 #type: ignore
    c_ops.append(np.sqrt(k72) * (GROUND[2] * ISC.dag()))  # N7 to N2 #type: ignore

    c_ops.append(
        np.sqrt(k63) * (GROUND[0] * EXCITED[0].dag())
    )  # N6 to N3 #type: ignore
    c_ops.append(np.sqrt(k73) * (GROUND[0] * ISC.dag()))  # N7 to N3    #type: ignore

    c_ops.append(np.sqrt(k47) * (ISC * EXCITED[1].dag()))  # N4 to N7   #type: ignore
    c_ops.append(np.sqrt(k57) * (ISC * EXCITED[2].dag()))  # N5 to N7   #type: ignore
    c_ops.append(np.sqrt(k67) * (ISC * EXCITED[0].dag()))  # N6 to N7   #type: ignore

    # Add collapse operators for decohereNCe
    c_ops.append(np.sqrt(GAMMA_GS[1]) * SZ_GS)
    c_ops.append(np.sqrt(GAMMA_GS[0] / 2) * (SM_GS))
    c_ops.append(np.sqrt(GAMMA_GS[0] / 2) * (SP_GS))
    c_ops.append(np.sqrt(GAMMA_ES[1]) * SZ_ES)
    c_ops.append(np.sqrt(GAMMA_ES[0] / 2) * (SM_ES))
    c_ops.append(np.sqrt(GAMMA_ES[0] / 2) * (SP_ES))
    return c_ops


def dynamics_mg(
    dt,
    init_state,
    om=None,
    om_r=None,
    w_p=None,
    k_index=K_IND,
    ti=0.0,
    mode="Free",
    progress_bar="ON",
    i=0,
):
    """
    Perform dynamics_mg simulation based on the given parameters, iNCluding optical transition rates index.
    Where, when using k_index=2 -> K_S[k_index]=[62.7,12.97,80.0,3.45,1.08], and the full K_S list is:
        K_S=[[66.0,0.0,57.0,1.0,0.7],
             [77.0,0.0,30.0,3.3,0.0],
             [62.7,12.97,80.0,3.45,1.08],
             [63.2,10.8,60.7,0.8,0.4],
             [67.4,9.9,96.6,4.83,1.055]]
    Parameters:
    - dt (float): Time step for the calculations.
    - init_state: INITial state for the simulation.
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
    if om is None:
        om = OMEGA_MG
    if om_r is None:
        om_r = OM_R
    if w_p is None:
        w_p = W_P

    # Arguments for the Hamiltonian
    args = {"w": om}

    # Define the time resolution
    t_bins = 1000 if dt <= 5 else 5000

    tf = ti + dt

    # Define collapse operators and Hamiltonian based on mode
    match mode:
        case "Free":
            c_ops = L_mg(0.0, k_index=k_index)
            Ham = H_mg(0.0)
        case "MW":
            c_ops = L_mg(0.0, k_index=k_index)
            Ham = H_mg(om_r)
        case "Laser":
            c_ops = L_mg(w_p, k_index=k_index)
            Ham = H_mg(0.0)
        case "Laser-MW":
            c_ops = L_mg(w_p, k_index=k_index)
            Ham = H_mg(om_r)
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
            print(f"{mode} {int(i + 1)} \n ti | tf \n {int(ti)} | {int(tf)}")
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


def H_mg_hf(om_r, A_GS=A_GS, A_ES=A_ES):
    """Returns the Hamiltonian of the system based on whether the MW is on or off
    Parameters:
        om_r (float) - Rabi frequeNCy

    Returns:
        Ham_0 (list) - list of the Hamiltonian terms and their time dependeNCe
    """
    Ham_0 = [
        [(0.5 * om_r * (GROUND[1] * GROUND[2].dag())) & ID_N15, "exp(1j*w*t)"],  # type: ignore
        [(0.5 * om_r * (GROUND[2] * GROUND[1].dag())) & ID_N15, "exp(-1j*w*t)"],  # type: ignore
        A_GS[0] * (SZ_GS & SZ_N) + A_GS[1] * ((SX_GS & SX_N) + (SY_GS & SY_N)),
        A_ES[0] * (SZ_ES & SZ_N) + A_ES[1] * ((SX_ES & SX_N) + (SY_ES & SY_N)),
    ]
    H_n = [
        [ID_NV & (0.5 * om_r * (MU_N / MU_E) * (NIT[0] * NIT[1].dag())), "exp(1j*w*t)"],  # type: ignore
        [
            ID_NV & (0.5 * om_r * (MU_N / MU_E) * (NIT[1] * NIT[0].dag())),
            "exp(-1j*w*t)",
        ],
    ]  # type: ignore
    return [*Ham_0, *H_n]


def L_mg_hf(w_p, k_index=K_IND, K_S=K_S):
    """Returns the Lindblad operators of the system
    Parameters:
        w_p (float) - Laser pump rate
    Returns:
        c_ops (list) - list of the Lindblad operators
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
        (np.sqrt(w_p) * (EXCITED[1] * GROUND[1].dag())) & ID_N15
    )  # N1 to N4 #type: ignore
    c_ops.append(
        (np.sqrt(w_p) * (EXCITED[2] * GROUND[2].dag())) & ID_N15
    )  # N2 to N5 #type: ignore
    c_ops.append(
        (np.sqrt(w_p) * (EXCITED[0] * GROUND[0].dag())) & ID_N15
    )  # N3 to N6 #type: ignore

    c_ops.append(
        (np.sqrt(k41) * (GROUND[1] * EXCITED[1].dag())) & ID_N15
    )  # N4 to N1 #type: ignore
    c_ops.append(
        (np.sqrt(k71) * (GROUND[1] * ISC.dag())) & ID_N15
    )  # N7 to N1 #type: ignore

    c_ops.append(
        (np.sqrt(k52) * (GROUND[2] * EXCITED[2].dag())) & ID_N15
    )  # N5 to N2 #type: ignore
    c_ops.append(
        (np.sqrt(k72) * (GROUND[2] * ISC.dag())) & ID_N15
    )  # N7 to N2 #type: ignore

    c_ops.append(
        (np.sqrt(k63) * (GROUND[0] * EXCITED[0].dag())) & ID_N15
    )  # N6 to N3 #type: ignore
    c_ops.append(
        (np.sqrt(k73) * (GROUND[0] * ISC.dag())) & ID_N15
    )  # N7 to N3 #type: ignore

    c_ops.append(
        (np.sqrt(k47) * (ISC * EXCITED[1].dag())) & ID_N15
    )  # N4 to N7 #type: ignore
    c_ops.append(
        (np.sqrt(k57) * (ISC * EXCITED[2].dag())) & ID_N15
    )  # N5 to N7 #type: ignore
    c_ops.append(
        (np.sqrt(k67) * (ISC * EXCITED[0].dag())) & ID_N15
    )  # N6 to N7 #type: ignore
    # Add collapse operators for decohereNCe
    c_ops.append((np.sqrt(GAMMA_GS[1]) * SZ_GS) & ID_N15)
    c_ops.append((np.sqrt(GAMMA_GS[0] / 2) * (SM_GS)) & ID_N15)
    c_ops.append((np.sqrt(GAMMA_GS[0] / 2) * (SP_GS)) & ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[1]) * SZ_ES) & ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[0] / 2) * (SM_ES)) & ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[0] / 2) * (SP_ES)) & ID_N15)
    c_ops.append(ID_NV & (np.sqrt(GAMMA_N[1]) * SZ_N))
    c_ops.append(ID_NV & (np.sqrt(GAMMA_N[0] / 2) * (SM_N)))
    c_ops.append(ID_NV & (np.sqrt(GAMMA_N[0] / 2) * (SP_N)))

    return c_ops


def dynamics_mg_hf(
    dt,
    init_state,
    om_r=None,
    om=None,
    w_p=None,
    k_index=K_IND,
    ti=0.0,
    mode="Free",
    progress_bar="ON",
    i=0,
):
    """
    Simulate the dynamics of a quantum system under hyperfine interaction using the Hamiltonian and collapse operators.
    iNCluding optical transition rates index.
    Where, when using k_index=2 -> K_S[k_index]=[62.7,12.97,80.0,3.45,1.08], and the full K_S list is:
        K_S=[[66.0,0.0,57.0,1.0,0.7],
             [77.0,0.0,30.0,3.3,0.0],
             [62.7,12.97,80.0,3.45,1.08],
             [63.2,10.8,60.7,0.8,0.4],
             [67.4,9.9,96.6,4.83,1.055]]
    Parameters:
    - dt (float): Total simulation time.
    - init_state (qutip.Qobj): INITial quantum state of the system.
    - om_r (float, optional): Rabi frequeNCy for microwave interactions. Defaults to OM_R.
    - om (float, optional): Angular frequeNCy of the system. Defaults to OMEGA.
    - w_p (float, optional): Laser frequeNCy. Defaults to W_P.
    - k_index(int, optional): Index for the optical transition rates. Defaults to K_IND.
    - ti (float, optional): INITial time of the simulation. Defaults to 0.0.
    - mode (str, optional): Simulation mode. Can be "Free", "MW", "Laser", or "Laser-MW". Defaults to "Free".
    - progress_bar (str, optional): Option to display a progress bar. Can be "ON" or "OFF". Defaults to "ON".
    - i (int, optional): Counter for the progress bar. Defaults to 0.

    Returns:
    - tf (float): Final time of the simulation.
    - result (qutip.solver.Result): Result object containing the simulation output.
    """
    # Default values
    if om_r is None:
        om_r = OM_R
    if om is None:
        om = OMEGA_MG
    if w_p is None:
        w_p = W_P

    # Time resolution based on dt
    t_bins = 1000 if dt <= 5 else 5000

    # Define Hamiltonian and collapse operators based on mode
    match mode:
        case "Free":
            Ham = H_mg_hf(0.0)
            c_ops = L_mg_hf(0.0, k_index=k_index)
        case "MW":
            Ham = H_mg_hf(om_r)
            c_ops = L_mg_hf(0.0, k_index=k_index)
        case "Laser":
            Ham = H_mg_hf(0.0)
            c_ops = L_mg_hf(w_p, k_index=k_index)
        case "Laser-MW":
            Ham = H_mg_hf(om_r)
            c_ops = L_mg_hf(w_p, k_index=k_index)
        case _:
            raise ValueError('mode must be one of "Free", "MW", "Laser", or "Laser-MW"')

    # Arguments for the Hamiltonian
    args = {"w": om}

    tf = ti + dt

    # Solve the master equation
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
            print(f"{mode} {int(i + 1)} \n ti | tf \n {int(ti)} | {int(tf)}")
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
