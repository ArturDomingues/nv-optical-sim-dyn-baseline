################### Imports #############################

import numpy as np
import qutip as qt
from .utils import B0, construct_spin_matrices

###################### Constants ###########################

# Strain terms (MHz) 
E_GS=0.0 #7.5 is an usual value
E_ES=70.0

# Zero-field splitting (MHz)
D_GS = 2870.0
D_ES = 1420.0

# Hyperfine interaction parameters (MHz)
# Model 1 & 3
A_GS=3.03,3.65
A_ES=-57.8,-39.2

# Model 2
A_PAR = 3.4,-58.1
A_PERP = 7.8,-77.0
A_ANI = 0.0, 0.0
A_PERP_PRIME = 0.0,0.0
PHI_H = 0.0

# Magnetic moments (MHz/G)
# NV
MU_E = 2.8 # (Mhz/G)
# ^{15} N
MU_N = 431.7*1e-6 # (MHz/G)

# t1 times (microseconds)
# NV
T1_GS=1e3
T1_ES=1e3
# ^{15} N
T1_N=1e5

# t2 times (microseconds)
# NV
T2_GS=1.5
T2_ES=6*1e-3
# ^{15} N
T2_N=1e3

# Dephasing and DecohereNCe rates (MHz)
# NV
GAMMA_GS = 1/T1_GS,1/T2_GS
GAMMA_ES = 1/T1_ES,1/T2_ES
# ^{15} N
GAMMA_N = 1/T1_N,1/T2_N

# Pump rate
W_P = 1.9

# Rabi frequeNCy (MHz)
OM_R = 15.7 # I had to modify this value to reproduce the results of the paper (original value was 15)

# Driving frequeNCy (MHz)
OMEGA_MG = 0.1

# Choice of model parameters (K_IND)
K_IND=2
# Transition rates
# k_41,k_52,k_63 - radiative rates (K_S[K_IND][0])
# k_47 - non-radiative rate from excided state to ISC (K_S[K_IND][1])
# k_57,k_67 - non-radiative rate from excided state to ISC (K_S[K_IND][2])
# k_71 - non-radiative rate from ISC to GROUND state (K_S[K_IND][3])
# k_72,k_73 - non-radiative rates from ISC to GROUND state (K_S[K_IND][4])
K_S=[[66.0,0.0,57.0,1.0,0.7],
    [77.0,0.0,30.0,3.3,0.0],
    [62.7,12.97,80.0,3.45,1.08],
    [63.2,10.8,60.7,0.8,0.4],
    [67.4,9.9,96.6,4.83,1.055],
    [64.0,11.8,79.8,5.6,0.0]]

# Dimension of the Hilbert space
DIM = 7

# States
# NV
EXCITED = qt.basis(DIM, 4), qt.basis(DIM, 3),qt.basis(DIM, 5) # |+1>_es, |0>_es, |-1>_es
ISC = qt.basis(DIM, 6)
GROUND = qt.basis(DIM, 1), qt.basis(DIM, 0), qt.basis(DIM, 2) # |+1>_gs, |0>_gs, |-1>_gs
# ^{15} N
NIT = qt.basis(2, 0), qt.basis(2, 1)

N1, N2, N3 = (
    GROUND[1] * GROUND[1].dag(), #type: ignore
    GROUND[2] * GROUND[2].dag(), #type: ignore
    GROUND[0] * GROUND[0].dag(), #type: ignore
)
N4, N5, N6 = (
    EXCITED[1] * EXCITED[1].dag(), #type: ignore
    EXCITED[2] * EXCITED[2].dag(), #type: ignore
    EXCITED[0] * EXCITED[0].dag(), #type: ignore
)
N7, NC = (
    ISC * ISC.dag(), #type: ignore
    GROUND[2] * GROUND[1].dag(), #type: ignore
)

# Operators
# MV
SX_GS, SY_GS, SZ_GS = construct_spin_matrices(GROUND)
SX_ES, SY_ES, SZ_ES = construct_spin_matrices(EXCITED)
SM_GS,SP_GS= SX_GS - 1j * SY_GS, SX_GS + 1j * SY_GS
SM_ES,SP_ES= SX_ES - 1j * SY_ES, SX_ES + 1j * SY_ES

S_GS=np.array([SX_GS, SY_GS, SZ_GS])
S_ES=np.array([SX_ES, SY_ES, SZ_ES])
ID_NV=qt.qeye(DIM)

# ^{15} N
SX_N, SY_N, SZ_N = qt.sigmax()*0.5, qt.sigmay()*0.5, qt.sigmaz()*0.5
SM_N,SP_N=SX_N-1j*SY_N,SX_N+1j*SY_N

S_N=np.array([SX_N,SY_N,SZ_N])
ID_N15 = qt.qeye(2)

# Static Magnetic Field (G)
B = B0(100.0,0.0,0.0) # (G)

###################### Models ###########################

##############################################
################ Model 1 #####################
##############################################

def H_mg(om_r):
    """Returns the Hamiltonian of the system based on whether the MW is on or off
    Parameters:
        om_r (float) - Rabi frequeNCy

    Returns:
        Ham_0 (list) - list of the Hamiltonian terms and their time dependeNCe
    """
    Ham_0 = [[0.5 * om_r * (GROUND[1]*GROUND[2].dag()), "exp(1j*w*t)"],  #type: ignore
           [0.5 * om_r * (GROUND[2]*GROUND[1].dag()), "exp(-1j*w*t)"]] #type: ignore
    return Ham_0


def L_mg(w_p,k_index=K_IND, K_S=K_S):
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

    c_ops.append(np.sqrt(w_p) * (EXCITED[1] * GROUND[1].dag()))  # N1 to N4 #type: ignore 
    c_ops.append(np.sqrt(w_p) * (EXCITED[2] * GROUND[2].dag()))  # N2 to N5 #type: ignore
    c_ops.append(np.sqrt(w_p) * (EXCITED[0] * GROUND[0].dag())) # N3 to N6  #type: ignore

    c_ops.append(np.sqrt(k41) * (GROUND[1] * EXCITED[1].dag()))  # N4 to N1 #type: ignore
    c_ops.append(np.sqrt(k71) * (GROUND[1] * ISC.dag()))  # N7 to N1    #type: ignore

    c_ops.append(np.sqrt(k52) * (GROUND[2] * EXCITED[2].dag()))  # N5 to N2 #type: ignore
    c_ops.append(np.sqrt(k72) * (GROUND[2] * ISC.dag()))  # N7 to N2 #type: ignore

    c_ops.append(np.sqrt(k63) * (GROUND[0] * EXCITED[0].dag()))  # N6 to N3 #type: ignore
    c_ops.append(np.sqrt(k73) * (GROUND[0] * ISC.dag()))  # N7 to N3    #type: ignore

    c_ops.append(np.sqrt(k47) * (ISC * EXCITED[1].dag()))  # N4 to N7   #type: ignore
    c_ops.append(np.sqrt(k57) * (ISC * EXCITED[2].dag()))  # N5 to N7   #type: ignore
    c_ops.append(np.sqrt(k67) * (ISC * EXCITED[0].dag()))  # N6 to N7   #type: ignore
    
    # Add collapse operators for decohereNCe
    c_ops.append(np.sqrt(GAMMA_GS[1]) * SZ_GS)
    c_ops.append(np.sqrt(GAMMA_GS[0]/2) * (SM_GS))
    c_ops.append(np.sqrt(GAMMA_GS[0]/2) * (SP_GS))
    c_ops.append(np.sqrt(GAMMA_ES[1]) * SZ_ES)
    c_ops.append(np.sqrt(GAMMA_ES[0]/2) * (SM_ES))
    c_ops.append(np.sqrt(GAMMA_ES[0]/2) * (SP_ES))
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
    if om is None: om = OMEGA_MG
    if om_r is None: om_r = OM_R
    if w_p is None: w_p = W_P

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

##############################################
############### Model 1 + HF #################
##############################################

def H_mg_hf(om_r,A_GS=A_GS,A_ES=A_ES):
    """Returns the Hamiltonian of the system based on whether the MW is on or off
    Parameters:
        om_r (float) - Rabi frequeNCy

    Returns:
        Ham_0 (list) - list of the Hamiltonian terms and their time dependeNCe
    """
    Ham_0 = [[(0.5 * om_r * (GROUND[1]*GROUND[2].dag()))&ID_N15, "exp(1j*w*t)"],  #type: ignore
           [(0.5 * om_r * (GROUND[2]*GROUND[1].dag()))&ID_N15, "exp(-1j*w*t)"], #type: ignore
           A_GS[0]*(SZ_GS&SZ_N) + A_GS[1]*((SX_GS&SX_N) + (SY_GS&SY_N)),
           A_ES[0]*(SZ_ES&SZ_N) + A_ES[1]*((SX_ES&SX_N) + (SY_ES&SY_N))]
    H_n=[[ID_NV&(0.5*om_r*(MU_N/MU_E)*(NIT[0]*NIT[1].dag())),"exp(1j*w*t)"],   #type: ignore
          [ID_NV&(0.5*om_r*(MU_N/MU_E)*(NIT[1]*NIT[0].dag())),"exp(-1j*w*t)"]] #type: ignore
    return [*Ham_0 , *H_n]

def L_mg_hf(w_p,k_index=K_IND, K_S=K_S):
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

    c_ops.append((np.sqrt(w_p) * (EXCITED[1] * GROUND[1].dag()))&ID_N15)  # N1 to N4 #type: ignore
    c_ops.append((np.sqrt(w_p) * (EXCITED[2] * GROUND[2].dag()))&ID_N15)  # N2 to N5 #type: ignore
    c_ops.append((np.sqrt(w_p) * (EXCITED[0] * GROUND[0].dag()))&ID_N15) # N3 to N6 #type: ignore

    c_ops.append((np.sqrt(k41) * (GROUND[1] * EXCITED[1].dag()))&ID_N15)  # N4 to N1 #type: ignore
    c_ops.append((np.sqrt(k71) * (GROUND[1] * ISC.dag()))&ID_N15)  # N7 to N1 #type: ignore

    c_ops.append((np.sqrt(k52) * (GROUND[2] * EXCITED[2].dag()))&ID_N15)  # N5 to N2 #type: ignore
    c_ops.append((np.sqrt(k72) * (GROUND[2] * ISC.dag()))&ID_N15)  # N7 to N2 #type: ignore

    c_ops.append((np.sqrt(k63) * (GROUND[0] * EXCITED[0].dag()))&ID_N15)  # N6 to N3 #type: ignore
    c_ops.append((np.sqrt(k73) * (GROUND[0] * ISC.dag()))&ID_N15)  # N7 to N3 #type: ignore

    c_ops.append((np.sqrt(k47) * (ISC * EXCITED[1].dag()))&ID_N15)  # N4 to N7 #type: ignore
    c_ops.append((np.sqrt(k57) * (ISC * EXCITED[2].dag()))&ID_N15)  # N5 to N7 #type: ignore
    c_ops.append((np.sqrt(k67) * (ISC * EXCITED[0].dag()))&ID_N15)  # N6 to N7 #type: ignore
    # Add collapse operators for decohereNCe   
    c_ops.append((np.sqrt(GAMMA_GS[1]) * SZ_GS)&ID_N15)
    c_ops.append((np.sqrt(GAMMA_GS[0]/2) * (SM_GS))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_GS[0]/2) * (SP_GS))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[1]) * SZ_ES)&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[0]/2) * (SM_ES))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[0]/2) * (SP_ES))&ID_N15)
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[1]) * SZ_N))
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[0]/2) * (SM_N)))
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[0]/2) * (SP_N)))

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
    if om_r is None: om_r = OM_R
    if om is None: om = OMEGA_MG
    if w_p is None: w_p = W_P

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

###############################################
############ Models 2 & 3 No-HF ###############
###############################################

# Driving frequeNCy
OMEGA = D_GS-MU_E*B[2]
# Driving phase
PHI = 0.0

def H_no(b, om_r):
    """
    Calculates the Hamiltonian for a given magnetic field and Rabi frequeNCy.
    
    Parameters:
    b (np.array): Magnetic field vector B0(B_amp,phi_B,theta_B).
    om_r (float): Rabi frequeNCy.
    
    Returns:
    list: A list containing the Hamiltonian Qobj terms.
    """
    Ham_0 = [D_GS*SZ_GS**2+E_GS*(SX_GS**2-SY_GS**2)+MU_E*np.dot(b,S_GS)+
           D_ES*SZ_ES**2+E_ES*(SX_ES**2-SY_ES**2)+MU_E*np.dot(b,S_ES)]
    H_int = [[np.sqrt(2)*om_r*(SX_GS), "cos(w*t)*cos(p)"],
           [np.sqrt(2)*om_r*(SY_GS), "cos(w*t)*sin(p)"],
           [np.sqrt(2)*om_r*(SX_ES), "cos(w*t)*cos(p)"],
           [np.sqrt(2)*om_r*(SY_ES), "cos(w*t)*sin(p)"]]
    return [*Ham_0,*H_int]

def L_no(w_p,k_index=K_IND, K_S=K_S):
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

    c_ops.append(np.sqrt(w_p) * (EXCITED[1] * GROUND[1].dag()))  # N1 to N4 #type: ignore
    c_ops.append(np.sqrt(w_p) * (EXCITED[2] * GROUND[2].dag()))  # N2 to N5 #type: ignore
    c_ops.append(np.sqrt(w_p) * (EXCITED[0] * GROUND[0].dag())) # N3 to N6 #type: ignore

    c_ops.append(np.sqrt(k41) * (GROUND[1] * EXCITED[1].dag()))  # N4 to N1 #type: ignore
    c_ops.append(np.sqrt(k71) * (GROUND[1] * ISC.dag()))  # N7 to N1 #type: ignore

    c_ops.append(np.sqrt(k52) * (GROUND[2] * EXCITED[2].dag()))  # N5 to N2 #type: ignore
    c_ops.append(np.sqrt(k72) * (GROUND[2] * ISC.dag()))  # N7 to N2 #type: ignore

    c_ops.append(np.sqrt(k63) * (GROUND[0] * EXCITED[0].dag()))  # N6 to N3 #type: ignore
    c_ops.append(np.sqrt(k73) * (GROUND[0] * ISC.dag()))  # N7 to N3 #type: ignore

    c_ops.append(np.sqrt(k47) * (ISC * EXCITED[1].dag()))  # N4 to N7 #type: ignore
    c_ops.append(np.sqrt(k57) * (ISC * EXCITED[2].dag()))  # N5 to N7 #type: ignore
    c_ops.append(np.sqrt(k67) * (ISC * EXCITED[0].dag()))  # N6 to N7 #type: ignore
    # Add collapse operators for decohereNCe
    c_ops.append(np.sqrt(GAMMA_GS[1]) * SZ_GS)
    c_ops.append(np.sqrt(GAMMA_GS[0]/2) * (SM_GS))
    c_ops.append(np.sqrt(GAMMA_GS[0]/2) * (SP_GS))
    c_ops.append(np.sqrt(GAMMA_ES[1]) * SZ_ES)
    c_ops.append(np.sqrt(GAMMA_ES[0]/2) * (SM_ES))
    c_ops.append(np.sqrt(GAMMA_ES[0]/2) * (SP_ES))
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
    if b is None: b = B
    if om is None: om = OMEGA
    if p is None: p = PHI
    if om_r is None: om_r = OM_R
    if w_p is None: w_p = W_P

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

###############################################
################## Model 2 ####################
###############################################

def H_dua_hf(b, om_r,phi_h):
    """
    Calculate the Hamiltonian for a given magnetic field and resonant frequeNCy, iNCluding hyperfine interactions.
    
    Parameters:
    b (numpy.ndarray): Magnetic field vector [bx, by, bz].
    om_r: Resonant frequeNCy.
    
    Returns:
    list: List of Hamiltonian terms.
    """
    Ham_0 = [((D_GS*SZ_GS**2 + E_GS*(SX_GS**2-SY_GS**2) + MU_E*np.dot(b, S_GS))&ID_N15)+
           ((D_ES*SZ_ES**2 + E_ES*(SX_ES**2-SY_ES**2) + MU_E*np.dot(b, S_ES))&ID_N15)+
           A_PAR[0]*(SZ_GS&SZ_N) + A_PERP[0]/4*((SP_GS&SM_N)+(SM_GS&SP_N))+
           A_PERP_PRIME[0]/4*((SP_GS&SP_N)*np.exp(-2j*phi_h) + (SM_GS&SM_N)*np.exp(2j*phi_h))+
           A_ANI[0]/2*((SP_GS&SZ_N) + (SZ_GS&SP_N)*np.exp(-1j*phi_h) + (SM_GS&SZ_N) + (SZ_GS&SM_N)*np.exp(1j*phi_h))+
           A_PAR[1]*(SZ_ES&SZ_N) + A_PERP[1]/4*((SP_ES&SM_N)+(SM_ES&SP_N))+
           A_PERP_PRIME[1]/4*((SP_ES&SP_N)*np.exp(-2j*phi_h) + (SM_ES&SM_N)*np.exp(2j*phi_h))+
           A_ANI[1]/2*((SP_ES&SZ_N) + (SZ_ES&SP_N)*np.exp(-1j*phi_h) + (SM_ES&SZ_N) + (SZ_ES&SM_N)*np.exp(1j*phi_h))]
    H_n = [ID_NV&MU_N*np.dot(b, S_N)]
    H_int = [[(np.sqrt(2)*om_r*SX_GS)&ID_N15, "cos(w*t)*cos(p)"],
             [(np.sqrt(2)*om_r*SY_GS)&ID_N15, "cos(w*t)*sin(p)"],
             [(np.sqrt(2)*om_r*SX_ES)&ID_N15, "cos(w*t)*cos(p)"],
             [(np.sqrt(2)*om_r*SY_ES)&ID_N15, "cos(w*t)*sin(p)"],
             [ID_NV&(2*om_r/MU_E*MU_N*SX_N), "cos(w*t)*cos(p)"],
             [ID_NV&(2*om_r/MU_E*MU_N*SY_N), "cos(w*t)*sin(p)"]]
    return [*Ham_0, *H_n, *H_int]

def L_dua_hf(w_p,k_index=K_IND, K_S=K_S):
    """
    Returns the Lindblad operators of the system, iNCluding optical transitions based on the given k_index.

    Parameters:
    - w_p (float): Laser pump rate.
    - k_index (int, optional): Index for the optical transition rates. Defaults to K_IND.

    Returns:
    - c_ops (list): List of Lindblad operators.
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

    c_ops.append((np.sqrt(w_p) * (EXCITED[1] * GROUND[1].dag()))&ID_N15)  # N1 to N4 #type: ignore
    c_ops.append((np.sqrt(w_p) * (EXCITED[2] * GROUND[2].dag()))&ID_N15)  # N2 to N5 #type: ignore
    c_ops.append((np.sqrt(w_p) * (EXCITED[0] * GROUND[0].dag()))&ID_N15) # N3 to N6 #type: ignore

    c_ops.append((np.sqrt(k41) * (GROUND[1] * EXCITED[1].dag()))&ID_N15)  # N4 to N1 #type: ignore
    c_ops.append((np.sqrt(k71) * (GROUND[1] * ISC.dag()))&ID_N15)  # N7 to N1 #type: ignore

    c_ops.append((np.sqrt(k52) * (GROUND[2] * EXCITED[2].dag()))&ID_N15)  # N5 to N2 #type: ignore
    c_ops.append((np.sqrt(k72) * (GROUND[2] * ISC.dag()))&ID_N15)  # N7 to N2 #type: ignore

    c_ops.append((np.sqrt(k63) * (GROUND[0] * EXCITED[0].dag()))&ID_N15)  # N6 to N3 #type: ignore
    c_ops.append((np.sqrt(k73) * (GROUND[0] * ISC.dag()))&ID_N15)  # N7 to N3 #type: ignore

    c_ops.append((np.sqrt(k47) * (ISC * EXCITED[1].dag()))&ID_N15)  # N4 to N7 #type: ignore
    c_ops.append((np.sqrt(k57) * (ISC * EXCITED[2].dag()))&ID_N15)  # N5 to N7 #type: ignore
    c_ops.append((np.sqrt(k67) * (ISC * EXCITED[0].dag()))&ID_N15)  # N6 to N7 #type: ignore
    # Add collapse operators for decohereNCe   
    c_ops.append((np.sqrt(GAMMA_GS[1]) * SZ_GS)&ID_N15)
    c_ops.append((np.sqrt(GAMMA_GS[0]/2) * (SM_GS))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_GS[0]/2) * (SP_GS))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[1]) * SZ_ES)&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[0]/2) * (SM_ES))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[0]/2) * (SP_ES))&ID_N15)
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[1]) * SZ_N))
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[0]/2) * (SM_N)))
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[0]/2) * (SP_N)))
    return c_ops

def dynamics_dua_hf(
    dt,
    init_state,
    b=None,
    om_r=None,
    om=None,
    p=None,
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
    - init_state (qt.Qobj): INITial quantum state of the system.
    - b (np.array, optional): Magnetic field vector. Defaults to B0(B_amp,phi_B,theta_B).
    - om_r (float, optional): Rabi frequeNCy for microwave interactions. Defaults to OM_R.
    - om (float, optional): Angular frequeNCy of the system. Defaults to OMEGA.
    - p (float, optional): Microwave phase. Defaults to 0.0.
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
    if b is None: b = B
    if om_r is None: om_r = OM_R
    if om is None: om = OMEGA
    if p is None: p = PHI
    if w_p is None: w_p = W_P
    
    # Arguments for the Hamiltonian
    args = {"w": om, "p": p}
    # Time resolution based on dt
    t_bins = 1000 if dt <= 5 else 5000
    
    tf = ti + dt
    
    # Define Hamiltonian and collapse operators based on mode
    match mode:
        case "Free":
            Ham = H_dua_hf(b, 0.0, PHI_H)
            c_ops = L_dua_hf(0.0, k_index=k_index)
        case "MW":
            Ham = H_dua_hf(b, om_r, PHI_H)
            c_ops = L_dua_hf(0.0, k_index=k_index)
        case "Laser":
            Ham = H_dua_hf(b, 0.0, PHI_H)
            c_ops = L_dua_hf(w_p, k_index=k_index)
        case "Laser-MW":
            Ham = H_dua_hf(b, om_r, PHI_H)
            c_ops = L_dua_hf(w_p, k_index=k_index)
        case _:
            raise ValueError('mode must be one of "Free", "MW", "Laser", or "Laser-MW"')
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

###############################################
################## Model 3 ####################
###############################################

def H_doh_hf(b, om_r):
    """
    Calculate the Hamiltonian for a given magnetic field and resonant frequeNCy.
    
    Parameters:
    b (numpy.ndarray): Magnetic field vector [bx,by,bz].
    om_r: Resonant frequeNCy.
    
    Returns:
    list: List of Hamiltonian terms.
    """
    
    Ham_0 = [((D_GS*SZ_GS**2+E_GS*(SX_GS**2-SY_GS**2)+MU_E*np.dot(b,S_GS))&ID_N15)+
           ((D_ES*SZ_ES**2+E_ES*(SX_ES**2-SY_ES**2)+MU_E*np.dot(b,S_ES))&ID_N15)+
           A_GS[0]*(SZ_GS&SZ_N) + A_GS[1]*((SX_GS&SX_N) + (SY_GS&SY_N))+
           A_ES[0]*(SZ_ES&SZ_N) + A_ES[1]*((SX_ES&SX_N) + (SY_ES&SY_N))]
    H_n = [ID_NV&(MU_N*np.dot(b,S_N))]
    H_int=[[(np.sqrt(2)*om_r*SX_GS)&ID_N15, "cos(w*t)*cos(p)"],
           [(np.sqrt(2)*om_r*SY_GS)&ID_N15, "cos(w*t)*sin(p)"],
           [(np.sqrt(2)*om_r*SX_ES)&ID_N15, "cos(w*t)*cos(p)"],
           [(np.sqrt(2)*om_r*SY_ES)&ID_N15, "cos(w*t)*sin(p)"],
           [ID_NV&(2*om_r/MU_E*MU_N*SX_N), "cos(w*t)*cos(p)"],
           [ID_NV&(2*om_r/MU_E*MU_N*SY_N), "cos(w*t)*sin(p)"]]
    return [*Ham_0,*H_n,*H_int] 

def L_doh_hf(w_p,k_index=K_IND, K_S=K_S):
    """
    Returns the Lindblad operators of the system, iNCluding optical transitions based on the given k_index.

    Parameters:
    - w_p (float): Laser pump rate.
    - k_index (int, optional): Index for the optical transition rates. Defaults to K_IND.

    Returns:
    - c_ops (list): List of Lindblad operators.
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

    c_ops.append((np.sqrt(w_p) * (EXCITED[1] * GROUND[1].dag()))&ID_N15)  # N1 to N4 #type: ignore
    c_ops.append((np.sqrt(w_p) * (EXCITED[2] * GROUND[2].dag()))&ID_N15)  # N2 to N5 #type: ignore
    c_ops.append((np.sqrt(w_p) * (EXCITED[0] * GROUND[0].dag()))&ID_N15) # N3 to N6 #type: ignore

    c_ops.append((np.sqrt(k41) * (GROUND[1] * EXCITED[1].dag()))&ID_N15)  # N4 to N1 #type: ignore
    c_ops.append((np.sqrt(k71) * (GROUND[1] * ISC.dag()))&ID_N15)  # N7 to N1 #type: ignore

    c_ops.append((np.sqrt(k52) * (GROUND[2] * EXCITED[2].dag()))&ID_N15)  # N5 to N2 #type: ignore
    c_ops.append((np.sqrt(k72) * (GROUND[2] * ISC.dag()))&ID_N15)  # N7 to N2 #type: ignore

    c_ops.append((np.sqrt(k63) * (GROUND[0] * EXCITED[0].dag()))&ID_N15)  # N6 to N3 #type: ignore
    c_ops.append((np.sqrt(k73) * (GROUND[0] * ISC.dag()))&ID_N15)  # N7 to N3 #type: ignore

    c_ops.append((np.sqrt(k47) * (ISC * EXCITED[1].dag()))&ID_N15)  # N4 to N7 #type: ignore
    c_ops.append((np.sqrt(k57) * (ISC * EXCITED[2].dag()))&ID_N15)  # N5 to N7 #type: ignore
    c_ops.append((np.sqrt(k67) * (ISC * EXCITED[0].dag()))&ID_N15)  # N6 to N7 #type: ignore
    # Collapse operators for decohereNCe   
    c_ops.append((np.sqrt(GAMMA_GS[1]) * SZ_GS)&ID_N15)
    c_ops.append((np.sqrt(GAMMA_GS[0]/2) * (SM_GS))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_GS[0]/2) * (SP_GS))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[1]) * SZ_ES)&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[0]/2) * (SM_ES))&ID_N15)
    c_ops.append((np.sqrt(GAMMA_ES[0]/2) * (SP_ES))&ID_N15)
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[1]) * SZ_N))
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[0]/2) * (SM_N)))
    c_ops.append(ID_NV&(np.sqrt(GAMMA_N[0]/2) * (SP_N)))
    return c_ops

def dynamics_doh_hf(
    dt,
    init_state,
    b=None,
    om_r=None,
    om=None,
    p=None,
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
    - init_state (qt.Qobj): INITial quantum state of the system.
    - b (np.array, optional): Magnetic field vector. Defaults to B0(B_amp,phi_B,theta_B).
    - om_r (float, optional): Rabi frequeNCy for microwave interactions. Defaults to OM_R.
    - om (float, optional): Angular frequeNCy of the system. Defaults to OMEGA.
    - p (float, optional): Microwave phase. Defaults to 0.0.
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
    if b is None: b = B
    if om_r is None: om_r = OM_R
    if om is None: om = OMEGA
    if p is None: p = PHI
    if w_p is None: w_p = W_P
    
    # Arguments for the Hamiltonian
    args = {"w": om, "p": p}
    # Time resolution based on dt
    t_bins = 1000 if dt <= 5 else 5000
    
    tf = ti + dt
    
    # Define Hamiltonian and collapse operators based on mode
    match mode:
        case "Free":
            Ham = H_doh_hf(b, 0.0)
            c_ops = L_doh_hf(0.0, k_index=k_index)
        case "MW":
            Ham = H_doh_hf(b, om_r)
            c_ops = L_doh_hf(0.0, k_index=k_index)
        case "Laser":
            Ham = H_doh_hf(b, 0.0)
            c_ops = L_doh_hf(w_p, k_index=k_index)
        case "Laser-MW":
            Ham = H_doh_hf(b, om_r)
            c_ops = L_doh_hf(w_p, k_index=k_index)
        case _:
            raise ValueError('mode must be one of "Free", "MW", "Laser", or "Laser-MW"')

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

#########################################################################