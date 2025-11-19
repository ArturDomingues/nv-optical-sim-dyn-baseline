################### Imports #############################

import numpy as np
import qutip as qt
from .utils import B0, construct_spin_matrices

###################### Constants ###########################

# Strain terms (MHz) 
E_gs=0.0 #7.5 is an usual value
E_es=70.0

# Zero-field splitting (MHz)
D_gs = 2870.0
D_es = 1420.0

# Hyperfine interaction parameters (MHz)
# Model 1 & 3
a_gs=3.03,3.65
a_es=-57.8,-39.2

# Model 2
A_par = 3.4,-58.1
A_perp = 7.8,-77.0
A_ani = 0.0, 0.0
A_perp_prime = 0.0,0.0
phi_H = 0.0

# Magnetic moments (MHz/G)
# NV
mu_e = 2.8 # (Mhz/G)
# ^{15} N
mu_n = 431.7*1e-6 # (MHz/G)

# t1 times (microseconds)
# NV
t1_gs=1e3
t1_es=1e3
# ^{15} N
t1_n=1e5

# t2 times (microseconds)
# NV
t2_gs=1.5
t2_es=6*1e-3
# ^{15} N
t2_n=1e3

# Dephasing and Decoherence rates (MHz)
# NV
gamma_gs = 1/t1_gs,1/t2_gs
gamma_es = 1/t1_es,1/t2_es
# ^{15} N
gamma_n = 1/t1_n,1/t2_n

# Pump rate
W_p = 1.9

# Rabi frequency (MHz)
Om_r = 15.7 # I had to modify this value to reproduce the results of the paper (original value was 15)

# Driving frequency (MHz)
omega = 0.1

# Choice of model parameters (k_ind)
k_ind=2
# Transition rates
# k_41,k_52,k_63 - radiative rates (K_s[k_ind][0])
# k_47 - non-radiative rate from excided state to ISC (K_s[k_ind][1])
# k_57,k_67 - non-radiative rate from excided state to ISC (K_s[k_ind][2])
# k_71 - non-radiative rate from ISC to ground state (K_s[k_ind][3])
# k_72,k_73 - non-radiative rates from ISC to ground state (K_s[k_ind][4])
K_s=[[66.0,0.0,57.0,1.0,0.7],
    [77.0,0.0,30.0,3.3,0.0],
    [62.7,12.97,80.0,3.45,1.08],
    [63.2,10.8,60.7,0.8,0.4],
    [67.4,9.9,96.6,4.83,1.055],
    [64.0,11.8,79.8,5.6,0.0]]

# Dimension of the Hilbert space
dim = 7

# States
# NV
excited = qt.basis(dim, 4), qt.basis(dim, 3),qt.basis(dim, 5) # |+1>_es, |0>_es, |-1>_es
isc = qt.basis(dim, 6)
ground = qt.basis(dim, 1), qt.basis(dim, 0), qt.basis(dim, 2) # |+1>_gs, |0>_gs, |-1>_gs
# ^{15} N
nit = qt.basis(2, 0), qt.basis(2, 1)

n1, n2, n3 = (
    ground[1] * ground[1].dag(), #type: ignore
    ground[2] * ground[2].dag(), #type: ignore
    ground[0] * ground[0].dag(), #type: ignore
)
n4, n5, n6 = (
    excited[1] * excited[1].dag(), #type: ignore
    excited[2] * excited[2].dag(), #type: ignore
    excited[0] * excited[0].dag(), #type: ignore
)
n7, nc = (
    isc * isc.dag(), #type: ignore
    ground[2] * ground[1].dag(), #type: ignore
)

# Operators
# MV
sx_gs, sy_gs, sz_gs = construct_spin_matrices(ground)
sx_es, sy_es, sz_es = construct_spin_matrices(excited)
sm_gs,sp_gs= sx_gs - 1j * sy_gs, sx_gs + 1j * sy_gs
sm_es,sp_es= sx_es - 1j * sy_es, sx_es + 1j * sy_es

S_gs=np.array([sx_gs, sy_gs, sz_gs])
S_es=np.array([sx_es, sy_es, sz_es])
IdNV=qt.qeye(dim)

# ^{15} N
sx_n, sy_n, sz_n = qt.sigmax()*0.5, qt.sigmay()*0.5, qt.sigmaz()*0.5
sm_n,sp_n=sx_n-1j*sy_n,sx_n+1j*sy_n

S_n=np.array([sx_n,sy_n,sz_n])
IdN15 = qt.qeye(2)

# Static Magnetic Field (G)
B = B0(100.0,0.0,0.0) # (G)

###################### Models ###########################

##############################################
################ Model 1 #####################
##############################################

def H_mg(om_r):
    """Returns the Hamiltonian of the system based on whether the MW is on or off
    Parameters:
        om_r (float) - Rabi frequency

    Returns:
        H_0 (list) - list of the Hamiltonian terms and their time dependence
    """
    H_0 = [[0.5 * om_r * (ground[1]*ground[2].dag()), "exp(1j*w*t)"],  #type: ignore
           [0.5 * om_r * (ground[2]*ground[1].dag()), "exp(-1j*w*t)"]] #type: ignore
    return H_0


def L_mg(w_p,k_index=k_ind, K_s=K_s):
    """Returns the Lindblad operators of the system.
    
    Parameters:
        w_p (float) - Laser pump rate
        
    Returns:
        c_ops (list) - list of the Lindblad operators
    """
    k41 = K_s[k_index][0]
    k52 = K_s[k_index][0]
    k63 = K_s[k_index][0]
    k57 = K_s[k_index][2]
    k67 = K_s[k_index][2]
    k47 = K_s[k_index][1]
    k71 = K_s[k_index][3]
    k72 = K_s[k_index][4]
    k73 = K_s[k_index][4]
    
    c_ops = []

    c_ops.append(np.sqrt(w_p) * (excited[1] * ground[1].dag()))  # n1 to n4 #type: ignore 
    c_ops.append(np.sqrt(w_p) * (excited[2] * ground[2].dag()))  # n2 to n5 #type: ignore
    c_ops.append(np.sqrt(w_p) * (excited[0] * ground[0].dag())) # n3 to n6  #type: ignore

    c_ops.append(np.sqrt(k41) * (ground[1] * excited[1].dag()))  # n4 to n1 #type: ignore
    c_ops.append(np.sqrt(k71) * (ground[1] * isc.dag()))  # n7 to n1    #type: ignore

    c_ops.append(np.sqrt(k52) * (ground[2] * excited[2].dag()))  # n5 to n2 #type: ignore
    c_ops.append(np.sqrt(k72) * (ground[2] * isc.dag()))  # n7 to n2 #type: ignore

    c_ops.append(np.sqrt(k63) * (ground[0] * excited[0].dag()))  # n6 to n3 #type: ignore
    c_ops.append(np.sqrt(k73) * (ground[0] * isc.dag()))  # n7 to n3    #type: ignore

    c_ops.append(np.sqrt(k47) * (isc * excited[1].dag()))  # n4 to n7   #type: ignore
    c_ops.append(np.sqrt(k57) * (isc * excited[2].dag()))  # n5 to n7   #type: ignore
    c_ops.append(np.sqrt(k67) * (isc * excited[0].dag()))  # n6 to n7   #type: ignore
    
    # Add collapse operators for decoherence
    c_ops.append(np.sqrt(gamma_gs[1]) * sz_gs)
    c_ops.append(np.sqrt(gamma_gs[0]/2) * (sm_gs))
    c_ops.append(np.sqrt(gamma_gs[0]/2) * (sp_gs))
    c_ops.append(np.sqrt(gamma_es[1]) * sz_es)
    c_ops.append(np.sqrt(gamma_es[0]/2) * (sm_es))
    c_ops.append(np.sqrt(gamma_es[0]/2) * (sp_es))
    return c_ops

def dynamics_mg(
    dt,
    init_state,
    om=None,
    om_r=None,
    w_p=None,
    k_index=k_ind,
    ti=0.0,    
    mode="Free",
    progress_bar="ON",
    i=0,
):
    """
    Perform dynamics_mg simulation based on the given parameters, including optical transition rates index.
    Where, when using k_index=2 -> K_s[k_index]=[62.7,12.97,80.0,3.45,1.08], and the full K_s list is:
        K_s=[[66.0,0.0,57.0,1.0,0.7],
             [77.0,0.0,30.0,3.3,0.0],
             [62.7,12.97,80.0,3.45,1.08],
             [63.2,10.8,60.7,0.8,0.4],
             [67.4,9.9,96.6,4.83,1.055]]
    Parameters:
    - dt (float): Time step for the calculations.
    - init_state: Initial state for the simulation.
    - om (float, optional): Angular frequency of the system. Defaults to omega.
    - om_r (float, optional): Angular frequency for MW-ON evolution. Defaults to Om_r.
    - w_p (float, optional): Frequency for laser-ON evolution. Defaults to W_p.
    - k_index=k_index (int, optional): Index for the optical transition rates. Defaults to k_ind.
    - ti (float, optional): Initial time for the simulation. Defaults to 0.0.
    - mode (str, optional): Mode of the simulation. Can be "Free", "MW", "Laser", or "Laser-MW". Defaults to "Free".
    - progress_bar (str, optional): Progress bar option. Can be "ON" or "OFF". Defaults to "ON".
    - i (int, optional): Iteration number. Defaults to 0.

    Returns:
    - tf (float): Final time of the simulation.
    - result: Result of the simulation.
    """
    # Default values
    if om is None: om = omega
    if om_r is None: om_r = Om_r
    if w_p is None: w_p = W_p

    # Arguments for the Hamiltonian
    args = {"w": om}
    
    # Define the time resolution
    t_bins = 1000 if dt <= 5 else 5000
    
    tf = ti + dt

    # Define collapse operators and Hamiltonian based on mode
    match mode:
        case "Free":
            c_ops = L_mg(0.0, k_index=k_index)
            H = H_mg(0.0)
        case "MW":
            c_ops = L_mg(0.0, k_index=k_index)
            H = H_mg(om_r)
        case "Laser":
            c_ops = L_mg(w_p, k_index=k_index)
            H = H_mg(0.0)
        case "Laser-MW":
            c_ops = L_mg(w_p, k_index=k_index)
            H = H_mg(om_r)
        case _:
            raise ValueError('mode must be one of "Free", "MW", "Laser", or "Laser-MW"')
    
    # Call the master equation solver
    match progress_bar:
        case "OFF":
            result = qt.mesolve(
                H,
                init_state,
                np.linspace(ti, tf, t_bins + 1),
                c_ops,
                args=args,
                options={"store_states": True},
            )
        case "ON":
            print(f"{mode} {int(i + 1)} \n ti | tf \n {int(ti)} | {int(tf)}")
            result = qt.mesolve(
                H,
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

def H_mg_hf(om_r,a_gs=a_gs,a_es=a_es):
    """Returns the Hamiltonian of the system based on whether the MW is on or off
    Parameters:
        om_r (float) - Rabi frequency

    Returns:
        H_0 (list) - list of the Hamiltonian terms and their time dependence
    """
    H_0 = [[(0.5 * om_r * (ground[1]*ground[2].dag()))&IdN15, "exp(1j*w*t)"],  #type: ignore
           [(0.5 * om_r * (ground[2]*ground[1].dag()))&IdN15, "exp(-1j*w*t)"], #type: ignore
           a_gs[0]*(sz_gs&sz_n) + a_gs[1]*((sx_gs&sx_n) + (sy_gs&sy_n)),
           a_es[0]*(sz_es&sz_n) + a_es[1]*((sx_es&sx_n) + (sy_es&sy_n))]
    H_n=[[IdNV&(0.5*om_r*(mu_n/mu_e)*(nit[0]*nit[1].dag())),"exp(1j*w*t)"],   #type: ignore
          [IdNV&(0.5*om_r*(mu_n/mu_e)*(nit[1]*nit[0].dag())),"exp(-1j*w*t)"]] #type: ignore
    return [*H_0 , *H_n]

def L_mg_hf(w_p,k_index=k_ind, K_s=K_s):
    """Returns the Lindblad operators of the system
    Parameters:
        w_p (float) - Laser pump rate
    Returns:
        c_ops (list) - list of the Lindblad operators
    """
    k41 = K_s[k_index][0]
    k52 = K_s[k_index][0]
    k63 = K_s[k_index][0]
    k57 = K_s[k_index][2]
    k67 = K_s[k_index][2]
    k47 = K_s[k_index][1]
    k71 = K_s[k_index][3]
    k72 = K_s[k_index][4]
    k73 = K_s[k_index][4]
    
    c_ops = []

    c_ops.append((np.sqrt(w_p) * (excited[1] * ground[1].dag()))&IdN15)  # n1 to n4 #type: ignore
    c_ops.append((np.sqrt(w_p) * (excited[2] * ground[2].dag()))&IdN15)  # n2 to n5 #type: ignore
    c_ops.append((np.sqrt(w_p) * (excited[0] * ground[0].dag()))&IdN15) # n3 to n6 #type: ignore

    c_ops.append((np.sqrt(k41) * (ground[1] * excited[1].dag()))&IdN15)  # n4 to n1 #type: ignore
    c_ops.append((np.sqrt(k71) * (ground[1] * isc.dag()))&IdN15)  # n7 to n1 #type: ignore

    c_ops.append((np.sqrt(k52) * (ground[2] * excited[2].dag()))&IdN15)  # n5 to n2 #type: ignore
    c_ops.append((np.sqrt(k72) * (ground[2] * isc.dag()))&IdN15)  # n7 to n2 #type: ignore

    c_ops.append((np.sqrt(k63) * (ground[0] * excited[0].dag()))&IdN15)  # n6 to n3 #type: ignore
    c_ops.append((np.sqrt(k73) * (ground[0] * isc.dag()))&IdN15)  # n7 to n3 #type: ignore

    c_ops.append((np.sqrt(k47) * (isc * excited[1].dag()))&IdN15)  # n4 to n7 #type: ignore
    c_ops.append((np.sqrt(k57) * (isc * excited[2].dag()))&IdN15)  # n5 to n7 #type: ignore
    c_ops.append((np.sqrt(k67) * (isc * excited[0].dag()))&IdN15)  # n6 to n7 #type: ignore
    # Add collapse operators for decoherence   
    c_ops.append((np.sqrt(gamma_gs[1]) * sz_gs)&IdN15)
    c_ops.append((np.sqrt(gamma_gs[0]/2) * (sm_gs))&IdN15)
    c_ops.append((np.sqrt(gamma_gs[0]/2) * (sp_gs))&IdN15)
    c_ops.append((np.sqrt(gamma_es[1]) * sz_es)&IdN15)
    c_ops.append((np.sqrt(gamma_es[0]/2) * (sm_es))&IdN15)
    c_ops.append((np.sqrt(gamma_es[0]/2) * (sp_es))&IdN15)
    c_ops.append(IdNV&(np.sqrt(gamma_n[1]) * sz_n))
    c_ops.append(IdNV&(np.sqrt(gamma_n[0]/2) * (sm_n)))
    c_ops.append(IdNV&(np.sqrt(gamma_n[0]/2) * (sp_n)))

    return c_ops

def dynamics_mg_hf(
    dt,
    init_state,
    om_r=None,
    om=None,
    w_p=None,
    k_index=k_ind,
    ti=0.0,
    mode="Free",
    progress_bar="ON",
    i=0,
):
    """
    Simulate the dynamics of a quantum system under hyperfine interaction using the Hamiltonian and collapse operators.
    including optical transition rates index.
    Where, when using k_index=2 -> K_s[k_index]=[62.7,12.97,80.0,3.45,1.08], and the full K_s list is:
        K_s=[[66.0,0.0,57.0,1.0,0.7],
             [77.0,0.0,30.0,3.3,0.0],
             [62.7,12.97,80.0,3.45,1.08],
             [63.2,10.8,60.7,0.8,0.4],
             [67.4,9.9,96.6,4.83,1.055]]
    Parameters:
    - dt (float): Total simulation time.
    - init_state (qutip.Qobj): Initial quantum state of the system.
    - om_r (float, optional): Rabi frequency for microwave interactions. Defaults to Om_r.
    - om (float, optional): Angular frequency of the system. Defaults to omega.
    - w_p (float, optional): Laser frequency. Defaults to W_p.
    - k_index(int, optional): Index for the optical transition rates. Defaults to k_ind.
    - ti (float, optional): Initial time of the simulation. Defaults to 0.0.
    - mode (str, optional): Simulation mode. Can be "Free", "MW", "Laser", or "Laser-MW". Defaults to "Free".
    - progress_bar (str, optional): Option to display a progress bar. Can be "ON" or "OFF". Defaults to "ON".
    - i (int, optional): Counter for the progress bar. Defaults to 0.

    Returns:
    - tf (float): Final time of the simulation.
    - result (qutip.solver.Result): Result object containing the simulation output.
    """
    # Default values
    if om_r is None: om_r = Om_r
    if om is None: om = omega
    if w_p is None: w_p = W_p

    # Time resolution based on dt
    t_bins = 1000 if dt <= 5 else 5000
    
    # Define Hamiltonian and collapse operators based on mode
    match mode:
        case "Free":
            H = H_mg_hf(0.0)
            c_ops = L_mg_hf(0.0, k_index=k_index)
        case "MW":
            H = H_mg_hf(om_r)
            c_ops = L_mg_hf(0.0, k_index=k_index)
        case "Laser":
            H = H_mg_hf(0.0)
            c_ops = L_mg_hf(w_p, k_index=k_index)
        case "Laser-MW":
            H = H_mg_hf(om_r)
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
                H,
                init_state,
                np.linspace(ti, tf, t_bins + 1),
                c_ops,
                args=args,
                options={"store_states": True},
            )
        case "ON":
            print(f"{mode} {int(i + 1)} \n ti | tf \n {int(ti)} | {int(tf)}")
            result = qt.mesolve(
                H,
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

# Driving frequency
omega = D_gs-mu_e*B[2]
# Driving phase
phi = 0.0

def H_no(b, om_r):
    """
    Calculates the Hamiltonian for a given magnetic field and Rabi frequency.
    
    Parameters:
    b (np.array): Magnetic field vector B0(B_amp,phi_B,theta_B).
    om_r (float): Rabi frequency.
    
    Returns:
    list: A list containing the Hamiltonian Qobj terms.
    """
    H_0 = [D_gs*sz_gs**2+E_gs*(sx_gs**2-sy_gs**2)+mu_e*np.dot(b,S_gs)+
           D_es*sz_es**2+E_es*(sx_es**2-sy_es**2)+mu_e*np.dot(b,S_es)]
    H_int = [[np.sqrt(2)*om_r*(sx_gs), "cos(w*t)*cos(p)"],
           [np.sqrt(2)*om_r*(sy_gs), "cos(w*t)*sin(p)"],
           [np.sqrt(2)*om_r*(sx_es), "cos(w*t)*cos(p)"],
           [np.sqrt(2)*om_r*(sy_es), "cos(w*t)*sin(p)"]]
    return [*H_0,*H_int]

def L_no(w_p,k_index=k_ind, K_s=K_s):
    """
    Returns the Lindblad operators of the system, including optical transitions based on the given k_index.

    Parameters:
    - w_p (float): Laser pump rate.
    - k_index (int, optional): Index for the optical transition rates. Defaults to k_ind.

    Returns:
    - c_ops (list): List of Lindblad operators Qobj.
    """
    k41 = K_s[k_index][0]
    k52 = K_s[k_index][0]
    k63 = K_s[k_index][0]
    k57 = K_s[k_index][2]
    k67 = K_s[k_index][2]
    k47 = K_s[k_index][1]
    k71 = K_s[k_index][3]
    k72 = K_s[k_index][4]
    k73 = K_s[k_index][4]
    
    c_ops = []

    c_ops.append(np.sqrt(w_p) * (excited[1] * ground[1].dag()))  # n1 to n4 #type: ignore
    c_ops.append(np.sqrt(w_p) * (excited[2] * ground[2].dag()))  # n2 to n5 #type: ignore
    c_ops.append(np.sqrt(w_p) * (excited[0] * ground[0].dag())) # n3 to n6 #type: ignore

    c_ops.append(np.sqrt(k41) * (ground[1] * excited[1].dag()))  # n4 to n1 #type: ignore
    c_ops.append(np.sqrt(k71) * (ground[1] * isc.dag()))  # n7 to n1 #type: ignore

    c_ops.append(np.sqrt(k52) * (ground[2] * excited[2].dag()))  # n5 to n2 #type: ignore
    c_ops.append(np.sqrt(k72) * (ground[2] * isc.dag()))  # n7 to n2 #type: ignore

    c_ops.append(np.sqrt(k63) * (ground[0] * excited[0].dag()))  # n6 to n3 #type: ignore
    c_ops.append(np.sqrt(k73) * (ground[0] * isc.dag()))  # n7 to n3 #type: ignore

    c_ops.append(np.sqrt(k47) * (isc * excited[1].dag()))  # n4 to n7 #type: ignore
    c_ops.append(np.sqrt(k57) * (isc * excited[2].dag()))  # n5 to n7 #type: ignore
    c_ops.append(np.sqrt(k67) * (isc * excited[0].dag()))  # n6 to n7 #type: ignore
    # Add collapse operators for decoherence
    c_ops.append(np.sqrt(gamma_gs[1]) * sz_gs)
    c_ops.append(np.sqrt(gamma_gs[0]/2) * (sm_gs))
    c_ops.append(np.sqrt(gamma_gs[0]/2) * (sp_gs))
    c_ops.append(np.sqrt(gamma_es[1]) * sz_es)
    c_ops.append(np.sqrt(gamma_es[0]/2) * (sm_es))
    c_ops.append(np.sqrt(gamma_es[0]/2) * (sp_es))
    return c_ops


def dynamics_no(
    dt,
    init_state,
    b=None,
    om=None,
    p=None,
    om_r=None,
    w_p=None,
    k_index=k_ind,
    ti=0.0,    
    mode="Free",
    progress_bar="ON",
    i=0,
):
    """
    Perform dynamics simulation based on the given parameters, including optical transition rates index.
    Where, when using k_index=2 -> K_s[k_index]=[62.7,12.97,80.0,3.45,1.08], and the full K_s list is:
        K_s=[[66.0,0.0,57.0,1.0,0.7],
             [77.0,0.0,30.0,3.3,0.0],
             [62.7,12.97,80.0,3.45,1.08],
             [63.2,10.8,60.7,0.8,0.4],
             [67.4,9.9,96.6,4.83,1.055]]
    Parameters:
    - dt (float): Time step for the calculations.
    - init_state: Initial state for the simulation.
    - b (np.array, optional): Magnetic field vector. Defaults to B0(B_amp,phi_B,theta_B).
    - om (float, optional): Angular frequency of the system. Defaults to omega.
    - om_r (float, optional): Angular frequency for MW-ON evolution. Defaults to Om_r.
    - w_p (float, optional): Frequency for laser-ON evolution. Defaults to W_p.
    - k_index=k_index (int, optional): Index for the optical transition rates. Defaults to k_ind.
    - ti (float, optional): Initial time for the simulation. Defaults to 0.0.
    - mode (str, optional): Mode of the simulation. Can be "Free", "MW", "Laser", or "Laser-MW". Defaults to "Free".
    - progress_bar (str, optional): Progress bar option. Can be "ON" or "OFF". Defaults to "ON".
    - i (int, optional): Iteration number. Defaults to 0.

    Returns:
    - tf (float): Final time of the simulation.
    - result: Result of the simulation.
    """
    # Default values
    if b is None: b = B
    if om is None: om = omega
    if p is None: p = phi
    if om_r is None: om_r = Om_r
    if w_p is None: w_p = W_p

    # Arguments for the Hamiltonian
    args = {"w": om, "p": p}
    
    # Define the time resolution   
    t_bins = 1000 if dt <= 5 else 5000
    
    tf = ti + dt

    # Define collapse operators and Hamiltonian based on mode
    match mode:
        case "Free":
            c_ops = L_no(0.0, k_index=k_index)
            H = H_no(b, 0.0)
        case "MW":
            c_ops = L_no(0.0, k_index=k_index)
            H = H_no(b, om_r)
        case "Laser":
            c_ops = L_no(w_p, k_index=k_index)
            H = H_no(b, 0.0)
        case "Laser-MW":
            c_ops = L_no(w_p, k_index=k_index)
            H = H_no(b, om_r)
        case _:
            raise ValueError('mode must be one of "Free", "MW", "Laser", or "Laser-MW"')

    # Call the master equation solver
    match progress_bar:
        case "OFF":
            result = qt.mesolve(
                H,
                init_state,
                np.linspace(ti, tf, t_bins + 1),
                c_ops,
                args=args,
                options={"store_states": True},
            )
        case "ON":
            print(f"{mode} {int(i + 1)} \n ti    |    tf \n {ti:.2f} | {tf:.2f}")
            result = qt.mesolve(
                H,
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
    Calculate the Hamiltonian for a given magnetic field and resonant frequency, including hyperfine interactions.
    
    Parameters:
    b (numpy.ndarray): Magnetic field vector [bx, by, bz].
    om_r: Resonant frequency.
    
    Returns:
    list: List of Hamiltonian terms.
    """
    H_0 = [((D_gs*sz_gs**2 + E_gs*(sx_gs**2-sy_gs**2) + mu_e*np.dot(b, S_gs))&IdN15)+
           ((D_es*sz_es**2 + E_es*(sx_es**2-sy_es**2) + mu_e*np.dot(b, S_es))&IdN15)+
           A_par[0]*(sz_gs&sz_n) + A_perp[0]/4*((sp_gs&sm_n)+(sm_gs&sp_n))+
           A_perp_prime[0]/4*((sp_gs&sp_n)*np.exp(-2j*phi_h) + (sm_gs&sm_n)*np.exp(2j*phi_h))+
           A_ani[0]/2*((sp_gs&sz_n) + (sz_gs&sp_n)*np.exp(-1j*phi_h) + (sm_gs&sz_n) + (sz_gs&sm_n)*np.exp(1j*phi_h))+
           A_par[1]*(sz_es&sz_n) + A_perp[1]/4*((sp_es&sm_n)+(sm_es&sp_n))+
           A_perp_prime[1]/4*((sp_es&sp_n)*np.exp(-2j*phi_h) + (sm_es&sm_n)*np.exp(2j*phi_h))+
           A_ani[1]/2*((sp_es&sz_n) + (sz_es&sp_n)*np.exp(-1j*phi_h) + (sm_es&sz_n) + (sz_es&sm_n)*np.exp(1j*phi_h))]
    H_n = [IdNV&mu_n*np.dot(b, S_n)]
    H_int = [[(np.sqrt(2)*om_r*sx_gs)&IdN15, "cos(w*t)*cos(p)"],
             [(np.sqrt(2)*om_r*sy_gs)&IdN15, "cos(w*t)*sin(p)"],
             [(np.sqrt(2)*om_r*sx_es)&IdN15, "cos(w*t)*cos(p)"],
             [(np.sqrt(2)*om_r*sy_es)&IdN15, "cos(w*t)*sin(p)"],
             [IdNV&(2*om_r/mu_e*mu_n*sx_n), "cos(w*t)*cos(p)"],
             [IdNV&(2*om_r/mu_e*mu_n*sy_n), "cos(w*t)*sin(p)"]]
    return [*H_0, *H_n, *H_int]

def L_dua_hf(w_p,k_index=k_ind, K_s=K_s):
    """
    Returns the Lindblad operators of the system, including optical transitions based on the given k_index.

    Parameters:
    - w_p (float): Laser pump rate.
    - k_index (int, optional): Index for the optical transition rates. Defaults to k_ind.

    Returns:
    - c_ops (list): List of Lindblad operators.
    """
    k41 = K_s[k_index][0]
    k52 = K_s[k_index][0]
    k63 = K_s[k_index][0]
    k57 = K_s[k_index][2]
    k67 = K_s[k_index][2]
    k47 = K_s[k_index][1]
    k71 = K_s[k_index][3]
    k72 = K_s[k_index][4]
    k73 = K_s[k_index][4]
    
    c_ops = []

    c_ops.append((np.sqrt(w_p) * (excited[1] * ground[1].dag()))&IdN15)  # n1 to n4 #type: ignore
    c_ops.append((np.sqrt(w_p) * (excited[2] * ground[2].dag()))&IdN15)  # n2 to n5 #type: ignore
    c_ops.append((np.sqrt(w_p) * (excited[0] * ground[0].dag()))&IdN15) # n3 to n6 #type: ignore

    c_ops.append((np.sqrt(k41) * (ground[1] * excited[1].dag()))&IdN15)  # n4 to n1 #type: ignore
    c_ops.append((np.sqrt(k71) * (ground[1] * isc.dag()))&IdN15)  # n7 to n1 #type: ignore

    c_ops.append((np.sqrt(k52) * (ground[2] * excited[2].dag()))&IdN15)  # n5 to n2 #type: ignore
    c_ops.append((np.sqrt(k72) * (ground[2] * isc.dag()))&IdN15)  # n7 to n2 #type: ignore

    c_ops.append((np.sqrt(k63) * (ground[0] * excited[0].dag()))&IdN15)  # n6 to n3 #type: ignore
    c_ops.append((np.sqrt(k73) * (ground[0] * isc.dag()))&IdN15)  # n7 to n3 #type: ignore

    c_ops.append((np.sqrt(k47) * (isc * excited[1].dag()))&IdN15)  # n4 to n7 #type: ignore
    c_ops.append((np.sqrt(k57) * (isc * excited[2].dag()))&IdN15)  # n5 to n7 #type: ignore
    c_ops.append((np.sqrt(k67) * (isc * excited[0].dag()))&IdN15)  # n6 to n7 #type: ignore
    # Add collapse operators for decoherence   
    c_ops.append((np.sqrt(gamma_gs[1]) * sz_gs)&IdN15)
    c_ops.append((np.sqrt(gamma_gs[0]/2) * (sm_gs))&IdN15)
    c_ops.append((np.sqrt(gamma_gs[0]/2) * (sp_gs))&IdN15)
    c_ops.append((np.sqrt(gamma_es[1]) * sz_es)&IdN15)
    c_ops.append((np.sqrt(gamma_es[0]/2) * (sm_es))&IdN15)
    c_ops.append((np.sqrt(gamma_es[0]/2) * (sp_es))&IdN15)
    c_ops.append(IdNV&(np.sqrt(gamma_n[1]) * sz_n))
    c_ops.append(IdNV&(np.sqrt(gamma_n[0]/2) * (sm_n)))
    c_ops.append(IdNV&(np.sqrt(gamma_n[0]/2) * (sp_n)))
    return c_ops

def dynamics_dua_hf(
    dt,
    init_state,
    b=None,
    om_r=None,
    om=None,
    p=None,
    w_p=None,
    k_index=k_ind,
    ti=0.0,
    mode="Free",
    progress_bar="ON",
    i=0,
):
    """
    Simulate the dynamics of a quantum system under hyperfine interaction using the Hamiltonian and collapse operators.
    including optical transition rates index.
    Where, when using k_index=2 -> K_s[k_index]=[62.7,12.97,80.0,3.45,1.08], and the full K_s list is:
        K_s=[[66.0,0.0,57.0,1.0,0.7],
             [77.0,0.0,30.0,3.3,0.0],
             [62.7,12.97,80.0,3.45,1.08],
             [63.2,10.8,60.7,0.8,0.4],
             [67.4,9.9,96.6,4.83,1.055]]
    Parameters:
    - dt (float): Total simulation time.
    - init_state (qt.Qobj): Initial quantum state of the system.
    - b (np.array, optional): Magnetic field vector. Defaults to B0(B_amp,phi_B,theta_B).
    - om_r (float, optional): Rabi frequency for microwave interactions. Defaults to Om_r.
    - om (float, optional): Angular frequency of the system. Defaults to omega.
    - p (float, optional): Microwave phase. Defaults to 0.0.
    - w_p (float, optional): Laser frequency. Defaults to W_p.
    - k_index(int, optional): Index for the optical transition rates. Defaults to k_ind.
    - ti (float, optional): Initial time of the simulation. Defaults to 0.0.
    - mode (str, optional): Simulation mode. Can be "Free", "MW", "Laser", or "Laser-MW". Defaults to "Free".
    - progress_bar (str, optional): Option to display a progress bar. Can be "ON" or "OFF". Defaults to "ON".
    - i (int, optional): Counter for the progress bar. Defaults to 0.

    Returns:
    - tf (float): Final time of the simulation.
    - result (qutip.solver.Result): Result object containing the simulation output.
    """
    # Default values
    if b is None: b = B
    if om_r is None: om_r = Om_r
    if om is None: om = omega
    if p is None: p = phi
    if w_p is None: w_p = W_p
    
    # Arguments for the Hamiltonian
    args = {"w": om, "p": p}
    # Time resolution based on dt
    t_bins = 1000 if dt <= 5 else 5000
    
    tf = ti + dt
    
    # Define Hamiltonian and collapse operators based on mode
    match mode:
        case "Free":
            H = H_dua_hf(b, 0.0, phi_H)
            c_ops = L_dua_hf(0.0, k_index=k_index)
        case "MW":
            H = H_dua_hf(b, om_r, phi_H)
            c_ops = L_dua_hf(0.0, k_index=k_index)
        case "Laser":
            H = H_dua_hf(b, 0.0, phi_H)
            c_ops = L_dua_hf(w_p, k_index=k_index)
        case "Laser-MW":
            H = H_dua_hf(b, om_r, phi_H)
            c_ops = L_dua_hf(w_p, k_index=k_index)
        case _:
            raise ValueError('mode must be one of "Free", "MW", "Laser", or "Laser-MW"')
    # Solve the master equation
    match progress_bar:
        case "OFF":
            result = qt.mesolve(
                H,
                init_state,
                np.linspace(ti, tf, t_bins + 1),
                c_ops,
                args=args,
                options={"store_states": True},
            )
        case "ON":
            print(f"{mode} {int(i + 1)} \n ti    |    tf \n {ti:.2f} | {tf:.2f}")
            result = qt.mesolve(
                H,
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
    Calculate the Hamiltonian for a given magnetic field and resonant frequency.
    
    Parameters:
    b (numpy.ndarray): Magnetic field vector [bx,by,bz].
    om_r: Resonant frequency.
    
    Returns:
    list: List of Hamiltonian terms.
    """
    
    H_0 = [((D_gs*sz_gs**2+E_gs*(sx_gs**2-sy_gs**2)+mu_e*np.dot(b,S_gs))&IdN15)+
           ((D_es*sz_es**2+E_es*(sx_es**2-sy_es**2)+mu_e*np.dot(b,S_es))&IdN15)+
           a_gs[0]*(sz_gs&sz_n) + a_gs[1]*((sx_gs&sx_n) + (sy_gs&sy_n))+
           a_es[0]*(sz_es&sz_n) + a_es[1]*((sx_es&sx_n) + (sy_es&sy_n))]
    H_n = [IdNV&(mu_n*np.dot(b,S_n))]
    H_int=[[(np.sqrt(2)*om_r*sx_gs)&IdN15, "cos(w*t)*cos(p)"],
           [(np.sqrt(2)*om_r*sy_gs)&IdN15, "cos(w*t)*sin(p)"],
           [(np.sqrt(2)*om_r*sx_es)&IdN15, "cos(w*t)*cos(p)"],
           [(np.sqrt(2)*om_r*sy_es)&IdN15, "cos(w*t)*sin(p)"],
           [IdNV&(2*om_r/mu_e*mu_n*sx_n), "cos(w*t)*cos(p)"],
           [IdNV&(2*om_r/mu_e*mu_n*sy_n), "cos(w*t)*sin(p)"]]
    return [*H_0,*H_n,*H_int] 

def L_doh_hf(w_p,k_index=k_ind, K_s=K_s):
    """
    Returns the Lindblad operators of the system, including optical transitions based on the given k_index.

    Parameters:
    - w_p (float): Laser pump rate.
    - k_index (int, optional): Index for the optical transition rates. Defaults to k_ind.

    Returns:
    - c_ops (list): List of Lindblad operators.
    """
    k41 = K_s[k_index][0]
    k52 = K_s[k_index][0]
    k63 = K_s[k_index][0]
    k57 = K_s[k_index][2]
    k67 = K_s[k_index][2]
    k47 = K_s[k_index][1]
    k71 = K_s[k_index][3]
    k72 = K_s[k_index][4]
    k73 = K_s[k_index][4]
    
    c_ops = []

    c_ops.append((np.sqrt(w_p) * (excited[1] * ground[1].dag()))&IdN15)  # n1 to n4 #type: ignore
    c_ops.append((np.sqrt(w_p) * (excited[2] * ground[2].dag()))&IdN15)  # n2 to n5 #type: ignore
    c_ops.append((np.sqrt(w_p) * (excited[0] * ground[0].dag()))&IdN15) # n3 to n6 #type: ignore

    c_ops.append((np.sqrt(k41) * (ground[1] * excited[1].dag()))&IdN15)  # n4 to n1 #type: ignore
    c_ops.append((np.sqrt(k71) * (ground[1] * isc.dag()))&IdN15)  # n7 to n1 #type: ignore

    c_ops.append((np.sqrt(k52) * (ground[2] * excited[2].dag()))&IdN15)  # n5 to n2 #type: ignore
    c_ops.append((np.sqrt(k72) * (ground[2] * isc.dag()))&IdN15)  # n7 to n2 #type: ignore

    c_ops.append((np.sqrt(k63) * (ground[0] * excited[0].dag()))&IdN15)  # n6 to n3 #type: ignore
    c_ops.append((np.sqrt(k73) * (ground[0] * isc.dag()))&IdN15)  # n7 to n3 #type: ignore

    c_ops.append((np.sqrt(k47) * (isc * excited[1].dag()))&IdN15)  # n4 to n7 #type: ignore
    c_ops.append((np.sqrt(k57) * (isc * excited[2].dag()))&IdN15)  # n5 to n7 #type: ignore
    c_ops.append((np.sqrt(k67) * (isc * excited[0].dag()))&IdN15)  # n6 to n7 #type: ignore
    # Collapse operators for decoherence   
    c_ops.append((np.sqrt(gamma_gs[1]) * sz_gs)&IdN15)
    c_ops.append((np.sqrt(gamma_gs[0]/2) * (sm_gs))&IdN15)
    c_ops.append((np.sqrt(gamma_gs[0]/2) * (sp_gs))&IdN15)
    c_ops.append((np.sqrt(gamma_es[1]) * sz_es)&IdN15)
    c_ops.append((np.sqrt(gamma_es[0]/2) * (sm_es))&IdN15)
    c_ops.append((np.sqrt(gamma_es[0]/2) * (sp_es))&IdN15)
    c_ops.append(IdNV&(np.sqrt(gamma_n[1]) * sz_n))
    c_ops.append(IdNV&(np.sqrt(gamma_n[0]/2) * (sm_n)))
    c_ops.append(IdNV&(np.sqrt(gamma_n[0]/2) * (sp_n)))
    return c_ops

def dynamics_doh_hf(
    dt,
    init_state,
    b=None,
    om_r=None,
    om=None,
    p=None,
    w_p=None,
    k_index=k_ind,
    ti=0.0,
    mode="Free",
    progress_bar="ON",
    i=0,
):
    """
    Simulate the dynamics of a quantum system under hyperfine interaction using the Hamiltonian and collapse operators.
    including optical transition rates index.
    Where, when using k_index=2 -> K_s[k_index]=[62.7,12.97,80.0,3.45,1.08], and the full K_s list is:
        K_s=[[66.0,0.0,57.0,1.0,0.7],
             [77.0,0.0,30.0,3.3,0.0],
             [62.7,12.97,80.0,3.45,1.08],
             [63.2,10.8,60.7,0.8,0.4],
             [67.4,9.9,96.6,4.83,1.055]]
    Parameters:
    - dt (float): Total simulation time.
    - init_state (qt.Qobj): Initial quantum state of the system.
    - b (np.array, optional): Magnetic field vector. Defaults to B0(B_amp,phi_B,theta_B).
    - om_r (float, optional): Rabi frequency for microwave interactions. Defaults to Om_r.
    - om (float, optional): Angular frequency of the system. Defaults to omega.
    - p (float, optional): Microwave phase. Defaults to 0.0.
    - w_p (float, optional): Laser frequency. Defaults to W_p.
    - k_index(int, optional): Index for the optical transition rates. Defaults to k_ind.
    - ti (float, optional): Initial time of the simulation. Defaults to 0.0.
    - mode (str, optional): Simulation mode. Can be "Free", "MW", "Laser", or "Laser-MW". Defaults to "Free".
    - progress_bar (str, optional): Option to display a progress bar. Can be "ON" or "OFF". Defaults to "ON".
    - i (int, optional): Counter for the progress bar. Defaults to 0.

    Returns:
    - tf (float): Final time of the simulation.
    - result (qutip.solver.Result): Result object containing the simulation output.
    """
    # Default values
    if b is None: b = B
    if om_r is None: om_r = Om_r
    if om is None: om = omega
    if p is None: p = phi
    if w_p is None: w_p = W_p
    
    # Arguments for the Hamiltonian
    args = {"w": om, "p": p}
    # Time resolution based on dt
    t_bins = 1000 if dt <= 5 else 5000
    
    tf = ti + dt
    
    # Define Hamiltonian and collapse operators based on mode
    match mode:
        case "Free":
            H = H_doh_hf(b, 0.0)
            c_ops = L_doh_hf(0.0, k_index=k_index)
        case "MW":
            H = H_doh_hf(b, om_r)
            c_ops = L_doh_hf(0.0, k_index=k_index)
        case "Laser":
            H = H_doh_hf(b, 0.0)
            c_ops = L_doh_hf(w_p, k_index=k_index)
        case "Laser-MW":
            H = H_doh_hf(b, om_r)
            c_ops = L_doh_hf(w_p, k_index=k_index)
        case _:
            raise ValueError('mode must be one of "Free", "MW", "Laser", or "Laser-MW"')

    # Solve the master equation
    match progress_bar:
        case "OFF":
            result = qt.mesolve(
                H,
                init_state,
                np.linspace(ti, tf, t_bins + 1),
                c_ops,
                args=args,
                options={"store_states": True},
            )
        case "ON":
            print(f"{mode} {int(i + 1)} \n ti    |    tf \n {ti:.2f} | {tf:.2f}")
            result = qt.mesolve(
                H,
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