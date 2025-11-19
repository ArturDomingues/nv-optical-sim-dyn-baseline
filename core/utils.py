################### Imports #############################

import numpy as np
import scipy as scp

###################### Definitions ######################

def B0(b_0, phi, theta):
    """
    Calculate the magnetic field vector in Cartesian coordinates.
    Parameters:
    - b_0 (float): The magnitude of the magnetic field.
    - theta (float): The azimuthal angle in radians.
    - phi (float): The polar angle in radians.
    Returns:
    - np.array: The magnetic field vector in Cartesian coordinates np.array([Bx, By, Bz]).
    """
    return np.array([b_0 * np.sin(theta) * np.cos(phi), b_0 * np.sin(theta) * np.sin(phi), b_0 * np.cos(theta)]) #type:ignore

def construct_spin_matrices(basis):
    """
    Constructs the spin matrices for a given basis.
    Parameters:
    - basis (list): A list of basis elements.
    Returns:
    - s_x (Qobj): The x-component of the spin matrix.
    - s_y (Qobj): The y-component of the spin matrix.
    - s_z (Qobj): The z-component of the spin matrix.
    Raises:
    - ValueError: If the number of basis elements is not 3.
    """
    if len(basis) == 3:
        # 3-level system (spin-1)
        s_x = (basis[0] * basis[1].dag() + basis[1] * basis[0].dag() +
               basis[1] * basis[2].dag() + basis[2] * basis[1].dag()) / np.sqrt(2)
        
        s_y = -1j * (basis[0] * basis[1].dag() - basis[1] * basis[0].dag() +
                     basis[1] * basis[2].dag() - basis[2] * basis[1].dag()) / np.sqrt(2)
        
        s_z = basis[0] * basis[0].dag() - basis[2] * basis[2].dag()
        return s_x, s_y, s_z

    else:
        raise ValueError("This function supports only 3 basis elements.")

################# Normalization Functions #####################
    
def normaliz(arr):
    """
    Normalize the input array to the range [0, 1].

    Parameters
    ----------
    arr : np.ndarray
        Input array to normalize.

    Returns
    -------
    np.ndarray
        Array with values scaled between 0 and 1.
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)).real

def stand(arr):
    """
    Standardize the input array to zero mean and unit variance.

    Parameters
    ----------
    arr : array-like
        Input array to standardize.

    Returns
    -------
    np.ndarray
        Standardized array with mean 0 and standard deviation 1.
    """
    return (arr - np.mean(arr)) / np.std(arr)

def max_reg(arr):
    """
    Normalize the input array by dividing by the maximum absolute value.

    Parameters
    ----------
    arr : np.ndarray
        Input array to normalize.

    Returns
    -------
    np.ndarray
        Normalized array with maximum absolute value scaled to 1.
    """
    return arr / np.max(np.abs(arr))

def l2_norm(arr):
    """
    Normalize the input array using the L2 norm.

    Parameters
    ----------
    arr : np.ndarray
        Input array to normalize.

    Returns
    -------
    np.ndarray
        Array normalized to unit L2 norm.
    """
    return arr / np.linalg.norm(arr)

#################### Fit Functions ##############################

def lorent(x,x0,gam,A,D):
    """Lorentzian function.
    Parameters:
    x (numpy.ndarray): Input array.
    x0 (float): Center of the Lorentzian peak.
    gam (float): Half-width at half-maximum (HWHM) of the Lorentzian peak.
    D (float): Offset value.
    Returns:
    numpy.ndarray: Lorentzian values.
    """
    return -A/(1+((x-x0)/gam)**2)*1/(gam*np.pi) + D

def gen_gauss(x,mu,alp,bet,A,D):
    """Generalized Gaussian function.
    Parameters:
    x (numpy.ndarray): Input array.
    mu (float): Mean of the distribution.
    alp (float): Scale parameter (related to the width).
    bet (float): Shape parameter (controls the "peakedness" of the distribution).
    D (float): Offset value.
    Returns:
    numpy.ndarray: Generalized Gaussian values.
    """
    return -A*np.exp(-np.abs((x-mu)/alp)**bet)*bet/(2*alp*np.sqrt(np.pi)*scp.special.gamma(1/bet)) + D

def damp_sin(x,A,omega,phi,tau, D):
    """
    Damped sine function.

    Parameters:
    x (numpy.ndarray): Input array.
    A (float): Amplitude.
    omega (float): Angular frequency.
    phi (float): Phase shift.
    tau (float): Damping time constant.
    D (float): Offset value.

    Returns:
    numpy.ndarray: Damped sine values.
    """
    return A * np.exp(-x/tau) * np.sin(omega*x + phi) + D

def damp_cos(x,A,omega,phi,tau, D):
    """
    Damped sine function.

    Parameters:
    x (numpy.ndarray): Input array.
    A (float): Amplitude.
    omega (float): Angular frequency.
    phi (float): Phase shift.
    tau (float): Damping time constant.
    D (float): Offset value.

    Returns:
    numpy.ndarray: Damped sine values.
    """
    return A * np.exp(-x/tau) * np.cos(omega*x + phi) + D

def expo(x,A, tau, D):
    """
    Exponential decay function.

    Parameters:
    x (numpy.ndarray): Input array.
    A (float): Amplitude.
    tau (float): Time constant.
    D (float): Offset value.

    Returns:
    numpy.ndarray: Exponential decay values.
    """
    return A * np.exp(-x/tau) + D

def sinn(x, x0, A, omega,phi, D):
    """
    Sin function.

    Parameters:
    x (numpy.ndarray): Input array.
    A (float): Amplitude.
    tau (float): Time constant.
    D (float): Offset value.

    Returns:
    numpy.ndarray: Exponential decay values.
    """
    return A * np.sin(omega*(x-x0) + phi) + D