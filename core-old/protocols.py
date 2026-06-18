import numpy as np
import qutip as qt
from tqdm.auto import tqdm
from functools import partial
from .models import B,W_P,OM_R,OMEGA,PHI

b_odmr=150
n_p_odmr=1
n_p_ramsey=1.5
w_p_odmr=n_p_odmr*1.9
w_p_ramsey=n_p_ramsey*1.9
#om_r_odmr=4*np.pi/t2_gs
om_r_odmr=15.7
W_p=W_P
Om_r=OM_R
omega=OMEGA
phi=PHI

def cwodmr_non_parallel(t, ws, model, init_state,exp_ops, 
           b=B, om_r=om_r_odmr, w_p=w_p_odmr):
    """
    Calculate the continuous wave optical detected resonance (CWODMR) dynamics.
    
    Parameters:
    - t (float): Time parameter.
    - ws (numpy.ndarray): Array of angular frequencies.
    - model (function): Function that defines the dynamics of the system.
    - init_state (object): Initial state of the system.
    - b (float): Parameter b (default: B).
    - om_r (float): Rabi frequency (default: Om_r).
    - w_p (float): Angular frequency (default: W_p).
    
    Returns:
    - ws (numpy.ndarray): Array of angular frequencies.
    - flur (numpy.ndarray): Array of PL values.
    - times (numpy.ndarray): Array of time values.
    - n_exp (numpy.ndarray): Array of expectation values.
    - results (numpy.ndarray): Array of simulation results.
    """
    #continuous MW and laser appplied together
    results = np.array([])
    flur = np.array([])
    for w in tqdm(ws):
        # Run the dynamics based on the mode choosen
        _, result = model(t, 
                           init_state,
                           b=b,
                           om_r=om_r,
                           om=w,
                           w_p=w_p/n_p_odmr,
                           ti=0.0, 
                           mode="Laser-MW", 
                           progress_bar="OFF") # type:ignore
        # Gather the expectation values from the results array
        results = np.append(results,result)
        fl=np.array([])
        fl=np.add(result.expect[3],result.expect[4])
        fl=np.add(fl,result.expect[5])
        flur=np.append(flur,np.sum(fl))
   
    n_exp = np.array([np.concatenate([qt.expect(M, res.states) for res in results],axis=0) for M in exp_ops])
    #for j in range(len(results[0].expect)):
    #    nn_exp = np.array([])
    #    for i in range(len(results)):
    #        nn_exp = np.append(nn_exp, results[i].expect[j])
    #    if j == 0:
    #        n_exp = np.array([nn_exp])
    #    else:
    #        n_exp = np.append(n_exp, [nn_exp], axis=0)
    # Gather the times for the plots
    times = results[0].times
    return ws,flur,times,n_exp,results
def run_single_frequency_cwodmr(w, model, t, init_state, b, om_r, w_p):
    # Run the dynamics for a single frequency w.
    _, result = model(t, init_state, b=b, om_r=om_r, om=w,
                      w_p=w_p/n_p_odmr, ti=0.0, mode="Laser-MW",
                      progress_bar="OFF")
    # Compute PL from the expectation values.
    return result
def cwodmr(t, ws, model, init_state, exp_ops, b=B, om_r=om_r_odmr, w_p=w_p_odmr):
    # Create a version of run_single_frequency with fixed model, t, etc.
    run_single = partial(run_single_frequency_cwodmr, model=model, t=t,
                         init_state=init_state, b=b, om_r=om_r, w_p=w_p)
    # Run the simulations in parallel over ws.
    outs = qt.loky_pmap(run_single, ws, progress_bar="tqdm",map_kw={'fail_fast': True})
    # Unpack results and PL values. 
    results = [out for out in outs] #type:ignore
    n_exp = np.array([[qt.expect(M, res.states) for M in exp_ops] for res in results]) #type:ignore
    flur = np.array([np.sum(ns[3]+ns[4]+ns[5]) for ns in n_exp])
    # Combine expectation values from each result.
    #for j in range(len(results[0].expect)):
    #    nn_exp = np.array([results[i].expect[j] for i in range(len(results))])
    #    if j == 0:
    #        n_exp = np.array([nn_exp])
    #    else:
    #        n_exp = np.append(n_exp, [nn_exp], axis=0)
    
    # Gather the time arrays.
    times = results[0].times #type:ignore
    return ws, flur, times, n_exp, results



def puodmr_non_parallel(tp, tw, tr, ws, model, init_state, exp_ops, b=B, om_r=om_r_odmr, w_p=W_p):
    """
    Perform a series of simulations for the puodmr experiment.
    
    Args:
        tp (float): Laser pulse duration.
        tw (float): Free evolution time.
        tr (float): Laser pulse duration.
        ws (array-like): Array of angular frequencies.
        model (function): Function that performs the simulation.
        init_state (object): Initial state of the system.
        b (float, optional): Magnetic field strength. Defaults to B.
        om_r (float, optional): Rabi frequency. Defaults to Om_r.
        w_p (float, optional): Probe frequency. Defaults to W_p.
    
    Returns:
        tuple: A tuple containing the following elements:
            - ws (array-like): Array of angular frequencies.
            - flur (array-like): Array of PL values.
            - times (array-like): Array of time values for the plots.
            - n_exps (array-like): Array of expectation values for each operator.
            - res (array-like): Array of simulation results.
    """
    flur=np.array([])
    for w in tqdm(ws):
        results=np.array([])
        time=np.array([])
        ti=0.0
        #laser tp
        tf,result=model(tp, 
                        init_state,
                        b=b,
                        om_r=om_r,
                        om=w,
                        w_p=w_p, 
                        ti=ti, 
                        mode="Laser", 
                        progress_bar="OFF")
        results=np.append(results,result)
        init_state = result.states[-1]
        #free tw
        ti=tf
        tf,result=model(tw, 
                        init_state,
                        b=b,
                        om_r=om_r,
                        om=w,
                        w_p=w_p,
                        ti=ti, 
                        mode="Free", 
                        progress_bar="OFF")
        results=np.append(results,result)
        init_state = result.states[-1]
        #pi pulse
        ti=tf
        tpi=np.pi/om_r
        tf,result=model(tpi, 
                        init_state,
                        b=b,
                        om_r=om_r,
                        om=w,
                        w_p=w_p, 
                        ti=ti, 
                        mode="MW", 
                        progress_bar="OFF")
        results=np.append(results,result)
        init_state = result.states[-1]
        #laser tr
        ti=tf
        tf,result=model(tr, 
                        init_state,
                        b=b,
                        om_r=om_r,
                        om=w,
                        w_p=w_p_odmr, 
                        ti=ti, 
                        mode="Laser", 
                        progress_bar="OFF")
        results=np.append(results,result)
        n_exp = np.array([np.concatenate([qt.expect(M, res.states) for res in results]) for M in exp_ops])
        #for j in range(len(results[0].expect)):
        #    nn_exp = np.array([])
        #    for i in range(len(results)):
        #        nn_exp = np.append(nn_exp, results[i].expect[j])
        #    if j == 0:
        #        n_exp = np.array([nn_exp])
        #    else:
        #        n_exp = np.append(n_exp, [nn_exp], axis=0)
        fl=n_exp[3]+n_exp[4]+n_exp[5]
        flur=np.append(flur,np.sum(fl))
        # Gather the times for the plots
        for i in range(len(results)):
            time = np.append(time, results[i].times)
        if w == ws[0]:
            times=np.array([time])
            res=np.array([results])
            n_exps=np.array([n_exp])
        else:    
            times=np.append(times, [time], axis=0)
            res=np.append(res, [results], axis=0)
            n_exps=np.append(n_exps, [n_exp], axis=0)
    return ws,flur,times,n_exps,res


def run_single_frequency_puodmr(w, tp, tw, tr, model, init_state, exp_ops, b, om_r, w_p):
    # Use a local copy of the initial state so that each frequency simulation is independent.
    current_state = init_state  # Consider current_state = init_state.copy() if needed
    results = np.array([])  # to collect simulation result objects
    ti = 0.0
    # 1. Laser pulse (duration tp, mode "Laser")
    tf, result = model(tp, current_state,
                         b=b, om_r=om_r, om=w, w_p=w_p/n_p_odmr,
                         ti=ti, mode="Laser", progress_bar="OFF")
    results = np.append(results, result)
    current_state = result.states[-1]
    # 2. Free evolution (duration tw, mode "Free")
    ti = tf
    tf, result = model(tw, current_state,
                         b=b, om_r=om_r, om=w, w_p=w_p,
                         ti=ti, mode="Free", progress_bar="OFF")
    results = np.append(results, result)
    current_state = result.states[-1]
    # 3. MW (pi pulse): duration pi/om_r, mode "MW"
    ti = tf
    tpi = np.pi/om_r
    tf, result = model(tpi, current_state,
                         b=b, om_r=om_r, om=w, w_p=w_p,
                         ti=ti, mode="MW", progress_bar="OFF")
    results = np.append(results, result)
    current_state = result.states[-1]
    # 4. Laser pulse (duration tr, mode "Laser") with different probe frequency
    ti = tf
    tf, result = model(tr, current_state,
                         b=b, om_r=om_r, om=w, w_p=w_p,  
                         ti=ti, mode="Laser", progress_bar="OFF")
    results = np.append(results, result)
    ## Build n_exp by concatenating the expectation values from each simulation.
    n_exp = np.array([np.concatenate([qt.expect(M, res.states) for res in results]) for M in exp_ops])
    #for j in range(len(results[0].expect)):
    #    nn_exp = np.array([])
    #    for i in range(len(results)):
    #        nn_exp = np.append(nn_exp, results[i].expect[j])
    #    if j == 0:
    #        n_exp = np.array([nn_exp])
    #    else:
    #        n_exp = np.append(n_exp, [nn_exp], axis=0)
    # Define PL as the sum of expectation values for indices 3, 4, and 5.
    fl = qt.expect(exp_ops[3]+exp_ops[4]+exp_ops[5], result.states)
    flur = np.sum(fl)
    # Gather times from each result.
    time_arr = np.concatenate([ress.times for ress in results])
    # Return a tuple: (w, flur, time_arr, n_exp, results)
    return (w, flur, time_arr, n_exp, results)

def puodmr(tp, tw, tr, ws, model, init_state,exp_ops, b=B, om_r=om_r_odmr, w_p=w_p_odmr):
    """
    Perform a series of simulations for the PUODMR experiment in parallel.
    
    Args:
        tp (float): Duration of the first (laser) pulse.
        tw (float): Duration of the free evolution.
        tr (float): Duration of the final (laser) pulse.
        ws (array-like): Array of angular frequencies.
        model (function): The simulation function.
        init_state: The initial state.
        b (float): Magnetic field.
        om_r (float): Rabi frequency.
        w_p (float): Probe frequency.
        
    Returns:
        tuple: (ws, flur, times, n_exps, res) where:
            - ws: Input frequencies.
            - flur: Array of PL values (one per frequency).
            - times: Array (or list) of time arrays.
            - n_exps: Array (or list) of expectation arrays.
            - res: Array (or list) of simulation results.
    """
    # Fix all parameters except frequency using partial.
    run_single = partial(run_single_frequency_puodmr,
                         tp=tp, tw=tw, tr=tr,
                         model=model, init_state=init_state,exp_ops=exp_ops,
                         b=b, om_r=om_r, w_p=w_p)
    # Use loky_pmap to run in parallel over ws.
    outs = qt.loky_pmap(run_single, ws, progress_bar="tqdm", map_kw={'fail_fast': True})
    
    # Now outs is a list of tuples: (w, flur, time_arr, n_exp, results) for each w.
    flur_all = np.array([item[1] for item in outs])         #type:ignore
    times_all = np.array([item[2] for item in outs])        #type:ignore
    n_exps_all = np.array([item[3] for item in outs])       #type:ignore
    res_all = np.array([item[4] for item in outs])          #type:ignore
    
    return ws, flur_all, times_all, n_exps_all, res_all


def ramsey(ts, tp, tw, tr, model, init_state,exp_ops,
           b=B, om_r=Om_r, w_p=W_p,om=omega):
    """
    Perform Ramsey experiment simulation.
    
    Parameters:
    - tp (float): Laser pulse duration for preparation.
    - tw (float): Free evolution time between the laser pulse and first pi/2 pulse.
    - ts (float): Free evolution time between the first and second pi/2 pulses.
    - tr (float): Laser pulse duration for readout.
    - model (function): Function that defines the simulation model.
    - init_state (object): Initial state of the system.
    - b (array): Parameter b.
    - om_r (float): Rabi frequency.
    - w_p (float): Parameter w_p.
    
    Returns:
    - ws (array): Array of frequencies.
    - flur (array): Array of PL values.
    - times (array): Array of times for the plots.
    - n_exps (array): Array of expectation values.
    - res (array): Array of simulation results.
    """
    flur=np.array([])
    #for t in tqdm(ts):
    results=np.array([])
    time=np.array([])
    ti=0.0
    #laser tp
    tf,result=model(tp, 
                    init_state,
                    b=b,
                    om_r=om_r,
                    om=om,
                    w_p=w_p, 
                    ti=ti, 
                    mode="Laser", 
                    progress_bar="OFF")
    results=np.append(results,result)
    init_state = result.states[-1]
    #free tw
    ti=tf
    tf,result=model(tw, 
                    init_state,
                    b=b,
                    om_r=om_r,
                    om=om,
                    w_p=w_p,
                    ti=ti, 
                    mode="Free", 
                    progress_bar="OFF")
    results=np.append(results,result)
    init_state = result.states[-1]
    #pi/2 pulse
    ti=tf
    tpi2=0.5*np.pi/om_r
    tf,result=model(tpi2, 
                    init_state,
                    b=b,
                    om_r=om_r,
                    om=om,
                    w_p=w_p,
                    ti=ti, 
                    mode="MW", 
                    progress_bar="OFF")
    results=np.append(results,result)
    init_state = result.states[-1]
    #free ts
    ti=tf
    tf,result=model(ts, 
                    init_state,
                    b=b,
                    om_r=om_r,
                    om=om,
                    w_p=w_p,
                    ti=ti, 
                    mode="Free", 
                    progress_bar="OFF")
    results=np.append(results,result)
    init_state = result.states[-1]
    #pi/2 pulse
    ti=tf
    tpi2=0.5*np.pi/om_r
    tf,result=model(tpi2, 
                    init_state,
                    b=b,
                    om_r=om_r,
                    om=om,
                    w_p=w_p,
                    ti=ti, 
                    mode="MW", 
                    progress_bar="OFF")
    results=np.append(results,result)
    init_state = result.states[-1]
    #laser tr
    ti=tf
    tf,result=model(tr, 
                    init_state,
                    b=b,
                    om_r=om_r,
                    om=om,
                    w_p=w_p_ramsey,
                    ti=ti, 
                    mode="Laser", 
                    progress_bar="OFF")
    results=np.append(results,result)
    n_exp = np.array([np.concatenate([qt.expect(M, res.states) for res in results]) for M in exp_ops])
    fl=qt.expect(exp_ops[3]+exp_ops[4]+exp_ops[5], result.states) 
    flur=np.append(flur,np.sum(fl))
    # Gather the times for the plots
    time = np.concatenate([res.times for res in results])

    return ts,flur,time,n_exp,results


def ramsey_parallel(tp, tw, ts, tr, model, init_state, exp_ops, b, om_r, w_p, om):
    """
    Run the Ramsey simulation in parallel over the free evolution time values in ts.
    
    Returns a tuple:
       (ts_out, flur_out, times_out, n_exps_out, res_out)
    where each element is an array (or list) with one entry per t.
    """
    # Create a partial function that fixes the parameters except t.
    run_ramsey = partial(ramsey, tp=tp, tw=tw, tr=tr,
                          model=model, init_state=init_state,exp_ops=exp_ops,
                          b=b, om_r=om_r, w_p=w_p, om=om)
    # Use loky_pmap to run in parallel over ts.
    outs = qt.loky_pmap(run_ramsey, ts, progress_bar="tqdm", map_kw={'fail_fast': True})
    
    # outs is a list of tuples: (t, flur, times, n_exp, results) for each t.
    ts_out = np.array([item[0] for item in outs])       #type:ignore
    flur_out = np.array([item[1] for item in outs])     #type:ignore
    times_out = np.array([item[2] for item in outs])    #type:ignore
    n_exps_out = np.array([item[3] for item in outs])   #type:ignore
    res_out = np.array([item[4] for item in outs])      #type:ignore
    
    return ts_out, flur_out, times_out, n_exps_out, res_out