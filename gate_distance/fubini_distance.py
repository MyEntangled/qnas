import numpy as np
import scipy

def minimize_sum_fidelity(X,Y,num_trials=5):
    '''
    Each row of X,Y contains a (normalized) pure state
    Find a unitary U to minimize \sum_i |⟨X_i|U|Y_i⟩|^2
    :param X:
    :param Y:
    :return:
    '''

    def obj_func(d, *args):
        verbose, = args

        phased_X = np.diag(np.exp(1j * d)) @ X
        M = phased_X.conj().T @ Y
        U, Sigma, V_dag = np.linalg.svd(M)
        Omega = U @ V_dag

        if verbose is True:
            print('state wise norm')
            print(np.linalg.norm(phased_X @ Omega - Y, axis=1) ** 2 / num_states)

        ## The return value = 2*num_states - 2*np.linalg.norm(M,ord='nuc'), but the way below is more efficient
        return np.linalg.norm(phased_X @ Omega - Y) ** 2 / num_states

    assert len(X) == len(Y), "The length of X and Y should be equal."
    num_states = len(X)

    min_res = np.inf

    for _ in range(num_trials):
        x0 = np.random.uniform(0, 2 * np.pi, num_states)
        res = scipy.optimize.minimize(obj_func, x0,args=(False,))
        if res.fun < min_res:
            min_res = res.fun
            optimal_phase = res.x

    #obj_func(optimal_phase, True)

    return min_res