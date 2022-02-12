import numpy as np
from qiskit.quantum_info import random_statevector
import scipy

def minimize_sum_fidelity(X,Y):
    '''
    Each row of X,Y contains a (normalized) pure state
    Find a unitary U to minimize \sum_i |⟨X_i|U|Y_i⟩|^2
    :param X:
    :param Y:
    :return:
    '''

    def obj_func(d, *args):
        phased_X = np.diag(np.exp(1j * d)) @ X
        M = phased_X.conj().T @ Y
        U, Sigma, V_dag = np.linalg.svd(M)
        Omega = U @ V_dag

        ## The return value = 2*num_states - 2*np.linalg.norm(M,ord='nuc'), but this way is more efficient
        return np.linalg.norm(phased_X @ Omega - Y) ** 2

    assert len(X) == len(Y), "The length of X and Y should be equal."
    num_states = len(X)
    num_trials = 5

    min_res = np.inf

    for _ in range(num_trials):
        x0 = np.random.uniform(0, 2 * np.pi, num_states)
        res = scipy.optimize.minimize(obj_func, x0)
        scipy.opti
        if res < min_res:
            min_res = res

    return min_res