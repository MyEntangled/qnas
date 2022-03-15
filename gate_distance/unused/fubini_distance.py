import numpy as np
import scipy

def unitary_procrustes(X, Y):
    assert X.shape == Y.shape
    # X = X.view(np.complex64)
    # Y = Y.view(np.complex64)
    M = (X.conj().T) @ Y
    #print(print(np.isnan(X).any(), np.isnan(Y).any()))
    U, Sigma, V_dag = scipy.linalg.svd(M)
    Omega = U @ V_dag

    ## The return value = 2*num_states - 2*np.linalg.norm(M,ord='nuc'), but the way below is more efficient
    #print("True", np.linalg.norm(X @ Omega - Y) ** 2)
    #num_states = X.shape[0]*Y.shape[0]
    #print("Dist via nuclear norm of M", 2*num_states - 2*np.linalg.norm(M,ord='nuc'), np.linalg.norm(M,ord='nuc'))
    #print("Dist via SVD", 2*num_states - 2*np.sum(Sigma), np.sum(Sigma))
    return Omega, np.linalg.norm(X @ Omega - Y) ** 2

def square_bures_distance(X, Y, phase_def:bool=False):
    """
    Compute \sum_i (2 - 2|⟨X_i|Y_i⟩|) = min_{d} \sum_i ||X_i * exp(i*d_i) - Y_i||
    :param X:
    :param Y:
    :return:
    """
    assert X.shape == Y.shape
    num_states = X.shape[0]

    if not phase_def:
        return np.sum(2 - 2*np.abs(np.sum(X.conj() * Y, axis=1)))

    else:
        euclidean_dist = lambda d: np.linalg.norm(np.diag(np.exp(1j * d)) @ X - Y)**2
        d0 = np.random.uniform(0, 2 * np.pi, num_states).astype(dtype=np.double)
        res = scipy.optimize.minimize(euclidean_dist, d0)
        return res.fun

def min_sum_square_bures_dist(X, Y, num_trials=5, get_unitary:bool=False):
    """
    Each row of X,Y contains a (normalized) pure state
    Find a unitary U to minimize \sum_i (2 - 2|⟨X_i|U|Y_i⟩|) = min_D || e^{iD}*X*U - Y ||^2
    :param X:
    :param Y:
    :param num_trials:
    :param get_unitary:
    :return:
    """

    def obj_func(d, *args):
        verbose, = args

        phased_X = np.diag(np.exp(1j * d)) @ X
        Omega, min_dist = unitary_procrustes(phased_X, Y)

        if verbose is True:
            print('verbose: ', min_dist)

        return Omega, min_dist

    assert X.shape == Y.shape
    num_states = len(X)

    min_square_dist = np.inf
    optimal_phase = np.zeros(num_states, dtype=np.double)

    for _ in range(num_trials):
        x0 = np.random.uniform(0, 2 * np.pi, num_states).astype(dtype=np.double)
        res = scipy.optimize.minimize(lambda x,args: obj_func(x,args)[1], x0,args=(False,))
        if res.fun < min_square_dist:
            min_square_dist = res.fun
            optimal_phase = res.x

    if get_unitary:
        optimal_unitary, min_square_dist = obj_func(optimal_phase, False)
        return optimal_unitary, min_square_dist
    else:
        return min_square_dist

def max_sum_sqrt_fidelity(X, Y, num_trials=5, get_unitary:bool=False):
    '''
    Each row of X,Y contains a (normalized) pure state
    Find a unitary U to maximize \sum_i |⟨X_i|U|Y_i⟩|
    :param X:
    :param Y:
    :return:
    '''
    assert X.shape == Y.shape
    num_states = X.shape[0]

    if get_unitary:
        optimal_unitary, min_square_dist = min_sum_square_bures_dist(X, Y, num_trials, get_unitary=True)
        return optimal_unitary, (2*num_states - min_square_dist)/2
    else:
        min_square_dist = min_sum_square_bures_dist(X, Y, num_trials, get_unitary=False)
        return (2*num_states - min_square_dist)/2