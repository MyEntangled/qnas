import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary, Operator

from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
import scipy

import sys
sys.path.append('/Users/erio/Dropbox/URP project/Code/PQC_composer/src')
from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES, ADMISSIBLE_GATES, DIRECTED_GATES, UNITARY
import gate_positioning
import MUBs

# def optimize_phases(X:np.ndarray, Y:np.ndarray):
#     """
#     Compute phases δ = (δ_1,...,δ_{N} such that ||exp(diag(δ))X - Y||^2 is minimized, which equals
#     \sum_{i=1}^{N} (2 - 2*abs(⟨X_i|Y_i⟩))
#
#         *** In practice N = d(d+1)T, where T is the number of theta's values.
#
#     :param X: np.ndarray of size N x d
#     :param Y: np.ndarray of size N x d
#     :return: np.ndarray of size N
#     """
#     assert X.shape == Y.shape and len(X.shape) == 2
#     N, d = X.shape
#
#     ### Modify code here
#     #U = np.inner(Y.conj(),X).diagonal()
#     U = np.sum(Y.conj() * X, axis=1)
#     phases = -np.angle(U)
#     #res = 2*N - 2*np.sum(np.abs(U))
#     res = np.linalg.norm(X * np.exp(1j*phases)[:,None] - Y)**2
#     assert np.isclose(res, 2*N - 2*np.sum(np.abs(U))), print(res, 2*N - 2*np.sum(np.abs(U)))
#     ###
#
#     return phases, res

def optimize_phases_v2(X:np.ndarray, Y:np.ndarray):
    """
    Compute phases δ = (δ_1,...,δ_{N} such that ||exp(diag(δ))X - Y||^2 is minimized, which equals
    \sum_{i=1}^{N} (2 - 2*abs(⟨X_i|Y_i⟩))

        *** In practice N = d(d+1)T, where T is the number of theta's values.

    :param X: np.ndarray of size d x N1
    :param Y: np.ndarray of size d x N2
    :return: np.ndarray of size N
    """
    assert X.shape[0] == Y.shape[0] and len(X.shape) == 2
    d,N1 = X.shape
    d,N2 = Y.shape

    U = np.sum(Y.conj() * X, axis=0)
    phases = -np.angle(U)

    #res = np.linalg.norm(X * np.exp(1j*phases)[:,None] - Y)**2
    res = np.linalg.norm(X @ np.diag(np.exp(1j*phases)) - Y)**2

    exact = N1+N2 - 2*np.sum(np.abs(U))
    assert abs(res-exact) < 1e-6, print('lalal', res, exact)

    ###

    return phases, res

# def optimize_unitary(X:np.ndarray, Y:np.ndarray):
#     """
#     Find a unitary matrix V to minimize ||XV - Y||^2 (Complex orthogonal Procrustes problem)
#
#     :param X: np.ndarray of size N x d
#     :param Y: np.ndarray of size N x d
#     :return: np.ndarray of size d x d
#     """
#     assert X.shape == Y.shape and len(X.shape) == 2
#     N, d = X.shape
#     #print("before procrustes: ", np.linalg.norm(X-Y)**2)
#     ### Modify code here
#
#     #V = np.zeros((d,d))
#     M = (X.conj().T) @ Y
#     U, Sigma, V_dag = scipy.linalg.svd(M)
#     Omega = U @ V_dag
#     res = np.linalg.norm(X @ Omega - Y) ** 2
#     ###
#     #print("after procrustes: ", res)
#     return Omega, res

def optimize_unitary_v2(X:np.ndarray, Y:np.ndarray):
    """
    Find a unitary matrix V to minimize ||VX - Y||^2 (Complex orthogonal Procrustes problem)

    :param X: np.ndarray of size d x KT
    :param Y: np.ndarray of size d x KT
    :return: np.ndarray of size d x d
    """
    assert X.shape == Y.shape and len(X.shape) == 2
    d, KT = X.shape
    #print("before procrustes: ", np.linalg.norm(X-Y)**2)

    #V = np.zeros((d,d))
    M = (X.conj().T) @ Y
    U, Sigma, V_dag = scipy.linalg.svd(M)
    Omega = U @ V_dag
    res = np.linalg.norm(X @ Omega - Y) ** 2
    ###
    #print("after procrustes: ", res)
    return Omega, res

def optimize_lincomb(L1:np.ndarray, L2:np.ndarray):
    """
    Find a unitary matrix V to minimize ||M @ L1 - L2||^2 (Complex orthogonal Procrustes problem)
    M: matrix of size d x K, where each column is normalized, and every pair of columns are not identical up to overall phases

    :param L1: np.ndarray of size K x KT
    :param L2: np.ndarray of size d x KT
    :return: np.ndarray of size d x K
    """
    d = L2.shape[0]
    K = L1.shape[0]

    R, Sigma, W_dag = scipy.linalg.svd(L1 @ L2.conj().T)
    sub_id = np.zeros((d,K), dtype=complex)
    M = W_dag.conj().T @ sub_id @ R.conj().T
    return M


def get_state_spectrum(num_qubits, V, qargs, thetas, anchor_states):
    '''
    Get V(theta)|anchor⟩ for various thetas and anchor states
    :param num_qubits:
    :param V:
    :param qargs:
    :param thetas:
    :param anchor_states:
    :return:
    '''
    assert V in ADMISSIBLE_GATES, f"V({V}) must belong to ADMISSIBLE_GATES({ADMISSIBLE_GATES})"

    output_states = np.zeros(shape=(len(anchor_states), len(thetas), 2 ** num_qubits), dtype=np.complex128)

    for i, anchor_state in enumerate(anchor_states):

        anchor = Statevector(anchor_state)  # initialize an anchor state

        for j, theta in enumerate(thetas):
            var_V_circ = QuantumCircuit(num_qubits)
            if V in SINGLE_QUBIT_DETERMINISTIC_GATES:  # one-qubit deterministic
                args = (*qargs,)
            elif V in SINGLE_QUBIT_VARIATIONAL_GATES:  # one-qubit variational
                args = (theta, *qargs)
            elif V in TWO_QUBIT_DETERMINISTIC_GATES:  # two-qubit deterministic
                args = (*qargs,)
            elif V in TWO_QUBIT_VARIATIONAL_GATES:  # two-qubit variational
                args = (theta, *qargs)

            getattr(var_V_circ, V)(*args)
            # print(var_V_circ.draw())
            output_states[i, j] = anchor.evolve(var_V_circ).data

    return np.array(output_states)

def get_double_spectrum(num_qubits, V, V1, V2, qargs1, qargs2, thetas, anchor_states):
    assert V1 in ADMISSIBLE_GATES, f"V({V1}) must belong to ADMISSIBLE_GATES({ADMISSIBLE_GATES})"
    assert V2 in ADMISSIBLE_GATES, f"V({V2}) must belong to ADMISSIBLE_GATES({ADMISSIBLE_GATES})"
    K = len(anchor_states)
    T = len(thetas)
    d = 2*num_qubits

    output_states = np.zeros(shape=(len(anchor_states), len(thetas), d), dtype=np.complex128)

    for i, anchor_state in enumerate(anchor_states):

        anchor = Statevector(anchor_state)  # initialize an anchor state

        temp = np.zeros((d,T), dtype=np.complex128)

        for j, theta in enumerate(thetas):
            var_V2_circ = QuantumCircuit(num_qubits)
            if V2 in SINGLE_QUBIT_DETERMINISTIC_GATES:  # one-qubit deterministic
                args = (*qargs2,)
            elif V2 in SINGLE_QUBIT_VARIATIONAL_GATES:  # one-qubit variational
                args = (theta, *qargs2)
            elif V2 in TWO_QUBIT_DETERMINISTIC_GATES:  # two-qubit deterministic
                args = (*qargs2,)
            elif V2 in TWO_QUBIT_VARIATIONAL_GATES:  # two-qubit variational
                args = (theta, *qargs2)

            getattr(var_V2_circ, V2)(*args)
            # print(var_V_circ.draw())
            temp[:,j] = anchor.evolve(var_V2_circ).data

        temp = V.conj().T @ temp

        for j, theta in enumerate(thetas):
            var_V1dag_circ = QuantumCircuit(num_qubits)
            if V1 in SINGLE_QUBIT_DETERMINISTIC_GATES:  # one-qubit deterministic
                args = (*qargs2,)
            elif V1 in SINGLE_QUBIT_VARIATIONAL_GATES:  # one-qubit variational
                args = (-theta, *qargs2)
            elif V1 in TWO_QUBIT_DETERMINISTIC_GATES:  # two-qubit deterministic
                args = (*qargs2,)
            elif V1 in TWO_QUBIT_VARIATIONAL_GATES:  # two-qubit variational
                args = (-theta, *qargs2)

            getattr(var_V1dag_circ, V1)(*args)
            output_states[i, j]  = Statevector(temp[:,j]).evolve(var_V1dag_circ).data

    return np.array(output_states)

# def optimization_routine(spectrum_V1, spectrum_V2, tol=10e-6):
#     K, T, d = spectrum_V1.shape
#     iter = 0
#
#     newX = spectrum_V1.copy().reshape(d, K*T)
#     newY = spectrum_V2.copy().reshape(d, K*T)
#     #print('raw: ', np.linalg.norm(newX - newY) ** 2)
#     phasesX, val = optimize_phases_v2(X=newX, Y=newY)
#     newX = newX @ np.diag(np.exp(1j*phasesX))
#     #print('init phase: ', val)
#
#     val_prev = val
#
#     tail_length = 0
#
#     while True:
#         iter += 1
#         #print('iteration:', iter)
#
#         phasesX, val = optimize_phases_v2(X=newX, Y=newY)
#         newX = newX @ np.diag(np.exp(1j*phasesX))
#         print('phase after perm: ', val)
#
#         V, val = optimize_unitary_v2(X=newX, Y=newY)
#         newX = V @ newX
#         print('unitary: ', val, V.shape)
#
#         # phasesX, val = optimize_phases_v2(X=newX, Y=newY)
#         # newX = newX @ np.diag(np.exp(1j*phasesX))
#         #print('phase after unitary: ', val)
#
#         #print(f'Iteration {iter}: optimal value = {val}')
#         if abs(val-val_prev) < tol:
#             tail_length += 1
#             if tail_length == 3:
#                 opt_sum_fid = (2 * spectrum_V1.shape[0] * spectrum_V1.shape[1] - val) / 2.
#                 return opt_sum_fid
#         else:
#             val_prev = val

def optimization_routine_v2(anchor_states, spectrum_V1, spectrum_V2, M, V1, V2, qargs1, qargs2, thetas, tol=10e-6):
    K, T, d = spectrum_V1.shape
    num_qubits = int(np.log2(d))
    iter = 0

    newX = spectrum_V1.copy().reshape(d, K*T)
    newY = spectrum_V2.copy().reshape(d, K*T)
    print('newY', newY.shape)
    val_prev = K*T*d

    tail_length = 0

    while True:
        iter += 1
        #print('iteration:', iter)

        ### Find M
        spectrum_M1 = get_state_spectrum(num_qubits=num_qubits,V=V1, qargs=qargs1, thetas=thetas, anchor_states=M.T) # K x T x d
        newM1 = spectrum_M1.reshape(d, K*T)
        print('spectrum_M1', spectrum_M1.shape)
        print('newM1', newM1.shape)

        phasesX, val = optimize_phases_v2(X=newM1, Y=newY)
        newM1 = newM1 @ np.diag(np.exp(1j*phasesX))
        print('phase involving M1 spectrum: ', val)

        V, val = optimize_unitary_v2(X=newX, Y=newY)
        newX = V @ newX
        print('unitary: ', val, V.shape)

        L2 = get_double_spectrum(num_qubits, V, V1, V2, qargs1, qargs2, thetas, anchor_states)
        L2 = L2.reshape(d,K*T)

        L1 = np.repeat(M, T, axis=1) ##

        phasesX, val = optimize_phases_v2(X=L1, Y=L2)
        L1 = L1 @ np.diag(np.exp(1j*phasesX))
        #print('phase after unitary: ', val)

        #print(f'Iteration {iter}: optimal value = {val}')
        if abs(val-val_prev) < tol:
            tail_length += 1
            if tail_length == 3:
                opt_sum_fid = (2 * spectrum_V1.shape[0] * spectrum_V1.shape[1] - val) / 2.
                return opt_sum_fid
        else:
            val_prev = val

def _shape_distance_with_config(num_qubits, V1, V2, qargs1, qargs2, num_theta_samples=4, num_trials=500):
    '''
    Return the shape distance between two quantum gates
    :param V1:
    :param V2:
    :return:
    '''

    assert V1 in ADMISSIBLE_GATES and V2 in ADMISSIBLE_GATES, "Input gates are not admissible."

    # Get the list of d(d+1) anchor states
    anchor_states = MUBs.get_anchor_states(num_qubits)
    num_anchors = len(anchor_states)

    lo_bound = -np.pi
    up_bound = np.pi
    thetas = np.linspace(lo_bound, up_bound, num_theta_samples, endpoint=False)

    # Generate a cluster of T=num_samples states from gates V1(theta) and V2(theta) for each anchor state.
    spectrum_V1 = get_state_spectrum(num_qubits, V1, qargs1, thetas, anchor_states)
    spectrum_V2 = get_state_spectrum(num_qubits, V2, qargs2, thetas, anchor_states)

    print(spectrum_V1.shape, spectrum_V2.shape)

    active_qubits = set(qargs1 + qargs2)
    count_active = 0
    for i in range(num_qubits):
        if i in active_qubits:
            count_active += 1
    assert count_active == len(active_qubits)
    active_d = 2**count_active
    print('active qubits', active_qubits)
    print('active dimension', active_d)

    tol = 10e-5
    K, T, d = spectrum_V1.shape

    opt_val_list = []
    for i in range(num_trials):
        #print(f'Trial {i+1}: ')

        if i == 0:
            perturbed_spectrum_V1 = spectrum_V1.copy()

            # perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(K*T,d) @ random_unitary(d).data
            # perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(K,T,d)
        else:
            perturbed_spectrum_V1 = spectrum_V1.reshape(K * T, d)

            rand_unitary = np.kron(np.eye(int(d / active_d)), random_unitary(active_d).data)
            #print(rand_unitary)
            perturbed_spectrum_V1 = perturbed_spectrum_V1 @ rand_unitary
            perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(K, T, d)


        rand_r = np.random.rand(d,K).astype(complex)
        rand_phase = np.random.uniform(0,2*np.pi,(d,K)).astype(complex)
        M = rand_r * np.exp(1j*rand_phase)

        opt_val = optimization_routine_v2(anchor_states=anchor_states,
                                          spectrum_V1=perturbed_spectrum_V1, spectrum_V2=spectrum_V2,
                                          M=M, V1=V1, V2=V2, qargs1=qargs1, qargs2=qargs2, thetas=thetas)
        opt_val_list.append(opt_val)

        if np.isclose(opt_val, K*T):
            break

    opt_val_list = np.array(opt_val_list)

    #print(f'best = {opt_val_list.max()}, mean = {opt_val_list.mean()}, sample variance = {opt_val_list.var()}')

    shape_distance = 1. - opt_val_list.max() / (K*T)

    if shape_distance < 10e-5:
        return 0
    else:
        return shape_distance

def compute_shape_distance(V1: str, V2: str, num_qubits: int, num_theta_samples:int, num_trials:int=500) -> dict:
    all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    all_distances = {}

    for pos_tag, qargs in all_positions.items():
        if (V1 in SINGLE_QUBIT_VARIATIONAL_GATES or V1 in TWO_QUBIT_VARIATIONAL_GATES) and \
            (V2 in SINGLE_QUBIT_VARIATIONAL_GATES or V2 in TWO_QUBIT_VARIATIONAL_GATES):

            all_distances[pos_tag] = _shape_distance_with_config(num_qubits=num_qubits, V1=V1, V2=V2, num_theta_samples=num_theta_samples,
                                                             qargs1=qargs[0], qargs2=qargs[1], num_trials=num_trials)
        elif (V1 in SINGLE_QUBIT_DETERMINISTIC_GATES or V1 in TWO_QUBIT_DETERMINISTIC_GATES) and \
            (V2 in SINGLE_QUBIT_DETERMINISTIC_GATES or V2 in TWO_QUBIT_DETERMINISTIC_GATES):

            all_distances[pos_tag] = 0

        else:
            all_distances[pos_tag] = np.inf

    return all_distances

if __name__ == '__main__':
    print(compute_shape_distance('rx', 'ry', num_qubits=3, num_theta_samples=12, num_trials=1000))