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

def optimize_phases(X:np.ndarray, Y:np.ndarray):
    """
    Compute phases δ = (δ_1,...,δ_{N} such that ||exp(diag(δ))X - Y||^2 is minimized, which equals
    \sum_{i=1}^{N} (2 - 2*abs(⟨X_i|Y_i⟩))

        *** In practice N = d(d+1)T, where T is the number of theta's values.

    :param X: np.ndarray of size N x d
    :param Y: np.ndarray of size N x d
    :return: np.ndarray of size N
    """
    assert X.shape == Y.shape and len(X.shape) == 2
    N, d = X.shape

    ### Modify code here
    #U = np.inner(Y.conj(),X).diagonal()
    U = np.sum(Y.conj() * X, axis=1)
    phases = -np.angle(U)
    #res = 2*N - 2*np.sum(np.abs(U))
    res = np.linalg.norm(X * np.exp(1j*phases)[:,None] - Y)**2
    assert np.isclose(res, 2*N - 2*np.sum(np.abs(U))), print(res, 2*N - 2*np.sum(np.abs(U)))
    ###

    return phases, res

def optimize_unitary(X:np.ndarray, Y:np.ndarray):
    """
    Find a unitary matrix V to minimize ||XV - Y||^2 (Complex orthogonal Procrustes problem)

    :param X: np.ndarray of size N x d
    :param V: np.ndarray of size N x d
    :return: np.ndarray of size d x d
    """
    assert X.shape == Y.shape and len(X.shape) == 2
    N, d = X.shape
    #print("before procrustes: ", np.linalg.norm(X-Y)**2)
    ### Modify code here

    #V = np.zeros((d,d))
    M = (X.conj().T) @ Y
    U, Sigma, V_dag = scipy.linalg.svd(M)
    Omega = U @ V_dag
    res = np.linalg.norm(X @ Omega - Y) ** 2
    ###
    #print("after procrustes: ", res)
    return Omega, res

def optimize_permutation(spectrumA:np.ndarray, spectrumB:np.ndarray):
    """
    Find a permutation π over anchor states to minimize \sum_{i=1}^{m} || spectrumA[i] - spectrumB[π[i]] ||^2 (assignment problem)
        *** In practice, m = d(d+1)
    :param spectrumA: np.ndarray of size m x T x d
    :param spectrumB: np.ndarray of size m x T x d
    :return:np.ndarray of size m
    """
    assert spectrumA.shape == spectrumB.shape and len(spectrumA.shape) == 3
    m, T, d = spectrumA.shape

    ### Modify code here
    #print('before permutation', np.linalg.norm(spectrumA - spectrumB)**2)


    ### This construction uses norms over permutations, minimization
    # cost_matrix = np.zeros(shape=(spectrumA.shape[0], spectrumB.shape[0]))
    # for i in range(len(cost_matrix)):
    #     for j in range(len(cost_matrix)):
    #         cost_matrix[i, j] = np.linalg.norm(spectrumA[i] - spectrumB[j])**2
    # row_ind, B_perm = linear_sum_assignment(cost_matrix)
    # print('after permutation', np.linalg.norm(spectrumA - spectrumB[B_perm]) ** 2)
    #return B_perm, cost_matrix[row_ind, B_perm].sum()

    ### This uses sum of fidelity, maximization. The optimal permutation is the same as above
    # cost_matrix = np.zeros(shape=(spectrumA.shape[0], spectrumB.shape[0]))
    # for i in range(len(cost_matrix)):
    #     for j in range(len(cost_matrix)):
    #         cost_matrix[i, j] = sum(np.abs(np.sum(spectrumB[j].conj() * spectrumA[i], axis=1)))
    # row_ind, B_perm = linear_sum_assignment(cost_matrix, maximize=True)


    # Equivalent, but faster
    cost_matrix = np.sum(np.abs(np.matmul(spectrumB.conj().transpose(1,0,2), spectrumA.transpose(1,2,0)).transpose(1,2,0)), axis=2).T
    row_ind, B_perm = linear_sum_assignment(cost_matrix, maximize=True)
    #print('after permutation', np.linalg.norm(spectrumA - spectrumB[B_perm]) ** 2)


    return B_perm, np.linalg.norm(spectrumA - spectrumB[B_perm]) ** 2

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

def optimization_routine(spectrum_V1, spectrum_V2, tol=10e-6):
    m, T, d = spectrum_V1.shape
    k = 0

    newX = spectrum_V1.copy().reshape(m * T, d)
    newY = spectrum_V2.copy().reshape(m * T, d)
    #print('raw: ', np.linalg.norm(newX - newY) ** 2)
    phasesX, val = optimize_phases(X=newX, Y=newY)
    newX = newX * np.exp(1j * phasesX)[:, None]
    #print('init phase: ', val)

    val_prev = val

    tail_length = 0

    while True:
        k += 1
        #print('iteration:', k)
        neworder, val = optimize_permutation(spectrumA=newX.reshape(m, T, d), spectrumB=newY.reshape(m, T, d))
        newY = newY.reshape(m, T, d)[neworder].reshape(m * T, d)

        phasesX, val = optimize_phases(X=newX, Y=newY)
        newX = newX * np.exp(1j * phasesX)[:, None]
        #print('phase after perm: ', val)

        V, val = optimize_unitary(X=newX, Y=newY)
        newX = newX @ V
        #print('unitary: ', val)

        phasesX, val = optimize_phases(X=newX, Y=newY)
        newX = newX * np.exp(1j * phasesX)[:, None]
        #print('phase after unitary: ', val)

        #print(f'Iteration {k}: optimal value = {val}')
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

    # Generate a cluster of T=num_samples states from gates V1(theta) and V2(theta) for each anchor state.
    lo_bound = -np.pi
    up_bound = np.pi
    spectrum_V1 = get_state_spectrum(num_qubits, V1, qargs1,
                                     np.linspace(lo_bound, up_bound, num_theta_samples, endpoint=False), anchor_states)
    spectrum_V2 = get_state_spectrum(num_qubits, V2, qargs2,
                                     np.linspace(lo_bound, up_bound, num_theta_samples, endpoint=False), anchor_states)
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
    m, T, d = spectrum_V1.shape

    opt_val_list = []
    for i in range(num_trials):
        #print(f'Trial {i+1}: ')

        if i == 0:
            perturbed_spectrum_V1 = spectrum_V1.copy()

            # perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(m*T,d) @ random_unitary(d).data
            # perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(m,T,d)
        else:
            random_order = np.random.permutation(num_anchors)
            perturbed_spectrum_V1 = spectrum_V1[random_order].reshape(m * T, d)

            rand_unitary = np.kron(np.eye(int(d / active_d)), random_unitary(active_d).data)
            #print(rand_unitary)
            perturbed_spectrum_V1 = perturbed_spectrum_V1 @ rand_unitary
            perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(m, T, d)



        opt_val = optimization_routine(spectrum_V1=perturbed_spectrum_V1, spectrum_V2=spectrum_V2, tol=tol)
        opt_val_list.append(opt_val)

        if np.isclose(opt_val, m*T):
            break

    opt_val_list = np.array(opt_val_list)

    #print(f'best = {opt_val_list.max()}, mean = {opt_val_list.mean()}, sample variance = {opt_val_list.var()}')

    shape_distance = 1. - opt_val_list.max() / (m*T)

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

def modify_shape_dist_dict(all_shape_distances):
    for pairname, dist in all_shape_distances.items():
        if dist == np.inf:
            all_shape_distances[pairname] = 10e4

    for pairname, dist in all_shape_distances.items():
        V1,V2,num_qubits,pos_tag = pairname.split('_')

        all_positioning = gate_positioning.all_relative_positions(V1=V1,V2=V2,num_qubits=int(num_qubits))

        if pairname in all_positioning:
            qargs = all_positioning[pairname]
            rev_qargs = [qargs[1], qargs[0]]

            rev_pos = gate_positioning.get_pos_from_gate_name(V1=V2,V2=V1,qargs=rev_qargs)

            rev_pairname = '_'.join([V2,V1,num_qubits,rev_pos])

            rev_dist = all_shape_distances[rev_pairname]
            #print(pairname, rev_pairname, qargs, rev_qargs, dist, rev_dist)
            #if not np.isclose(dist, rev_dist):
            if ADMISSIBLE_GATES.index(V1) <= ADMISSIBLE_GATES.index(V2) and abs(dist - rev_dist) >= 0.001:
                print(pairname, rev_pairname, qargs, rev_qargs, dist, rev_dist)

            all_shape_distances[pairname] = min(all_shape_distances[pairname], all_shape_distances[rev_pairname])
    return all_shape_distances


if __name__ == '__main__':
    import sys
    sys.path.append('/Users/erio/Dropbox/URP project/Code/PQC_composer')
    np.random.seed(20)

    num_theta_samples = 12

    #print(compute_shape_distance('rxx', 'rxx', num_qubits=4, num_theta_samples=num_theta_samples, num_trials=1000))
    # print(compute_shape_distance('rz', 'rx', num_qubits=2, num_theta_samples=num_theta_samples))

    import pickle

    filename = './' + str(num_theta_samples) + 'theta_raw_all_shape_distances.pkl'

    try:
        with open(filename, 'rb') as f:
            ALL_SHAPE_DISTANCES = pickle.load(f)
    except:
        with open(filename, 'wb') as f:
            ALL_SHAPE_DISTANCES = {}
            pickle.dump(ALL_SHAPE_DISTANCES, f)
    #
    #
    # for q in [1,2,3,4]:
    #     for i,V1 in enumerate(ADMISSIBLE_GATES):
    #         for j,V2 in enumerate(ADMISSIBLE_GATES):
    #             if q == 1:
    #                 if V1 in TWO_QUBIT_DETERMINISTIC_GATES or V1 in TWO_QUBIT_VARIATIONAL_GATES \
    #                     or V2 in TWO_QUBIT_DETERMINISTIC_GATES or V2 in TWO_QUBIT_VARIATIONAL_GATES:
    #                         continue ## not enough qubit
    #
    #             positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=q)
    #             if any([pos not in ALL_SHAPE_DISTANCES for pos in positions.keys()]):
    #                 try:
    #                     shape_distance = compute_shape_distance(V1,V2, num_qubits=q, num_theta_samples=num_theta_samples)
    #                 except:
    #                     continue
    #                 else:
    #                     ALL_SHAPE_DISTANCES.update(shape_distance)
    #                     with open(filename, 'wb') as f:
    #                         pickle.dump(ALL_SHAPE_DISTANCES, f)
    #
    # print(len(ALL_SHAPE_DISTANCES))
    # print(ALL_SHAPE_DISTANCES)
    # with open(filename, 'rb') as f:
    #     loaded_dict = pickle.load(f)
    #     print(loaded_dict == ALL_SHAPE_DISTANCES)


    ## Modify
    ALL_SHAPE_DISTANCES = modify_shape_dist_dict(ALL_SHAPE_DISTANCES)
    with open(filename, 'wb') as f:
        pickle.dump(ALL_SHAPE_DISTANCES, f)

    # V1 = 'rx'
    # V2 = 'ry'
    # num_qubits = 3
    # rel_pos = 's'
    # all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    #
    # print(all_positions)
    # pos = '_'.join([V1,V2,str(num_qubits),rel_pos])
    # print(pos)
    # two_qargs = all_positions[pos]
    # dshape = _shape_distance_with_config(num_qubits=num_qubits, V1=V1, V2=V2, num_theta_samples=num_theta_samples,
    #                             qargs1=two_qargs[0], qargs2=two_qargs[1], num_trials=1000)
    # print(dshape)
    #
    # with open(filename, 'rb') as f:
    #     ALL_SHAPE_DISTANCES = pickle.load(f)
    #
    # if dshape <= ALL_SHAPE_DISTANCES[pos]:
    #     ALL_SHAPE_DISTANCES[pos] = dshape
    #
    # with open(filename, 'wb') as f:
    #     pickle.dump(ALL_SHAPE_DISTANCES, f)


