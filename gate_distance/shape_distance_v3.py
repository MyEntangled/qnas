import MUBs
import fubini_distance
import time
import gate_positioning

import numpy as np
import scipy

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary, Operator

from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
import scipy

import MUBs

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES, ADMISSIBLE_GATES, DIRECTED_GATES, UNITARY


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

def get_counter_unitary(spectrum_V1, spectrum_V2, V1, V2, qargs1, qargs2):
    m, T, d = spectrum_V1.shape
    num_qubits = int(np.log2(d))

    qc = QuantumCircuit(num_qubits)
    if V1 == 'rx' and V2 == 'ry':
        print('counter unitary exists')
        qc.rz(np.pi/2, qargs1)
    if V1 == 'rx' and V2 == 'rz':
        print('counter unitary exists')
        qc.ry(np.pi/2, qargs1)

    counter_unitary = Operator(qc).data
    return counter_unitary

def path_unitary(V1, V2, qargs1, qargs2, num_qubits, theta, t):
    qc1 = QuantumCircuit(num_qubits)
    args1 = (theta, *qargs1)
    getattr(qc1, V1)(*args1)

    qc2 = QuantumCircuit(num_qubits)
    args2 = (theta, *qargs2)
    getattr(qc2, V2)(*args2)

    V1_theta_matrix = Operator(qc1).data
    V2_theta_matrix = Operator(qc2).data

    U_path = scipy.linalg.expm(t * scipy.linalg.logm(V2_theta_matrix @ V1_theta_matrix.conj().T))
    #print(theta, t, np.linalg.norm(U_path - V1_theta_matrix), np.linalg.norm(U_path - V2_theta_matrix))
    return U_path

def _shape_distance_with_config(num_qubits, V1, V2, qargs1, qargs2, num_samples=4, num_trials=200):
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
                                     np.linspace(lo_bound, up_bound, num_samples, endpoint=False), anchor_states)
    spectrum_V2 = get_state_spectrum(num_qubits, V2, qargs2,
                                     np.linspace(lo_bound, up_bound, num_samples, endpoint=False), anchor_states)
    print(spectrum_V1.shape, spectrum_V2.shape)

    tol = 10e-5
    m, T, d = spectrum_V1.shape

    opt_val_list = []
    for i in range(num_trials):
        print(f'Trial {i+1}: ')


        if i > 0:
            #random_U = random_unitary(2 ** num_qubits).data
            #random_order = np.random.permutation(num_anchors)
            #random_phases = np.random.uniform(0, 2 * np.pi, m * T)

            #perturbed_spectrum_V1 = spectrum_V1[random_order].reshape(m * T, d)
            #perturbed_spectrum_V1 = perturbed_spectrum_V1 * np.exp(1j * random_phases)[:, None]
            #perturbed_spectrum_V1 = perturbed_spectrum_V1 @ random_U
            #perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(m, T, d)

            random_order = np.random.permutation(num_anchors)
            perturbed_spectrum_V1 = spectrum_V1[random_order].reshape(m * T, d)
            theta = np.random.uniform(0,2*np.pi)
            t = np.random.rand()
            perturbed_spectrum_V1 = perturbed_spectrum_V1 @ path_unitary(V1, V2, qargs1, qargs2, num_qubits, theta, t)
            perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(m, T, d)

        # elif i == 1:
        #     print('HELLO')
        #     counter_unitary = get_counter_unitary(spectrum_V1, spectrum_V2, V1, V2, qargs1, qargs2)
        #     perturbed_spectrum_V1 = spectrum_V1.reshape(m * T, d)
        #     perturbed_spectrum_V1 = perturbed_spectrum_V1 @ counter_unitary
        #     perturbed_spectrum_V1 = perturbed_spectrum_V1.reshape(m, T, d)

        else:
            perturbed_spectrum_V1 = spectrum_V1.copy()

        opt_val = optimization_routine(spectrum_V1=perturbed_spectrum_V1, spectrum_V2=spectrum_V2, tol=tol)
        opt_val_list.append(opt_val)

        if np.isclose(opt_val, m*T):
            break

    opt_val_list = np.array(opt_val_list)

    print(f'best = {opt_val_list.max()}, mean = {opt_val_list.mean()}, sample variance = {opt_val_list.var()}')

    shape_distance = 1. - opt_val_list.max() / (m*T)

    if shape_distance < 10e-5:
        return 0
    else:
        return shape_distance



def compute_shape_distance(V1: str, V2: str, num_qubits: int) -> dict:
    all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    all_distances = {}

    for pos_tag, qargs in all_positions.items():
        if (V1 in SINGLE_QUBIT_VARIATIONAL_GATES or V1 in TWO_QUBIT_VARIATIONAL_GATES) and \
            (V2 in SINGLE_QUBIT_VARIATIONAL_GATES or V2 in TWO_QUBIT_VARIATIONAL_GATES):

            all_distances[pos_tag] = _shape_distance_with_config(num_qubits=num_qubits, V1=V1, V2=V2,
                                                             qargs1=qargs[0], qargs2=qargs[1])
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
            if not np.isclose(dist, rev_dist):
                print(pairname, rev_pairname, qargs, rev_qargs, dist, rev_dist)

            all_shape_distances[pairname] = min(all_shape_distances[pairname], all_shape_distances[rev_pairname])
    return all_shape_distances


if __name__ == '__main__':

    print(compute_shape_distance('rx', 'rz', num_qubits=3))
    print(compute_shape_distance('rz', 'rx', num_qubits=3))

    import pickle

    filename = './raw_all_shape_distances.pkl'

    try:
        with open(filename, 'rb') as f:
            ALL_SHAPE_DISTANCES = pickle.load(f)
    except:
        with open(filename, 'wb') as f:
            ALL_SHAPE_DISTANCES = {}
            pickle.dump(ALL_SHAPE_DISTANCES, f)


    for q in range(1,5):
        for i,V1 in enumerate(ADMISSIBLE_GATES):
            for j,V2 in enumerate(ADMISSIBLE_GATES):

                if q == 1:
                    if V1 in TWO_QUBIT_DETERMINISTIC_GATES or V1 in TWO_QUBIT_VARIATIONAL_GATES \
                        or V2 in TWO_QUBIT_DETERMINISTIC_GATES or V2 in TWO_QUBIT_VARIATIONAL_GATES:
                            continue ## not enough qubit

                positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=q)
                #if list(positions.keys())[0] not in ALL_SHAPE_DISTANCES:
                if not all(pos in ALL_SHAPE_DISTANCES for pos in list(positions.keys())):
                    try:
                        shape_distance = compute_shape_distance(V1,V2, num_qubits=q)
                    except:
                        continue
                    else:
                        ALL_SHAPE_DISTANCES.update(shape_distance)
                        with open(filename, 'wb') as f:
                            pickle.dump(ALL_SHAPE_DISTANCES, f)

    print(len(ALL_SHAPE_DISTANCES))
    print(ALL_SHAPE_DISTANCES)
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)
        print(loaded_dict == ALL_SHAPE_DISTANCES)

    ALL_SHAPE_DISTANCES = modify_shape_dist_dict(ALL_SHAPE_DISTANCES)
    # with open('all_shape_distances.pkl', 'wb') as f:
    #     pickle.dump(ALL_SHAPE_DISTANCES, f)