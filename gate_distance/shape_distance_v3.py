import MUBs
import fubini_distance
import time
import gate_positioning

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes

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
    print(np.abs(U))
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

    ### Modify code here
    V = np.zeros((d,d))
    V, res = orthogonal_procrustes(X,Y)
    ###

    return V, res**2

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

    ### This construction uses norms over permutations, minimization
    cost_matrix = np.zeros(shape=(spectrumA.shape[0], spectrumB.shape[0]))
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix)):
            cost_matrix[i, j] = np.linalg.norm(spectrumA[i] - spectrumB[j])**2
    row_ind, B_perm = linear_sum_assignment(cost_matrix)

    ### This uses sum of fidelity, maximization. The optimal permutation is the same as above
    # cost_matrix = np.zeros(shape=(spectrumA.shape[0], spectrumB.shape[0]))
    # for i in range(len(cost_matrix)):
    #     for j in range(len(cost_matrix)):
    #         cost_matrix[i, j] = sum(np.abs(np.sum(spectrumB[j].conj() * spectrumA[i], axis=1)))
    # row_ind, B_perm = linear_sum_assignment(cost_matrix, maximize=True)

    # Equivalent, but faster
    # cost_matrix = np.sum(np.abs(np.matmul(spectrumB.conj().transpose(1,0,2), spectrumA.transpose(1,2,0)).transpose(1,2,0)), axis=2).T
    # row_ind, B_perm = linear_sum_assignment(cost_matrix, maximize=True)

    return B_perm, cost_matrix[row_ind, B_perm].sum()

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

    output_states = np.zeros(shape=(len(anchor_states), len(thetas), 2 ** num_qubits), dtype=np.complex_)

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

def _shape_distance_with_config(num_qubits, V1, V2, qargs1, qargs2, num_samples=4):
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


    m, T, d = spectrum_V1.shape
    tol = 10e-6
    k = 0

    newX = spectrum_V1.copy().reshape(m*T,d)
    newY = spectrum_V2.copy().reshape(m*T,d)

    phasesX, val = optimize_phases(X=newX, Y=newY)
    newX = newX * phasesX[:, None]

    #val = np.linalg.norm(newX - newY)**2
    val_prev = val
    print('init phase: ', val)
    while val_prev > tol:
        k += 1
        print(k)
        neworder,val = optimize_permutation(spectrumA=newX.reshape(m, T, d), spectrumB=newY.reshape(m, T, d))
        print('perm: ', val)

        phasesX,val = optimize_phases(X=newX, Y=newY.reshape(m,T,d)[neworder].reshape(m*T,d))
        newX = newX * phasesX[:, None]
        print('phase after perm: ', val)

        V,val = optimize_unitary(X=newX, Y=newY)
        print('unitary: ', val)
        phasesX,val = optimize_phases(X=newX @ V, Y=newY)
        newX = newX * phasesX[:, None]
        print('phase after unitary: ', val)

        if abs(val_prev - val) < tol:
            opt_sum_fid = (2*spectrum_V1.shape[0]*spectrum_V1.shape[1] - val)/2.
            print(val_prev, val)
            return opt_sum_fid
        else:
            val_prep = val



def compute_shape_distance(V1: str, V2: str, num_qubits: int) -> dict:
    all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    all_distances = {}

    for pos_tag, qargs in all_positions.items():
        all_distances[pos_tag] = _shape_distance_with_config(num_qubits=num_qubits, V1=V1, V2=V2,
                                                             qargs1=qargs[0], qargs2=qargs[1])

    return all_distances

if __name__ == '__main__':
    print(compute_shape_distance('ry', 'rx', num_qubits=1))