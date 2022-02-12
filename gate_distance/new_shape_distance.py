import MUBs
import fubini_distance
import gate_positioning

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scipy.optimize import linear_sum_assignment

import MUBs

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES, ADMISSIBLE_GATES, DIRECTED_GATES, UNITARY
'''
MUBs as anchor states: https://arxiv.org/pdf/quant-ph/0512217.pdf
'''

MAX_DISTANCE = 1

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
            #print(var_V_circ.draw())
            output_states[i, j] = anchor.evolve(var_V_circ).data

    return np.array(output_states)

def minimize_permutation(spectrum_A, spectrum_B, axis, anchor_states):
    #print("Initial cost", np.linalg.norm(spectrum_B - spectrum_A) ** 2 / (spectrum_A.shape[0] * spectrum_A.shape[1]))

    spectrum_A = np.moveaxis(spectrum_A, axis, 0)
    spectrum_B = np.moveaxis(spectrum_B, axis, 0)

    cost_matrix = np.zeros(shape=(spectrum_A.shape[0], spectrum_B.shape[0]))
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix)):
            #cost_matrix[i, j] = np.linalg.norm(spectrum_B[i] - spectrum_A[j]) ** 2  ## Row for B and Column for A
            cost_matrix[i,j] = fubini_distance.minimize_sum_fidelity(X=spectrum_B[i],Y=spectrum_A[j],num_trials=1)
            print(i, j, anchor_states[i], anchor_states[j], cost_matrix[i,j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print(row_ind, col_ind)
    print(cost_matrix[row_ind, col_ind].sum())
    permuted_spectrum_A = spectrum_A[col_ind]

    spectrum_A = np.moveaxis(spectrum_A, 0, axis)
    permuted_spectrum_A = np.moveaxis(permuted_spectrum_A, 0, axis)
    spectrum_B = np.moveaxis(spectrum_B, 0, axis)

    # print("Minimized cost",
    #       np.linalg.norm(spectrum_B - permuted_spectrum_A) ** 2 / (spectrum_A.shape[0] * spectrum_A.shape[1]))
    min_cost = fubini_distance.minimize_sum_fidelity(X=permuted_spectrum_A.reshape(-1, spectrum_A.shape[-1]),
                                                                  Y=spectrum_B.reshape(-1, spectrum_B.shape[-1]), num_trials=1)
    print('min', min_cost)
    return min_cost

def check_if_unitary_exists(oldvecs, newvecs):
    min_fid = fubini_distance.minimize_sum_fidelity(X=oldvecs,
                                          Y=newvecs, num_trials=1)
    print(min_fid)
    if np.isclose(min_fid,0):
        return True
    else:
        return False

def reorder_anchors(spectrum_A, spectrum_B, anchor_states):
    assert spectrum_A.shape == spectrum_B.shape
    num_anchors = spectrum_A.shape[0]
    num_samples = spectrum_A.shape[1]

    circular_fidel_A = np.zeros(shape=(num_anchors, num_samples), dtype=np.double)
    circular_fidel_B = np.zeros(shape=(num_anchors, num_samples), dtype=np.double)
    cost_matrix = np.zeros(shape=(num_anchors, num_anchors),dtype=np.double)

    for k in range(num_anchors):
        for i in range(num_samples):
            circular_fidel_A[k,i] = np.abs(spectrum_A[k, i % num_samples].conj() @ spectrum_A[k, (i+1) % num_samples])
            circular_fidel_B[k,i] = np.abs(spectrum_B[k, i % num_samples].conj() @ spectrum_B[k, (i+1) % num_samples])

    for i in range(num_anchors):
        for j in range(num_anchors):
            cost_matrix[i,j] = np.sum(np.abs(circular_fidel_A[i] - circular_fidel_B[j]))

    # row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # col_ind = np.array([2,3,1,0,4,5])

    eps = 1e-10
    modified_cost_matrix = cost_matrix
    #modified_cost_matrix = cost_matrix.copy()
    all_sols = []
    m = 0
    num_steps = 100
    for m in range(num_steps):
        row_ind, col_ind = linear_sum_assignment(modified_cost_matrix)
        all_sols.append(col_ind)
        print('sum cost matrix: s', cost_matrix[row_ind, col_ind].sum())
        if check_if_unitary_exists(anchor_states, anchor_states[col_ind]):
           break
        else:
            #modified_cost_matrix = cost_matrix.copy()
            for i in range(num_anchors):
                modified_cost_matrix[row_ind[i],col_ind[i]] += (1. - m/num_steps)*eps

    distinct_sols = [np.array(x) for x in set(tuple(x) for x in all_sols)]
    is_valid = [check_if_unitary_exists(anchor_states, anchor_states[x]) for x in distinct_sols]
    print(list(zip(distinct_sols, is_valid)))
    #print(row_ind, col_ind)
    #print(cost_matrix[row_ind, col_ind].sum())
    return col_ind


def _get_shape_distance(num_qubits, V1, V2, qargs1, qargs2, num_samples=4):
    '''
    Return the shape distance between two quantum gates
    :param V1:
    :param V2:
    :return:
    '''

    assert V1 in ADMISSIBLE_GATES and V2 in ADMISSIBLE_GATES, "Input gates are not admissible."

    anchor_states = MUBs.get_anchor_states(num_qubits)
    #print(anchor_states)
    lo_bound = -np.pi
    up_bound = np.pi
    spectrum_V1 = get_state_spectrum(num_qubits, V1, qargs1, np.linspace(lo_bound, up_bound, num_samples, endpoint=False), anchor_states)
    spectrum_V2 = get_state_spectrum(num_qubits, V2, qargs2, np.linspace(lo_bound, up_bound, num_samples, endpoint=False), anchor_states)
    print(spectrum_V1.shape, spectrum_V2.shape)

    dist = fubini_distance.minimize_sum_fidelity(X=spectrum_V1.reshape(-1, 2 ** num_qubits),
                                                 Y=spectrum_V2.reshape(-1, 2 ** num_qubits), num_trials=1)
    print('same anchor order: ', dist)

    new_anchor_order = reorder_anchors(spectrum_A=spectrum_V1, spectrum_B=spectrum_V2, anchor_states=anchor_states)

    #print(check_if_unitary_exists(anchor_states, anchor_states[new_anchor_order]))

    #new_anchor_order = np.array([2,3,1,0,4,5]) # +z(0) to +x(2), -z(1) to -x(3), +x(2) to -z(1), -x(3) to +z(0), +y(5) to +y(4), -y(5) to -y(5).

    #print(check_if_unitary_exists(anchor_states, anchor_states[new_anchor_order]))

    new_dist = fubini_distance.minimize_sum_fidelity(X=spectrum_V1.reshape(-1, 2 ** num_qubits),
                                                     Y=spectrum_V2[new_anchor_order].reshape(-1, 2 ** num_qubits), num_trials=1)
    #print(spectrum_V2)
    #print(spectrum_V2[new_anchor_order])
    print('reordered anchors: ', new_dist)
    return (dist, new_dist)


    #return minimize_permutation(spectrum_A=spectrum_V1, spectrum_B=spectrum_V2, axis=0, anchor_states=anchor_states)






def compute_shape_distance(V1:str, V2:str, num_qubits:int) -> dict:
    all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    all_distances = {}

    for pos_tag, qargs in all_positions.items():
        all_distances[pos_tag] = _get_shape_distance(num_qubits=num_qubits,V1=V1,V2=V2,
                                                     qargs1=qargs[0],qargs2=qargs[1])

    return all_distances



if __name__ == '__main__':
    print(compute_shape_distance('rz', 'rx', num_qubits=1))



