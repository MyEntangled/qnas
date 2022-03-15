import MUBs
import fubini_distance
import gate_positioning

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scipy.optimize import linear_sum_assignment

import MUBs

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES, ADMISSIBLE_GATES, DIRECTED_GATES, UNITARY

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
            # print(var_V_circ.draw())
            output_states[i, j] = anchor.evolve(var_V_circ).data

    return np.array(output_states)


def maximize_fidelity_permutation(spectrum_A, spectrum_B, circular_fidel_A, circular_fidel_B):
    assert spectrum_A.shape == spectrum_B.shape
    num_states = spectrum_A.shape[1]

    cost_matrix = np.zeros(shape=(spectrum_A.shape[0], spectrum_B.shape[0]))
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix)):
            # cost_matrix[i, j] = np.linalg.norm(spectrum_B[i] - spectrum_A[j]) ** 2  ## Row for B and Column for A
            # cost_matrix[i,j] = fubini_distance.minimize_sum_fidelity(X=spectrum_A[i],Y=spectrum_B[j],num_trials=1)
            cost_matrix[i, j] = sum(np.abs(np.sum(spectrum_B[j].conj() * spectrum_A[i],
                                                  axis=1)))  # + 2*np.sum(np.abs(circular_fidel_A[i] - circular_fidel_B[j]))

    cost_matrix.ravel()[::cost_matrix.shape[1]+1] -= 10e-6

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # print(col_ind)
    #print('new U, old P: ', cost_matrix[row_ind, row_ind].sum())
    #print('new U, new P: ', cost_matrix[row_ind, col_ind].sum())
    assert cost_matrix[row_ind, col_ind].sum() >= cost_matrix[row_ind, row_ind].sum() - 1e-6

    return col_ind, cost_matrix[row_ind, col_ind].sum()


def check_if_unitary_exists(oldvecs, newvecs):
    min_fid = fubini_distance.max_sum_sqrt_fidelity(X=oldvecs,
                                                    Y=newvecs, num_trials=1)
    # print(min_fid)
    if np.isclose(min_fid, 0):
        return True
    else:
        return False


def reorder_anchors(spectrum_A, spectrum_B, anchor_states):
    assert spectrum_A.shape == spectrum_B.shape
    num_anchors = spectrum_A.shape[0]
    num_samples = spectrum_A.shape[1]

    circular_fidel_A = np.zeros(shape=(num_anchors, num_samples), dtype=np.double)
    circular_fidel_B = np.zeros(shape=(num_anchors, num_samples), dtype=np.double)
    cost_matrix = np.zeros(shape=(num_anchors, num_anchors), dtype=np.double)

    for k in range(num_anchors):
        for i in range(num_samples):
            circular_fidel_A[k, i] = np.abs(
                spectrum_A[k, i % num_samples].conj() @ spectrum_A[k, (i + 1) % num_samples])
            circular_fidel_B[k, i] = np.abs(
                spectrum_B[k, i % num_samples].conj() @ spectrum_B[k, (i + 1) % num_samples])

    for i in range(num_anchors):
        for j in range(num_anchors):
            cost_matrix[i, j] = np.sum(np.abs(circular_fidel_A[i] - circular_fidel_B[j]))

    # row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # col_ind = np.array([2,3,1,0,4,5])

    eps = 1e-7
    modified_cost_matrix = cost_matrix
    # modified_cost_matrix = cost_matrix.copy()
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
            # modified_cost_matrix = cost_matrix.copy()
            for i in range(num_anchors):
                modified_cost_matrix[row_ind[i], col_ind[i]] += (1. - m / num_steps) * eps

    distinct_sols = [np.array(x) for x in set(tuple(x) for x in all_sols)]
    is_valid = [check_if_unitary_exists(anchor_states, anchor_states[x]) for x in distinct_sols]
    print(list(zip(distinct_sols, is_valid)))
    print(list(zip(distinct_sols, is_valid)))

    # print(row_ind, col_ind)
    # print(cost_matrix[row_ind, col_ind].sum())
    return col_ind


def _shape_distance_with_config(num_qubits, V1, V2, qargs1, qargs2, num_samples=4):
    '''
    Return the shape distance between two quantum gates
    :param V1:
    :param V2:
    :return:
    '''

    assert V1 in ADMISSIBLE_GATES and V2 in ADMISSIBLE_GATES, "Input gates are not admissible."

    anchor_states = MUBs.get_anchor_states(num_qubits)
    num_anchors = len(anchor_states)
    # print(anchor_states)
    lo_bound = -np.pi
    up_bound = np.pi
    spectrum_V1 = get_state_spectrum(num_qubits, V1, qargs1,
                                     np.linspace(lo_bound, up_bound, num_samples, endpoint=False), anchor_states)
    spectrum_V2 = get_state_spectrum(num_qubits, V2, qargs2,
                                     np.linspace(lo_bound, up_bound, num_samples, endpoint=False), anchor_states)
    print(spectrum_V1.shape, spectrum_V2.shape)

    transformed_spectrum_V1 = spectrum_V1.copy()
    permuted_spectrum_V2 = spectrum_V2.copy()

    U_pred_all = np.eye(2 ** num_qubits, dtype=np.complex128)
    reorder_all = np.array(range(num_anchors))

    U_optimal = U_pred_all.copy()
    order_optimal = reorder_all.copy()
    max_fid = -np.inf

    for i in range(30):
        print(f'iteration {i}')
        oU_oP = np.sum(np.abs(np.sum(permuted_spectrum_V2.conj() * transformed_spectrum_V1, axis=2)))
        print('old U, old P: ', oU_oP)
        #print(np.isnan(transformed_spectrum_V1).any(), np.isnan(permuted_spectrum_V2).any())

        U_pred, transformed_fid = fubini_distance.max_sum_sqrt_fidelity(
            X=transformed_spectrum_V1.reshape(-1, 2 ** num_qubits),
            Y=permuted_spectrum_V2.reshape(-1, 2 ** num_qubits),
            num_trials=1, get_unitary=True)
        transformed_spectrum_V1 = transformed_spectrum_V1 @ U_pred
        U_pred_all = U_pred_all @ U_pred
        nU_oP = np.sum(np.abs(np.sum(permuted_spectrum_V2.conj() * transformed_spectrum_V1, axis=2)))
        print('new U, old P: ', nU_oP)
        #print( sum([ np.abs(np.sum(transformed_spectrum_V1[i].conj() * permuted_spectrum_V2[i])) for i in range(num_anchors)]) )
        reorder, permuted_fid = maximize_fidelity_permutation(spectrum_A=transformed_spectrum_V1,
                                                              spectrum_B=permuted_spectrum_V2,
                                                              circular_fidel_A=None, circular_fidel_B=None)
        print(reorder)
        permuted_spectrum_V2 = permuted_spectrum_V2[reorder]
        reorder_all = reorder[reorder_all]
        nU_nP = np.sum(np.abs(np.sum(permuted_spectrum_V2.conj() * transformed_spectrum_V1, axis=2)))
        print('new U, new P: ', nU_nP)

        #assert oU_oP <= nU_oP + 1e-6, f"oU_oP({oU_oP}) should be not greater than nU_oP({nU_oP})"
        #assert nU_oP <= nU_nP + 1e-6, f"nU_oP({nU_oP}) should be not greater than nU_nP({nU_nP})"

        if permuted_fid > max_fid:
            max_fid = permuted_fid
            U_optimal = U_pred_all.copy()
            order_optimal = reorder_all.copy()

    # best_fid = fubini_distance.max_sum_sqrt_fidelity(X=(spectrum_V1 @ U_optimal).reshape(-1, 2 ** num_qubits),
    #                                                  Y=spectrum_V2[order_optimal].reshape(-1, 2 ** num_qubits), num_trials=1)

    # print('reordered anchors: ', order_optimal, best_fid)
    return max_fid



    # new_anchor_order = np.array([2,3,1,0,4,5]) # +z(0) to +x(2), -z(1) to -x(3), +x(2) to -z(1), -x(3) to +z(0), +y(5) to +y(4), -y(5) to -y(5).
    # ORDERS THAT WORK: [2 3 5 4 1 0], [2 3 4 5 0 1], [2 3 0 1 5 4]
    # new_anchor_order = np.array([3, 2, 5, 4, 0, 1]) ## FAIL
    # new_anchor_order = np.array([3, 2, 0, 1, 5, 4]) ## FAIL

    # transformed_spectrum_V1 = (spectrum_V1.reshape(-1, 2 ** num_qubits) @ U_pred).reshape(spectrum_V1.shape)
    # new_fid = fubini_distance.max_sum_sqrt_fidelity(X=spectrum_V1.reshape(-1, 2 ** num_qubits),
    #                                                  Y=spectrum_V2[new_anchor_order].reshape(-1, 2 ** num_qubits), num_trials=1)
    #
    # print('reordered anchors: ', new_anchor_order, new_fid)
    # return (fid, new_fid)


def compute_shape_distance(V1: str, V2: str, num_qubits: int) -> dict:
    all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    all_distances = {}

    for pos_tag, qargs in all_positions.items():
        all_distances[pos_tag] = _shape_distance_with_config(num_qubits=num_qubits, V1=V1, V2=V2,
                                                             qargs1=qargs[0], qargs2=qargs[1])

    return all_distances


if __name__ == '__main__':
    print(compute_shape_distance('ry', 'rx', num_qubits=3))
