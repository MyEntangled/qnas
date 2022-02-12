import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
import clifford_group, state_utility

SINGLE_QUBIT_DETERMINISTIC_GATES = ['i', 'h', 's', 'x', 'y', 'z', 't']
SINGLE_QUBIT_VARIATIONAL_GATES = ['rx', 'ry', 'rz']
TWO_QUBIT_DETERMINISTIC_GATES = ['cx', 'cy', 'cz']
TWO_QUBIT_VARIATIONAL_GATES = ['crx', 'cry', 'crz', 'rxx', 'ryy', 'rzz']
ADMISSIBLE_GATES = SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_VARIATIONAL_GATES

# def apply_clifford(qc, clifford):
#     cliff_circ = clifford.to_circuit()
#     qc = qc.compose(cliff_circ)
#     # print(qc.draw())
#     return qc

def get_state_spectrum(num_qubits, V, applied_qubits, thetas, anchor_states):
    '''
    Get V(theta)C|0⟩ for various thetas.
    :param num_qubits:
    :param V:
    :param applied_qubits:
    :param thetas:
    :param anchor_states:
    :return:
    '''
    assert V in ADMISSIBLE_GATES, "The input gate is not valid!"

    output_states = np.zeros(shape=(len(anchor_states), len(thetas), 2 ** num_qubits), dtype=np.complex_)

    for i, anchor_state in enumerate(anchor_states):

        anchor = Statevector(anchor_state)  # initialize an anchor state

        for j, theta in enumerate(thetas):
            var_V_circ = QuantumCircuit(num_qubits)
            if V in SINGLE_QUBIT_DETERMINISTIC_GATES:  # one-qubit deterministic
                args = (*applied_qubits,)
            elif V in SINGLE_QUBIT_VARIATIONAL_GATES:  # one-qubit variational
                args = (theta, *applied_qubits)
            elif V in TWO_QUBIT_DETERMINISTIC_GATES:  # two-qubit deterministic
                args = (*applied_qubits,)
            elif V in TWO_QUBIT_VARIATIONAL_GATES:  # two-qubit variational
                args = (theta, *applied_qubits)

            getattr(var_V_circ, V)(*args)
            output_states[i, j] = anchor.evolve(var_V_circ).data

    return np.array(output_states)


def minimize_permutation(spectrum_A, spectrum_B, axis):
    print("Initial cost", np.linalg.norm(spectrum_B - spectrum_A) ** 2 / (spectrum_A.shape[0] * spectrum_A.shape[1]))

    spectrum_A = np.moveaxis(spectrum_A, axis, 0)
    spectrum_B = np.moveaxis(spectrum_B, axis, 0)

    cost_matrix = np.zeros(shape=(spectrum_A.shape[0], spectrum_B.shape[0]))
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix)):
            cost_matrix[i, j] = np.linalg.norm(spectrum_B[i] - spectrum_A[j]) ** 2  ## Row for B and Column for A

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    permuted_spectrum_A = spectrum_A[col_ind]

    spectrum_A = np.moveaxis(spectrum_A, 0, axis)
    permuted_spectrum_A = np.moveaxis(permuted_spectrum_A, 0, axis)
    spectrum_B = np.moveaxis(spectrum_B, 0, axis)

    # print("Permutation difference", np.linalg.norm(spectrum_A - permuted_spectrum_A))
    print("Minimized cost",
          np.linalg.norm(spectrum_B - permuted_spectrum_A) ** 2 / (spectrum_A.shape[0] * spectrum_A.shape[1]))

    return permuted_spectrum_A, spectrum_B


def minimize_unitary(spectrum_A, spectrum_B):
    """
    Min |A x Omega - B|^2, where Omega is a unitary matrix
    """
    A = spectrum_A.reshape(-1, spectrum_A.shape[2])
    B = spectrum_B.reshape(-1, spectrum_B.shape[2])
    print("Initial cost", np.linalg.norm(B - A) ** 2 / A.shape[0])

    M = A.conj().T @ B
    U, Sigma, V_d = np.linalg.svd(M)
    rank = sum(Sigma > 10e-6)  # =2

    #     W = np.eye(U.shape[1], dtype=np.complex_)
    #     if U.shape[1] - rank > 1:
    #         W[rank:, rank:] = random_unitary(U.shape[1] - rank).spectrum
    #     Omega = U @ W @ V_d # == U @ V_d
    Omega = U @ V_d

    transformed_A = A @ Omega
    transformed_spectrum_A = transformed_A.reshape(spectrum_A.shape[0], spectrum_A.shape[1], -1)

    # print("Unitary transform difference", np.linalg.norm(spectrum_A - transformed_spectrum_A))
    print("Minimized cost", np.linalg.norm(spectrum_B - transformed_spectrum_A) ** 2 / A.shape[0])

    return transformed_spectrum_A, spectrum_B


def minimize_unitary_lib(spectrum_A, spectrum_B):
    A = spectrum_A.reshape(-1, spectrum_A.shape[2])
    B = spectrum_B.reshape(-1, spectrum_B.shape[2])
    print("Initial cost", np.linalg.norm(B - A) / A.shape[0])

    Omega, scale = orthogonal_procrustes(A, B)

    transformed_A = A @ Omega
    print("Minimized cost before reshaping", np.linalg.norm(B - transformed_A))

    transformed_spectrum_A = transformed_A.reshape(spectrum_A.shape[0], spectrum_A.shape[1], -1)

    # print("Unitary transform difference", np.linalg.norm(spectrum_A - transformed_spectrum_A))
    print("Minimized cost", np.linalg.norm(spectrum_B - transformed_spectrum_A) / A.shape[0])
    print("\n")
    return transformed_spectrum_A, spectrum_B

def compute_shape_distance(V1, V2, num_samples, num_iters):
    '''
    Return the shape distance between two quantum gates
    :param V1:
    :param V2:
    :return:
    '''

    assert V1 in ADMISSIBLE_GATES and V2 in ADMISSIBLE_GATES, "Input gates are not admissible."
    num_qubits_V1 = 1 if V1 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    num_qubits_V2 = 1 if V2 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    is_variational_V1 = 1 if V1 in SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_VARIATIONAL_GATES else 0
    is_variational_V2 = 1 if V2 in SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_VARIATIONAL_GATES else 0

    if (num_qubits_V1 != num_qubits_V2) or (is_variational_V1 != is_variational_V2):
        return 10e9
    if is_variational_V1 == 0:
        return 0

    anchor_states = clifford_group.get_anchor_states(num_qubits_V1, False)
    spectrum_V1 = state_utility.reformat_statedata(get_state_spectrum(num_qubits_V1, V1, range(num_qubits_V1), np.linspace(-eps, eps, num_samples), anchor_states))
    spectrum_V2 = state_utility.reformat_statedata(get_state_spectrum(num_qubits_V2, V2, range(num_qubits_V2), np.linspace(-eps, eps, num_samples), anchor_states))

    data_A, data_B = spectrum_V1, spectrum_V2
    print(data_A.shape, data_B.shape)

    min_distance = 10e9

    for i in range(num_iters):
        #print(np.linalg.norm(data_A - data_B) ** 2 / (1 * data_A.shape[0]))
        #print(np.linalg.norm(data_A - data_B) ** 2 / (1 * data_A.shape[1]))
        data_A, data_B = minimize_permutation(data_A, data_B, axis=0)

        #print(np.linalg.norm(data_A - data_B) ** 2 / (1 * data_A.shape[0]))
        #print(np.linalg.norm(data_A - data_B) ** 2 / (1 * data_A.shape[1]))
        data_A, data_B = minimize_permutation(data_A, data_B, axis=1)

        #print(np.linalg.norm(data_A - data_B) ** 2 / (1 * data_A.shape[0]))
        #print(np.linalg.norm(data_A - data_B) ** 2 / (1 * data_A.shape[1]))
        data_A, data_B = minimize_unitary(data_A, data_B)

        distance = np.linalg.norm(data_A - data_B) ** 2 / (data_A.shape[0] * data_A.shape[1])
        if distance < 10e-6:
            return 0
        if distance < min_distance:
            min_distance = distance

        data_A = state_utility.reformat_statedata(data_A)


    return min(min_distance, np.linalg.norm(data_A - data_B) ** 2 / (data_A.shape[0] * data_A.shape[1]))




if __name__ == '__main__':
    eps = np.pi
    num_samples = 100
    num_iters = 100

    # print("ONE QUBIT-GATE SPECTRUM")
    # anchor_states = clifford_group.get_anchor_states(1, False)
    # spectrum = get_state_spectrum(1, 'rx', [0], np.linspace(-eps, eps, num_samples), anchor_states)
    # spectrum = state_utility.reformat_statedata(spectrum)
    # print(spectrum.shape)
    # print(spectrum)
    #
    # print("TWO QUBIT-GATE SPECTRUM")
    # anchor_states = clifford_group.get_anchor_states(2, False)
    # spectrum = get_state_spectrum(2, 'rxx', [0,1], np.linspace(-eps, eps, num_samples), anchor_states)
    # spectrum = state_utility.reformat_statedata(spectrum)
    # print(spectrum.shape)
    # print(spectrum)

    #print(compute_shape_distance('crx', 'rxx', num_samples))

    gate_distance = np.zeros((len(ADMISSIBLE_GATES), len(ADMISSIBLE_GATES)))
    for i,V1 in enumerate(ADMISSIBLE_GATES):
        for j,V2 in enumerate(ADMISSIBLE_GATES):
            if j>=i:
                gate_distance[i,j] = compute_shape_distance(V1,V2,num_samples,num_iters)
            else:
                gate_distance[i,j] = gate_distance[j,i]


    print(ADMISSIBLE_GATES)
    print(gate_distance)
    np.savetxt("gate_shape_distance.csv", gate_distance, delimiter=",")