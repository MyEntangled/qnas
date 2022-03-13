import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

import gate_positioning
from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES, ADMISSIBLE_GATES, DIRECTED_GATES, UNITARY

'''
https://arxiv.org/pdf/2001.07202.pdf
'''

def get_normalized_Hamiltonian(U):
    '''
    Compute H, where U = exp(iHt), where H is Hermitian and has trace norm of 1.
    :param U:
    :return:
    '''

    assert len(U.shape) == 2 and U.shape[0] == U.shape[1], "The input matrix is not square."
    assert np.linalg.norm(U @ U.conj().T - np.eye(U.shape[0])) < 10e-6, "The input matrix is not unitary."

    iHt = scipy.linalg.logm(U)
    t = np.linalg.norm(iHt,'nuc')

    if t != 0:
        H = -1j * iHt / t
        return H,t
    else:
        return iHt,0

def get_unitary_matrix(V, num_qubits, qargs):
    qc = QuantumCircuit(num_qubits)

    if V in SINGLE_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_DETERMINISTIC_GATES:
        qc.append(UNITARY[V](), qargs)
    else:
        qc.append(UNITARY[V](1), qargs) # 1 is just a placeholder parameter

    op = qi.Operator(qc)
    matrix = op.data
    return matrix

def _get_core_distance(matrix1, matrix2, normtype='nuc'):
    H1, t1 = get_normalized_Hamiltonian(matrix1)
    H2, t2 = get_normalized_Hamiltonian(matrix2)
    return np.linalg.norm(H1 - H2, ord=normtype)

def _core_distance_with_config(num_qubits, V1, V2, qargs1, qargs2):
    """

    :param num_qubits:
    :param V1:
    :param V2:
    :param qargs1:
    :param qargs2:
    :return:
    """
    assert V1 in ADMISSIBLE_GATES and V2 in ADMISSIBLE_GATES, "Input gates are not admissible."

    matrix1 = get_unitary_matrix(V=V1, num_qubits=num_qubits, qargs=qargs1)
    matrix2 = get_unitary_matrix(V=V2, num_qubits=num_qubits, qargs=qargs2)
    return _get_core_distance(matrix1, matrix2)


def compute_core_distance(V1: str, V2: str, num_qubits: int) -> dict:
    all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    all_distances = {}

    for pos_tag, qargs in all_positions.items():
        all_distances[pos_tag] = 0.5 * _core_distance_with_config(num_qubits=num_qubits, V1=V1, V2=V2,
                                                             qargs1=qargs[0], qargs2=qargs[1])

    print(all_distances)
    return all_distances


if __name__ == '__main__':

    import pickle

    ALL_CORE_DISTANCES = {}
    for q in range(1,5):
        for i,V1 in enumerate(ADMISSIBLE_GATES):
            for j,V2 in enumerate(ADMISSIBLE_GATES):
                try:
                    core_distance = compute_core_distance(V1,V2, num_qubits=q)
                except:
                    continue
                else:
                    ALL_CORE_DISTANCES.update(core_distance)

    print(len(ALL_CORE_DISTANCES))
    print(ALL_CORE_DISTANCES)

    with open('all_core_distances.pkl', 'wb') as f:
        pickle.dump(ALL_CORE_DISTANCES, f)

    with open('all_core_distances.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    print(ALL_CORE_DISTANCES == loaded_dict)











