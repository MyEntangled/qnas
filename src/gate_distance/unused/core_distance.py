import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

import gate_positioning
from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES, ADMISSIBLE_GATES, DIRECTED_GATES, UNITARY

import pickle

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

def compute_core_distance(V1, V2):
    assert V1 in ADMISSIBLE_GATES and V2 in ADMISSIBLE_GATES, "Input gates are not admissible."

    num_qubits_1 = 1 if V1 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    num_qubits_2 = 1 if V2 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    # is_variational_1 = 1 if V1 in SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_VARIATIONAL_GATES else 0
    # is_variational_2 = 1 if V2 in SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_VARIATIONAL_GATES else 0
    # is_directed_1 = 1 if V1 in DIRECTED_GATES else 0
    # is_directed_2 = 1 if V2 in DIRECTED_GATES else 0

    ALL_DISTANCES = {}

    if num_qubits_1 == 1:
        if num_qubits_2 == 1:
            ## On the same register (same qubit)
            matrix1 = get_unitary_matrix(V=V1,num_qubits=1,qargs=[0])
            matrix2 = get_unitary_matrix(V=V2,num_qubits=1,qargs=[0])
            ALL_DISTANCES['_'.join([V1, V2, 's'])] = _get_core_distance(matrix1, matrix2)

            ## On different registers (different registers)
            matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0])
            matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[1])
            ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

        else:  ## num_qubits_2 == 2
            if V2 in DIRECTED_GATES:
                ## V1 on the first qubit of the register V2 applied on
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 'up'])] = _get_core_distance(matrix1, matrix2)

                ## V1 on the second qubit of the register V2 applied on
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 'lo'])] = _get_core_distance(matrix1, matrix2)

                ## On different registers
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[1,2])
                ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

            else: ## V2 is NON-DIRECTED
                ## Overlap on one qubit
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 'over'])] = _get_core_distance(matrix1, matrix2)

                ## On different registers
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[1,2])
                ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

    else: ## num_qubits_1 == 2
        if num_qubits_2 == 1:
            if V1 in DIRECTED_GATES:
                ## V2 on the first qubit of the register V1 applied on
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0])
                ALL_DISTANCES['_'.join([V1, V2, 'up'])] = _get_core_distance(matrix1, matrix2)

                ## V2 on the second qubit of the register V1 applied on
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[1])
                ALL_DISTANCES['_'.join([V1, V2, 'lo'])] = _get_core_distance(matrix1, matrix2)

                ## On different registers
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[2])
                ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

            else: ## V1 is NON-DIRECTED
                ## Overlap on one qubit
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0])
                ALL_DISTANCES['_'.join([V1, V2, 'over'])] = _get_core_distance(matrix1, matrix2)

                ## On different registers
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[2])
                ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

        else: ## num_qubits_2 == 2:
            if (V1 in DIRECTED_GATES) and (V2 in DIRECTED_GATES):
                ## Same register, aligning
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 'alig'])] = _get_core_distance(matrix1, matrix2)

                ## Same register, anti-align
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[1,0])
                ALL_DISTANCES['_'.join([V1, V2, 'anti'])] = _get_core_distance(matrix1, matrix2)

                ## Overlap on the first qubit of V1 register, aligning
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[1,2])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 'up', 'alig'])] = _get_core_distance(matrix1, matrix2)

                ## Overlap on the first qubit of V1 register, anti-align
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[1,2])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[1,0])
                ALL_DISTANCES['_'.join([V1, V2, 'up', 'anti'])] = _get_core_distance(matrix1, matrix2)

                ## Overlap on the second qubit of V1 register, aligning
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[1,2])
                ALL_DISTANCES['_'.join([V1, V2, 'lo', 'alig'])] = _get_core_distance(matrix1, matrix2)

                ## Overlap on the second qubit of V1 register, anti-align
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[2,1])
                ALL_DISTANCES['_'.join([V1, V2, 'lo', 'anti'])] = _get_core_distance(matrix1, matrix2)

                ## Different register
                matrix1 = get_unitary_matrix(V=V1,num_qubits=4,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=4,qargs=[2,3])
                ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

            elif (V1 in DIRECTED_GATES) and (V2 not in DIRECTED_GATES):
                ## Same register
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 's'])] = _get_core_distance(matrix1, matrix2)

                ## Overlap on the first qubit of V1 register
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[1,2])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 'up'])] = _get_core_distance(matrix1, matrix2)

                ## Overlap on the second qubit of V1 register
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[1,2])
                ALL_DISTANCES['_'.join([V1, V2, 'lo'])] = _get_core_distance(matrix1, matrix2)

                ## Different registers
                matrix1 = get_unitary_matrix(V=V1,num_qubits=4,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=4,qargs=[2,3])
                ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

            elif (V1 not in DIRECTED_GATES) and (V2 in DIRECTED_GATES):
                ## Same register
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 's'])] = _get_core_distance(matrix1, matrix2)

                # Overlap on the first qubit of V2 register
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[1,2])
                ALL_DISTANCES['_'.join([V1, V2, 'up'])] = _get_core_distance(matrix1, matrix2)

                # Overlap on the second qubit of V2 register
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[1,2])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 'lo'])] = _get_core_distance(matrix1, matrix2)

                # Different registers
                matrix1 = get_unitary_matrix(V=V1,num_qubits=4,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=4,qargs=[2,3])
                ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

            else: ## 2 UNDIRECTED GATES
                ## Same register
                matrix1 = get_unitary_matrix(V=V1,num_qubits=2,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=2,qargs=[0,1])
                ALL_DISTANCES['_'.join([V1, V2, 's'])] = _get_core_distance(matrix1, matrix2)

                ## Overlap on 1 qubit
                matrix1 = get_unitary_matrix(V=V1,num_qubits=3,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=3,qargs=[1,2])
                ALL_DISTANCES['_'.join([V1, V2, 'over'])] = _get_core_distance(matrix1, matrix2)

                # Different registers
                matrix1 = get_unitary_matrix(V=V1,num_qubits=4,qargs=[0,1])
                matrix2 = get_unitary_matrix(V=V2,num_qubits=4,qargs=[2,3])
                ALL_DISTANCES['_'.join([V1, V2, 'd'])] = _get_core_distance(matrix1, matrix2)

    return ALL_DISTANCES



if __name__ == '__main__':


    ALL_CORE_DISTANCES = {}
    for i,V1 in enumerate(ADMISSIBLE_GATES):
        for j,V2 in enumerate(ADMISSIBLE_GATES):
            ALL_CORE_DISTANCES.update(compute_core_distance(V1,V2))
    print(len(ALL_CORE_DISTANCES))
    print(ALL_CORE_DISTANCES)

    with open('all_core_distances.pkl', 'wb') as f:
        pickle.dump(ALL_CORE_DISTANCES, f)

    with open('all_core_distances.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    print(ALL_CORE_DISTANCES == loaded_dict)










