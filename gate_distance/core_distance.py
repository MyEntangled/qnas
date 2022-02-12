import numpy as np
import scipy.linalg
from qiskit.circuit.library import IGate,HGate,SGate,XGate,YGate,ZGate,TGate
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.circuit.library import CXGate, CYGate, CZGate
from qiskit.circuit.library import CRXGate, CRYGate, CRZGate, RXXGate, RYYGate, RZZGate

SINGLE_QUBIT_DETERMINISTIC_GATES = ['i', 'h', 's', 'x', 'y', 'z', 't']
SINGLE_QUBIT_VARIATIONAL_GATES = ['rx', 'ry', 'rz']
TWO_QUBIT_DETERMINISTIC_GATES = ['cx', 'cy', 'cz']
TWO_QUBIT_VARIATIONAL_GATES = ['crx', 'cry', 'crz', 'rxx', 'ryy', 'rzz']
ADMISSIBLE_GATES = SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_VARIATIONAL_GATES
GATE_DICT = {'i':IGate, 'h':HGate, 's':SGate, 'x':XGate, 'y':YGate, 'z':ZGate, 't':TGate,
             'rx':RXGate, 'ry':RYGate, 'rz':RZGate,
             'cx':CXGate, 'cy':CYGate, 'cz':CZGate,
             'crx':CRXGate, 'cry':CRYGate, 'crz':CRZGate, 'rxx':RXXGate, 'ryy':RYYGate, 'rzz':RZZGate}

def operator_distance(A,B, norm_type):
    '''
    Compute distance between two linear operators A and B induced by a specified norm.
    :param A:
    :param B:
    :param norm_type:
    :return:
    '''
    assert norm_type in ['fro', 'nuc'], "Distance should be induced from 'fro' or 'nuc' norm."
    d = np.linalg.norm(A-B, ord = norm_type)
    return d

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

def compute_core_distance(V1, V2):

    assert V1 in ADMISSIBLE_GATES and V2 in ADMISSIBLE_GATES, "Input gates are not admissible."
    num_qubits_V1 = 1 if V1 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    num_qubits_V2 = 1 if V2 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    is_variational_V1 = 1 if V1 in SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_VARIATIONAL_GATES else 0
    is_variational_V2 = 1 if V2 in SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_VARIATIONAL_GATES else 0

    if num_qubits_V1 == num_qubits_V2:
        if is_variational_V1:
            H_V1, t_V1 = get_normalized_Hamiltonian(GATE_DICT[V1](1).to_matrix()) # 1 is just a placeholder parameter
        else:
            H_V1, t_V1 = get_normalized_Hamiltonian(GATE_DICT[V1]().to_matrix())

        if is_variational_V2:
            print(GATE_DICT[V2])
            H_V2, t_V2 = get_normalized_Hamiltonian(GATE_DICT[V2](1).to_matrix())
        else:
            H_V2, t_V2 = get_normalized_Hamiltonian(GATE_DICT[V2]().to_matrix())

    else: # V1 and V2 have different dimension
        if num_qubits_V1 == 1: # V1 for 1 qubit, V2 for 2 qubits
            if is_variational_V1:
                H_V1, t_V1 = get_normalized_Hamiltonian(
                    np.kron(np.eye(2, dtype=np.complex_), GATE_DICT[V1](1).to_matrix()))
            else:
                H_V1, t_V1 = get_normalized_Hamiltonian(
                    np.kron(np.eye(2, dtype=np.complex_), GATE_DICT[V1]().to_matrix()))

            if is_variational_V2:
                H_V2, t_V2 = get_normalized_Hamiltonian(GATE_DICT[V2](1).to_matrix())
            else:
                H_V2, t_V2 = get_normalized_Hamiltonian(GATE_DICT[V2]().to_matrix())

        else: # V1 for 2 qubits, V2 for 1 qubit
            if is_variational_V1:
                H_V1, t_V1 = get_normalized_Hamiltonian(GATE_DICT[V1](1).to_matrix())
            else:
                H_V1, t_V1 = get_normalized_Hamiltonian(GATE_DICT[V1]().to_matrix())

            if is_variational_V2:
                H_V2, t_V2 = get_normalized_Hamiltonian(
                    np.kron(np.eye(2, dtype=np.complex_), GATE_DICT[V2](1).to_matrix()))
            else:
                H_V2, t_V2 = get_normalized_Hamiltonian(
                    np.kron(np.eye(2, dtype=np.complex_), GATE_DICT[V2]().to_matrix()))

    return operator_distance(H_V1,H_V2,'nuc')


if __name__ == '__main__':
    # Z = np.array([[1,0],[0,-1]], dtype=np.complex_)
    # H_Z, t = get_normalized_Hamiltonian(Z)
    # rec_Z = scipy.linalg.expm(1j * H_Z * t)
    # print(H_Z, np.linalg.norm(H_Z, ord='fro'), np.linalg.norm(H_Z, ord='nuc'))
    # print(rec_Z)
    #
    # CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    # H_CNOT, t = get_normalized_Hamiltonian(CNOT)
    # rec_CNOT = scipy.linalg.expm(1j * H_CNOT * t)
    # print(H_CNOT, np.linalg.norm(H_CNOT, ord='fro'), np.linalg.norm(H_CNOT, ord='nuc'))
    # print(rec_CNOT)
    #
    # RX_pi2 = qiskit.circuit.library.RXGate(np.pi / 2).to_matrix()
    # RX_pi3 = qiskit.circuit.library.RXGate(np.pi / 3).to_matrix()
    # H_pi2 = get_normalized_Hamiltonian(RX_pi2)
    # H_pi3 = get_normalized_Hamiltonian(RX_pi3)
    # print(H_pi2)
    # print(H_pi3)
    #
    # X = qiskit.circuit.library.XGate().to_matrix()
    # Y = qiskit.circuit.library.YGate().to_matrix()
    # Z = qiskit.circuit.library.ZGate().to_matrix()
    # H = qiskit.circuit.library.HGate().to_matrix()
    # H_X = get_normalized_Hamiltonian(X)[0]
    # H_Y = get_normalized_Hamiltonian(Y)[0]
    # H_Z = get_normalized_Hamiltonian(Z)[0]
    # H_H = get_normalized_Hamiltonian(H)[0]
    # print(operator_distance(H_X, H_Y, 'nuc'), operator_distance(H_X, H_Y, 'fro'))
    # print(operator_distance(H_X, H_Z, 'nuc'), operator_distance(H_X, H_Z, 'fro'))
    # print(operator_distance(H_Y, H_Z, 'nuc'), operator_distance(H_Y, H_Z, 'fro'))
    # print(operator_distance(H_X, H_H, 'nuc'), operator_distance(H_X, H_H, 'fro'))
    # print(operator_distance(H_Y, H_H, 'nuc'), operator_distance(H_Y, H_H, 'fro'))
    # print(operator_distance(H_Z, H_H, 'nuc'), operator_distance(H_Z, H_H, 'fro'))




    gate_distance = np.zeros((len(ADMISSIBLE_GATES), len(ADMISSIBLE_GATES)))
    for i,V1 in enumerate(ADMISSIBLE_GATES):
        for j,V2 in enumerate(ADMISSIBLE_GATES):
            if j>=i:
                print(V1,V2)
                gate_distance[i,j] = compute_core_distance(V1,V2)
            else:
                gate_distance[i,j] = gate_distance[j,i]

    print(ADMISSIBLE_GATES)
    print(gate_distance)
    np.savetxt("gate_core_distance.csv", gate_distance, delimiter=",")













