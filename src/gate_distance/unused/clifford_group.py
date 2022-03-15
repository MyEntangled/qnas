import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_clifford
from state_utility import reformat_statedata
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def generate_cliffords(num_qubits):
    assert num_qubits in [1,2], "The number of qubits must be 1 or 2."
    if num_qubits == 1:
        max_cliffords = 24
    elif num_qubits == 2:
        max_cliffords = 11520

    clifford_group = []
    while len(clifford_group) < max_cliffords:
        new_clifford = random_clifford(num_qubits)
        if new_clifford not in clifford_group:
            clifford_group.append(new_clifford)

    filename = 'cliffords_' + str(num_qubits) + 'qubits.pkl'
    save_object(clifford_group, filename)

    return

def get_cliffords(num_qubits):
    '''
    Get all elements of the Clifford group of dimension 2**num_qubits
    :param num_qubits:
    :return:
    '''
    assert num_qubits in [1, 2], 'The number of qubits must be 1 or 2!'

    filename = 'cliffords_' + str(num_qubits) + 'qubits.pkl'
    with open(filename, 'rb') as f:
        cliffords = pickle.load(f)

    return cliffords


def generate_anchors(num_qubits):
    '''
    Generate all distinct states produced by applying the Clifford groups on \ket{0} and the corresponding Cliffords.
    :param num_qubits:
    :return:
    '''
    cliffords = get_cliffords(num_qubits)
    states = np.zeros(shape=(len(cliffords), 2 ** num_qubits), dtype=np.complex_)

    for i, clifford in enumerate(cliffords):
        vec = Statevector.from_label('0' * num_qubits)

        cliff_circ = clifford.to_circuit()
        # print(vec, cliff_circ)
        states[i] = reformat_statedata(vec.evolve(cliff_circ).data)

    anchor_states, inds = np.unique(states, return_index=True, axis=0)
    anchor_cliffords = np.array(cliffords)[inds]

    filename = 'anchor_states_' + str(num_qubits) + 'qubits.npy'
    with open(filename, 'wb') as f:
        np.save(f, anchor_states)

    filename = 'anchor_cliffords_' + str(num_qubits) + 'qubits.pkl'
    save_object(anchor_cliffords, filename)

    return


def get_anchor_states(num_qubits, get_anchor_cliffords: bool = False):
    assert num_qubits in [1, 2], 'The number of qubits must be 1 or 2!'

    filename = 'anchor_states_' + str(num_qubits) + 'qubits.npy'
    with open(filename, 'rb') as f:
        anchor_states = np.load(f)

    if not get_anchor_cliffords:
        return anchor_states
    else:
        filename = 'anchor_cliffords_' + str(num_qubits) + 'qubits.pkl'
        with open(filename, 'rb') as f:
            anchor_cliffords = pickle.load(f)
        return anchor_states, anchor_cliffords



if __name__ == '__main__':
    generate_cliffords(1)
    generate_anchors(1)

    states, cliffords = get_anchor_states(1, True)
    print("Single qubit Cliffords")
    print(len(cliffords))  # =6
    print(states)

########################

    generate_anchors(2)

    states, cliffords = get_anchor_states(2, True)
    print("Two qubit Cliffords")
    print(len(cliffords))  # =172
    print(states)
