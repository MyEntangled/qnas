import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import qiskit.circuit.library as library

from utility.ansatz_template import AnsatzTemplate
from utility.data_encoding import FeatureMap
from utility.quantum_nn import QuantumNeuralNetwork

from typing import Union
import warnings
warnings.filterwarnings("ignore") # to surpass a decapration warning when calling qubit.index

GATE_TYPE = {'id':0, 'h':1, 'rx':2, 'ry':3, 'rz':4, 'cx_U':5, 'cx_D':6, 'cy_U':7, 'cy_D':8, 'cz_U':9, 'cz_D':10}
GATE_TYPE_IDX = {0:'id', 1:'h', 2:'rx', 3:'ry', 4:'rz', 5:'cx_U', 6:'cx_D', 7:'cy_U', 8:'cy_D', 9:'cz_U', 10:'cz_D'}
UNITARY = {'id': library.IGate, 'h': library.HGate, 'rx': library.RXGate, 'ry': library.RYGate, 'rz': library.RZGate,
           'cx_U': library.CXGate, 'cx_D':library.CXGate, 'cy_U':library.CYGate, 'cy_D':library.CYGate, 'cz_U':library.CZGate, 'cz_D':library.CZGate}
NUM_PARAM_PER_GATE = {0:0, 1:0, 2:1, 3:1, 4:1, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
SINGLE_QUBIT_GATE_TYPE = ['id','h','rx','ry','rz']
TWO_QUBIT_GATE_TYPE = ['cx','cy','cz']

def circuit_to_tensor(model, max_gates):
    num_gates = model.PQC.size()

    if max_gates:
        assert num_gates <= max_gates, f"Number of gates in the ansatz exceeds the limit of {max_gates}."
        embedding = np.zeros((max_gates, model.num_qubits, len(GATE_TYPE)))
    else:
        embedding = np.zeros((num_gates, model.num_qubits , len(GATE_TYPE)))

    for time_idx, gate in enumerate(model.PQC): # encode PQC into tensor
        if gate[0].name in TWO_QUBIT_GATE_TYPE: # handle two-qubit gates
            ctrl, target = [qubit.index for qubit in gate[1]] # DECAPRATION WARNING
            if target == (ctrl-1) % model.num_qubits:
                gate_type_idx = GATE_TYPE[gate[0].name + '_U']
                embedding[time_idx, ctrl, gate_type_idx] = 1
            elif target == (ctrl+1) % model.num_qubits:
                gate_type_idx = GATE_TYPE[gate[0].name + '_D']
                embedding[time_idx, ctrl, gate_type_idx] = 1

        elif gate[0].name in GATE_TYPE: # single-qubit gates
            qubit = gate[1][0].index # DECAPRATION WARNING
            gate_type_idx = GATE_TYPE[gate[0].name]
            embedding[time_idx, qubit, gate_type_idx] = 1

    if num_gates < embedding.shape[0]: # fill remaining gate positions by identities
        #num_id_gates = embedding.shape[0] - num_gates
        gate_type_idx = GATE_TYPE['id']
        embedding[num_gates:, 0, gate_type_idx] = 1

    return embedding

def _operations_to_circuit(num_qubits, operations):

    param_dim = np.sum(np.array(operations)[:, 2])
    param = ParameterVector('θ', param_dim)
    theta_idx = 0

    qc = QuantumCircuit(num_qubits)

    for qubit, gate_type_idx, num_param_of_gate in operations:
        # print(qubit, int(gate_type_idx), num_param_per_gate)

        gate_type_idx = int(gate_type_idx)
        gate_type = GATE_TYPE_IDX[gate_type_idx]

        if gate_type in SINGLE_QUBIT_GATE_TYPE:

            if num_param_of_gate == 0: # insert normal gate
                gate = UNITARY[gate_type]()

            elif num_param_of_gate == 1: # insert param gate
                gate = UNITARY[gate_type](param[theta_idx])
                theta_idx += 1

            else:
                raise Exception("Gates should have at most 1 parameter}")

            qc.append(gate, [qubit])

        else:

            suffix = gate_type[-1] # 'U' or 'D'
            t = -1 if suffix == 'U' else 1

            if num_param_of_gate == 0: # insert normal 2-q gate
                gate = UNITARY[gate_type]()
            elif num_param_of_gate == 1: # insert param 2-q gate
                gate = UNITARY[gate_type](param[theta_idx])
                theta_idx += 1
            else:
                raise Exception("Gates should have at most 1 parameter}")

            qc.append(gate, [qubit, (qubit + t) % num_qubits])

    return qc


def tensor_to_circuit(embedding):

    num_gates, num_qubits, _ = embedding.shape
    operations = []

    for time_idx in range(num_gates):

        qubit, gate_type_idx = np.unravel_index(embedding[time_idx].argmax(), embedding[time_idx].shape)

        if gate_type_idx != 0:
            num_param_of_gate = NUM_PARAM_PER_GATE[gate_type_idx]
            operations.append((qubit, gate_type_idx, num_param_of_gate))

    param_dim = np.sum(np.array(operations)[:, 2])
    param = np.zeros(param_dim)

    qc = _operations_to_circuit(num_qubits, operations)
    return qc


if __name__ == '__main__':
    feature_map = FeatureMap('PauliFeatureMap', 4, 1)

    template = AnsatzTemplate()
    template.construct_simple_template(4, 1)

    model = QuantumNeuralNetwork(feature_map, template)

    print(model.PQC.draw())

    tensor = circuit_to_tensor(model, 25)

    # for mat in tensor:
    #     print(mat) ## correct
    print(tensor)

    retcon = tensor_to_circuit(tensor)
    print(retcon.draw())




