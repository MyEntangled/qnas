import qiskit.circuit.library as library
import numpy as np

SINGLE_QUBIT_DETERMINISTIC_GATES = ['h', 'x', 'y', 'z']
SINGLE_QUBIT_VARIATIONAL_GATES = ['rx', 'ry', 'rz']
TWO_QUBIT_DETERMINISTIC_GATES = ['cx', 'cy', 'cz']
TWO_QUBIT_VARIATIONAL_GATES = ['crx', 'cry', 'crz', 'rxx', 'ryy', 'rzz']
#ADMISSIBLE_GATES = SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_VARIATIONAL_GATES
ADMISSIBLE_GATES = SINGLE_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_VARIATIONAL_GATES

DIRECTED_GATES = ['cx', 'cy', 'cz', 'crx', 'cry', 'crz']

UNITARY = {'h': library.HGate, 'x': library.XGate, 'y': library.YGate, 'z': library.ZGate,
           'rx': library.RXGate, 'ry': library.RYGate, 'rz': library.RZGate,
           'cx': library.CXGate, 'cy':library.CYGate, 'cz':library.CZGate,
           'crx':library.CRXGate, 'cry':library.CRYGate, 'crz':library.CRZGate,
           'rxx':library.RXXGate, 'ryy':library.RYYGate, 'rzz':library.RZZGate}

## This dictionary will be generated by the following function
# GATE_DIM = {'h':2, 'x':2, 'y':2, 'z':2,
#              'rx':2, 'ry':2, 'rz':2,
#              'cx':4, 'cy':4, 'cz':4,
#              'crx':4, 'cry':4, 'crz':4, 'rxx':4, 'ryy':4, 'rzz':4}
GATE_DIM = {}
for gate in ADMISSIBLE_GATES:
    if gate in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES:
        GATE_DIM[gate] = 2
    else:
        GATE_DIM[gate] = 4

## This dictionary will be generated by the following function
# GATE_COMP_UNIT = {'h':0.1, 'x':0.1, 'y':0.1, 'z':0.1,
#              'rx':1, 'ry':1, 'rz':1,
#              'cx':0.1, 'cy':0.1, 'cz':0.1,
#              'crx':1, 'cry':1, 'crz':1, 'rxx':1, 'ryy':1, 'rzz':1}
GATE_COMP_UNIT = {}
for gate in ADMISSIBLE_GATES:
    if gate in SINGLE_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_DETERMINISTIC_GATES:
        GATE_COMP_UNIT[gate] = 0.1
    else:
        GATE_COMP_UNIT[gate] = 1


# def create_node_dict(num_qubits):
#     NODE_DICT = {}
#     num_gates = len(ADMISSIBLE_GATES)
#
#     op_scalars = np.linspace(0.1,0.9, 2*num_gates+1)[1::2]
#     for i,gate in enumerate(ADMISSIBLE_GATES):
#         NODE_DICT[gate] = op_scalars[i]
#
#     in_scalars = np.linspace(0,0.1, 2*num_qubits+1)[1::2]
#     out_scalars = np.linspace(0.9,1, 2*num_qubits+1)[1::2]
#     for i in range(num_qubits):
#         NODE_DICT['_'.join(['in',str(i)])] = in_scalars[i]
#         NODE_DICT['_'.join(['out', str(i)])] = out_scalars[i]
#     return NODE_DICT

def create_op_node_dict():
    OP_NODE_DICT = {}
    num_ops = len(ADMISSIBLE_GATES)

    #print(np.linspace(-1,1, 2*num_ops+1)[1::2])
    #op_scalars = np.arctanh(np.linspace(0,1, 2*num_ops+1)[1::2])
    op_scalars = np.linspace(0,1, 2*num_ops+1)[1::2]
    for i, gate in enumerate(ADMISSIBLE_GATES):
        OP_NODE_DICT[gate] = op_scalars[i]

    return OP_NODE_DICT
OP_NODE_DICT = create_op_node_dict()

if __name__ == '__main__':
    print(ADMISSIBLE_GATES)
    print(GATE_DIM)
    print(GATE_COMP_UNIT)
    #print(create_node_dict(4))
    print(create_op_node_dict())