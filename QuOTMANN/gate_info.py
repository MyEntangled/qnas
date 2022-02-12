import qiskit.circuit.library as library
import numpy as np

# SINGLE_QUBIT_DETERMINISTIC_GATES = ['i', 'h', 's', 'x', 'y', 'z', 't']
# SINGLE_QUBIT_VARIATIONAL_GATES = ['rx', 'ry', 'rz']
# TWO_QUBIT_DETERMINISTIC_GATES = ['cx', 'cy', 'cz']
# TWO_QUBIT_VARIATIONAL_GATES = ['crx', 'cry', 'crz', 'rxx', 'ryy', 'rzz']
# ADMISSIBLE_GATES = SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_VARIATIONAL_GATES

SINGLE_QUBIT_DETERMINISTIC_GATES = ['h', 'x', 'y', 'z']
SINGLE_QUBIT_VARIATIONAL_GATES = ['rx', 'ry', 'rz']
TWO_QUBIT_DETERMINISTIC_GATES = ['cx', 'cy', 'cz']
TWO_QUBIT_VARIATIONAL_GATES = ['crx', 'cry', 'crz', 'rxx', 'ryy', 'rzz']
ADMISSIBLE_GATES = SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES + TWO_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_VARIATIONAL_GATES

UNITARY = {'h': library.HGate, 'x': library.XGate, 'y': library.YGate, 'z': library.ZGate,
           'rx': library.RXGate, 'ry': library.RYGate, 'rz': library.RZGate,
           'cx': library.CXGate, 'cy':library.CYGate, 'cz':library.CZGate,
           'crx':library.CRXGate, 'cry':library.CRYGate, 'crz':library.CRZGate,
           'rxx':library.RXXGate, 'ryy':library.RYYGate, 'rzz':library.RZZGate}

# GATE_DIM = {'i':2, 'h':2, 's':2, 'x':2, 'y':2, 'z':2, 't':2,
#              'rx':2, 'ry':2, 'rz':2,
#              'cx':4, 'cy':4, 'cz':4,
#              'crx':4, 'cry':4, 'crz':4, 'rxx':4, 'ryy':4, 'rzz':4}
GATE_DIM = {}
for gate in ADMISSIBLE_GATES:
    if gate in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES:
        GATE_DIM[gate] = 2
    else:
        GATE_DIM[gate] = 4

# GATE_COMP_UNIT = {'i':0.1, 'h':0.1, 's':0.1, 'x':0.1, 'y':0.1, 'z':0.1, 't':0.1,
#              'rx':1, 'ry':1, 'rz':1,
#              'cx':0.1, 'cy':0.1, 'cz':0.1,
#              'crx':1, 'cry':1, 'crz':1, 'rxx':1, 'ryy':1, 'rzz':1}
GATE_COMP_UNIT = {}
for gate in ADMISSIBLE_GATES:
    if gate in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES:
        GATE_COMP_UNIT[gate] = 0.05
    else:
        GATE_COMP_UNIT[gate] = 1.

NODE_DICT = {}
def create_node_dict():
    num_gates = len(ADMISSIBLE_GATES)
    scalars = np.linspace(0,1, num_gates+2)[1:-1]
    for i,gate in enumerate(ADMISSIBLE_GATES):
        NODE_DICT[gate] = scalars[i]
create_node_dict()

if __name__ == '__main__':
    print(ADMISSIBLE_GATES)
    print(GATE_DIM)
    print(GATE_COMP_UNIT)
    print(NODE_DICT)