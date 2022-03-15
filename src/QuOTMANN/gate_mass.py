import numpy as np
from QuOTMANN.gate_info import GATE_DIM, GATE_COMP_UNIT

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, \
    SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, \
    TWO_QUBIT_VARIATIONAL_GATES, \
    ADMISSIBLE_GATES, \
    DIRECTED_GATES, \
    UNITARY, \
    GATE_DIM, \
    GATE_COMP_UNIT

# def gate_mass(gate):
#     return GATE_DIM[gate], GATE_COMP_UNIT[gate]

def gate_mass(gate, d):
    if gate in SINGLE_QUBIT_VARIATIONAL_GATES or gate in TWO_QUBIT_VARIATIONAL_GATES:
        gate_params = 1
    else:
        gate_params = 0

    gate_dim = GATE_DIM[gate]
    return gate_params * (gate_dim ** 2 - 1.) #/ (d ** 2 - 1.)

