import numpy as np
from QuOTMANN.gate_info import GATE_DIM, GATE_COMP_UNIT

def gate_mass(gate):
    return GATE_DIM[gate], GATE_COMP_UNIT[gate]