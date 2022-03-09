from typing import List
#from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.dagcircuit.dagnode import DAGNode

import numpy as np
from qiskit import QuantumCircuit
from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES, ADMISSIBLE_GATES, DIRECTED_GATES, UNITARY

def _append_to_positions(positioning:dict, V1:str, V2:str, num_qubits:int, pos:str, qargs:List, ignore_warning=True):
    '''
    Append a relative position to a position list if a condition is satisfied.
    Conditions include num_qubit is enough for that relative position
    :param positioning:
    :param V1:
    :param V2:
    :param num_qubits:
    :param pos:
    :return:
    '''
    qarg1 = qargs[0]
    qarg2 = qargs[1]

    if ignore_warning:
        if all(qubit_idx < num_qubits for qubit_idx in qarg1) and all(qubit_idx < num_qubits for qubit_idx in qarg2):
            positioning['_'.join([V1, V2, str(num_qubits), pos])] = qargs
    else:
        assert all(qubit_idx < num_qubits for qubit_idx in qarg1), f"qarg1 contains a qubit out of range ({num_qubits}) " \
                                                               f"for V1({V1}), V2({V2}), pos({pos})."
        assert all(qubit_idx < num_qubits for qubit_idx in qarg2), f"qarg2 contains a qubit out of range ({num_qubits}) " \
                                                               f"for V1({V1}), V2({V2}), pos({pos})."
        positioning['_'.join([V1, V2, str(num_qubits), pos])] = qargs

    return positioning

def all_relative_positions(V1:str, V2:str, num_qubits:int) -> List:
    assert V1 in ADMISSIBLE_GATES, f'gate1 ({V1}) does not belong to ADMISSIBLE_GATES'
    assert V2 in ADMISSIBLE_GATES, f'gate2 ({V2}) does not belong to ADMISSIBLE_GATES'
    assert isinstance(num_qubits, int) and num_qubits > 0, f"num_qubits({num_qubits}) must be a positive integer."

    num_qubits_1 = 1 if V1 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    num_qubits_2 = 1 if V2 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2

    positioning = {}

    if num_qubits == 1:
        assert num_qubits_1 == 1, \
            f'gate1 ({V1}) requires more qubits than num_qubits ({num_qubits})'
        assert num_qubits_2 == 1, \
            f'gate1 ({V2}) requires more qubits than num_qubits ({num_qubits})'

        positioning = _append_to_positions(positioning, V1, V2, num_qubits, 's', [[0], [0]])
        return positioning


    if num_qubits_1 == 1:
        if num_qubits_2 == 1:
            ## On the same register (same qubit)
            positioning = _append_to_positions(positioning, V1, V2, num_qubits, 's', [[0], [0]])
            ## On different registers (different registers)
            positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'd', [[0], [1]])

        else: # num_qubits_2 == 2
            if V2 in DIRECTED_GATES:
                ## V1 on the first qubit of the register V2 applied on
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'up', [[0], [0, 1]])
                ## V1 on the second qubit of the register V2 applied on
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'lo', [[1], [0, 1]])
                ## On different registers
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'd', [[2], [0, 1]]) # or [[0], [1,2]]
            else: # V2 is NON-DIRECTED
                ## Overlap on one qubit
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'over', [[0], [0, 1]])
                ## On different registers
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'over', [[2], [0, 1]])

    else: # num_qubits_1 == 2
        if num_qubits_2 == 1:
            if V1 in DIRECTED_GATES:
                ## V2 on the first qubit of the register V1 applied on
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'up', [[0, 1], [0]])
                ## V2 on the second qubit of the register V1 applied on
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'lo', [[0, 1], [1]])
                ## On different registers
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'd', [[0, 1], [2]])
            else: # V1 is NON-DIRECTED
                ## Overlap on one qubit
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'over', [[0, 1], [0]])
                ## On different registers
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'd', [[0, 1], [2]])
        else: # num_qubits_2 == 2
            if (V1 in DIRECTED_GATES) and (V2 in DIRECTED_GATES):
                ## Same register, aligning
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'alig', [[0, 1], [0, 1]])
                ## Same register, anti-aligning
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'anti', [[0, 1], [1, 0]])
                ## Overlap on the first qubit of V1 register, aligning (mutual qubit in same position in qargs)
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'upalig', [[0, 1], [0, 2]])
                ## Overlap on the first qubit of V1 register, anti-align (mutual qubit in different positions in qargs)
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'upanti', [[0, 1], [2, 0]])
                ## Overlap on the second qubit of V1 register, aligning (mutual qubit in same position in qargs)
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'loalig', [[0, 1], [2, 1]])
                ## Overlap on the second qubit of V1 register, anti-align (mutual qubit in different positions in qargs)
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'loanti', [[0, 1], [1, 2]])
                ## Different registers
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'd', [[0, 1], [2, 3]])
            elif (V1 in DIRECTED_GATES) and (V2 not in DIRECTED_GATES):
                ## Same register
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 's', [[0, 1], [0, 1]])
                ## Overlap on the first qubit of V1 register
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'up', [[1, 2], [0, 1]])
                ## Overlap on the second qubit of V1 register
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'lo', [[0, 1], [1, 2]])
                ## Different registers
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'd', [[0, 1], [2, 3]])
            elif (V1 not in DIRECTED_GATES) and (V2 in DIRECTED_GATES):
                ## Same register
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 's', [[0, 1], [0, 1]])
                ## Overlap on the first qubit of V2 register
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'up', [[0, 1], [1, 2]])
                ## Overlap on the second qubit of V1 register
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'lo', [[1, 2], [0, 1]])
                ## Different registers
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'd', [[0, 1], [2, 3]])
            else: # V1 and V2 are NON-DIRECTED
                ## Same register
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 's', [[0, 1], [0, 1]])
                ## Overlap on one qubit
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'over', [[0, 1], [1, 2]])
                ## Different registers
                positioning = _append_to_positions(positioning, V1, V2, num_qubits, 'd', [[0, 1], [2, 3]])

    return positioning

def get_pos_from_gate_name(V1:str, V2:str, qargs:List) -> dict:
    assert V1 in ADMISSIBLE_GATES, f'gate1 ({V1}) does not belong to ADMISSIBLE_GATES'
    assert V2 in ADMISSIBLE_GATES, f'gate2 ({V2}) does not belong to ADMISSIBLE_GATES'

    num_qubits_1 = 1 if V1 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    num_qubits_2 = 1 if V2 in SINGLE_QUBIT_DETERMINISTIC_GATES + SINGLE_QUBIT_VARIATIONAL_GATES else 2
    is_directed_1 = 1 if V1 in DIRECTED_GATES else 0
    is_directed_2 = 1 if V2 in DIRECTED_GATES else 0
    qarg1 = qargs[0]
    qarg2 = qargs[1]

    assert len(qarg1) == num_qubits_1, f"qarg1({qarg1}) must have length equal to V1' num_qubits({num_qubits_1})"
    assert len(qarg2) == num_qubits_2, f"qarg2({qarg2}) must have length equal to V2' num_qubits({num_qubits_2})"
    assert len(qarg1) == len(set(qarg1)), f"qarg1({qarg1}) contains a duplicated qubit register."
    assert len(qarg2) == len(set(qarg2)), f"qarg2({qarg2}) contains a duplicated qubit register."

    if num_qubits_1 == 1 and num_qubits_2 == 1:
        return 's' if qarg1[0] == qarg2[0] else 'd'

    if num_qubits_1 == 1 and num_qubits_2 == 2:
        if is_directed_2:
            if qarg1[0] == qarg2[0]:
                return 'up'
            elif qarg1[0] == qarg2[1]:
                return 'lo'
            else: # qarg1[0] not in qarg2
                return 'd'
        else: # V2 is NON-DIRECTED
            if qarg1[0] in qarg2:
                return 'over'
            else: # qarg1[0] not in qarg2:
                return 'd'

    if num_qubits_1 == 2 and num_qubits_2 == 1:
        if is_directed_1:
            if qarg2[0] == qarg1[0]:
                return 'up'
            elif qarg2[0] == qarg1[1]:
                return 'lo'
            else: # qarg[2] not in qarg1
                return 'd'
        else: # V1 is NON-DIRECTED
            if qarg2[0] in qarg1:
                return 'over'
            else: # qarg2[0] not in qarg1
                return 'd'

    if num_qubits_1 == 2 and num_qubits_2 == 2:
        if is_directed_1 and is_directed_2:
            if qarg1[0] == qarg2[0] and qarg1[1] == qarg2[1]:
                return 'alig'
            elif qarg1[0] == qarg2[1] and qarg1[1] == qarg2[0]:
                return 'anti'
            elif qarg1[0] == qarg2[0] and qarg1[1] != qarg2[1]:
                return 'upalig'
            elif qarg1[0] == qarg2[1] and qarg1[1] != qarg2[0]:
                return 'upanti'
            elif qarg1[1] == qarg2[1] and qarg1[0] != qarg2[0]:
                return 'loalig'
            elif qarg1[1] == qarg2[0] and qarg1[0] != qarg2[1]:
                return 'loanti'
            else:
                return 'd'
        elif is_directed_1 and not is_directed_2:
            if qarg1[0] in qarg2 and qarg1[1] in qarg2:
                return 's'
            elif qarg1[0] in qarg2 and qarg1[1] not in qarg2:
                return 'up'
            elif qarg1[1] in qarg2 and qarg1[0] not in qarg2:
                return 'lo'
            else:
                return 'd'
        elif not is_directed_1 and is_directed_2:
            if qarg2[0] in qarg1 and qarg2[1] in qarg1:
                return 's'
            elif qarg2[0] in qarg1 and qarg2[1] not in qarg1:
                return 'up'
            elif qarg2[1] in qarg1 and qarg2[0] not in qarg1:
                return 'lo'
            else:
                return 'd'
        else: # V1 and V2 are NON-DIRECTED
            if qarg1[0] in qarg2 and qarg1[0] in qarg2:
                return 's'
            elif qarg1[0] in qarg2 or qarg1[1] in qarg2: # in this elif, only either one can happen
                return 'over'
            else:
                return 'd'

def get_pos_from_gate_DAGobj(node1:DAGNode, node2:DAGNode) -> dict:
    V1 = node1.name
    V2 = node2.name
    qarg1 = [qubit.index for qubit in node1.qargs]
    qarg2 = [qubit.index for qubit in node2.qargs]
    qargs = [qarg1, qarg2]
    return get_pos_from_gate_name(V1, V2, qargs)

