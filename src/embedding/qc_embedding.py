import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import torch

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, \
    SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, \
    TWO_QUBIT_VARIATIONAL_GATES, \
    ADMISSIBLE_GATES, \
    DIRECTED_GATES, \
    UNITARY, \
    OP_NODE_DICT

#INV_OP_NODE_DICT = {v:k for k,v in OP_NODE_DICT.items()}
#OP_VALUES = np.array(list(OP_NODE_DICT.values()))
#OP_VALUES_1Q = np.array([val for node,val in OP_NODE_DICT.items() if node in SINGLE_QUBIT_DETERMINISTIC_GATES or node in SINGLE_QUBIT_VARIATIONAL_GATES])

INV_OP_NODE_DICT_TORCH = {v:k for k,v in OP_NODE_DICT.items()}
OP_VALUES_TORCH = torch.tensor(list(OP_NODE_DICT.values()))
OP_VALUES_1Q_TORCH = torch.tensor([val for node,val in OP_NODE_DICT.items() if node in SINGLE_QUBIT_DETERMINISTIC_GATES or node in SINGLE_QUBIT_VARIATIONAL_GATES])
print(INV_OP_NODE_DICT_TORCH)
def qc_to_enc(qc: QuantumCircuit, MAX_OP_NODES:int = None) -> np.ndarray:
    if MAX_OP_NODES is None:
        MAX_OP_NODES = qc.size()
    else:
        if MAX_OP_NODES < qc.size():
            raise ValueError(f'MAX_OP_NODES ({MAX_OP_NODES}) is necessarily greater than or equal to the number of quantum operations ({qc.size()})')

    encoding = np.zeros((qc.num_qubits + 1, MAX_OP_NODES)) # later flattened

    for idx,(inst,qargs,cargs) in enumerate(qc.data):
        gatename = inst.name
        assert gatename in ADMISSIBLE_GATES, f"{gatename} is not in ADMISSIBLE_GATES"

        encoding[-1, idx] = OP_NODE_DICT[gatename]

        if (gatename in SINGLE_QUBIT_DETERMINISTIC_GATES) or (gatename in SINGLE_QUBIT_VARIATIONAL_GATES):
            encoding[qargs[0].index, idx] = 1
        elif DIRECTED_GATES:
            ctrl_qubit = qargs[0].index
            applied_qubit = qargs[1].index
            encoding[ctrl_qubit, idx] = 0.75
            encoding[applied_qubit, idx] = 0.25
        else:
            encoding[qargs[0].index, idx] = 0.5
            encoding[qargs[1].index, idx] = 0.5

    return encoding.ravel()

#
# def enc_to_qc(num_qubits: int, encoding: np.ndarray) -> QuantumCircuit:
#     encoding = encoding.reshape(num_qubits+1, -1)
#     qc = QuantumCircuit(num_qubits)
#     theta = ParameterVector('theta',0)
#
#     infolist = [] ## list of gatenames
#     for i,code in enumerate(encoding[-1]):
#         if code <= 0:
#             infolist.append('none')
#         else: ## code > 0
#             if num_qubits > 1:
#                 closest_mark = OP_VALUES[torch.abs(OP_VALUES - code).argmin()]
#                 infolist.append(INV_OP_NODE_DICT[closest_mark])
#             else:
#                 closest_mark = OP_VALUES_1Q[torch.abs(OP_VALUES_1Q - code).argmin()]
#                 infolist.append(INV_OP_NODE_DICT[closest_mark])
#
#
#     for idx,gatename in enumerate(infolist):
#
#         if gatename in SINGLE_QUBIT_DETERMINISTIC_GATES:
#             qargs = [qc.qubits[np.argmax(encoding[:-1,idx])]]
#             qc.append(UNITARY[gatename](), qargs=qargs, cargs=[])
#
#         elif gatename in SINGLE_QUBIT_VARIATIONAL_GATES:
#             qargs = [qc.qubits[np.argmax(encoding[:-1,idx])]]
#             theta.resize(len(theta) + 1)
#             qc.append(UNITARY[gatename](theta[-1]), qargs=qargs, cargs=[])
#
#         elif gatename in TWO_QUBIT_DETERMINISTIC_GATES:
#             two_highest_idx = np.argpartition(-encoding[:-1,idx],kth=1)
#             highest_idx = two_highest_idx[0]
#             second_highest_idx = two_highest_idx[1]
#
#             qargs = [qc.qubits[highest_idx], qc.qubits[second_highest_idx]]
#             qc.append(UNITARY[gatename](), qargs=qargs, cargs=[])
#
#         elif gatename in TWO_QUBIT_VARIATIONAL_GATES:
#             two_highest_idx = np.argpartition(-encoding[:-1,idx],kth=1)
#
#             highest_idx = two_highest_idx[0]
#             second_highest_idx = two_highest_idx[1]
#
#             qargs = [qc.qubits[highest_idx], qc.qubits[second_highest_idx]]
#             theta.resize(len(theta) + 1)
#
#             qc.append(UNITARY[gatename](theta[-1]), qargs=qargs, cargs=[])
#
#         else: ## gatename == 'none'
#             pass
#
#     return qc

def enc_to_qc_torch(num_qubits: int, encoding: torch.tensor) -> QuantumCircuit:
    OP_VALUES_TORCH_dev = OP_VALUES_TORCH.to(encoding)
    OP_VALUES_1Q_TORCH_dev = OP_VALUES_1Q_TORCH.to(encoding)

    encoding = encoding.reshape(num_qubits+1, -1)
    qc = QuantumCircuit(num_qubits)
    theta = ParameterVector('theta',0)

    infolist = [] ## list of gatenames
    for i,code in enumerate(encoding[-1]):
        if code <= 0:
            infolist.append('none')
        else: ## code > 0
            if num_qubits > 1:
                closest_mark = OP_VALUES_TORCH_dev[torch.abs(OP_VALUES_TORCH_dev - code).argmin()].item()
                infolist.append(INV_OP_NODE_DICT_TORCH[closest_mark])
            else:
                closest_mark = OP_VALUES_1Q_TORCH_dev[torch.abs(OP_VALUES_1Q_TORCH_dev - code).argmin()].item()
                infolist.append(INV_OP_NODE_DICT_TORCH[closest_mark])


    for idx,gatename in enumerate(infolist):

        if gatename in SINGLE_QUBIT_DETERMINISTIC_GATES:
            qargs = [qc.qubits[torch.argmax(encoding[:-1,idx])]]
            qc.append(UNITARY[gatename](), qargs=qargs, cargs=[])

        elif gatename in SINGLE_QUBIT_VARIATIONAL_GATES:
            qargs = [qc.qubits[torch.argmax(encoding[:-1,idx])]]
            theta.resize(len(theta) + 1)
            qc.append(UNITARY[gatename](theta[-1]), qargs=qargs, cargs=[])

        elif gatename in TWO_QUBIT_DETERMINISTIC_GATES:
            #two_highest_idx = np.argpartition(-encoding[:-1,idx],kth=1)
            _, two_highest_idx = torch.topk(encoding[:-1,idx], k=2, sorted=True)
            highest_idx = two_highest_idx[0]
            second_highest_idx = two_highest_idx[1]

            qargs = [qc.qubits[highest_idx], qc.qubits[second_highest_idx]]
            qc.append(UNITARY[gatename](), qargs=qargs, cargs=[])

        elif gatename in TWO_QUBIT_VARIATIONAL_GATES:
            #two_highest_idx = np.argpartition(-encoding[:-1,idx],kth=1)
            _, two_highest_idx = torch.topk(encoding[:-1, idx], k=2, sorted=True)
            highest_idx = two_highest_idx[0]
            second_highest_idx = two_highest_idx[1]

            qargs = [qc.qubits[highest_idx], qc.qubits[second_highest_idx]]
            theta.resize(len(theta) + 1)

            qc.append(UNITARY[gatename](theta[-1]), qargs=qargs, cargs=[])

        else: ## gatename == 'none'
            pass

    return qc

if __name__ == '__main__':
    qc = QuantumCircuit(4)
    qc.cx(0,1)
    qc.cx(3,2)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.cx(1,0)
    print(qc.draw())

    encoding = qc_to_enc(qc, MAX_OP_NODES=None)
    print(encoding.reshape(5,5))

    rec_qc = enc_to_qc(4, encoding)

    print(rec_qc.draw())

    np.random.seed(10)
    rand_encoding = np.random.standard_normal((5,5))
    print(rand_encoding)
    rand_rec_qc = enc_to_qc(4, rand_encoding)
    print(rand_rec_qc.draw())