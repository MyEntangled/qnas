import numpy as np

from QuOTMANN.dag_utility import create_dag
from gate_distance import gate_positioning

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, \
    SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, \
    TWO_QUBIT_VARIATIONAL_GATES, \
    ADMISSIBLE_GATES, \
    DIRECTED_GATES, \
    UNITARY

import pickle


def label_mismatch_cost_matrix(PQC_1, PQC_2):
    assert PQC_1.num_qubits == PQC_2.num_qubits, "Two quantum circuits must have the same number of qubits."
    num_qubits_str = str(PQC_1.num_qubits)

    with open('src/gate_distance/all_shape_distances.pkl', 'rb') as f:
        all_shape_distances = pickle.load(f)
    with open('src/gate_distance/all_core_distances.pkl', 'rb') as f:
        all_core_distances = pickle.load(f)

    dag_1, nx_dag_1, in_nodes_1, out_nodes_1 = create_dag(PQC_1)
    dag_2, nx_dag_2, in_nodes_2, out_nodes_2 = create_dag(PQC_2)

    op_nodes_1 = dag_1.op_nodes()
    op_nodes_2 = dag_2.op_nodes()
    C_lmm = np.zeros((len(op_nodes_1), len(op_nodes_2)))

    for i, op_node_1 in enumerate(op_nodes_1):
        for j, op_node_2 in enumerate(op_nodes_2):
            pos = gate_positioning.get_pos_from_gate_DAGobj(op_node_1, op_node_2)
            shape_dist = all_shape_distances['_'.join([op_node_1.name, op_node_2.name, num_qubits_str, pos])]
            core_dist = all_core_distances['_'.join([op_node_1.name, op_node_2.name, num_qubits_str, pos])]
            C_lmm[i,j] = (core_dist + shape_dist)/2.

    return C_lmm

if __name__ == '__main__':
    pass

