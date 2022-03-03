import numpy as np
import pandas as pd

from utility.ansatz_template import AnsatzTemplate
from utility.data_encoding import FeatureMap
from utility.quantum_nn import QuantumNeuralNetwork

from QuOTMANN.dag_utility import create_dag

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, \
    SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, \
    TWO_QUBIT_VARIATIONAL_GATES, \
    ADMISSIBLE_GATES, \
    DIRECTED_GATES, \
    UNITARY

def label_mismatch_cost_matrix(PQC_1, PQC_2):
    shape_distance = pd.read_csv('/Users/Erio/Dropbox/URP project/Code/PQC_composer/gate_distance/gate_shape_distance.csv',header=None).values
    core_distance = pd.read_csv('/Users/Erio/Dropbox/URP project/Code/PQC_composer/gate_distance/gate_core_distance.csv',header=None).values
    gate_distance = core_distance + shape_distance

    #print(core_distance)


    dag_1, nx_dag_1, in_nodes_1, out_nodes_1 = create_dag(PQC_1)
    dag_2, nx_dag_2, in_nodes_2, out_nodes_2 = create_dag(PQC_2)

    op_nodes_1 = dag_1.op_nodes()
    op_nodes_2 = dag_2.op_nodes()
    C_lmm = np.zeros((len(op_nodes_1), len(op_nodes_2)))

    for i,op_node_1 in enumerate(op_nodes_1):
        for j,op_node_2 in enumerate(op_nodes_2):
            shape_dist = shape_distance[ADMISSIBLE_GATES.index(op_node_1.name), ADMISSIBLE_GATES.index(op_node_2.name)]
            core_dist = core_distance[ADMISSIBLE_GATES.index(op_node_1.name), ADMISSIBLE_GATES.index(op_node_2.name)]
            C_lmm[i,j] = gate_distance[ADMISSIBLE_GATES.index(op_node_1.name), ADMISSIBLE_GATES.index(op_node_2.name)]
            #print((i,j), (op_node_1.name, op_node_2.name), (ADMISSIBLE_GATES.index(op_node_1.name), ADMISSIBLE_GATES.index(op_node_2.name)), 'shape distance: ', shape_dist, 'core distance: ', core_dist)

    return C_lmm

if __name__ == '__main__':
    feature_map_1 = FeatureMap('ZZFeatureMap', feature_dim=4, reps=1)
    template_1 = AnsatzTemplate()
    template_1.construct_simple_template(num_qubits=4, num_layers=1)
    model_1 = QuantumNeuralNetwork(feature_map_1, template_1, platform='Qiskit')
    #model_1.visualize()
    print(model_1.num_qubits, model_1.input_dim, model_1.param_dim)

    feature_map_2 = FeatureMap('ZZFeatureMap', feature_dim=4, reps=1)
    template_2 = AnsatzTemplate()
    template_2.construct_simple_template(num_qubits=4, num_layers=2)
    model_2 = QuantumNeuralNetwork(feature_map_2, template_2, platform='Qiskit')
    #model_2.visualize()
    print(model_2.num_qubits, model_2.input_dim, model_2.param_dim)

    C_lmm = label_mismatch_cost_matrix(model_1.PQC, model_2.PQC)
    #print(C_lmm.shape)
    #print(C_lmm)


