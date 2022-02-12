from utility.ansatz_template import AnsatzTemplate
from utility.data_encoding import FeatureMap
from utility.quantum_nn import QuantumNeuralNetwork
import networkx as nx
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
import numpy as np

def create_dag(PQC):
    dag = circuit_to_dag(PQC)
    nx_dag = dag.to_networkx()

    in_nodes = [0] * dag.num_qubits()
    out_nodes = [0] * dag.num_qubits()

    for node in nx_dag.nodes():
        if node.type in ['in', 'out']:
            #print(node.type, node.wire.index)

            if node.type == 'in':
                in_nodes[node.wire.index] = node
            if node.type == 'out':
                out_nodes[node.wire.index] = node

        elif node.type == 'op':
            1
            #print(node.name, [qubit.index for qubit in node.qargs])

    return dag, nx_dag, in_nodes, out_nodes

def longest_simple_path_length(dag, graph, source, target) -> int:
    longest_paths = []
    longest_path_length = 0
    all_paths = list(nx.all_simple_paths(graph, source=source, target=target))

    if len(all_paths) == 0:
        return len(dag.longest_path())

    for path in all_paths:
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_paths.clear()
            longest_paths.append(path)
        elif len(path) == longest_path_length:
            longest_paths.append(path)
    return longest_path_length

def shortest_simple_path_length(dag, graph, source, target) -> int:
    shortest_paths = []
    shortest_path_length = 10e9
    all_paths = list(nx.all_simple_paths(graph, source=source, target=target))

    if len(all_paths) == 0:
        return len(dag.longest_path())

    for path in all_paths:
        if len(path) < shortest_path_length:
            shortest_path_length = len(path)
            shortest_paths.clear()
            shortest_paths.append(path)
        elif len(path) == shortest_path_length:
            shortest_paths.append(path)
    return shortest_path_length

def random_walk_path_length(graph, source, target) -> int:
    ##TODO
    return 0


def structural_path_lengths(dag, nx_dag, in_nodes, out_nodes):
    n = dag.num_qubits()

    op_nodes = dag.op_nodes()
    lengths_to_in_nodes = np.zeros((len(op_nodes), n, 2))
    lengths_to_out_nodes = np.zeros((len(op_nodes), n, 2))

    for i, op_node in enumerate(op_nodes):
        for j, in_node in enumerate(in_nodes):
            lengths_to_in_nodes[i, j, 0] = longest_simple_path_length(dag, nx_dag, in_node, op_node)
            lengths_to_in_nodes[i, j, 1] = shortest_simple_path_length(dag, nx_dag, in_node, op_node)
            #lengths_to_in_nodes[i, j, 2] = random_walk_path_length(nx_dag, in_node, op_node)
        for k, out_node in enumerate(out_nodes):
            lengths_to_out_nodes[i, j, 0] = longest_simple_path_length(dag, nx_dag, op_node, out_node)
            lengths_to_out_nodes[i, j, 1] = shortest_simple_path_length(dag, nx_dag, op_node, out_node)
            #lengths_to_out_nodes[i, j, 2] = random_walk_path_length(nx_dag, op_node, out_node)
    return lengths_to_in_nodes, lengths_to_out_nodes

def average_path_length(PQC_1, PQC_2):
    dag_1, nx_dag_1, in_nodes_1, out_nodes_1 = create_dag(PQC_1)
    dag_2, nx_dag_2, in_nodes_2, out_nodes_2 = create_dag(PQC_2)

    num_path_types = 2
    lengths_to_in_nodes_1, lengths_to_out_nodes_1 = structural_path_lengths(dag_1, nx_dag_1, in_nodes_1, out_nodes_1)
    lengths_to_in_nodes_2, lengths_to_out_nodes_2 = structural_path_lengths(dag_2, nx_dag_2, in_nodes_2, out_nodes_2)

    avg_lengths = np.zeros((lengths_to_in_nodes_1.shape[0], lengths_to_out_nodes_2.shape[0]))

    for i in range(lengths_to_in_nodes_1.shape[0]):
        for j in range(lengths_to_out_nodes_2.shape[0]):
            avg_lengths[i,j] = np.sum(np.abs(lengths_to_in_nodes_1[i] - lengths_to_in_nodes_2[j])) + np.sum(np.abs(lengths_to_out_nodes_1[i] - lengths_to_out_nodes_2[j])) / (2*dag_1.num_qubits()*num_path_types)
    return avg_lengths


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

    print(average_path_length(model_1.PQC, model_2.PQC))










