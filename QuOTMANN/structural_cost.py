import networkx as nx
import numpy as np

from QuOTMANN.dag_utility import create_dag



def longest_simple_path_length(dag, graph, source, target, no_path_cost) -> int:
    longest_paths = []
    longest_path_length = 0
    all_paths = list(nx.all_simple_paths(graph, source=source, target=target))

    if len(all_paths) == 0:
        #return len(dag.longest_path())
        return no_path_cost

    for path in all_paths:
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_paths.clear()
            longest_paths.append(path)
        elif len(path) == longest_path_length:
            longest_paths.append(path)
    return longest_path_length

def shortest_simple_path_length(dag, graph, source, target, no_path_cost) -> int:
    shortest_paths = []
    shortest_path_length = 10e9
    all_paths = list(nx.all_simple_paths(graph, source=source, target=target))

    if len(all_paths) == 0:
        #return len(dag.longest_path())
        return no_path_cost

    for path in all_paths:
        if len(path) < shortest_path_length:
            shortest_path_length = len(path)
            shortest_paths.clear()
            shortest_paths.append(path)
        elif len(path) == shortest_path_length:
            shortest_paths.append(path)
    return shortest_path_length


def random_walk_path_length(dag, graph, source, target, no_path_cost) -> int:
    all_paths = list(nx.all_simple_paths(graph, source=source, target=target))

    if len(all_paths) == 0:
        #return len(dag.longest_path())
        return no_path_cost

    return np.average([len(path) for path in all_paths])


def three_path_lengths(graph, source, target, no_path_cost):
    all_paths = list(nx.all_simple_paths(graph, source=source, target=target))
    if len(all_paths) == 0:
        return 0,0,0

    all_paths_length = [len((path)) for path in all_paths]

    longest_path_length = max(all_paths_length)
    shortest_path_length = min(all_paths_length)
    mean_path_length = sum(all_paths_length) / len(all_paths_length)
    return longest_path_length, shortest_path_length, mean_path_length




def structural_path_lengths(dag, nx_dag, in_nodes, out_nodes):
    n = dag.num_qubits()

    op_nodes = dag.op_nodes()
    lengths_to_in_nodes = np.zeros((len(op_nodes), n, 3))
    lengths_to_out_nodes = np.zeros((len(op_nodes), n, 3))

    no_path_cost = len(dag.longest_path())

    for i, op_node in enumerate(op_nodes):
        for j, in_node in enumerate(in_nodes):
            #lengths_to_in_nodes[i, j, 0] = longest_simple_path_length(dag, nx_dag, in_node, op_node, no_path_cost)
            #lengths_to_in_nodes[i, j, 1] = shortest_simple_path_length(dag, nx_dag, in_node, op_node, no_path_cost)
            #lengths_to_in_nodes[i, j, 2] = random_walk_path_length(dag, nx_dag, in_node, op_node, no_path_cost)
            lengths_to_in_nodes[i, j, :] = three_path_lengths(graph=nx_dag,source=in_node,target=op_node,no_path_cost=no_path_cost)
        for k, out_node in enumerate(out_nodes):
            #lengths_to_out_nodes[i, k, 0] = longest_simple_path_length(dag, nx_dag, op_node, out_node, no_path_cost)
            #lengths_to_out_nodes[i, k, 1] = shortest_simple_path_length(dag, nx_dag, op_node, out_node, no_path_cost)
            #lengths_to_out_nodes[i, k, 2] = random_walk_path_length(dag, nx_dag, op_node, out_node, no_path_cost)
            lengths_to_in_nodes[i, j, :] = three_path_lengths(graph=nx_dag,source=op_node,target=out_node,no_path_cost=no_path_cost)
    return lengths_to_in_nodes, lengths_to_out_nodes

def structural_cost_matrix(PQC_1, PQC_2):
    dag_1, nx_dag_1, in_nodes_1, out_nodes_1 = create_dag(PQC_1)
    dag_2, nx_dag_2, in_nodes_2, out_nodes_2 = create_dag(PQC_2)

    num_path_types = 3
    lengths_to_in_nodes_1, lengths_to_out_nodes_1 = structural_path_lengths(dag_1, nx_dag_1, in_nodes_1, out_nodes_1)
    lengths_to_in_nodes_2, lengths_to_out_nodes_2 = structural_path_lengths(dag_2, nx_dag_2, in_nodes_2, out_nodes_2)
    #print(lengths_to_in_nodes_1.shape, lengths_to_out_nodes_1.shape, lengths_to_in_nodes_2.shape, lengths_to_out_nodes_2.shape)

    #C_str = np.zeros((lengths_to_in_nodes_1.shape[0], lengths_to_out_nodes_2.shape[0]))
    C_str = np.zeros((len(dag_1.op_nodes()), len(dag_2.op_nodes())))

    normalization = 2 * lengths_to_in_nodes_1.shape[1] * lengths_to_in_nodes_1.shape[2]

    for i in range(lengths_to_in_nodes_1.shape[0]):
        for j in range(lengths_to_out_nodes_2.shape[0]):
            C_str[i,j] = (np.sum(np.abs(lengths_to_in_nodes_1[i] - lengths_to_in_nodes_2[j])) + np.sum(np.abs(lengths_to_out_nodes_1[i] - lengths_to_out_nodes_2[j]))) / normalization
    return C_str


if __name__ == '__main__':
    # feature_map_1 = FeatureMap('ZZFeatureMap', feature_dim=4, reps=1)
    # template_1 = AnsatzTemplate()
    # template_1.construct_simple_template(num_qubits=4, num_layers=1)
    # model_1 = QuantumNeuralNetwork(feature_map_1, template_1, platform='Qiskit')
    # #model_1.visualize()
    # print(model_1.num_qubits, model_1.input_dim, model_1.param_dim)
    #
    # feature_map_2 = FeatureMap('ZZFeatureMap', feature_dim=4, reps=1)
    # template_2 = AnsatzTemplate()
    # template_2.construct_simple_template(num_qubits=4, num_layers=2)
    # model_2 = QuantumNeuralNetwork(feature_map_2, template_2, platform='Qiskit')
    # #model_2.visualize()
    # print(model_2.num_qubits, model_2.input_dim, model_2.param_dim)
    #
    # print(structural_cost_matrix(model_1.PQC, model_2.PQC))

    qc1 = QuantumCircuit(4)
    theta1 = Parame










