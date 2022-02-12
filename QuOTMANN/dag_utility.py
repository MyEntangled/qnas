from qiskit.converters import circuit_to_dag

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

    return dag, nx_dag, in_nodes, out_nodes