import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.visualization import dag_drawer
import qiskit.circuit.library as library
import networkx as nx
from qiskit.circuit import Parameter, ParameterVector

from QuOTMANN import optimal_transport
from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, \
    SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, \
    TWO_QUBIT_VARIATIONAL_GATES, \
    ADMISSIBLE_GATES, \
    DIRECTED_GATES, \
    UNITARY, \
    OP_NODE_DICT

INV_OP_NODE_DICT = {v:k for k,v in OP_NODE_DICT.items()}
OP_VALUES = np.array(list(OP_NODE_DICT.values()))

'''
https://arxiv.org/pdf/2007.04965.pdf
'''

# def topological_sort_grouped(G):
#     '''
#     Return a list of list which contains operations in the same topological layer.
#     :param G:
#     :return:
#     '''
#     indegree_map = {v: d for v, d in G.in_degree() if d > 0}
#     zero_indegree = [v for v, d in G.in_degree() if d == 0]
#     while zero_indegree:
#         yield zero_indegree
#         new_zero_indegree = []
#         for v in zero_indegree:
#             for _, child in G.edges(v):
#                 indegree_map[child] -= 1
#                 if not indegree_map[child]:
#                     new_zero_indegree.append(child)
#         zero_indegree = new_zero_indegree

def circuit_to_vector(qc, MAX_OP_NODES:int = None):

    dag = circuit_to_dag(qc)
    dag_nx = dag.to_networkx()

    # Compute a topological node list and corresponding adjacency matrix
    #nodelist = [node for layer in list(topological_sort_grouped(dag_nx)) for node in layer] ## DONT USE THIS

    allnodelist = list(dag.topological_nodes())
    in_nodes_ids = []
    in_nodes = []
    op_nodes_ids = []
    op_nodes = []
    for i,node in enumerate(allnodelist):
        if node.type == 'in':
            in_nodes_ids.append(i)
            in_nodes.append(node)
        elif node.type == 'op':
            op_nodes_ids.append(i)
            op_nodes.append(node)
    in_nodes_ids = np.array(in_nodes_ids)
    op_nodes_ids = np.array(op_nodes_ids)

    ## Create adj_matrix
    all_adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(dag_nx, allnodelist).toarray()
    op_adj_matrix = all_adj_matrix[op_nodes_ids[:, None], op_nodes_ids]
    in_op_adj_matrix = all_adj_matrix[in_nodes_ids[:, None], op_nodes_ids]

    adj_matrix = - np.ones(shape=(len(in_nodes) + len(op_nodes), len(op_nodes)))
    adj_matrix[len(in_nodes): , :] = op_adj_matrix
    adj_matrix[:len(in_nodes) , :] = in_op_adj_matrix

    ## Create adj_encoding from adj_matrix
    adj_encoding, padded_adj_matrix = _encode_adj(adj_matrix=adj_matrix, nodelist=op_nodes, num_qubits=qc.num_qubits, MAX_OP_NODES=MAX_OP_NODES)

    return padded_adj_matrix, adj_encoding

# def _node_info(node):
#     nodetype = node.type
#     if nodetype in ['in', 'out']:
#         qubit = node.wire.index
#         info = '_'.join([nodetype, str(qubit)])
#     else: # i.e. nodetype == 'op'
#         info = node.op.name
#     return info

def _encode_adj(adj_matrix, nodelist, num_qubits, MAX_OP_NODES:int = None):
    '''

    :param adj_matrix:
    :param nodelist:
    :param MAX_OP_NODES:
    :return:
    '''

    if MAX_OP_NODES == None:
        MAX_OP_NODES = len(nodelist)
    else:
        assert len(nodelist) <= MAX_OP_NODES, "nodelist has more nodes than MAX_OP_NODES"

    infolist = [node.op.name for node in nodelist]

    ADJ_ENCODING_LENGTH = int((MAX_OP_NODES**2 + (2*num_qubits-1)*MAX_OP_NODES) // 2)
    ENCODING_LENGTH = ADJ_ENCODING_LENGTH + MAX_OP_NODES
    adj_encoding = np.zeros(ENCODING_LENGTH)

    padded_adj_matrix = - np.ones(shape=(MAX_OP_NODES+num_qubits, MAX_OP_NODES))
    padded_adj_matrix[0:adj_matrix.shape[0], 0:adj_matrix.shape[1]] = adj_matrix

    ## The first part contains q-upper diagonal entries of adj matrix (i.e. excluded q (main and low) diagonals)
    iu = np.triu_indices_from(padded_adj_matrix, k=-(num_qubits - 1))
    adj_encoding[:ADJ_ENCODING_LENGTH] = padded_adj_matrix[iu]

    ## The second part is the coded nodelist
    for i in range(len(infolist)):
        adj_encoding[ADJ_ENCODING_LENGTH + i] = (OP_NODE_DICT[infolist[i]])

    print(infolist)
    return adj_encoding, padded_adj_matrix

def _enc_to_adjmatrix(adj_encoding, num_qubits, MAX_OP_NODES:int=None):
    '''
    Compute the adj_matrix and infolist from the adj_encoding.
    The desired shape(adj_matrix) = (N+q,N) and len(infolist) = N
    Note that len(adj_encoding) = N(N+q) - N(N+1)/2 + N = (N^2 + (2q+1)N)/2,
    where N is the (maximum) number of operations and q number of qubits.

    :param adj_encoding:
    :param MAX_N_VERTICES:
    :return:
    '''
    ## Solve N from the equation l = N^2/2 + (2q+1)N/2
    if MAX_OP_NODES is None:
        MAX_OP_NODES = int( (-(2*num_qubits+1)+np.sqrt((2*num_qubits+1)**2 + 8*len(adj_encoding))) // 2 )
    else:
        assert MAX_OP_NODES == int( (-(2*num_qubits+1)+np.sqrt((2*num_qubits+1)**2 + 8*len(adj_encoding))) // 2 ), 'MAX_OP_NODES is inconsistent with len(adj_encoding).'

    adj_code = adj_encoding[:-MAX_OP_NODES] # part of adj
    #node_code = np.tanh(adj_encoding[-MAX_OP_NODES:]) # part of infolist
    node_code = adj_encoding[-MAX_OP_NODES:]

    ## Reconstruct the infolist containing valid op_nodes
    infolist = []
    for i,code in enumerate(node_code):
        if code <= 0:
            infolist.append('none')
        else: ## code > 0
            closest_mark = OP_VALUES[np.abs(OP_VALUES - code).argmin()]
            infolist.append(INV_OP_NODE_DICT[closest_mark])

    ## Reconstruct a valid adj_matrix
    rec_adj_matrix = - np.ones(shape=(MAX_OP_NODES+num_qubits, MAX_OP_NODES))
    iu = np.triu_indices_from(rec_adj_matrix, k=-(num_qubits-1))
    rec_adj_matrix[iu] = adj_code

    return rec_adj_matrix, infolist

def enc_to_qc(num_qubits, adj_encoding, MAX_OP_NODES:int=None):
    #print(adj_encoding)
    raw_adj_matrix, infolist = _enc_to_adjmatrix(adj_encoding=adj_encoding,num_qubits=num_qubits,MAX_OP_NODES=MAX_OP_NODES)

    assert raw_adj_matrix.shape[0] == raw_adj_matrix.shape[1] + num_qubits, 'adj_matrix should have shape (N+q,N)'
    assert raw_adj_matrix.shape[1] == len(infolist)

    #raw_adj_matrix = np.exp(raw_adj_matrix)
    print(raw_adj_matrix)
    print(infolist)

    ## Initialize circuit
    qc = QuantumCircuit(num_qubits)
    theta = ParameterVector('theta',0)

    ## Initialize dagcircuit
    dagcircuit = DAGCircuit()
    dagcircuit.add_qubits(qc.qubits)
    dagcircuit.add_clbits(qc.clbits)
    for register in qc.qregs:
        dagcircuit.add_qreg(register)
    for register in qc.cregs:
        dagcircuit.add_creg(register)

    ## List of list of remaining (available) qubits for use
    rem_qubits = []
    nodelist = []

    ## preprocessing for input nodes
    for i in range(num_qubits):
        in_node = DAGNode(type='in', wire=qc.qubits[i])
        nodelist.append(in_node)
        rem_qubits.append([qc.qubits[i]])

    ## Iterate the adj_matrix for all op_nodes
    for c,adj_list in list(enumerate(raw_adj_matrix.T)): ## Iterate over some columns but keep original index
        nodename = infolist[c]
        print(nodename)

        if (nodename in SINGLE_QUBIT_DETERMINISTIC_GATES) or (nodename in SINGLE_QUBIT_VARIATIONAL_GATES):
            op_num_qubits = 1
        elif (nodename in TWO_QUBIT_DETERMINISTIC_GATES) or (nodename in TWO_QUBIT_VARIATIONAL_GATES):
            op_num_qubits = 2
        else:
            op_num_qubits = 0

        feasible = np.ones(len(adj_list))
        feasible[c+num_qubits:] = 0
        for i in range(c+num_qubits):
            if len(rem_qubits[i]) == 0:
                feasible[i] = 0
        fea_adj_list = adj_list * feasible
        print(fea_adj_list)
        print(rem_qubits)

        parents_node = []
        if op_num_qubits == 1:
            parents_ids = [np.argmax(fea_adj_list)] # only 1 parent
            parents_node = [nodelist[idx] for idx in parents_ids]
            num_edges_to_parents = np.array([1])
        elif op_num_qubits == 2:
            two_highest_idx = np.argpartition(-fea_adj_list,kth=1)
            highest_idx = two_highest_idx[0]
            second_highest_idx = two_highest_idx[1]
            highest = fea_adj_list[highest_idx]
            second_highest = fea_adj_list[second_highest_idx]

            if (highest < 2*second_highest) or (len(rem_qubits[highest_idx]) < 2): # two distinct parents then
                parents_ids = [highest_idx, second_highest_idx]
                #print(parents_ids)
                parents_node = [nodelist[idx] for idx in parents_ids]
                num_edges_to_parents = np.array([1,1])
            else:
                parents_ids = [highest_idx]
                #print(parents_ids)
                #print(len(nodelist))
                parents_node = [nodelist[idx] for idx in parents_ids]
                num_edges_to_parents = np.array([2])
        else: ## null node, i.e. op_num_qubits == 0
            pass

        print(parents_node)
        if op_num_qubits != 0:
            #print('parents_ids: ', parents_ids)
            #print('num_edges_to_parents: ', num_edges_to_parents)

            qargs = []
            for j, parent in enumerate(parents_node):
                num_edges = num_edges_to_parents[j]
                # print('feed to qargs', rem_qubits[parents_ids[j]][:num_edges])
                #print(parents_ids[j])
                qargs += rem_qubits[parents_ids[j]][:num_edges]
                rem_qubits[parents_ids[j]] = rem_qubits[parents_ids[j]][num_edges:]

            op_node = DAGNode(type='op', op=UNITARY[nodename], qargs=qargs, cargs=[])
            op_node.op.name = nodename

            nodelist.append(op_node)
            rem_qubits.append(qargs)

            print('qargs: ', qargs)
            ## Insert gate to qc
            if nodename in SINGLE_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_DETERMINISTIC_GATES:
                qc.append(UNITARY[nodename](), qargs=qargs, cargs=[])
            else:
                theta.resize(len(theta) + 1)
                qc.append(UNITARY[nodename](theta[-1]), qargs=qargs, cargs=[])

        else: # valid gates
            nodelist.append([])
            rem_qubits.append([])

    ## Iterate adj_matrix for out_nodes
    for i,rem_qubit in reversed(list(enumerate(rem_qubits))): ## Iterate rem_qubits in reverse order but keep the original index
        for qubit in rem_qubit:
            out_node = DAGNode(type='out',wire=qubit)
            nodelist.append(out_node)

    return nodelist, qc, circuit_to_dag(qc)


# def _decode_adj(adj_encoding, MAX_OP_NODES:int=None):
#     if MAX_OP_NODES == None:
#         MAX_OP_NODES = int((-1 + np.sqrt(1 + 8 * len(adj_encoding))) // 2)
#     else:
#         assert MAX_OP_NODES == (
#                     -1 + np.sqrt(1 + 8 * len(adj_encoding))) // 2, "MAX_OP_NODES is incompatible with adj_code"
#
#     true_code = adj_encoding[:-MAX_OP_NODES]
#     node_code = adj_encoding[-MAX_OP_NODES:]
#
#     infolist = []
#     for code in node_code:
#         if code != -1:
#             infolist.append(INV_NODE_DICT[code])
#
#     ## Reconstruct adj_matrix
#     rec_padded_adj_matrix = -np.ones(shape=(MAX_OP_NODES, MAX_OP_NODES))
#     iu = np.triu_indices(MAX_OP_NODES, k=1)
#     rec_padded_adj_matrix[iu] = true_code
#
#     # Fill out 0 to subdiagonal entries (diagonal included)
#     il = np.tril_indices(MAX_OP_NODES, k=0)
#     rec_padded_adj_matrix[il] = 0
#     rec_padded_adj_matrix[len(infolist):, :] = -1
#
#     rec_adj_matrix = rec_padded_adj_matrix[:len(infolist), :len(infolist)]
#
#     return rec_adj_matrix, infolist

# def adjmatrix_to_qc(adj_matrix, infolist):
#
#     ## Compute the number of qubits and initialize the circuit
#     assert len([nodename for nodename in infolist if 'in' in nodename]) == len([nodename for nodename in infolist if 'out' in nodename])
#     n_qubits = len([nodename for nodename in infolist if 'in' in nodename])
#     adj_matrix = adj_matrix.astype(int)
#
#     qc = QuantumCircuit(n_qubits)
#
#     theta = ParameterVector('theta',0)
#
#     dagcircuit = DAGCircuit()
#     dagcircuit.add_qubits(qc.qubits)
#     dagcircuit.add_clbits(qc.clbits)
#     for register in qc.qregs:
#         dagcircuit.add_qreg(register)
#     for register in qc.cregs:
#         dagcircuit.add_creg(register)
#
#     '''
#     for instruction, qargs, cargs in qc.data:
#         dagcircuit.apply_operation_back(instruction.copy(), qargs, cargs)
#     dagcircuit.draw()
#     '''
#
#     G = nx.convert_matrix.from_numpy_matrix(adj_matrix)
#     node_labeling = {i: infolist[i] for i in G.nodes()}
#
#     nodelist = []
#     rem_qubits = [] #, list of list of remaining qubits going out from nodes.
#
#     for i,nodename in enumerate(infolist):
#         #print(i,nodename)
#         if 'in' in nodename:
#             qubit_idx = int(nodename.split('_')[1])
#             rem_qubits.append([qc.qubits[qubit_idx]])
#
#             in_node = DAGNode(type='in',wire=qc.qubits[qubit_idx])
#             nodelist.append(in_node)
#         elif 'out' in nodename:
#             qubit_idx = int(nodename.split('_')[1])
#             rem_qubits.append([])
#
#             # adj_list = adj_matrix[:,i]
#             # parents_ids, = np.where(adj_list > 0)
#             # parents_node = [nodelist[idx] for idx in parents_ids]
#             # num_edges_to_parents = adj_list[parents_ids]
#
#             out_node = DAGNode(type='out',wire=qc.qubits[qubit_idx])
#             nodelist.append(out_node)
#         else:
#             adj_list = adj_matrix[:,i]
#             parents_ids, = np.where(adj_list > 0)
#             parents_node = [nodelist[idx] for idx in parents_ids]
#             num_edges_to_parents = adj_list[parents_ids]
#
#             #print(rem_qubits)
#             #print(num_edges_to_parents)
#
#             qargs = []
#             for j,parent in enumerate(parents_node):
#                 num_edges = num_edges_to_parents[j]
#
#                 #print('feed to qargs', rem_qubits[parents_ids[j]][:num_edges])
#                 qargs += (rem_qubits[parents_ids[j]][:num_edges])
#                 rem_qubits[parents_ids[j]] = rem_qubits[parents_ids[j]][num_edges:]
#             #print(qargs)
#             op_node = DAGNode(type='op',op=UNITARY[nodename],
#                               qargs=qargs,cargs=[])
#             op_node.op.name = nodename
#
#             nodelist.append(op_node)
#             rem_qubits.append(qargs)
#
#             try:
#                 qc.append(UNITARY[nodename](), qargs=qargs, cargs=[])
#             except:
#                 theta.resize(len(theta)+1)
#                 qc.append(UNITARY[nodename](theta[-1]), qargs=qargs, cargs=[])
#
#             #dagcircuit.apply_operation_back(UNITARY[nodename], qargs=qargs, cargs=[])
#
#     return nodelist, qc, circuit_to_dag(qc)

if __name__ == '__main__':
    qc = QuantumCircuit(4)
    qc.cx(1, 0)
    qc.cx(2, 3)
    qc.cx(1, 2)
    theta = Parameter('θ')
    qc.rxx(theta, 2, 1)
    qc.cx(1, 0)
    qc.cx(2, 3)

    dag = circuit_to_dag(qc)

    print(qc.draw())
    #print(dag.draw())
    adj_matrix, encoding = circuit_to_vector(qc, MAX_OP_NODES=8)

    print(adj_matrix)
    print(encoding)

    print('-----------------')

    #synthetic_encoding = np.random.uniform(-2,2,size=encoding.shape)
    #synthetic_encoding[-6:] = np.tanh(synthetic_encoding[-6:])
    rec_nodelist, rec_qc, dag = enc_to_qc(num_qubits=4,adj_encoding=encoding, MAX_OP_NODES=8)
    print(rec_qc.draw())
    print(rec_nodelist)

    rec_adjmatrix, rec_encoding = circuit_to_vector(qc, MAX_OP_NODES=8)
    print(rec_adjmatrix)

    print('Is rec_adjmatrix correct: ', (adj_matrix==rec_adjmatrix).all())
    print('-----------------')


    print("Are two graphs isomorphic: ", nx.algorithms.isomorphism.is_isomorphic(dag.to_networkx(), circuit_to_dag(rec_qc).to_networkx()))
    print("QuOTMANN distance: ", optimal_transport.circuit_distance(qc, rec_qc))
    #
    # print('-----------------')