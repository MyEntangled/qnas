import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.dagcircuit import DAGCircuit, DAGNode
from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, \
    SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, \
    TWO_QUBIT_VARIATIONAL_GATES, \
    ADMISSIBLE_GATES, \
    DIRECTED_GATES, \
    UNITARY, \
    OP_NODE_DICT

from qiskit.converters import circuit_to_dag, dag_to_circuit

INV_OP_NODE_DICT = {v:k for k,v in OP_NODE_DICT.items()}
INV_OP_VALUES = np.array(list(INV_OP_NODE_DICT.values()))

def transform_encoding(adj_encoding, num_qubits, MAX_OP_NODES=None):
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
    node_code = np.tanh(adj_encoding[-MAX_OP_NODES:]) # part of infolist

    ## Reconstruct the infolist containing valid op_nodes
    infolist = []
    valid_op_idx = []
    for i,code in enumerate(node_code):
        if code < 0:
            infolist.append(['none'])
        else: ## code >= 0
            valid_op_idx.append[i]
            closest_mark = INV_OP_VALUES[np.abs(INV_OP_VALUES - code).argmin()]
            infolist.append(INV_OP_NODE_DICT[closest_mark])
    #valid_op_idx = np.array(valid_op_idx)

    ## Reconstruct a valid adj_matrix
    rec_adj_matrix = -np.ones(shape=(MAX_OP_NODES+num_qubits, MAX_OP_NODES))
    iu = np.triu_indices_from(rec_adj_matrix, k=-(num_qubits-1))
    rec_adj_matrix[iu] = adj_code

    return rec_adj_matrix, infolist


def reconstruct(num_qubits, raw_adj_matrix, infolist):
    assert raw_adj_matrix.shape[0] == raw_adj_matrix.shape[1], 'adj_matrix must be square.'
    assert raw_adj_matrix.shape[0] - 2*num_qubits == len(infolist)
    raw_adj_matrix = np.exp(raw_adj_matrix)

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
    ## Nodelist
    rem_qubits = []
    nodelist = []

    ## preprocessing for input nodes
    for i in range(num_qubits):
        in_node = DAGNode(type='in', wire=qc.qubits[i])
        nodelist.append(in_node)
        rem_qubits.append([qc.qubits[i]])

    ## Iterate the adj_matrix for all op_nodes
    for c,adj_list in list(enumerate(raw_adj_matrix.T))[num_qubits:-num_qubits]: ## Iterate over some columns but keep original index
        nodename = infolist[c-num_qubits]
        print(nodename)

        if (nodename in SINGLE_QUBIT_DETERMINISTIC_GATES) or (nodename in SINGLE_QUBIT_VARIATIONAL_GATES):
            op_num_qubits = 1
        else:
            op_num_qubits = 2

        feasible = np.ones(len(adj_list))
        feasible[c:] = 0
        for i in range(c):
            if len(rem_qubits[i]) == 0:
                feasible[i] = 0
        fea_adj_list = adj_list * feasible
        print(fea_adj_list)
        print(rem_qubits)

        if op_num_qubits == 1:
            parents_ids = [np.argmax(fea_adj_list)] # only 1 parent
            parents_node = [nodelist[idx] for idx in parents_ids]
            num_edges_to_parents = np.array([1])
        else: # i.e. op_num_qubits == 2
            two_highest_idx = np.argpartition(-fea_adj_list,kth=1)
            highest_idx = two_highest_idx[0]
            second_highest_idx = two_highest_idx[1]
            highest = fea_adj_list[highest_idx]
            second_highest = fea_adj_list[second_highest_idx]
            print(highest_idx, second_highest_idx)

            if (highest < 2*second_highest) or (len(rem_qubits[highest_idx]) < 2): # two distinct parents then
                parents_ids = [highest_idx, second_highest_idx]
                parents_node = [nodelist[idx] for idx in parents_ids]
                num_edges_to_parents = np.array([1,1])
            else:
                parents_ids  = [highest_idx]
                parents_node = [nodelist[idx] for idx in parents_ids]
                num_edges_to_parents = np.array([2])

        print('parents_ids: ', parents_ids)
        print('num_edges_to_parents: ', num_edges_to_parents)
        qargs = []
        for j, parent in enumerate(parents_node):
            num_edges = num_edges_to_parents[j]
            # print('feed to qargs', rem_qubits[parents_ids[j]][:num_edges])
            qargs += (rem_qubits[parents_ids[j]][:num_edges])
            rem_qubits[parents_ids[j]] = rem_qubits[parents_ids[j]][num_edges:]

        op_node = DAGNode(type='op', op=UNITARY[nodename], qargs=qargs, cargs=[])
        op_node.op.name = nodename

        nodelist.append(op_node)
        rem_qubits.append(qargs)

        #print(nodename)
        print('qargs: ', qargs)
        ## Insert gate to qc
        if nodename in SINGLE_QUBIT_DETERMINISTIC_GATES + TWO_QUBIT_DETERMINISTIC_GATES:
            qc.append(UNITARY[nodename](), qargs=qargs, cargs=[])
        else:
            theta.resize(len(theta) + 1)
            qc.append(UNITARY[nodename](theta[-1]), qargs=qargs, cargs=[])

    ## Iterate adj_matrix for out_nodes
    for i,rem_qubit in reversed(list(enumerate(rem_qubits))): ## Iterate rem_qubits in reverse order but keep the original index
        for qubit in rem_qubit:
            out_node = DAGNode(type='out',wire=qubit)
            nodelist.append(out_node)

    return nodelist, qc, circuit_to_dag(qc)


if __name__ == '__main__':
    #np.random.seed(0)
    num_qubits = 4
    infolist = ['x','rxx','crx','y','cry','ryy','crz','rx','z']

    dim = 2*num_qubits + len(infolist)

    for i in range(1):
        adj_matrix = np.random.standard_normal((dim, dim))
        print(adj_matrix)
        nodelist, qc, dag = reconstruct(num_qubits=num_qubits,raw_adj_matrix=adj_matrix,infolist=infolist)
        print(qc.draw())





