import ot
import numpy as np

from qiskit.converters import circuit_to_dag
from itertools import groupby

from QuOTMANN.dag_utility import create_dag
from QuOTMANN.structural_cost import structural_cost_matrix
from QuOTMANN.label_mismatch_cost import label_mismatch_cost_matrix
from QuOTMANN.gate_mass import gate_mass

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES

from typing import Union, List

import time


def adjust_parametrized_gate_duplicates_mass(PQC, gate_mass):
    dag = circuit_to_dag(PQC)
    op_nodes = dag.op_nodes()
    op_nodes_dict = {node: idx for idx, node in enumerate(op_nodes)}

    exempt = []
    for q in range(PQC.num_qubits):
        nodes = []
        for node in dag.nodes_on_wire(PQC.qubits[q], only_ops=True):
            if node.name in ['rxx','ryy','rzz']:
                nodes.append((node.name, set(node.qargs), op_nodes_dict[node]))
            else:
                nodes.append((node.name, node.qargs, op_nodes_dict[node]))

        for group in groupby(nodes, key=lambda node:node[:-1]):
            name,_,dict_idx = zip(*group[1])
            dict_idx = list(dict_idx)
            if len(name) > 1:
                if name[0] in TWO_QUBIT_VARIATIONAL_GATES:
                    if dict_idx not in exempt:
                        gate_mass[list(dict_idx)] /= len(name)
                        exempt.append(dict_idx)
                        # print(exempt)
                else:
                    gate_mass[list(dict_idx)] /= len(name)
    return gate_mass


def circuit_distance_POT(PQC_1, PQC_2, eta:float=.1, nas_cost:float=1., nu_list:Union[List,float]=0.1):

    dag_1, nx_dag_1, in_nodes_1, out_nodes_1 = create_dag(PQC_1)
    dag_2, nx_dag_2, in_nodes_2, out_nodes_2 = create_dag(PQC_2)

    op_nodes_1 = dag_1.op_nodes()
    op_nodes_2 = dag_2.op_nodes()

    d1 = 2**dag_1.num_qubits()
    d2 = 2**dag_2.num_qubits()

    if not isinstance(nu_list, List):
        nu_list = [nu_list]

    ## Get individual gate mass (0 for deterministic gate)
    gate_mass_1 = np.array([gate_mass(op.name, d1) for op in op_nodes_1])
    # print(gate_mass_1)
    gate_mass_1 = adjust_parametrized_gate_duplicates_mass(PQC_1, gate_mass_1)
    # print(gate_mass_1)
    gate_mass_2 = np.array([gate_mass(op.name, d2) for op in op_nodes_2])
    # print(gate_mass_2)
    gate_mass_2 = adjust_parametrized_gate_duplicates_mass(PQC_2, gate_mass_2)
    # print(gate_mass_2)


    if len(gate_mass_1) == 0 and len(gate_mass_2) == 0: ## Two empty circuits
        return [0.]*len(nu_list), [0.]*len(nu_list)

    if len(gate_mass_1) > 0 and len(gate_mass_2) == 0: ## Second circuit is empty
        num_deterministic_gates_1 = np.count_nonzero(gate_mass_1 == 0)
        if num_deterministic_gates_1 > 0:
            gate_mass_1[gate_mass_1 == 0] = eta * gate_mass_1.sum() / num_deterministic_gates_1
        return [nas_cost * gate_mass_1.sum()]*len(nu_list), [nas_cost]*len(nu_list)

    if len(gate_mass_2) > 0 and len(gate_mass_1) == 0: ## First circuit is empty
        num_deterministic_gates_2 = np.count_nonzero(gate_mass_2 == 0)
        if num_deterministic_gates_2 > 0:
            gate_mass_2[gate_mass_2 == 0] = eta * gate_mass_2.sum() / num_deterministic_gates_2
        return [nas_cost * gate_mass_2.sum()]*len(nu_list), [nas_cost]*len(nu_list)

    ## Gate mass of each deterministic gate = 0.1/num_deterministic_gates * sum(variational gate mass)
    num_deterministic_gates_1 = np.count_nonzero(gate_mass_1 == 0)
    if num_deterministic_gates_1 > 0:
        gate_mass_1[gate_mass_1 == 0] = eta * gate_mass_1.sum() / num_deterministic_gates_1

    num_deterministic_gates_2 = np.count_nonzero(gate_mass_2 == 0)
    if num_deterministic_gates_2 > 0:
            gate_mass_2[gate_mass_2 == 0] = eta * gate_mass_2.sum() / num_deterministic_gates_2

    total_mass_1 = sum(gate_mass_1)
    total_mass_2 = sum(gate_mass_2)
    if total_mass_1 + total_mass_2 == 0.:
        return [0]*len(nu_list),[0]*len(nu_list)

    C_lmm = label_mismatch_cost_matrix(PQC_1, PQC_2)
    #start = time.time()
    C_str = structural_cost_matrix(PQC_1, PQC_2)
    #print(time.time() - start)



    y1 = np.append(gate_mass_1, total_mass_2)
    y2 = np.append(gate_mass_2, total_mass_1)

    C_lmm_pad = np.pad(C_lmm, [(0, 1), (0, 1)], mode='constant', constant_values=0)
    C_str_pad = np.pad(C_str, [(0, 1), (0, 1)], mode='constant', constant_values=0)
    C_nas_pad = np.zeros_like(C_lmm_pad)
    C_nas_pad[-1,:] = nas_cost
    C_nas_pad[:,-1] = nas_cost
    C_nas_pad[-1,-1] = 0

    all_dist = []
    all_distnorm = []


    for nu in nu_list:
        C = C_lmm_pad + nu*C_str_pad + C_nas_pad
        #start = time.time()
        dist = ot.emd2(y1, y2, C)
        #print(time.time() -start)
        distnorm = dist / (total_mass_1 + total_mass_2)
        all_dist.append(dist)
        all_distnorm.append(distnorm)

    return all_dist, all_distnorm



