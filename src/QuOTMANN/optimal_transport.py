import cvxpy as cp
import ot
import numpy as np

from QuOTMANN.dag_utility import create_dag
from QuOTMANN.structural_cost import structural_cost_matrix
from QuOTMANN.label_mismatch_cost import label_mismatch_cost_matrix
from QuOTMANN.gate_mass import gate_mass

from typing import Union, List

import time


def construct_cvx_prob(C_lmm, C_str, gate_mass_1, gate_mass_2, nas_cost):
    Z = cp.Variable(C_lmm.shape)
    nu = cp.Parameter()

    phi_lmm = cp.sum(cp.multiply(Z,C_lmm))
    phi_str = cp.sum(cp.multiply(Z,C_str))
    phi_nas = nas_cost * (cp.sum(gate_mass_1) + cp.sum(gate_mass_2) - 2*cp.sum(Z))

    constraints = []
    constraints.append(Z >= 0)

    for i in range(len(gate_mass_1)):
        constraints.append( cp.sum(Z[i,:]) <= gate_mass_1[i] )
    for j in range(len(gate_mass_2)):
        constraints.append( cp.sum(Z[:,j]) <= gate_mass_2[j] )

    objective_min = cp.Minimize(phi_lmm + phi_nas + nu * phi_str)
    return cp.Problem(objective_min, constraints), Z, nu

def circuit_distance(PQC_1, PQC_2, eta:float=.1, nas_cost:float=1., nu_list:Union[List,float]=0.1):
    dag_1, nx_dag_1, in_nodes_1, out_nodes_1 = create_dag(PQC_1)
    dag_2, nx_dag_2, in_nodes_2, out_nodes_2 = create_dag(PQC_2)

    op_nodes_1 = dag_1.op_nodes()
    op_nodes_2 = dag_2.op_nodes()

    d1 = 2**dag_1.num_qubits()
    d2 = 2**dag_2.num_qubits()


    ## Get individual gate mass (0 for deterministic gate)
    gate_mass_1 = np.array([gate_mass(op.name, d1) for op in op_nodes_1])
    gate_mass_2 = np.array([gate_mass(op.name, d2) for op in op_nodes_2])

    if len(gate_mass_1) == 0 and len(gate_mass_2) == 0: ## Two empty circuits
        return 0, 0

    if len(gate_mass_1) > 0 and len(gate_mass_2) == 0: ## Second circuit is empty
        num_deterministic_gates_1 = np.count_nonzero(gate_mass_1 == 0)
        if num_deterministic_gates_1 > 0:
            gate_mass_1[gate_mass_1 == 0] = eta * gate_mass_1.sum() / num_deterministic_gates_1
        return nas_cost * gate_mass_1.sum(), nas_cost

    if len(gate_mass_2) > 0 and len(gate_mass_1) == 0: ## First circuit is empty
        num_deterministic_gates_2 = np.count_nonzero(gate_mass_2 == 0)
        if num_deterministic_gates_2 > 0:
            gate_mass_2[gate_mass_2 == 0] = eta * gate_mass_2.sum() / num_deterministic_gates_2
        return nas_cost * gate_mass_2.sum(), nas_cost

    ## Gate mass of each deterministic gate = 0.1/num_deterministic_gates * sum(variational gate mass)
    num_deterministic_gates_1 = np.count_nonzero(gate_mass_1 == 0)
    if num_deterministic_gates_1 > 0:
        gate_mass_1[gate_mass_1 == 0] = eta * gate_mass_1.sum() / num_deterministic_gates_1

    num_deterministic_gates_2 = np.count_nonzero(gate_mass_2 == 0)
    if num_deterministic_gates_2 > 0:
            gate_mass_2[gate_mass_2 == 0] = eta * gate_mass_2.sum() / num_deterministic_gates_2


    # print('Mass 1', gate_mass_1)
    # print('Mass 2', gate_mass_2)

    C_lmm = label_mismatch_cost_matrix(PQC_1, PQC_2)
    # print(C_lmm)
    C_str = structural_cost_matrix(PQC_1, PQC_2)

    #print("Label mismatch cost matrix", C_lmm)
    #print("Structural cost matrix", C_str)


    prob, Z_var, nu_param = construct_cvx_prob(C_lmm, C_str, gate_mass_1, gate_mass_2, nas_cost)
    if not isinstance(nu_list, List):
        nu_list = [nu_list]

    all_dist = []
    all_distnorm = []
    for nu in nu_list:
        nu_param.value = nu
        #start = time.time()
        prob.solve()
        #print(time.time() - start)

        dist = prob.value
        # print('Z ', np.array(Z_var.value))
        # print(np.sum(Z_var.value * C_lmm))
        # print(np.sum(Z_var.value * C_str))
        # print(np.sum(gate_mass_1) + np.sum(gate_mass_2) - 2*np.sum(Z_var.value))
        all_dist.append(dist)
        all_distnorm.append(dist / (gate_mass_1.sum() + gate_mass_2.sum()))

    return all_dist, all_distnorm

def circuit_distance_POT(PQC_1, PQC_2, eta:float=.1, nas_cost:float=1., nu_list:Union[List,float]=0.1):

    dag_1, nx_dag_1, in_nodes_1, out_nodes_1 = create_dag(PQC_1)
    dag_2, nx_dag_2, in_nodes_2, out_nodes_2 = create_dag(PQC_2)

    op_nodes_1 = dag_1.op_nodes()
    op_nodes_2 = dag_2.op_nodes()

    d1 = 2**dag_1.num_qubits()
    d2 = 2**dag_2.num_qubits()


    ## Get individual gate mass (0 for deterministic gate)
    gate_mass_1 = np.array([gate_mass(op.name, d1) for op in op_nodes_1])
    gate_mass_2 = np.array([gate_mass(op.name, d2) for op in op_nodes_2])

    if len(gate_mass_1) == 0 and len(gate_mass_2) == 0: ## Two empty circuits
        return [0], [0]

    if len(gate_mass_1) > 0 and len(gate_mass_2) == 0: ## Second circuit is empty
        num_deterministic_gates_1 = np.count_nonzero(gate_mass_1 == 0)
        if num_deterministic_gates_1 > 0:
            gate_mass_1[gate_mass_1 == 0] = eta * gate_mass_1.sum() / num_deterministic_gates_1
        return [nas_cost * gate_mass_1.sum()], [nas_cost]

    if len(gate_mass_2) > 0 and len(gate_mass_1) == 0: ## First circuit is empty
        num_deterministic_gates_2 = np.count_nonzero(gate_mass_2 == 0)
        if num_deterministic_gates_2 > 0:
            gate_mass_2[gate_mass_2 == 0] = eta * gate_mass_2.sum() / num_deterministic_gates_2
        return [nas_cost * gate_mass_2.sum()], [nas_cost]

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
        return [0],[0]

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

    if not isinstance(nu_list, List):
        nu_list = [nu_list]

    for nu in nu_list:
        C = C_lmm_pad + nu*C_str_pad + C_nas_pad
        #start = time.time()
        dist = ot.emd2(y1, y2, C)
        #print(time.time() -start)
        distnorm = dist / (total_mass_1 + total_mass_2)
        all_dist.append(dist)
        all_distnorm.append(distnorm)

    return all_dist, all_distnorm

if __name__ == '__main__':

    from embedding import qc_embedding

    num_qubits = 4
    MAX_OP_NODES = 30

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = np.array([[-.2] * encoding_length, [1.0] * encoding_length])
    num_trials = 100

    count_dist_er = 0
    count_dist_norm_er = 0

    x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(encoding_length)
    qc0 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x)
    qc1 = qc0.copy()
    dist01, dist_norm01 = circuit_distance(qc0, qc1, nu_list=0)
    print(dist01, dist_norm01)

    dist01, dist_norm01 = circuit_distance_POT(qc0, qc1, nu_list=0)
    print(dist01, dist_norm01)

    num_circuits = 100
    x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(num_circuits,encoding_length)
    y = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(num_circuits,encoding_length)
    for i in range(num_circuits):
        qc0 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[i])
        qc1 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=y[i])

        dist01, dist_norm01 = circuit_distance(qc0, qc1, nu_list=0.1)
        print(f"Turn {i}: ", dist01, dist_norm01)

        dist01, dist_norm01 = circuit_distance_POT(qc0, qc1, nu_list=0.1)
        print(f"Turn {i}: ", dist01, dist_norm01)

    ## Changing order of qubits will change the distance, since C_lmm takes into account relative difference in applied qubits
    # from qiskit import QuantumCircuit
    # qc0 = QuantumCircuit(4)
    # qc0.h(0); qc0.rx(0.1,0); qc0.cry(0.2,1,2); qc0.rzz(0.3,2,3)
    # qc1 = QuantumCircuit(4)
    # qc1.h(3); qc1.rx(0.1,3); qc1.cry(0.2,2,1); qc1.rzz(0.3,1,0)
    #
    # dist01, dist_norm01 = circuit_distance(qc0, qc1, nu_list=0)
    # print(dist01, dist_norm01)

    # for k in range(num_trials):
    #     print('k =', k)
    #     x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(3, encoding_length)
    #     qc0 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[0])
    #     qc1 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[1])
    #     qc2 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[2])
    #     dist01, dist_norm01 = circuit_distance(qc0, qc1)
    #     dist01, dist_norm01 = dist01[0], dist_norm01[0]
    #
    #     dist02, dist_norm02 = circuit_distance(qc0, qc2)
    #     dist02, dist_norm02 = dist02[0], dist_norm02[0]
    #
    #     dist12, dist_norm12 = circuit_distance(qc1, qc2)
    #     dist12, dist_norm12 = dist12[0], dist_norm12[0]
    #     print(dist01, dist_norm01)
    #     print(dist02, dist_norm02)
    #     print(dist12, dist_norm12)
    #     if (dist01+dist02 < dist12) or (dist01+dist12 < dist02) or (dist02+dist12 < dist01):
    #         count_dist_er += 1
    #         print('dist', x)
    #     if (dist_norm01+dist_norm02 < dist_norm12) or (dist_norm01+dist_norm12 < dist_norm02) or (dist_norm02+dist_norm12 < dist_norm01):
    #         print('dist', x)
    #         count_dist_norm_er += 1
    # print(count_dist_er, count_dist_norm_er)


