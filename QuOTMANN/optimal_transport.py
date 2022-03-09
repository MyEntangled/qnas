import cvxpy as cp
import numpy as np

from QuOTMANN.dag_utility import create_dag
from QuOTMANN.structural_cost import structural_cost_matrix
from QuOTMANN.label_mismatch_cost import label_mismatch_cost_matrix
from QuOTMANN.gate_mass import gate_mass

from typing import Union, List


def construct_cvx_prob(C_lmm, C_str, gate_mass_1, gate_mass_2):
    Z = cp.Variable(C_lmm.shape)
    nu = cp.Parameter()

    phi_lmm = cp.sum(cp.multiply(Z,C_lmm))
    phi_str = cp.sum(cp.multiply(Z,C_str))
    phi_nas = cp.sum(gate_mass_1) + cp.sum(gate_mass_2) - 2*cp.sum(Z)

    constraints = []
    constraints.append(Z >= 0)

    for i in range(len(gate_mass_1)):
        constraints.append( cp.sum(Z[i,:]) <= gate_mass_1[i] )
    for j in range(len(gate_mass_2)):
        constraints.append( cp.sum(Z[:,j]) <= gate_mass_2[j] )

    objective_min = cp.Minimize(phi_lmm + phi_nas + nu * phi_str)
    return cp.Problem(objective_min, constraints), nu

def circuit_distance(PQC_1, PQC_2, nas_cost:np.double=1, nu_list:Union[List,float]=0.1):
    dag_1, nx_dag_1, in_nodes_1, out_nodes_1 = create_dag(PQC_1)
    dag_2, nx_dag_2, in_nodes_2, out_nodes_2 = create_dag(PQC_2)

    op_nodes_1 = dag_1.op_nodes()
    op_nodes_2 = dag_2.op_nodes()

    gate_property_1 = np.array([gate_mass(op.name) for op in op_nodes_1])
    gate_property_2 = np.array([gate_mass(op.name) for op in op_nodes_2])

    if len(gate_property_1) == 0 and len(gate_property_2) == 0: # Two empty circuits
        return 0,0
    if len(gate_property_1) == 0 and len(gate_property_2) > 0: # First circuit is empty
        gate_mass_2 = gate_property_2[:,0] * gate_property_2[:,1] * 2**dag_2.num_qubits()
        return nas_cost*gate_mass_2, nas_cost
    if len(gate_property_1) > 0 and len(gate_property_2) == 0: # Second circuit is empty
        gate_mass_1 = gate_property_1[:, 0] * gate_property_1[:, 1] * 2 ** dag_1.num_qubits()
        return nas_cost*gate_mass_1, nas_cost


    gate_mass_1 = gate_property_1[:,0] * gate_property_1[:,1] * 2**dag_1.num_qubits()
    gate_mass_2 = gate_property_2[:,0] * gate_property_2[:,1] * 2**dag_2.num_qubits()

    #print('Mass 1', gate_mass_1)
    #print('Mass 2', gate_mass_2)

    C_lmm = label_mismatch_cost_matrix(PQC_1, PQC_2)
    C_str = structural_cost_matrix(PQC_1, PQC_2)

    #print("Label mismatch cost matrix", C_lmm)
    #print("Structural cost matrix", C_str)

    # Z = cp.Variable((len(op_nodes_1), len(op_nodes_2)))
    #
    # phi_lmm = cp.sum(cp.multiply(Z,C_lmm))
    # phi_str = cp.sum(cp.multiply(Z,C_str))
    # phi_nas = cp.sum(gate_mass_1) + cp.sum(gate_mass_2) - 2*cp.sum(Z)
    #
    # constraints = []
    # constraints.append( Z >= 0)
    #
    # for i in range(len(gate_mass_1)):
    #     constraints.append( cp.sum(Z[i,:]) <= gate_mass_1[i] )
    # for j in range(len(gate_mass_2)):
    #     constraints.append( cp.sum(Z[:,j]) <= gate_mass_2[j] )
    #
    # objective_min = cp.Minimize(phi_lmm + phi_nas + nu_str * phi_str)
    # prob = cp.Problem(objective_min, constraints)

    prob, nu_param = construct_cvx_prob(C_lmm, C_str, gate_mass_1, gate_mass_2)
    if not isinstance(nu_list, List):
        nu_list = [nu_list]


    #print('Z ', np.array(Z.value))
    # print(np.sum(Z_value * C_lmm))
    # print(np.sum(Z_value * C_str))
    # print(np.sum(gate_mass_1) + np.sum(gate_mass_2) - 2*np.sum(Z_value))

    all_dist = []
    all_distnorm = []
    for nu in nu_list:
        #print(nu)
        nu_param.value = nu
        prob.solve()

        dist = prob.value
        all_dist.append(dist)
        all_distnorm.append(dist / (gate_mass_1.sum() + gate_mass_2.sum()))

    return all_dist, all_distnorm

if __name__ == '__main__':

    from embedding import qc_embedding

    num_qubits = 4
    MAX_OP_NODES = 10

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = np.array([[-.2] * encoding_length, [1.0] * encoding_length])
    num_trials = 100

    count_dist_er = 0
    count_dist_norm_er = 0

    for k in range(num_trials):
        print('k =', k)
        x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(3, encoding_length)
        qc0 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[0])
        qc1 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[1])
        qc2 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[2])
        dist01, dist_norm01 = circuit_distance(qc0, qc1)
        dist01, dist_norm01 = dist01[0], dist_norm01[0]

        dist02, dist_norm02 = circuit_distance(qc0, qc2)
        dist02, dist_norm02 = dist02[0], dist_norm02[0]

        dist12, dist_norm12 = circuit_distance(qc1, qc2)
        dist12, dist_norm12 = dist12[0], dist_norm12[0]
        print(dist01, dist_norm01)
        print(dist02, dist_norm02)
        print(dist12, dist_norm12)
        if (dist01+dist02 < dist12) or (dist01+dist12 < dist02) or (dist02+dist12 < dist01):
            count_dist_er += 1
            print('dist', x)
        if (dist_norm01+dist_norm02 < dist_norm12) or (dist_norm01+dist_norm12 < dist_norm02) or (dist_norm02+dist_norm12 < dist_norm01):
            print('dist', x)
            count_dist_norm_er += 1
    print(count_dist_er, count_dist_norm_er)


