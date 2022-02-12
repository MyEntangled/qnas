import cvxpy as cp
import numpy as np

from QuOTMANN.dag_utility import create_dag
from QuOTMANN.structural_cost import structural_cost_matrix
from QuOTMANN.label_mismatch_cost import label_mismatch_cost_matrix
from QuOTMANN.gate_mass import gate_mass


def construct_cvx_prob(C_lmm, C_str, gate_mass_1, gate_mass_2, nu_str):
    Z = cp.Variable(C_lmm.shape)

    phi_lmm = cp.sum(cp.multiply(Z,C_lmm))
    phi_str = cp.sum(cp.multiply(Z,C_str))
    phi_nas = cp.sum(gate_mass_1) + cp.sum(gate_mass_2) - 2*cp.sum(Z)

    constraints = []
    constraints.append(Z >= 0)

    for i in range(len(gate_mass_1)):
        constraints.append( cp.sum(Z[i,:]) <= gate_mass_1[i] )
    for j in range(len(gate_mass_2)):
        constraints.append( cp.sum(Z[:,j]) <= gate_mass_2[j] )

    objective_min = cp.Minimize(phi_lmm + phi_nas + nu_str * phi_str)
    return cp.Problem(objective_min, constraints)

def circuit_distance(PQC_1, PQC_2, nas_cost:np.double=1, nu_str:np.double=0.05):
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

    prob = construct_cvx_prob(C_lmm, C_str, gate_mass_1, gate_mass_2, nu_str)
    prob.solve()

    #print('Z ', np.array(Z.value))
    # print(np.sum(Z_value * C_lmm))
    # print(np.sum(Z_value * C_str))
    # print(np.sum(gate_mass_1) + np.sum(gate_mass_2) - 2*np.sum(Z_value))

    dist = prob.value
    dist_norm = dist / (gate_mass_1.sum() + gate_mass_2.sum())
    return dist, dist_norm

if __name__ == '__main__':

    # import qc_embedding

    # num_qubits = 4
    # MAX_OP_NODES = 10
    #
    # encoding_length = (num_qubits + 1) * MAX_OP_NODES
    # bounds = np.array([[-.2] * encoding_length, [1.0] * encoding_length])
    #
    # alpha = 1.
    # beta = .1
    #
    # num_trials = 100
    # num_samples = 20
    #
    # #x2 = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(num_samples,encoding_length)
    # count = 0
    # for k in range(num_trials):
    #     x1 = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(num_samples, encoding_length)
    #     dist = np.zeros(shape=(x1.shape[0], x1.shape[0]))
    #     dist_norm = np.zeros(shape=(x1.shape[0], x1.shape[0]))
    #     for i in range(dist.shape[0]):
    #         for j in range(dist.shape[1]):
    #             qc1 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x1[i])
    #             qc2 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x1[j])
    #             dist[i, j], dist_norm[i,j] = program(qc1, qc2)
    #
    #     print(k, dist, dist_norm)
    #     gram = alpha * np.exp(-beta * dist)
    #     eigvals = np.linalg.eigvalsh(gram)
    #
    #     gram_norm = alpha * np.exp(-beta * dist_norm)
    #     eigvals_norm = np.linalg.eigvalsh(gram_norm)
    #     if (eigvals > -1e-4).all() and (eigvals_norm > -1e-4).all():
    #         count += 1
    #
    # print(count/num_trials)

    from embedding import qc_embedding

    num_qubits = 4
    MAX_OP_NODES = 10

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = np.array([[-.2] * encoding_length, [1.0] * encoding_length])
    num_trials = 100

    count_dist_er = 0
    count_dist_norm_er = 0

    for k in range(num_trials):
        print(k)
        x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(3, encoding_length)
        qc0 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[0])
        qc1 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[1])
        qc2 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x[2])
        dist01, dist_norm01 = circuit_distance(qc0, qc1)
        dist02, dist_norm02 = circuit_distance(qc0, qc2)
        dist12, dist_norm12 = circuit_distance(qc1, qc2)
        if (dist01+dist02 < dist12) or (dist01+dist12 < dist02) or (dist02+dist12 < dist01):
            count_dist_er += 1
            print('dist', x)
        if (dist_norm01+dist_norm02 < dist_norm12) or (dist_norm01+dist_norm12 < dist_norm02) or (dist_norm02+dist_norm12 < dist_norm01):
            print('dist', x)
            count_dist_norm_er += 1
    print(count_dist_er, count_dist_norm_er)


