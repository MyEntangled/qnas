import numpy as np
import scipy.stats
from qiskit.quantum_info import Statevector, state_fidelity

# def sample_pqc(num_samples, PQC):
#     param_dim = PQC.num_parameters
#     state_dim = 2 ** PQC.num_qubits
#     thetamin = 0
#     thetamax = 2 * np.pi
#
#     params = np.random.uniform(thetamin, thetamax, (2 * num_samples, param_dim))
#
#     result = np.zeros(num_samples)
#     id = 0
#     #initial_state = np.zeros(state_dim)
#     #initial_state[0] = 1
#     initial_state = Statevector.from_label('0'*PQC.num_qubits)
#     for i in range(0,len(params),2):
#         circuit_1 = PQC.bind_parameters(params[i])
#         #result_1 = (Operator(circuit_1).data)@initial_state
#         final_state_1 = initial_state.evolve(circuit_1)
#         circuit_2 = PQC.bind_parameters(params[i+1])
#         #result_2 = (Operator(circuit_2).data)@initial_state
#         final_state_2 = initial_state.evolve(circuit_2)
#         #result[id] = (np.abs(result_1.conj().T @ result_2))**2
#         result[id] = state_fidelity(final_state_1, final_state_2)
#         id += 1
#     return result


def sample_pqc_fidelities(num_samples, PQC):
    param_dim = PQC.num_parameters
    state_dim = 2 ** PQC.num_qubits
    thetamin = 0
    thetamax = 2 * np.pi

    params = np.random.uniform(thetamin, thetamax, (2 * num_samples, param_dim))

    id = 0

    init_state = Statevector.from_label('0'*PQC.num_qubits)
    final_states_1 = np.zeros(shape=(num_samples, state_dim), dtype=complex)
    final_states_2 = np.zeros(shape=(num_samples, state_dim), dtype=complex)

    for i in range(0,len(params),2):
        circuit_1 = PQC.bind_parameters(params[i])
        circuit_2 = PQC.bind_parameters(params[i + 1])

        final_states_1[id] = init_state.evolve(circuit_1).data
        final_states_2[id] = init_state.evolve(circuit_2).data
        id += 1

    fidelities = np.abs((final_states_1.conj() * final_states_2).sum(axis=1))**2
    return fidelities

def get_expressibility(PQC, num_samples: int, num_bins: int):
    state_dim = 2 ** PQC.num_qubits
    PQC_samples = sample_pqc_fidelities(num_samples,PQC)

    hist_PQC, bin_edges = np.histogram(a=PQC_samples,bins=num_bins,range=(0.,1.))
    hist_PQC = hist_PQC / sum(hist_PQC)

    terms = -(1-bin_edges)**(state_dim-1)
    hist_haar = np.ediff1d(terms)
    kl_div = scipy.stats.entropy(hist_PQC, hist_haar)

    return kl_div

if __name__ == '__main__':
    # from embedding import qc_embedding
    #
    # num_qubits = 4
    # MAX_OP_NODES = 20
    #
    # encoding_length = (num_qubits + 1) * MAX_OP_NODES
    # bounds = np.array([[-.2] * encoding_length, [1.0] * encoding_length])
    # x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(encoding_length)
    # qc = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x)


    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    qc = QuantumCircuit(2)
    theta = ParameterVector('theta',2)
    qc.h(0)
    qc.rz(theta[0],0)
    qc.rx(theta[1],0)

    print(qc.draw())
    expr = get_expressibility(qc, 5000, 75)
    print(expr)


    # samples_list = [10000,20000,30000,40000,50000]
    # expr_list = []
    # for num_samples in samples_list:
    #     expr = get_expressibility(qc,num_samples,75)
    #     expr_list.append(expr)
    #
    # import matplotlib.pyplot as plt
    # print(expr_list)
    # plt.plot(samples_list, expr_list)
    # plt.show()