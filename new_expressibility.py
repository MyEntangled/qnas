import numpy as np
from qiskit.quantum_info import Operator
from scipy.special import kl_div

from utility.ansatz_template import AnsatzTemplate
from utility.data_encoding import FeatureMap
from utility.quantum_nn import QuantumNeuralNetwork


def sample_haar(dim, num_samples):
    def get_F_from_cdf(p):
        return 1 - (1 - p) ** (1 / (dim - 1))

    cum_prob = np.random.uniform(0, 1, num_samples)
    return np.vectorize(get_F_from_cdf)(cum_prob)

def sample_pqc(num_samples, PQC):
    param_dim = PQC.num_parameters
    state_dim = 2 ** PQC.num_qubits
    thetamin = 0
    thetamax = 2 * np.pi
    params = np.random.uniform(thetamin, thetamax, (2 * num_samples, param_dim))
    result = np.zeros(num_samples)
    id = 0
    initial_state = np.zeros(state_dim)
    initial_state[0] = 1
    for i in range(0,len(params),2):
        circuit_1 = PQC.bind_parameters(params[i])
        result_1 = (Operator(circuit_1).data)@initial_state
        circuit_2 = PQC.bind_parameters(params[i+1])
        result_2 = (Operator(circuit_2).data)@initial_state
        result[id] = (np.abs(result_1.conj().T @ result_2))**2
        id += 1
    return result

def get_expressibility(PQC, num_samples: int, indices_len: int ):
    state_dim = 2 ** PQC.num_qubits
    indices = np.linspace(0,1,indices_len)
    unit_indices = float(1/(indices_len**2))
    PQC_indices = np.zeros(indices_len)
    Haar_indices = np.zeros(indices_len)

    PQC_samples = sample_pqc(num_samples,PQC)
    Haar_samples = sample_haar(state_dim,num_samples)

    for i in range(num_samples):
        Haar_indices[int(Haar_samples[i]/indices[1])] += unit_indices
        PQC_indices[int(PQC_samples[i]/indices[1])] += unit_indices
    print(PQC_samples)
    print(Haar_samples)
    return kl_div(Haar_indices,PQC_indices)
if __name__ == '__main__':

    feature_map = FeatureMap('PauliFeatureMap', 4, 1)

    template = AnsatzTemplate()
    template.construct_simple_template(4, 1)

    model = QuantumNeuralNetwork(feature_map, template)
    model.visualize()

    print(model.num_qubits, model.input_dim, model.param_dim)

    from embedding import qc_embedding

    num_qubits = 4
    MAX_OP_NODES = 6

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = np.array([[-.2] * encoding_length, [1.0] * encoding_length])
    x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(encoding_length)
    qc = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x)

    qc.draw()
    express = get_expressibility(qc,10000,75)
    print(express)