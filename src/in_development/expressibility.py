import numpy as np
from scipy.special import digamma
import matplotlib.pyplot as plt
import time
from typing import List, Union
import multiprocessing as mp

from qiskit.quantum_info import Statevector

def sample_haar(dim, num_samples):
    """

    :param dim:
    :param num_samples:
    :return:
    """

    def get_F_from_cdf(p):
        return 1 - (1 - p) ** (1 / (dim - 1))

    cum_prob = np.random.uniform(0, 1, num_samples)
    return np.vectorize(get_F_from_cdf)(cum_prob)


def whitening(s1, s2):
    """

    :param s1:
    :param s2:
    :return:
    """
    n = len(s1)
    m = len(s2)
    mu = (np.sum(s1) + np.sum(s2)) / (n + m)
    C = (1. / (n + m - 1)) * (np.sum((s1 - mu) ** 2) + np.sum((s2 - mu) ** 2))

    s1 = (s1 - mu) / np.sqrt(C)
    s2 = (s2 - mu) / np.sqrt(C)

    return s1, s2


def KL_knn_estimator(s1, s2, knn=1):
    """ KL-Divergence estimator using brute-force (numpy) k-NN
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        knn: Number of neighbours considered in 1-th index (default 1)
        return: estimated D(P|Q)
    """
    s1, s2 = whitening(s1, s2)
    n, m = len(s1), len(s2)
    D = np.log(m / (n - 1))
    d = 1

    temp_s1 = np.array([s1])
    temp_s2 = np.array([s2])

    distance_s1 = abs(temp_s1.T - temp_s1)
    distance_s2 = abs(temp_s1.T - temp_s2)

    for idx, p1 in enumerate(s1):

        distance_s1[idx] = np.partition(distance_s1[idx], knn)
        distance_s2[idx] = np.partition(distance_s2[idx], knn - 1)

        epsilon = max(distance_s1[idx][knn], distance_s2[idx][knn - 1])

        if epsilon == distance_s1[idx][knn]:
            l = knn
            k = np.sum(distance_s2[idx] <= epsilon)

        else:  # epsilon == distance_s2[idx][knn-1]
            l = np.sum(distance_s1[idx] <= epsilon) - 1
            k = knn

        distance_s2[idx] = np.partition(distance_s2[idx], k - 1)
        distance_s1[idx] = np.partition(distance_s1[idx], l)

        nu = distance_s2[idx, k - 1]  # -1 because 'p1' is not in 's2'
        rho = distance_s1[idx, l]

        D += (d / n) * np.log(nu / (rho + 10e-12)) + (1. / n) * (digamma(l) - digamma(k))
    return D


# def compute_expressibility(template, wires, num_samples, num_iterations):
#     '''
#     Return expressibility score of the template PQC
#     :param template: predefined PQC
#     :param wires: number of qubits
#     :param num_samples: number of iterations
#     :return: KL divergence between Haar distribution and empirical fidelity distribution
#     '''
#
#     dev = qml.device('default.qubit', wires=wires)
#
#     state_dim = 2 ** wires
#     param_dim = wires * 4
#
#     s = 0
#
#     for iteration in range(num_iterations):
#
#         PQC_samples = []
#
#         ## This sampling part is extremely time-consuming.
#         ## Fix: Adopt qiskit.evolve + multiprocessing
#         for sample in range(num_samples):
#             param = np.random.uniform(0, 2 * np.pi, param_dim)
#             state_1 = find_output_state(template, dev, wires, param)
#
#             param = np.random.uniform(0, 2 * np.pi, param_dim)
#             state_2 = find_output_state(template, dev, wires, param)
#
#             F = abs(state_1.conj().T @ state_2) ** 2
#             PQC_samples.append(F)
#
#         PQC_samples = np.array(PQC_samples)
#         haar_samples = sample_haar(state_dim, num_samples)
#
#         s += KL_knn_estimator(PQC_samples, haar_samples, knn=1)
#
#     mean_expr = s / num_iterations
#
#     return min(mean_expr, state_dim)


def _get_expr(inds, params, PQC, results):
    for i, param in zip(inds, params):
        # circuit_ = model.PQC.bind_parameters(_compose_param_dict(model,param))
        #param_dict = compose_param_dict(model.PQC.parameters, param)
        #circuit_ = model.PQC.bind_parameters(param_dict)
        circuit_ = PQC.bind_parameters(param)

        result = Statevector.from_label('0' * PQC.num_qubits).evolve(circuit_)

        start = 2 * i * (2**PQC.num_qubits)
        end = (2 * i + 2) * (2**PQC.num_qubits)

        results[start:end:2] = np.real(result.data)
        results[start + 1:end:2] = np.imag(result.data)

        # print(result.data)
        # print(results[start:end])


def get_expressibility(PQC, num_samples: int, num_iterations: Union[int, List, np.ndarray] = 1):

    param_dim = PQC.num_parameters
    state_dim = 2 ** PQC.num_qubits
    thetamin = 0
    thetamax = 2 * np.pi

    if isinstance(num_iterations, int):
        iter_list = np.array([num_iterations])
    else:
        iter_list = np.array(num_iterations)

    num_iterations = max(iter_list)

    kl_div = np.zeros(num_iterations)

    num_processes = mp.cpu_count()

    for n in range(num_iterations):

        params = np.random.uniform(thetamin, thetamax, (2 * num_samples, param_dim))

        # multiprocessing preparation
        indices = prepare_multiprocesses(2 * num_samples, num_processes)
        results = mp.Array('d', (2 * 2 * num_samples * state_dim))

        processes = [mp.Process(target=_get_expr, args=(inds, params[inds], PQC, results))
                     for inds in indices]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        ## Reformat the multithread array into an array with complex values at half of the original size
        results = np.ctypeslib.as_array(results.get_obj()).view(np.complex128)

        PQC_samples = np.zeros(num_samples)

        for i in range(num_samples):
            start_bra = i * state_dim
            end_bra = (i + 1) * state_dim

            start_ket = (i + num_samples) * state_dim
            end_ket = (i + 1 + num_samples) * state_dim

            PQC_samples[i] = abs(results[start_bra:end_bra].conj().T @ results[start_ket:end_ket]) ** 2

        haar_samples = sample_haar(state_dim, num_samples)

        kl_div[n] = min(KL_knn_estimator(PQC_samples, haar_samples, knn=1), state_dim)
    mean_expr = np.zeros(len(iter_list))

    for i, num_iter in enumerate(iter_list):
        mean_expr[i] = sum(kl_div[:num_iter]) / num_iter

    if len(mean_expr) == 1:
        mean_expr = mean_expr[0]

    return mean_expr


if __name__ == '__main__':

    # feature_map = FeatureMap('PauliFeatureMap', 4, 1)
    #
    # template = AnsatzTemplate()
    # template.construct_simple_template(4, 1)
    #
    # model = QuantumNeuralNetwork(feature_map, template, platform='Qiskit')
    # # model.visualize()
    #
    # print(model.num_qubits, model.input_dim, model.param_dim)

    from embedding import qc_embedding

    num_qubits = 4
    MAX_OP_NODES = 6

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = np.array([[-.2] * encoding_length, [1.0] * encoding_length])
    x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(encoding_length)
    qc = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x)

    new_expr = []

    start_time = time.time()

    sample_range = range(2000,10001,2000)

    for sample in sample_range:
        new_expr.append(get_expressibility(qc, sample, 10))

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.plot(sample_range, new_expr, label='new method')
    # plt.plot(range(step,iterations,step), old_expr, label='used method')
    plt.xlabel("Number of fidelity samples")
    plt.ylabel("Average expressibility")
    plt.show()

    print(new_expr)
