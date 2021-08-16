import numpy as np
from scipy.special import logsumexp
from typing import Union, List

from utility.quantum_nn import QuantumNeuralNetwork
from utility.data_encoding import FeatureMap
from utility.ansatz_template import AnsatzTemplate

import time
import matplotlib.pyplot as plt


def compute_data_effdim(model,
                        fisher_type: str,
                        inputs_: Union[int, List, np.ndarray],
                        params_: Union[int, List, np.ndarray],
                        num_samples: Union[List, np.ndarray, int]):
    assert fisher_type in ['info', 'quantum'], 'fisher_type must be either "info" or "quantum".'

    if isinstance(num_samples, int):
        num_samples = [num_samples]

    if isinstance(params_, int):
        num_params = params_
    else:
        num_params = len(params_)

    d = model.param_dim

    f_hats = model.get_empirical_fishers(fisher_type, inputs_, params_, 'input', trace_normalize=True)

    eff_dim = []

    for n_samples in num_samples:
        Fhat = f_hats * n_samples / (2 * np.pi * np.log(n_samples))
        one_plus_F = np.eye(d) + Fhat
        _, det = np.linalg.slogdet(one_plus_F)  # log det to avoid overflow
        r = det / 2  # divide by 2 because of sqrt

        ed = 2 * (logsumexp(r) - np.log(num_params)) / np.log(n_samples / (2 * np.pi * np.log(n_samples)))
        eff_dim.append(ed)
    eff_dim = np.array(eff_dim) / d

    if len(eff_dim) == 1:
        eff_dim = eff_dim[0]

    return eff_dim


def _trace_renormalize(fishers):
    d = fishers.shape[-1]  # param_dim
    fisher_trace = np.trace(np.average(fishers, axis=0))
    f_hats = d * fishers / fisher_trace
    return f_hats


def _scale_effdim(f_hats, num_params, num_samples):
    d = f_hats.shape[-1]  ## param_dim

    eff_dim = []

    for n_samples in num_samples:
        Fhat = f_hats * n_samples / (2 * np.pi * np.log(n_samples))
        one_plus_F = np.eye(d) + Fhat
        _, det = np.linalg.slogdet(one_plus_F)  # log det to avoid overflow
        r = det / 2  # divide by 2 because of sqrt

        ed = 2 * (logsumexp(r) - np.log(num_params)) / np.log(n_samples / (2 * np.pi * np.log(n_samples)))
        eff_dim.append(ed)
    eff_dim = np.array(eff_dim) / d

    if len(eff_dim) == 1:
        eff_dim = eff_dim[0]

    return eff_dim


def compute_quantum_effdim(model,
                           inputs_: Union[int, List, np.ndarray],
                           params_: Union[int, List, np.ndarray],
                           rank_effdim=False,
                           trace_effdim=False,
                           scale_effdim=False,
                           num_samples: Union[List, np.ndarray, int] = None):
    """
    template for calculating types of effective dimension
    """

    assert rank_effdim + trace_effdim + scale_effdim > 0, "You shouldn't call this without calculating any kind of effective dimension."

    if scale_effdim:
        assert num_samples, "num_samples must be provided for scale-dependent effective dimension."

    avg_fishers = model.get_empirical_fishers('quantum', inputs_, params_, 'input',
                                              trace_normalize=False)  # output in shape (num_params, param_dim, param_dim)
    quantum_effdim = {}

    if rank_effdim:
        quantum_effdim['q_rank'] = np.average([softrank(fisher) for fisher in avg_fishers])
    if trace_effdim:
        quantum_effdim['q_trace'] = np.average([np.trace(fisher) for fisher in avg_fishers])
    if scale_effdim:
        if isinstance(num_samples, int):
            num_samples = [num_samples]

        if isinstance(params_, int):
            num_params = params_
        else:
            num_params = len(params_)

        f_hats = _trace_renormalize(avg_fishers)
        quantum_effdim['q_scale'] = _scale_effdim(f_hats, num_params, num_samples)

    return quantum_effdim


def compute_stat_effdim(model,
                        inputs_: Union[int, List, np.ndarray],
                        params_: Union[int, List, np.ndarray],
                        rank_effdim=False,
                        trace_effdim=False,
                        scale_effdim=False,
                        num_samples: Union[List, np.ndarray, int] = None,
                        observables: Union[str, List] = 'all'):
    """
    template for calculating types of effective dimension
    """

    assert rank_effdim + trace_effdim + scale_effdim > 0, "You shouldn't call this without calculating any kind of effective dimension."

    if scale_effdim:
        assert num_samples, "num_samples must be provided for scale-dependent effective dimension."
        assert observables, "A set of observables corresponding to a valid probability space should be specified."

    # avg_fishers = model.get_empirical_fishers('info', inputs_, params_, 'input',
    #                                           trace_normalize=False)

    avg_fishers = model.get_empirical_fishers('info', inputs_, params_, 'input',
                                              trace_normalize=False,
                                              observables=observables) # output in shape (num_params, param_dim, param_dim)

    stat_effdim = {}

    if rank_effdim:
        stat_effdim['s_rank'] = np.average([softrank(fisher) for fisher in avg_fishers])

    if trace_effdim:
        stat_effdim['s_trace'] = np.average([np.trace(fisher) for fisher in avg_fishers])

    if scale_effdim:
        if isinstance(num_samples, int):
            num_samples = [num_samples]

        if isinstance(params_, int):
            num_params = params_
        else:
            num_params = len(params_)

        f_hats = _trace_renormalize(avg_fishers)
        stat_effdim['s_scale'] = _scale_effdim(f_hats, num_params, num_samples)

    return stat_effdim


def softrank(matrix, z=1e-9):
    """
    Return number of eigenvalues significantly
    larger than threshold z

    :param matrix: a matrix, duh
    :param z: threshold, 1e-9 by default
    :return: effective dimension of a symmetric matrix
    """
    f = lambda eig: eig / (eig + z)
    f = np.vectorize(f)
    eigvals = np.real(np.linalg.eigvals(matrix))
    softrank = sum(f(eigvals))

    return softrank


if __name__ == '__main__':
    feature_map = FeatureMap('PauliFeatureMap', feature_dim=4, reps=1)

    template = AnsatzTemplate()
    template.construct_simple_template(num_qubits=4, num_layers=1)

    model = QuantumNeuralNetwork(feature_map, template, platform='Qiskit')
    # model.visualize()

    print(model.num_qubits, model.input_dim, model.param_dim)

    ############

    num_inputs = 10
    input_dim = model.input_dim
    inputs = np.random.normal(0, 1, size=(num_inputs, input_dim))

    num_params = 10

    #num_samples = 100
    num_samples = [100, 1000, 10000, 100000, 1000000]

    ## This will return a dictionary with keys 'q_rank', 'q_trace', 'q_det', 'q_scale'.
    ## 'q_scale' corresponds to either a value or a list depending on type(num_samples)
    quantum_effdim = compute_quantum_effdim(model, inputs, num_params, rank_effdim=True, trace_effdim=True,
                                            scale_effdim=True, num_samples=num_samples)
    print(quantum_effdim)

    ## This will return a dictionary with keys 's_rank', 's_trace', 's_det', 's_scale'.
    ## 's_scale' corresponds to either a value or a list depending on type(num_samples)
    stat_effdim = compute_stat_effdim(model, inputs, num_params, rank_effdim=True, trace_effdim=True,
                                      scale_effdim=True, num_samples=num_samples, observables='all')

    print(stat_effdim)
