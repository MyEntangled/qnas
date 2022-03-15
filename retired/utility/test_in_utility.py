from retired.utility.ansatz_template import AnsatzTemplate
from retired.utility.data_encoding import FeatureMap
from retired.utility.quantum_nn import QuantumNeuralNetwork

from qiskit.opflow import I,X
from qiskit.opflow.primitive_ops import CircuitOp

import numpy as np
import time

from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector

if __name__ == '__main__':
    np.random.seed(0)
    feature_map = FeatureMap('ZZFeatureMap', feature_dim=3, reps=1)

    #feature_map.visualize()
    #print(feature_map.circ)

    template = AnsatzTemplate()
    template.construct_simple_template(num_qubits=3, num_layers=1)

    model = QuantumNeuralNetwork(None, template)

    model.visualize()

    print(model.num_qubits, model.input_dim, model.param_dim)

    ######### Test forward(): compute output states and output probabilities

    num_inputs = 5
    num_params = 5
    param_dim = 12
    thetamin = 0
    thetamax = 2 * np.pi
    input_dim = 4

    rep_range = np.tile(np.array([num_inputs]), num_params)
    params = np.random.uniform(thetamin, thetamax, size=(num_params, param_dim))
    grid_params = np.repeat(params, repeats=rep_range, axis=0)
    inputs = np.random.normal(0, 1, size=(num_inputs, input_dim))
    grid_inputs = np.tile(inputs, (num_params, 1))

    init_state = Statevector.from_label('1' * 3)
    qft_circ = QFT(num_qubits=3, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name=None)
    #qft_state = init_state.evolve(qft_circ)
    qft_projector = CircuitOp(qft_circ)
    output_states, output_exps = model.forward(grid_inputs, grid_params, observables=[qft_projector,I^3,I^I^X])
    print('TEST FORWARD')
    #print(output_states.shape, output_exps.shape)
    #print(np.sum(output_exps, axis=1))
    print('------------------------')

    ######## Test get_gradients(): compute output gradients for final measurement in standard basis
    start_time = time.time()
    observables = ['0' * model.num_qubits, '1' * model.num_qubits]
    #output_grads = model.get_gradients(grid_inputs, grid_params,
    #                                   observables='all')  # observables = 'all' for every measurement
    output_grads = model.get_gradients(grid_inputs, grid_params, observables=[qft_projector,I^3,I^I^X])
    print('TEST GRADIENT')
    print(output_grads.shape)
    # print(output_grads)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('------------------------')

    ######## Test get_fishers(): compute QFI
    start_time = time.time()
    output_fishers = model.get_quantum_fishers(grid_inputs, grid_params)
    print('TEST QFI')
    print(output_fishers.shape)
    print(output_fishers)
    print("--- %s seconds ---" % (time.time() - start_time))

    ######## Test empirical info Fishers
    start_time = time.time()
    empirical_info_fishers = model.get_empirical_fishers('info', 3, 5, 'input', trace_normalize=True, observables='parity')
    print('TEST EMPIRICAL CFI')
    print(empirical_info_fishers.shape)
    print("--- %s seconds ---" % (time.time() - start_time))

    ######## Test empirical quantum Fishers
    start_time = time.time()
    print('TEST EMPIRICAL QFI')
    empirical_quantum_fishers = model.get_empirical_fishers('quantum', 3, 5, 'input', trace_normalize=True)
    print(empirical_quantum_fishers.shape)
    #print(empirical_quantum_fishers)
    print("--- %s seconds ---" % (time.time() - start_time))




