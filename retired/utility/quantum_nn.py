import numpy as np
from qiskit import QuantumCircuit
from typing import List, Union
import multiprocessing as mp
from retired.utility.model_base import Model

from retired.utility.ansatz_template import AnsatzTemplate
from retired.utility.data_encoding import FeatureMap
from qiskit.opflow import I, X
from surfer.gradient import ReverseGradient
from surfer.qfi import ReverseQFI
import time
from retired.utility.tools import (prepare_multiprocesses, compose_param_dict, get_measurement_operator, generate_bitstrings, get_parity_observables)

np.random.seed(0)


class QuantumNeuralNetwork(Model):

    def __init__(self, feature_map, template):

        if feature_map is not None:
            assert feature_map.num_qubits == template.num_qubits, "Feature map and PQC must have the same number of qubits"

        super(QuantumNeuralNetwork, self).__init__()

        self.num_qubits = template.num_qubits
        self.state_dim = 2 ** self.num_qubits
        if feature_map is not None:
            self.input_dim = feature_map.input_dim
        else:
            self.input_dim = 0
        self.param_dim = len(template.PQC.parameters)

        self.param_min = 0
        self.param_max = 2*np.pi

        if feature_map is not None:
            self.feature_map_circ = feature_map.circ
        else:
            self.feature_map_circ = QuantumCircuit(self.num_qubits)
        self.PQC = template.PQC
        self.circuit = QuantumCircuit(self.num_qubits).compose(self.feature_map_circ).compose(self.PQC)

        self.sv = Statevector.from_label('0' * self.num_qubits)
        self.output_state = None
        self.rev_grad = None
        self.rev_QFI = None

    def _get_output_states(self, inds, inputs, params, results):
        for i, input, param in zip(inds, inputs, params):

            # circuit_ = circuit.bind_parameters(self._compose_param_dict(input, param))

            input_dict = compose_param_dict(self.feature_map_circ.parameters, input)
            param_dict = compose_param_dict(self.PQC.parameters, param)
            circuit_ = self.circuit.bind_parameters({**input_dict, **param_dict})

            result = self.sv.evolve(circuit_)

            start = 2 * i * 2 ** self.num_qubits
            end = (2 * i + 2) * 2 ** self.num_qubits

            results[start:end:2] = np.real(result.data)
            results[start + 1:end:2] = np.imag(result.data)

    def forward(self, inputs: Union[List, np.ndarray], params: Union[List, np.ndarray],
                observables: Union[str, List]='all'):
        """
        Compute final quantum states and measurement probabilitues
        :param inputs: sets of data inputs
        :param params: lists of parameter
        :return: Quantum states and Measurement probabilities, the number of them given by the number of input sets / param sets
        """

        assert len(params) == len(inputs), "The number of parameter sets must be equal to the number of input sets"

        state_dim = 2 ** self.num_qubits
        num_inputs = len(inputs)

        # map input to arrays
        params = np.array(params)
        inputs = np.array(inputs)

        # specify number of parallel processes
        num_processes = mp.cpu_count()
        # construct index set per process
        indices = prepare_multiprocesses(num_inputs, num_processes)

        # initialize shared array to store output states (only supports 1D-array, needs reshaping later)
        results = mp.Array('d', (
                    2 * num_inputs * state_dim))  ## Results for output state vectors (Re and Im parts separately)

        # construct processes to be run in parallel
        processes = [mp.Process(target=self._get_output_states, args=(inds, inputs[inds], params[inds], results))
                     for inds in indices]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        ## Reformat the multithread array into an array with complex values at half of the original size
        results = np.ctypeslib.as_array(results.get_obj()).view(np.complex128)

        output_states = np.zeros((num_inputs, state_dim), dtype=np.complex_)
        
        if observables == 'all':
            output_probs = np.zeros((num_inputs, state_dim))

            for i in range(num_inputs):
                start = i * state_dim
                end = (i + 1) * state_dim
                output_states[i] = results[start:end]
                output_probs[i] = abs(results[start:end]) ** 2
            return output_states, output_probs

        elif observables == 'parity':
            observable_ops = get_parity_observables(self.num_qubits)
            output_exps = np.zeros((num_inputs, len(observable_ops)), dtype=np.complex_)
            observables_mats = np.array([ob.to_matrix() for ob in observable_ops])

            for i in range(num_inputs):
                start = i * state_dim
                end = (i + 1) * state_dim
                output_states[i] = results[start:end]
                output_exps[i] = output_states[i].conj().T @ observables_mats @ output_states[i]
            return output_states, output_exps

        else: ## list of operators
            output_exps = np.zeros((num_inputs, len(observables)), dtype=np.complex_)
            observables_mats = []
            for ob in observables:
                try:
                    observables_mats.append(ob.to_matrix())
                except:
                    observables_mats.append(ob.data)
            observables_mats = np.array(observables_mats)
            #observables_mats = np.array([ob.to_matrix() for ob in observable_ops])
            for i in range(num_inputs):
                start = i * state_dim
                end = (i + 1) * state_dim
                output_states[i] = results[start:end]
                output_exps[i] = output_states[i].conj().T @ observables_mats @ output_states[i]
            return output_states, output_exps


    def _get_derivatives(self, inds, inputs, params, operator, results):

        for i, input, param in zip(inds, inputs, params):

            # circuit_ = circuit.bind_parameters(self._compose_param_dict(input, param, input_dict_only=True))

            input_dict = compose_param_dict(self.feature_map_circ.parameters, input)
            circuit_ = self.circuit.bind_parameters(input_dict)

            grad = self.rev_grad.compute(operator, circuit_, param)
            start = i * self.param_dim
            end = (i + 1) * self.param_dim

            results[start:end] = grad


    def get_gradients(self, inputs, params,
                      observables: Union[str, List]='all'):
        """
        Compute gradients of measurement probabilities wrt to parameter
        :param inputs: sets of data input
        :param params: sets of parameter
        :param observables: list of projective measurement of interest
        :return: Gradient vectors, the number of them given by the number of input sets / param sets
        """

        assert len(inputs) == len(params), 'The number of parameter sets must be equal to the number of input sets'
        num_inputs = len(inputs)

        self.rev_grad = ReverseGradient()

        num_processes = mp.cpu_count()
        indices = prepare_multiprocesses(num_inputs, num_processes)

        if observables == 'all':
            observables = generate_bitstrings(self.num_qubits) ## bitstring form
            observable_ops = [get_measurement_operator(observable) for observable in observables] ## operator form

        elif observables == 'parity':
            observable_ops = get_parity_observables(self.num_qubits) ## operator form

        else: ## observables given as list of operator
            observable_ops = observables

        num_observables = len(observable_ops)

        output_gradients = np.zeros((num_inputs, num_observables, self.param_dim))

        for op_idx, operator in enumerate(observable_ops):

            results = mp.Array('d', (num_inputs * self.param_dim))

            processes = [mp.Process(target=self._get_derivatives, args=(inds, inputs[inds], params[inds], operator, results))
                         for inds in indices]

            for p in processes:
                p.start()
            for p in processes:
                p.join()

            for i in range(num_inputs):
                start = i * self.param_dim
                end = (i + 1) * self.param_dim
                output_gradients[i, op_idx] = results[start:end]

        return output_gradients


    def get_info_fishers(self, inputs, params,
                         observables: Union[str, List]='all'):

        assert len(inputs) == len(params), 'The number of input sets must equal to the number of param sets.'

        if observables == 'all':
            observables = generate_bitstrings(self.num_qubits)  ## bitstring form
            observable_ops = [get_measurement_operator(observable) for observable in observables]  ## operator form

        elif observables == 'parity':
            observable_ops = get_parity_observables(self.num_qubits)  ## operator form

        else:  ## observables given as list of operator
            observable_ops = observables

            sum_op = sum(observables).to_matrix()
            iden = (I ^ self.num_qubits).to_matrix()
            assert (sum_op == iden).all(), "Observables regarding statistical Fisher information must sum up to identity."


        _, output_exps = self.forward(inputs, params, observable_ops)
        output_exps = output_exps.real ## because imag part is zero since observables must form a PSD partition

        output_grads = self.get_gradients(inputs, params, observable_ops)

        fishers = np.zeros((len(inputs), self.param_dim, self.param_dim))

        for i in range(output_grads.shape[0]): # over len(inputs)
            grads = output_grads[i]
            exps = output_exps[i]

            for l in range(output_grads.shape[1]): # over num of observables, = state_dim for standard measurement
                fishers[i] += np.array(np.outer(grads[l], grads[l])) / exps[l]

        return fishers


    def _get_fisher(self, inds, inputs, params, results):

        for i, input, param in zip(inds, inputs, params):

            input_dict = compose_param_dict(self.feature_map_circ.parameters, input)
            circuit_ = self.circuit.bind_parameters(input_dict)

            fisher = self.rev_QFI.compute(circuit_, param)

            start = i * self.param_dim * self.param_dim
            end = (i + 1) * self.param_dim * self.param_dim

            results[start:end] = fisher.flatten()


    def get_quantum_fishers(self, inputs, params):
        """
        Compute Quantum Fisher Information Matrix
        :param inputs: sets of data input
        :param params: sets of parameter
        :return: Fisher matrices, the number of them given by the number of input sets / param sets
        """

        self.rev_QFI = ReverseQFI()

        assert len(inputs) == len(params), 'The number of parameter sets must be equal to the number of input sets'
        num_inputs = len(inputs)

        num_processes = mp.cpu_count()
        indices = prepare_multiprocesses(num_inputs, num_processes)

        results = mp.Array('d', (num_inputs * self.param_dim * self.param_dim))

        processes = [mp.Process(target=self._get_fisher,
                                args=(inds, inputs[inds], params[inds], results))
                     for inds in indices]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        output_fishers = np.zeros((num_inputs, self.param_dim, self.param_dim))

        for i in range(num_inputs):
            start = i * self.param_dim * self.param_dim
            end = (i + 1) * self.param_dim * self.param_dim
            output_fishers[i] = np.array(results[start:end]).reshape(self.param_dim, self.param_dim)

        return output_fishers

    def _normalize_fisher(self, fishers, fishers_shape, axis, trace_normalize):
        d = self.param_dim

        if trace_normalize:
            fisher_trace = np.trace(np.average(fishers, axis=0))
            fisher = np.average(np.reshape(fishers, fishers_shape),
                                axis=axis)  # fishers_shape = (num_params, num_inputs, d, d)
            f_hats = d * fisher / fisher_trace
            return f_hats

        else:
            fisher = np.average(np.reshape(fishers, fishers_shape),
                                axis=axis)  # fishers_shape = (num_params, num_inputs, d, d)
            return fisher

    def get_empirical_fishers(self,
                              fisher_type: str,
                              inputs_: Union[int, List, np.ndarray],
                              params_: Union[int, List, np.ndarray],
                              avg_axis: str,
                              trace_normalize: bool,
                              observables: Union[str, List]=None):

        assert fisher_type in ['info', 'quantum'], 'fisher_type must be either "info" or "quantum".'
        assert avg_axis in ['input', 'param'], 'avg_axis must be "input" or "param".'
        if fisher_type == 'info':
            assert observables, 'Please specify observables'

        if avg_axis == 'input':
            avg_axis = 1
        elif avg_axis == 'param':
            avg_axis = 0

        if isinstance(inputs_, int):
            inputs_ = np.random.normal(0, 1, size=(inputs_, self.input_dim))
        if isinstance(params_, int):
            params_ = np.random.uniform(0, 2 * np.pi, size=(params_, self.param_dim))

        d = self.param_dim
        num_inputs = len(inputs_)
        num_params = len(params_)

        rep_range = np.tile(np.array([num_inputs]), num_params)
        grid_params = np.repeat(params_, repeats=rep_range, axis=0)
        grid_inputs = np.tile(inputs_, (num_params, 1))

        if fisher_type == 'info':

            if observables == 'all':
                observables = generate_bitstrings(self.num_qubits)  ## bitstring form
                observable_ops = [get_measurement_operator(observable) for observable in observables]  ## operator form
            elif observables == 'parity':
                observable_ops = get_parity_observables(self.num_qubits)  ## operator form
            else:  ## observables given as list of operator
                observable_ops = observables

            fishers = self.get_info_fishers(grid_inputs, grid_params, observable_ops)

        elif fisher_type == 'quantum':
            fishers = self.get_quantum_fishers(grid_inputs, grid_params)

        f_hats = self._normalize_fisher(fishers, (num_params, num_inputs, d, d), avg_axis, trace_normalize)

        return f_hats

    def visualize(self, output=None):
        print(self.circuit.draw(output=output))


if __name__ == '__main__':
    feature_map = FeatureMap('ZZFeatureMap', feature_dim=4, reps=1)
    #feature_map.visualize()
    #print(feature_map.circ)

    template = AnsatzTemplate()
    template.construct_simple_template(num_qubits=4, num_layers=1)

    model = QuantumNeuralNetwork(None, template)

    model.visualize()

    print(model.num_qubits, model.input_dim, model.param_dim)

    ######### Test forward(): compute output states and output probabilities

    num_inputs = 5
    num_params = 5
    param_dim = 16
    thetamin = 0
    thetamax = 2 * np.pi
    input_dim = 4

    rep_range = np.tile(np.array([num_inputs]), num_params)
    params = np.random.uniform(thetamin, thetamax, size=(num_params, param_dim))
    grid_params = np.repeat(params, repeats=rep_range, axis=0)
    inputs = np.random.normal(0, 1, size=(num_inputs, input_dim))
    grid_inputs = np.tile(inputs, (num_params, 1))

    ## QFT observable
    from qiskit.circuit.library import QFT
    from qiskit.quantum_info import Statevector

    qft_circ = QFT(num_qubits=4, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name=None)
    input_state = Statevector.from_label('0000')
    qft_state = input_state.evolve(qft_circ)
    qft_projector = qft_state.to_operator()

    output_states, output_exps = model.forward(grid_inputs, grid_params, observables=[qft_projector, I^4,I^I^I^X])
    print('TEST FORWARD')
    print(output_states.shape, output_exps.shape)
    print(output_exps[0])
    print('------------------------')

    # ######## Test get_gradients(): compute output gradients for final measurement in standard basis
    # start_time = time.time()
    # observables = ['0' * model.num_qubits, '1' * model.num_qubits]
    # #output_grads = model.get_gradients(grid_inputs, grid_params,
    # #                                   observables='all')  # observables = 'all' for every measurement
    # output_grads = model.get_gradients(grid_inputs, grid_params, observables=[I^4,I^I^I^X])
    # print('TEST GRADIENT')
    # print(output_grads.shape)
    # # print(output_grads)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print('------------------------')

    ######## Test get_gradients(): compute gradients for QFT projector observable


    start_time = time.time()
    output_grads = model.get_gradients(grid_inputs, grid_params, observables=[qft_projector])
    print('TEST GRADIENT')
    print(output_grads.shape)
    print(output_grads)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('------------------------')


    # ######## Test get_fishers(): compute QFI
    # start_time = time.time()
    # output_fishers = model.get_quantum_fishers(grid_inputs, grid_params)
    # print('TEST QFI')
    # print(output_fishers.shape)
    # print(output_fishers)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # 
    # ######## Test empirical info Fishers
    # start_time = time.time()
    # empirical_info_fishers = model.get_empirical_fishers('info', 3, 5, 'input', trace_normalize=True, observables='parity')
    # print('TEST EMPIRICAL CFI')
    # print(empirical_info_fishers.shape)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # 
    # ######## Test empirical quantum Fishers
    # start_time = time.time()
    # print('TEST EMPIRICAL QFI')
    # empirical_quantum_fishers = model.get_empirical_fishers('quantum', 3, 5, 'input', trace_normalize=True)
    # print(empirical_quantum_fishers.shape)
    # #print(empirical_quantum_fishers)
    # print("--- %s seconds ---" % (time.time() - start_time))
