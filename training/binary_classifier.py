from utility.tools import *
import numpy as np
from qiskit.algorithms.optimizers import GradientDescent, ADAM

from utility.ansatz_template import AnsatzTemplate
from utility.data_encoding import FeatureMap
from utility.quantum_nn import QuantumNeuralNetwork

from qiskit.opflow import I, X, Y, Z

import time


class BinaryClassifier():

    def __init__(self, model):

        self.model = model
        self.inputs = None
        self.labels = None
        self.observable = None

        self.circuits_ = []

        self.num_batches = None
        self.batch_inputs = None
        self.batch_labels = None
        self.batch_idx = None
        self.num_inputs_per_batch = None

        self.num_epochs = None

        self.epoch_obj = None
        self.batch_obj = None

        self.fx = None # Only used for ADAM optimizer since it has no callback
        self.evaluate_obj_adam = None

    def evaluate_objective(self, param):
        grid_params = np.tile(param, (self.num_inputs_per_batch[self.batch_idx], 1))

        # if self.cost_type == 'local_expectation':
        #     _, expectations = self.model.forward(self.batch_inputs[self.batch_idx], grid_params,
        #                                          observables=[self.observable])
        #
        #     expectations[self.batch_labels[self.batch_idx] == -1] *= -1  # negate those of the second class

        _, expectations = self.model.forward(self.batch_inputs[self.batch_idx], grid_params,
                                             observables=[self.observable])

        expectations[self.batch_labels[self.batch_idx] == -1] *= -1  # negate those of the second class

        objective = np.real(np.average(expectations, axis=0))

        return objective

    def evaluate_gradient(self, param):
        grid_params = np.tile(param, (self.num_inputs_per_batch[self.batch_idx], 1))

        # if self.cost_type == 'local_expectation':
        #     gradients = self.model.get_gradients(self.batch_inputs[self.batch_idx], grid_params,
        #                                          observables=[self.observable])
        #
        #     gradients[self.batch_labels[self.batch_idx] == -1] *= -1  # negate those of the second class

        gradients = self.model.get_gradients(self.batch_inputs[self.batch_idx], grid_params,
                                             observables=[self.observable])

        gradients[self.batch_labels[self.batch_idx] == -1] *= -1  # negate those of the second class

        avg_gradient = np.average(gradients, axis=0)

        if self.optimizer == 'ADAM':
            self._adam_callback(param)  # manually call back for adam

        return -np.real(avg_gradient)

    def evaluate_natural_gradient(self, param):
        grid_params = np.tile(param, (self.num_inputs_per_batch[self.batch_idx], 1))

        # if self.cost_type == 'local_expectation':
        #     gradients = self.model.get_gradients(self.batch_inputs[self.batch_idx], grid_params,
        #                                          observables=[self.observable])
        #     gradients[self.batch_labels[self.batch_idx] == -1] *= -1  # negate those of the second class

        gradients = self.model.get_gradients(self.batch_inputs[self.batch_idx], grid_params,
                                             observables=[self.observable])
        gradients[self.batch_labels[self.batch_idx] == -1] *= -1  # negate those of the second class


        quantum_fishers = self.model.get_quantum_fishers(self.batch_inputs[self.batch_idx], grid_params)

        natural_gradients = np.array(
            [np.linalg.pinv(qfi, hermitian=True) @ gradients[idx].ravel() for idx, qfi in enumerate(quantum_fishers)])

        avg_natural_gradient = np.average(natural_gradients, axis=0)

        if self.optimizer == 'ADAM':
            self._adam_callback(param)  # manually call back for adam

        return -np.real(avg_natural_gradient)

    def _gd_callback(self, nfevs, x, fx, stepsize):

        if self.batch_idx == 0:  # first batch in epoch
            self.batch_obj = []

        self.batch_obj.append(fx)

        if self.batch_idx == self.num_batches - 1:  # last batch in epoch
            self.epoch_obj.append(np.average(self.batch_obj))
            self._create_input_batches()

            print("Epoch: {}/{}".format(self.epoch+1, self.num_epochs))
            print("Obj: {}".format(self.epoch_obj[-1]))

            self.epoch += 1

        self.batch_idx = (self.batch_idx + 1) % self.num_batches

    def _adam_callback(self, param):

        if self.evaluate_obj_adam: # compute the objective again

            self.fx = self.evaluate_objective(param)

            if self.batch_idx == 0:  # first batch in epoch
                self.batch_obj = []

            self.batch_obj.append(self.fx)

            if self.batch_idx == self.num_batches - 1:  # last batch in epoch
                self.epoch_obj.append(np.average(self.batch_obj))
                self._create_input_batches()

                print("Epoch: {}/{}".format(self.epoch+1, self.num_epochs))
                print("Obj: {}".format(self.epoch_obj[-1]))

                self.epoch += 1

            self.batch_idx = (self.batch_idx + 1) % self.num_batches

        else:
            if self.batch_idx == self.num_batches - 1:  # last batch in epoch
                self._create_input_batches()

                print("Epoch: {}/{}".format(self.epoch+1, self.num_epochs))

                self.epoch += 1

            self.batch_idx = (self.batch_idx + 1) % self.num_batches

    def _create_input_batches(self):
        perm = np.random.permutation(len(self.inputs))
        self.inputs = self.inputs[perm]
        self.labels = self.labels[perm]

        self.batch_inputs = [self.inputs[start_idx: start_idx + self.num_inputs_per_batch[i]] for i, start_idx in
                             enumerate(self.input_batch_starts)]
        self.batch_labels = [self.labels[start_idx: start_idx + self.num_inputs_per_batch[i]] for i, start_idx in
                             enumerate(self.input_batch_starts)]

    def _bind_inputs(self):
        for input in self.inputs:
            input_dict = compose_param_dict(self.model.feature_map_circ.parameters, input)
            self.circuits_.append(self.model.circuit.bind_parameters(input_dict))

    def train(self, inputs, labels,
              optimizer='GradientDescent',
              cost_type='local_expectation',
              batch_size=None, epochs=100, lr=0.01,
              initial_param=None, nat_gradient=False,
              evaluate_obj_adam=True):

        self.inputs = np.array(inputs)
        self.labels = np.array(labels)

        if cost_type == 'local_expectation':
            self.cost_type = cost_type
            self.observable = Z ^ (I ^ (self.model.num_qubits - 1))
        elif cost_type == 'global_expectation':
            self.cost_type = cost_type
            self.observable = Z ^ self.model.num_qubits

        self.num_epochs = epochs
        self.batch_idx = 0
        self.epoch = 0
        self.epoch_obj = []

        self._bind_inputs()  # bind inputs to create param-only circuits

        if batch_size is None:
            self.batch_size = len(self.inputs)
        else:
            self.batch_size = batch_size

        self.num_batches = int(np.ceil(len(self.inputs) / self.batch_size))  # number of batches
        self.num_inputs_per_batch = [self.batch_size] * (len(self.inputs) // self.batch_size) + [
            len(self.inputs) % self.batch_size]  # list contains number of inputs per batch
        self.input_batch_starts = np.concatenate(([0], np.cumsum(self.num_inputs_per_batch)[
                                                       :self.num_batches - 1]))  # list contains the starting position every batch

        # print(self.num_batches, self.num_inputs_per_batch, self.input_batch_starts)

        self._create_input_batches()

        # Create initial parameter set
        if initial_param is None:
            initial_param = np.random.random((self.model.param_dim))

        # max iterations
        maxiter = epochs * self.num_batches

        # parameter bound
        param_bound = [(self.model.param_min, self.model.param_max)] * self.model.param_dim

        # callback for adam
        self.evaluate_obj_adam = evaluate_obj_adam

        # gd = GradientDescent(maxiter=maxiter, learning_rate=lr, callback=self._gd_callback)

        if optimizer == 'GradientDescent':
            self.optimizer = optimizer
            opt = GradientDescent(maxiter=maxiter, learning_rate=lr, callback=self._gd_callback)

            if nat_gradient:
                x_opt, fx_opt, nfevs = opt.optimize(self.model.param_dim,  # number of parameters
                                                    self.evaluate_objective,  # function to minimize
                                                    gradient_function=self.evaluate_natural_gradient,
                                                    # function to evaluate the natural gradient
                                                    variable_bounds=param_bound,
                                                    initial_point=initial_param)  # initial point
            else:
                x_opt, fx_opt, nfevs = opt.optimize(self.model.param_dim,  # number of parameters
                                                    self.evaluate_objective,  # function to minimize
                                                    gradient_function=self.evaluate_gradient,
                                                    # function to evaluate the gradient
                                                    variable_bounds=param_bound,
                                                    initial_point=initial_param)  # initial point

        elif optimizer == 'ADAM':
            self.optimizer = optimizer
            opt = ADAM(maxiter=maxiter, lr=lr, amsgrad=False)

            if nat_gradient:
                x_opt, fx_opt, nfevs = opt.minimize(self.evaluate_objective,  # function to minimize
                                                    initial_point=initial_param,  # initial point
                                                    gradient_function=self.evaluate_natural_gradient, # function to evaluate natural gradient
                                                    )
            else:
                x_opt, fx_opt, nfevs = opt.minimize(self.evaluate_objective,  # function to minimize
                                                    initial_point=initial_param,  # initial point
                                                    gradient_function=self.evaluate_gradient, # function to evaluate vanilla gradient
                                                    )
        else:
            raise Exception("The specified optimizer is unknown.")

        return x_opt, fx_opt, nfevs

    def evaluate_objective_eval(self, param, inputs, labels):
        grid_params = np.tile(param, (len(inputs), 1))

        # if self.cost_type == 'local_expectation':
        #     _, expectations = self.model.forward(inputs, grid_params,
        #                                          observables=[self.observable])
        #
        #     expectations[labels == -1] *= -1  # negate those of the second class

        _, expectations = self.model.forward(inputs, grid_params,
                                             observables=[self.observable])

        expectations[labels == -1] *= -1  # negate those of the second class

        objective = np.average(expectations, axis=0)

        return np.real(objective)

    def predict(self, param, inputs, labels):

        grid_params = np.tile(param, (len(inputs), 1))

        # if self.cost_type == 'local_expectation':
        #     _, expectations = self.model.forward(inputs, grid_params,
        #                                          observables=[self.observable])

        _, expectations = self.model.forward(inputs, grid_params,
                                             observables=[self.observable])

        pred = np.array([1 if exp > 0 else -1 for exp in expectations])

        accuracy = sum(pred == labels) * 100. / len(labels)
        objective = self.evaluate_objective_eval(param, inputs, labels)[0]

        return pred, objective, accuracy


if __name__ == '__main__':
    feature_map = FeatureMap('PauliFeatureMap', feature_dim=4, reps=5)

    template = AnsatzTemplate()
    template.construct_simple_template(num_qubits=4, num_layers=2)

    model = QuantumNeuralNetwork(feature_map, template, platform='Qiskit')
    # model.visualize()

    print(model.num_qubits, model.input_dim, model.param_dim)

    #####################

    ## GENERATE DATA
    input_dim = 4
    num_samples = 48

    # Generate random input coordinates (X) and binary labels (y)
    X = 2 * np.random.rand(num_samples, input_dim) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}, y01 will be used for CircuitQNN example
    y = 2 * y01 - 1  # in {-1, +1}, y will be used for OplowQNN example

    ####################

    start_time = time.time()

    qiskit.utils.algorithm_globals.random_seed = 0

    model_trainer = BinaryClassifier(model)
    param_opt, obj_opt, _ = model_trainer.train(X, y, optimizer='GradientDescent',  # 'GradientDescent' or 'ADAM'
                                                cost_type='global_expectation', # dont change this
                                                batch_size=8, epochs=5, lr=0.1,
                                                nat_gradient=False,  # Vanilla gradients preferred
                                                evaluate_obj_adam=True)  # When using ADAM, no computation of objective by default
                                                                         # True for observing change in objective value.
                                                                         # False for training large dataset.

    pred, obj, acc = model_trainer.predict(param_opt, X, y)

    print("Labels: {}".format(y))
    print("Predictions: {}".format(pred))
    print("Objective: {}, Accuracy: {}%".format(obj, acc))
    print("Epochs obj: ", model_trainer.epoch_obj)

    print("--- %s seconds ---" % (time.time() - start_time))
