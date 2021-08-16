# NECESSARY IMPORTS

# basic
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Union
import pickle
import os.path
from time import time

# qiskit
import qiskit
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter

# qiskit machine learning
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

# sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# from code
from effective_dimension import compute_stat_effdim
from effective_dimension import compute_quantum_effdim

from utility.quantum_nn import QuantumNeuralNetwork
from utility.data_encoding import FeatureMap
from utility.ansatz_template import AnsatzTemplate

from expressibility import get_expressibility

from training.binary_classifier import BinaryClassifier

# --------------------------------------------------------------------------------------

# LOAD IRIS DATA

def iris(n_features):
    """
    :param n_features: int: number of features to extract from iris
    :return: x_train, x_test, y_train, y_test
    """
    data, target = datasets.load_iris(return_X_y=True)

    x_train, x_test, y_train, y_test = \
        train_test_split(data[:100], target[:100], test_size=0.25, random_state=42)

    # Now we standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(x_train)
    x_train = std_scale.transform(x_train)
    x_test = std_scale.transform(x_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n_features).fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    # Scale to the range (-1, +1)
    samples = np.append(x_train, x_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    x_train = minmax_scale.transform(x_train)
    x_test = minmax_scale.transform(x_test)

    return x_train, x_test, y_train, y_test


# -----------------------------------------------------------------------------------------
# PICKLE & INDICATORS

def pickle_instances(instances, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(instances, outp, pickle.HIGHEST_PROTOCOL)


def unpickle_instances(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)


def save_circuit_data(instances, pkl_file_name, csv_file_name,
                      inputs_, params_, num_samples):
    """
    Take a list of instances, pickle them, and store indicator
    data in a csv file
    :param num_samples: *
    :param inputs_: *
    :param params_: *
    :param instances: a list of QNN instances
    :param pkl_file_name: a path to the pickle file
    :param csv_file_name: a path to the csv file
    :return:
    """
    if pkl_file_name:
        assert not os.path.isfile(pkl_file_name), f'{pkl_file_name} already exists'
    if csv_file_name:
        assert not os.path.isfile(csv_file_name), f'{csv_file_name} already exists'

    if pkl_file_name:
        pickle_instances(instances, pkl_file_name)
    if csv_file_name:
        with open(csv_file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', )
            for instance in instances:
                # expr = get_expressibility(instance, num_samples=2000, num_iterations=3)

                # Effective dimension
                # ed = compute_stat_effdim(instance,
                #                          inputs_=inputs_, params_=NUM_PARAMS,
                #                          num_samples=NUM_SAMPLES,
                #                          scale_effdim=True,
                #                          observables='parity')['s_scale']

                quantum_things = compute_quantum_effdim(instance,
                                                        inputs_=inputs_, params_=params_,
                                                        num_samples=num_samples,
                                                        rank_effdim=True, trace_effdim=True,
                                                        scale_effdim=True)

                qed = quantum_things['q_scale']
                eqd = quantum_things['q_rank']
                tqd = quantum_things['q_trace']

                # writer.writerow([expr, ed, qed, eqd, tqd])
                writer.writerow([qed, eqd, tqd])


if __name__ == '__main__':

    # -------------------------------------------------------------------------------------
    # CONSTANTS & DATA

    qiskit.utils.algorithm_globals.random_seed = 0

    MAX_NUM_LAYERS = 11
    NUM_QUBITS = 4
    x_train, x_test, y_train, y_test = iris(n_features=4)
    NUM_SAMPLES = len(x_train)
    NUM_PARAMS = 100
    NUM_TRIALS = 10  # number of random initialization trials

    # --------------------------------------------------------------------------------------
    # COMPUTING INDICATORS EXAMPLE

    # The common practice is to store objects in a dict, tuple, or list
    # and pickle the entire ensemble at once
    # models = []
    # for num_layers in range(1, MAX_NUM_LAYERS + 1):
    #     feature_map = FeatureMap('ZZFeatureMap', feature_dim=NUM_QUBITS, reps=1)
    #     template = AnsatzTemplate()
    #     template.construct_simple_template(num_qubits=NUM_QUBITS, num_layers=num_layers)
    #     model = QuantumNeuralNetwork(feature_map, template, platform='Qiskit')
    #     models.append(model)
    #
    # save_circuit_data(models, None,
    #                   csv_file_name='circuit-data/indicators-quantum.csv',
    #                   inputs_=x_train, params_=NUM_PARAMS, num_samples=NUM_SAMPLES)
    #
    # del models  # delete instances
    #
    # models = unpickle_instances('circuit-data/try.pkl')  # recover instances
    # for model in models:
    #     model.visualize()

    # ----------------------------------------------------------------------------------------
    # COMPUTING LEARNING COST

    # Log the running time of one epoch with a full batch
    # for 1-layer simple template, with NUM_TRIALS initializations

    # create circuit
    feature_map = FeatureMap('ZZFeatureMap', feature_dim=NUM_QUBITS, reps=1)
    template = AnsatzTemplate()
    template.construct_simple_template(num_qubits=NUM_QUBITS, num_layers=1)
    model = QuantumNeuralNetwork(feature_map, template, platform='Qiskit')

    with open('circuit-data/running_time.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        for num_trials in range(1, NUM_TRIALS + 1):
            start = time()

            # train
            model_trainer = BinaryClassifier(model)
            param_opt, obj_opt, _ = model_trainer.train(x_train, y_train, optimizer='ADAM',
                                                        cost_type='local_expectation',
                                                        batch_size=75, epochs=1, lr=0.1,
                                                        nat_gradient=False, evaluate_obj_adam=True)

            end = time()
            writer.writerow([end - start])

            # pred, obj, acc = model_trainer.predict(param_opt, x_test, y_test)
            # number of layers, objective, accuracy
            # print('test set accuracy: ', acc)
