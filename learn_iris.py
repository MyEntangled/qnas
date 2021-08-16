# NECESSARY IMPORTS

# basic
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Union

# qiskit
from qiskit import Aer, QuantumCircuit
from qiskit.opflow import Z, I, StateFn
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter, Instruction
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA

# qiskit machine learning
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC

# sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# from code
from utility.ansatz_template import AnsatzTemplate


# --------------------------------------------------------------------------------------

# LOAD IRIS DATA

def iris(n_features):
    """
    :param n_features: int: number of features to extract from iris
    :return: x_train, x_test, y_train, y_test
    """
    data, target = datasets.load_iris(return_X_y=True)

    x_train, x_test, y_train, y_test = \
        train_test_split(data[:100], target[:100], test_size=0.33, random_state=42)

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


# ------------------------------------------------------------------------------------

# DEFINE QNN MODEL

NUM_QUBITS = 4
MAX_NUM_LAYERS = 20

# declare quantum instance
quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator'), shots=1024)

# collect data
X_train, X_test, Y_train, Y_test = iris(n_features=NUM_QUBITS)

# construct feature map
feature_map: ZZFeatureMap = ZZFeatureMap(NUM_QUBITS, reps=1)

# construct ansatz
ansatz = TwoLocal(NUM_QUBITS, ['rx', 'ry'], 'cx', 'linear', reps=1)

# construct quantum circuit
qc = QuantumCircuit(NUM_QUBITS)
qc.append(feature_map, range(NUM_QUBITS))
qc.h(range(NUM_QUBITS))
qc.append(ansatz, range(NUM_QUBITS))

print(ansatz)

# # parity maps bitstrings to 0 or 1
# def parity(x):
#     return '{:b}'.format(x).count('1') % 2
#
#
# output_shape = 2  # corresponds to the number of classes, possible outcomes of the (parity) mapping.
#
# # construct QNN
# circuit_qnn = CircuitQNN(circuit=qc,
#                          input_params=feature_map.parameters,
#                          weight_params=ansatz.parameters,
#                          interpret=parity,
#                          output_shape=output_shape,
#                          quantum_instance=quantum_instance)
#
# # ------------------------------------------------------------------------------
#
# # LEARN
#
# NUM_ITERS = np.arange(10, 51, 10)
#
# for num_iters in [1]:
#
#     Optimizer = COBYLA(maxiter=num_iters)
#
#     # construct classifier
#     circuit_classifier = NeuralNetworkClassifier(neural_network=circuit_qnn,
#                                                  optimizer=Optimizer)
#
#     # fit classifier to data
#     circuit_classifier.fit(X_train, Y_train)
#
#     # score classifier
#     score = circuit_classifier.score(X_test, Y_test)
#     with open(f'circuit-data/zz{NUM_QUBITS}-1layer-iris-accuracy.csv', 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',', )
#         writer.writerow([num_iters, score])

# # evaluate data points
# Y_predict = circuit_classifier.predict(X_test)
#
# # plot results
# # red == wrongly classified
# for x, y_target, y_p in zip(X_test, Y_test, Y_predict):
#     if y_target == 1:
#         plt.plot(x[0], x[1], 'bo')
#     else:
#         plt.plot(x[0], x[1], 'go')
#     if y_target != y_p:
#         plt.scatter(x[0], x[1], s=200, facecolors='none', edgecolors='r', linewidths=2)
# plt.show()
