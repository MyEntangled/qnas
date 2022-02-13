import unittest
from numpy.testing import *

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from QuOTMANN.optimal_transport import circuit_distance
from embedding import qc_embedding

class Test(unittest.TestCase):
    def test_distance_as_a_metric(self):
        ## CASE 1: symmetry
        qc1 = QuantumCircuit(4)
        theta1 = ParameterVector('theta', length=3)
        qc1.x(1)
        qc1.rz(theta1[0], 3)
        qc1.crx(theta1[1], 1, 2)
        qc1.cy(3, 0)
        qc1.rxx(theta1[2], 0, 1)

        qc2 = QuantumCircuit(4)
        theta2 = ParameterVector('theta', length=3)
        qc2.x(1)
        qc2.rz(theta2[0], 3)
        qc2.crx(theta2[1], 1, 2)
        qc2.cy(3, 0)
        qc2.rxx(theta2[2], 0, 1)

        qc3 = QuantumCircuit(4)
        theta3 = ParameterVector('theta', length=3)
        qc3.x(2)
        qc3.rz(theta3[0], 0)
        qc3.crx(theta3[1], 2, 3)
        qc3.cy(0, 1)
        qc3.rxx(theta3[2], 1, 2)

        dist12 = circuit_distance(qc1, qc2)
        dist21 = circuit_distance(qc2, qc1)
        dist13 = circuit_distance(qc1, qc3)
        dist31 = circuit_distance(qc3, qc1)
        dist23 = circuit_distance(qc2, qc3)
        dist32 = circuit_distance(qc3, qc2)

        ## Nonnegavity
        assert dist12[0] >= 0
        assert dist23[0] >= 0
        assert dist13[0] >= 0

        ## Identity of indiscernible
        assert_almost_equal(dist12[0], 0)
        assert_almost_equal(dist12[1], 0)

        ## Symmetricity
        assert_equal(dist12[0], dist21[0])
        assert_equal(dist12[1], dist21[1])

        assert_equal(dist23[0], dist32[0])
        assert_equal(dist23[1], dist32[1])

        assert_equal(dist13[0], dist31[0])
        assert_equal(dist13[1], dist31[1])

        ## Triangle inequality
        assert dist12[0] + dist23[0] >= dist13[0]
        assert dist12[1] + dist23[1] >= dist13[1]

        assert dist12[0] + dist13[0] >= dist23[0]
        assert dist12[1] + dist13[1] >= dist23[1]

        assert dist13[0] + dist23[0] >= dist12[0]
        assert dist13[1] + dist23[1] >= dist12[1]
    def test_positive_definite_kernel(self, num_trials=3):
        num_qubits = 4
        MAX_OP_NODES = 10

        encoding_length = (num_qubits + 1) * MAX_OP_NODES
        bounds = np.array([[-.2] * encoding_length, [1.0] * encoding_length])

        alpha = 1.
        beta = 1.
        num_samples = 10

        count = 0
        for k in range(num_trials):
            x1 = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(num_samples, encoding_length)
            dist = np.zeros(shape=(x1.shape[0], x1.shape[0]))
            dist_norm = np.zeros(shape=(x1.shape[0], x1.shape[0]))
            for i in range(dist.shape[0]):
                for j in range(dist.shape[1]):
                    qc1 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x1[i])
                    qc2 = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=x1[j])
                    dist[i, j], dist_norm[i,j] = circuit_distance(qc1, qc2)

            gram = alpha * np.exp(-beta * dist)
            eigvals = np.linalg.eigvalsh(gram)

            gram_norm = alpha * np.exp(-beta * dist_norm)
            eigvals_norm = np.linalg.eigvalsh(gram_norm)
            if (eigvals > -1e-4).all() and (eigvals_norm > -1e-4).all():
                count += 1

        assert_equal(count, num_trials)

test = Test()
test.test_distance_as_a_metric()
test.test_positive_definite_kernel(num_trials=2)

print("Unit test for OTMANN distance: Done!")