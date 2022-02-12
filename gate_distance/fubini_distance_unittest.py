import unittest
import numpy as np
from numpy.testing import *

import fubini_distance
from qiskit.quantum_info import random_statevector, random_unitary

class Test(unittest.TestCase):
    def test_min_with_perturbed_phases(self):
        num_qubits = 4
        d = 2**num_qubits
        num_states = d*(d+1)
        U = random_unitary(dims=(2 ** num_qubits)).data
        X = np.zeros(shape=(num_states, d), dtype=np.complex64)
        Y = np.zeros(shape=(num_states, d), dtype=np.complex64)
        for i in range(num_states):
            X[i] = random_statevector(dims=2 ** num_qubits).data
        for i in range(num_states):
            Y[i] = U @ X[i]

        random_phase = np.random.uniform(0, 2 * np.pi, num_states)
        for i in range(len(Y)):
            Y[i] *= np.exp(1j * random_phase[i])

        res = fubini_distance.minimize_sum_fidelity(X,Y, num_trials=2)
        assert_almost_equal(res, 0)


