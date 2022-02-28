import unittest
import numpy as np
from numpy.testing import *

import fubini_distance
from qiskit.quantum_info import random_statevector, random_unitary

class Test(unittest.TestCase):
    def test_unitary_procrustes(self):
        num_qubits = 4
        d = 2 ** num_qubits
        num_states = d * (d + 1)

        ## Case 1: Perfect unitary U exists
        U = random_unitary(dims=d).data
        X = np.zeros(shape=(num_states, d), dtype=np.complex128)
        Y = np.zeros(shape=(num_states, d), dtype=np.complex128)
        for i in range(num_states):
            X[i] = random_statevector(dims=d).data
        for i in range(num_states):
            Y[i] = X[i] @ U

        U_pred, min_dist = fubini_distance.unitary_procrustes(X,Y)
        assert_almost_equal(min_dist, 0)
        assert_array_almost_equal(U_pred @ U_pred.conj().T, np.eye(d))

        ## Case 2: No perfect U exists
        X = np.zeros(shape=(num_states, d), dtype=np.complex128)
        Y = np.zeros(shape=(num_states, d), dtype=np.complex128)
        for i in range(num_states):
            X[i] = random_statevector(dims=d).data
        for i in range(num_states):
            Y[i] = random_statevector(dims=d).data

        U_pred, min_dist = fubini_distance.unitary_procrustes(X,Y)

        U_random = []
        dist_random = []
        for _ in range(1000):
            U_random.append(random_unitary(dims=d).data)
            dist_random.append(np.linalg.norm(X @ U - Y) ** 2)
        assert min_dist <= min(dist_random)

    def test_square_bures_distance(self):
        ## CASE 1: One vector state
        X = np.array([[1.,0.]], dtype=np.complex128)
        Y = np.array([[1., -1j]]/np.sqrt(2), dtype=np.complex128)
        bures_from_inner_prod = fubini_distance.square_bures_distance(X,Y, phase_def=False)
        bures_from_phase = fubini_distance.square_bures_distance(X,Y,phase_def=True)

        assert_almost_equal(bures_from_inner_prod, 2-np.sqrt(2))
        assert_almost_equal(bures_from_phase, bures_from_inner_prod)


        ## CASE 2: Many vector states
        dups = 100
        X = np.array([[1., 0.]], dtype=np.complex128)
        X = np.repeat(X, repeats=dups, axis=0)
        Y = np.array([[1., -1j]] / np.sqrt(2), dtype=np.complex128)
        Y = np.repeat(Y, repeats=dups, axis=0)
        bures_from_inner_prod = fubini_distance.square_bures_distance(X, Y, phase_def=False)
        bures_from_phase = fubini_distance.square_bures_distance(X, Y, phase_def=True)

        assert_almost_equal(bures_from_inner_prod, dups*(2 - np.sqrt(2)))
        assert_almost_equal(bures_from_phase, bures_from_inner_prod)

    def test_max_sum_sqrt_fidelity(self):
        num_qubits = 2
        d = 2**num_qubits
        num_states = d*(d+1)

        ## CASE 1: A perfect unitary mapping U exists
        U = random_unitary(dims=(2 ** num_qubits)).data
        X = np.zeros(shape=(num_states, d), dtype=np.complex128)
        Y = np.zeros(shape=(num_states, d), dtype=np.complex128)
        for i in range(num_states):
            X[i] = random_statevector(dims=2 ** num_qubits).data
        for i in range(num_states):
            #Y[i] = U @ X[i]
            Y[i] = X[i] @ U

        random_phase = np.random.uniform(0, 2 * np.pi, num_states)
        for i in range(len(Y)):
            Y[i] *= np.exp(1j * random_phase[i])

        U_pred, max_fid = fubini_distance.max_sum_sqrt_fidelity(X, Y, num_trials=2, get_unitary=True)
        assert_almost_equal(max_fid, num_states)

        phased_identity = U @ U_pred.conj().T
        assert_almost_equal(phased_identity / phased_identity[0,0], np.eye(d), decimal=4)

        # Sum of square root fidelity
        sum_fid = 0
        for i in range(num_states):
            #sum_fid += np.abs(np.inner(Y[i].conj(), U_pred.T @ X[i]))
            sum_fid += np.abs(np.inner(Y[i].conj(), X[i] @ U_pred))

        assert_almost_equal(sum_fid, max_fid)

        # Sum of square root fidelity, vectorized
        X_transformed = (X @ U_pred)
        all_fid = sum(np.abs(np.sum(Y.conj() * X_transformed, axis=1)))
        assert_almost_equal(all_fid, max_fid)



        ## CASE 2: No such U exists.
        X = np.zeros(shape=(num_states, d), dtype=np.complex128)
        Y = np.zeros(shape=(num_states, d), dtype=np.complex128)
        for i in range(num_states):
            X[i] = random_statevector(dims=d).data
        for i in range(num_states):
            Y[i] = random_statevector(dims=d).data

        random_phase = np.random.uniform(0, 2 * np.pi, num_states)
        for i in range(len(Y)):
            Y[i] *= np.exp(1j * random_phase[i])

        U_pred, max_fid = fubini_distance.max_sum_sqrt_fidelity(X, Y, num_trials=1000, get_unitary=True)
        assert max_fid > 0 and max_fid < num_states

        assert_almost_equal(U_pred @ U_pred.conj().T, np.eye(d), decimal=4)

        # Sum of square root fidelity
        sum_fid = 0
        for i in range(num_states):
            sum_fid += np.abs(np.inner(Y[i].conj(), X[i] @ U_pred))

        assert_almost_equal(sum_fid, max_fid)

        # Sum of square root fidelity, vectorized
        X_transformed = (X @ U_pred)
        all_fid = sum(np.abs(np.sum(Y.conj() * X_transformed, axis=1)))
        assert_almost_equal(all_fid, max_fid)


        U_random = []
        fid_random = []
        for _ in range(10000):
            U_random.append(random_unitary(dims=d).data)
            X_transformed = (X @ U_random[-1])
            fid_random.append(sum(np.abs(np.sum(Y.conj() * X_transformed, axis=1))))
        print(max_fid / num_states, max(fid_random) / num_states)
        assert max_fid >= max(fid_random)