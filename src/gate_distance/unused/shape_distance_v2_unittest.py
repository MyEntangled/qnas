import unittest
import numpy as np
from numpy.testing import *

import shape_distance_v2
import shape_distance_v3
from qiskit.quantum_info import random_statevector, random_unitary

class Test(unittest.TestCase):

    def test_minimize_permutation(self):
        num_qubits = 1
        d = 2**num_qubits
        num_anchors = d*(d+1)
        num_samples = 10

        ## Initialize spectrum_A == spectrum_B
        spectrum_A = np.zeros(shape=(num_anchors,num_samples,d), dtype=np.complex128)
        for i in range(spectrum_A.shape[0]):
            for j in range(spectrum_A.shape[1]):
                spectrum_A[i,j] = random_statevector(d).data
        spectrum_B = spectrum_A.copy()

        ## Reorder anchors of spectrum_B
        new_anchor_order = np.random.permutation(num_anchors)
        spectrum_B = spectrum_B[new_anchor_order]

        #inds = shape_distance_v2.maximize_fidelity_permutation(spectrum_A, spectrum_B, None, None)
        inds = shape_distance_v3.optimize_permutation(spectrumA=spectrum_A, spectrumB=spectrum_B)
        print(inds)
        assert_equal(inds[new_anchor_order], np.array(range(num_anchors)))


