import unittest
import numpy as np
from numpy.testing import *

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter,ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from qc_embedding import qc_to_enc, enc_to_qc

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, \
    SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, \
    TWO_QUBIT_VARIATIONAL_GATES, \
    ADMISSIBLE_GATES, \
    DIRECTED_GATES, \
    UNITARY, \
    OP_NODE_DICT


class Test(unittest.TestCase):
    def compare_qc_data(self, qc1:QuantumCircuit, qc2:QuantumCircuit):
        self.assertEqual(len(qc1.data), len(qc2.data), "Unequal data length")
        for idx in range(len(qc1.data)):
            inst1, qarg1, carg1 = qc1.data[idx]
            inst2, qarg2, carg2 = qc2.data[idx]
            self.assertEqual(qarg1, qarg2, "Unmatched qargs")
            self.assertEqual(carg1, carg2, "Unmatched cargs")
            self.assertEqual(inst1.name, inst2.name, "Unmatched instruction names")
            self.assertEqual(len(inst1.params), len(inst2.params), "Unmatched parameter lengths")
            params1 = inst1.params
            params2 = inst2.params

            for param_idx in range(len(inst1.params)):
                if not (type(params1[param_idx]) in [ParameterVectorElement, Parameter] and type(params2[param_idx]) in [ParameterVectorElement, Parameter]):
                    self.assertEqual(params1[param_idx], params2[param_idx])
    def test_encoding_1qubit(self):
        ## Case 1: 1 qubit, no parameter
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.x(0)
        enc1a = qc_to_enc(qc1)
        enc1b = qc_to_enc(qc1, MAX_OP_NODES=4)
        assert_equal(enc1a.shape, (2*2,))
        assert_equal(enc1b.shape, (2*4,))

        enc1a = enc1a.reshape(2,2)
        enc1b = enc1b.reshape(2,4)

        desired_1a = np.array([[1,1],[OP_NODE_DICT['h'], OP_NODE_DICT['x']]])
        desired_1b = np.array([[1,1,0,0], [OP_NODE_DICT['h'], OP_NODE_DICT['x'], 0, 0]])
        assert_equal(enc1a, desired_1a)
        assert_equal(enc1b, desired_1b)

        ## Case 2: 1 qubit, with parameters
        qc2 = QuantumCircuit(1)
        qc2.h(0)
        qc2.ry(15,0)
        qc2.x(0)
        qc2.rz(1.1,0)

        enc2a = qc_to_enc(qc2)
        enc2b = qc_to_enc(qc2, MAX_OP_NODES=5)
        assert_equal(enc2a.shape, (2*4,))
        assert_equal(enc2b.shape, (2*5,))

        enc2a = enc2a.reshape(2,4)
        enc2b = enc2b.reshape(2,5)

        desired_2a = np.array([[1,1,1,1],[OP_NODE_DICT['h'], OP_NODE_DICT['ry'], OP_NODE_DICT['x'], OP_NODE_DICT['rz']]])
        desired_2b = np.array([[1,1,1,1,0], [OP_NODE_DICT['h'], OP_NODE_DICT['ry'], OP_NODE_DICT['x'], OP_NODE_DICT['rz'], 0]])
        assert_equal(enc2a, desired_2a)
        assert_equal(enc2b, desired_2b)
    def test_encoding_manyqubits(self):
        ## Case 1: 4 qubit, with parameters
        qc = QuantumCircuit(4)
        qc.x(1)
        qc.rz(0.4, 3)
        qc.crx(0.7, 1, 2)
        qc.cy(3, 0)
        qc.rxx(0.4, 0, 1)
        enca = qc_to_enc(qc)
        encb = qc_to_enc(qc, MAX_OP_NODES=6)
        assert_equal(enca.shape, (5*5,))
        assert_equal(encb.shape, (5*6,))

        enca = enca.reshape(5,5)
        encb = encb.reshape(5,6)

        desired_a = np.array([[0,0,0,0.25,0.75],[1,0,0.75,0,0.25],[0,0,0.25,0,0],[0,1,0,0.75,0],[OP_NODE_DICT['x'], OP_NODE_DICT['rz'], OP_NODE_DICT['crx'], OP_NODE_DICT['cy'], OP_NODE_DICT['rxx']]])
        desired_b = np.array([[0,0,0,0.25,0.75,0],[1,0,0.75,0,0.25,0],[0,0,0.25,0,0,0],[0,1,0,0.75,0,0],[OP_NODE_DICT['x'], OP_NODE_DICT['rz'], OP_NODE_DICT['crx'], OP_NODE_DICT['cy'], OP_NODE_DICT['rxx'], 0]])
        assert_equal(enca, desired_a)
        assert_equal(encb, desired_b)
    def test_decoding_1qubit(self):
        ## CASE 1: 1 QUBIT, NO PARAMETER
        enc1a = np.array([[1,1],[OP_NODE_DICT['h'], OP_NODE_DICT['x']]])
        enc1b = np.array([[1,0,1,0], [OP_NODE_DICT['h'], 0, OP_NODE_DICT['x'], 0]])

        qc1a = enc_to_qc(num_qubits=1, encoding=enc1a)
        qc1b = enc_to_qc(num_qubits=1, encoding=enc1b)

        desired_1a = QuantumCircuit(1)
        desired_1a.h(0)
        desired_1a.x(0)

        desired_1b = QuantumCircuit(1)
        desired_1b.h(0)
        desired_1b.x(0)

        assert qc1a.data == desired_1a.data
        assert qc1b.data == desired_1b.data


        ## CASE 2: 1 QUBIT, WITH PARAMETER
        enc2a = np.array([[1,1,1,1],[OP_NODE_DICT['h'], OP_NODE_DICT['ry'], OP_NODE_DICT['x'], OP_NODE_DICT['rz']]])
        enc2b = np.array([[1,0,1,1,1],[OP_NODE_DICT['h'], 0, OP_NODE_DICT['ry'], OP_NODE_DICT['x'], OP_NODE_DICT['rz']]])

        qc2a = enc_to_qc(num_qubits=1, encoding=enc2a)
        qc2b = enc_to_qc(num_qubits=1, encoding=enc2b)

        desired_2a = QuantumCircuit(1)
        theta_2a = ParameterVector('theta', length=0)
        desired_2a.h(0)
        theta_2a.resize(len(theta_2a)+1)
        desired_2a.ry(theta_2a[-1],0)
        desired_2a.x(0)
        theta_2a.resize(len(theta_2a) + 1)
        desired_2a.rz(theta_2a[-1],0)

        desired_2b = QuantumCircuit(1)
        theta_2b = ParameterVector('theta', length=0)
        desired_2b.h(0)
        theta_2b.resize(len(theta_2a)+1)
        desired_2b.ry(theta_2a[-1],0)
        desired_2b.x(0)
        theta_2b.resize(len(theta_2a) + 1)
        desired_2b.rz(theta_2a[-1],0)

        self.compare_qc_data(qc2a,desired_2a)
        self.compare_qc_data(qc2b,desired_2b)
    def test_decoding_manyqubits(self):
        ## MANY QUBITS, WITH PARAMETERS
        enca = np.array([[0,0,0,0.25,0.75],[1,0,0.75,0,0.25],[0,0,0.25,0,0],[0,1,0,0.75,0],[OP_NODE_DICT['x'], OP_NODE_DICT['rz'], OP_NODE_DICT['crx'], OP_NODE_DICT['cy'], OP_NODE_DICT['rxx']]])
        encb = np.array([[0,0,0,0.25,0.75,0],[1,0,0.75,0,0.25,0],[0,0,0.25,0,0,0],[0,1,0,0.75,0,0],[OP_NODE_DICT['x'], OP_NODE_DICT['rz'], OP_NODE_DICT['crx'], OP_NODE_DICT['cy'], OP_NODE_DICT['rxx'], 0]])

        qca = enc_to_qc(num_qubits=4, encoding=enca)
        qcb = enc_to_qc(num_qubits=4, encoding=encb)

        desired_a = QuantumCircuit(4)
        theta_a = ParameterVector('theta', length=3)
        desired_a.x(1)
        desired_a.rz(theta_a[0], 3)
        desired_a.crx(theta_a[1], 1, 2)
        desired_a.cy(3, 0)
        desired_a.rxx(theta_a[2], 0, 1)

        desired_b = QuantumCircuit(4)
        theta_b = ParameterVector('theta', length=3)
        desired_b.x(1)
        desired_b.rz(theta_b[0], 3)
        desired_b.crx(theta_b[1], 1, 2)
        desired_b.cy(3, 0)
        desired_b.rxx(theta_b[2], 0, 1)


        self.compare_qc_data(qca,desired_a)
        self.compare_qc_data(qcb,desired_b)
    def test_encoding_decoding(self):
        ## CASE: Encoding first, then decoding
        qc = QuantumCircuit(4)
        theta = ParameterVector('theta', length=3)
        qc.x(1)
        qc.rz(theta[0], 3)
        qc.crx(theta[1], 1, 2)
        qc.cy(3, 0)
        qc.rxx(theta[2], 0, 1)

        enc = qc_to_enc(qc)
        rec_qc = enc_to_qc(4,enc)
        self.compare_qc_data(qc, rec_qc)
    def test_decoding_encoding(self):
        ## CASE: Decoding first, encoding again
        enc = np.array([[0, 0, 0, 0.25, 0.75], [1, 0, 0.75, 0, 0.25], [0, 0, 0.25, 0, 0], [0, 1, 0, 0.75, 0],
                         [OP_NODE_DICT['x'], OP_NODE_DICT['rz'], OP_NODE_DICT['crx'], OP_NODE_DICT['cy'],
                          OP_NODE_DICT['rxx']]])
        qc = enc_to_qc(4, enc)
        rec_enc = qc_to_enc(qc)

        assert_equal(enc.ravel(),rec_enc)
    def test_decoding_random_enc(self):
        np.random.seed(20)
        ## CASE 1: 1 QUBIT
        for _ in range(1000):
            enc = np.random.uniform(-0.2,1,size=(2,10))
            qc = enc_to_qc(1, enc)
            assert isinstance(qc, QuantumCircuit)




test = Test()

test.test_encoding_1qubit()
test.test_encoding_manyqubits()

test.test_decoding_1qubit()
test.test_decoding_manyqubits()

test.test_encoding_decoding()
test.test_decoding_encoding()

test.test_decoding_random_enc()
print('Unit test for quantum circuit embedding: Done!')