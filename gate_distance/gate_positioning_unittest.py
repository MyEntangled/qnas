import unittest
import numpy as np
from numpy.testing import *

from gate_positioning import *

from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class Test(unittest.TestCase):
    def test_all_relative_positions(self):
        ## CASE 1: Two single-qubit gates
        res1a = all_relative_positions('x','rx',1)
        desired_1a = {'x_rx_1_s':[[0],[0]]}
        assert_equal(res1a, desired_1a)
        res1b = all_relative_positions('x','rx',3)
        desired_1b = {'x_rx_3_s':[[0],[0]], 'x_rx_3_d':[[0],[1]]}
        assert_equal(res1b, desired_1b)

        ## CASE 2: One single, one two-qubit gate
        res2a = all_relative_positions('rx','cy',2)
        desired_2a = {'rx_cy_2_up':[[0],[0,1]], 'rx_cy_2_lo':[[1],[0,1]]}
        assert_equal(res2a, desired_2a)

        res2b = all_relative_positions('rx', 'cy',3)
        desired_2b = {'rx_cy_3_up': [[0], [0, 1]], 'rx_cy_3_lo': [[1], [0, 1]], 'rx_cy_3_d':[[2],[0,1]]}
        assert_equal(res2b, desired_2b)

        res2c = all_relative_positions('rx','ryy',2)
        desired_2c = {'rx_ryy_2_over':[[0],[0,1]]}
        assert_equal(res2c, desired_2c)

        res2d = all_relative_positions('rx','ryy',3)
        desired_2d = {'rx_ryy_3_over': [[0], [0,1]], 'rx_ryy_3_d': [[2], [0,1]]}
        assert_equal(res2c, desired_2c)

        ## CASE 3: Two two-qubit gates
        res3a = all_relative_positions('cx','cy',4)
        desired_3a = {'cx_cy_4_alig':[[0,1],[0,1]], 'cx_cy_4_anti':[[0,1],[1,0]],
                      'cx_cy_4_upalig':[[0,1],[0,2]], 'cx_cy_4_upanti':[[0,1],[2,0]],
                      'cx_cy_4_loalig':[[0,1],[2,1]], 'cx_cy_4_loanti':[[0,1],[1,2]],
                      'cx_cy_4_d':[[0,1],[2,3]]}
        assert_equal(res3a, desired_3a)

        res3b = all_relative_positions('ryy','crx',4)
        desired_3b = {'ryy_crx_4_up':[[0,1],[1,2]], 'ryy_crx_4_lo':[[1,2],[0,1]],
                      'ryy_crx_4_s':[[0,1],[0,1]], 'ryy_crx_4_d':[[0,1],[2,3]]}
        assert_equal(res3b, desired_3b)

        res3c = all_relative_positions('rxx','ryy',4)
        desired_3c = {'rxx_ryy_4_s':[[0,1],[0,1]], 'rxx_ryy_4_over':[[0,1],[1,2]],
                      'rxx_ryy_4_d':[[0,1],[2,3]]}
        assert_equal(res3c, desired_3c)
    def test_get_pos_from_gate_name(self):
        ## CASE 1: Two single-qubit gates
        res1a = get_pos_from_gate_name('x','rx',[[3],[3]])
        desired_1a = 's'
        assert_equal(res1a, desired_1a)

        res1b = get_pos_from_gate_name('x', 'rx', [[3],[0]])
        desired_1b = 'd'
        assert_equal(res1b, desired_1b)

        ## CASE 2: One single, one two-qubit gate
        res2a = get_pos_from_gate_name('rx','cry', [[1],[4,1]])
        desired_2a = 'lo'
        assert_equal(res2a, desired_2a)

        res2b = get_pos_from_gate_name('rx','cry',[[1],[1,4]])
        desired_2b = 'up'
        assert_equal(res2b, desired_2b)

        res2c = get_pos_from_gate_name('rx','cry',[[1],[0,4]])
        desired_2c = 'd'
        assert_equal(res2c, desired_2c)

        res2d = get_pos_from_gate_name('rx','ryy',[[3],[3,1]])
        desired_2d = 'over'
        assert_equal(res2d, desired_2d)

        ## CASE 3: Two two-qubit gates
        res3a = get_pos_from_gate_name('crx','crx',[[3,2],[3,2]])
        desired_3a = 'alig'
        assert_equal(res3a, desired_3a)

        res3b = get_pos_from_gate_name('crx','crx', [[3,2], [1,3]])
        desired_3b = 'upanti'
        assert_equal(res3b, desired_3b)

        res3c = get_pos_from_gate_name('crx', 'crx', [[3,2], [4,2]])
        desired_3c = 'loalig'
        assert_equal(res3c, desired_3c)
    def test_get_pos_from_gate_obj(self):
        qc1 = QuantumCircuit(4)
        qc1.rx(0.1, 2)
        qc1.cry(0.2, 3, 1)
        qc1.rzz(0.3, 2, 0)
        dag1 = circuit_to_dag(qc1)

        qc2 = QuantumCircuit(4)
        qc2.rx(0.1, 0)
        qc2.cry(0.2, 2, 3)
        qc2.rzz(0.3, 0, 2)
        dag2 = circuit_to_dag(qc2)

        for node1 in dag1.op_nodes():
            for node2 in dag2.op_nodes():
                print(node1.name, node2.name, get_pos_from_gate_DAGobj(node1, node2))


