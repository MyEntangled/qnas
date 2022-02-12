import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import pennylane as qml
import qiskit


class AnsatzTemplate():

    def __init__(self, PQC=None):
        if PQC:
            self.PQC = PQC
            self.num_qubits = PQC.num_qubits

        else:
            print('hi')
            self.PQC = None
            self.num_qubits = None

    def _apply_qubit_wise(self, qc, gate, theta):
        if gate == 'rx':
            for i in range(self.num_qubits):
                qc.rx(theta[i], i)
        elif gate == 'ry':
            for i in range(self.num_qubits):
                qc.ry(theta[i], i)
        elif gate == 'rz':
            for i in range(self.num_qubits):
                qc.rz(theta[i], i)
        return

    def construct_simple_template(self, num_qubits, num_layers):
        # self.num_qubits = num_qubits
        #
        # qc = QuantumCircuit(self.num_qubits)
        # param_dim = (2 * self.num_qubits) * num_layers + 2 * self.num_qubits
        # theta = ParameterVector('θ', param_dim)
        #
        # start = 0
        # end = 0
        # qc.h(range(self.num_qubits))
        #
        # for i in range(num_layers):
        #     start = end
        #     end = start + self.num_qubits
        #     self._apply_qubit_wise(qc, 'rx', theta[start:end])
        #
        #     start = end
        #     end = start + self.num_qubits
        #     self._apply_qubit_wise(qc, 'ry', theta[start:end])
        #
        #     for q in range(num_qubits - 1):
        #         qc.cnot(q, q + 1)
        #
        # start = end
        # end = start + self.num_qubits
        # self._apply_qubit_wise(qc, 'rx', theta[start:end])
        #
        # start = end
        # end = start + self.num_qubits
        # self._apply_qubit_wise(qc, 'ry', theta[start:end])
        # # qc.ry(range(self.num_qubits), theta[start:end])
        #
        # self.PQC = qc

        self.num_qubits = num_qubits
        twolocal = qiskit.circuit.library.TwoLocal(self.num_qubits, ['rx', 'ry'], 'cx', 'circular', reps=num_layers)
        qc = QuantumCircuit(self.num_qubits).compose(twolocal)
        self.PQC = qc.decompose()

    def visualize(self, output=None):
        print(self.PQC.draw(output=output))


if __name__ == '__main__':

    template = AnsatzTemplate()
    template.construct_simple_template(4, 1)
    template.visualize()

    # ##### Test construction of qiskit ansatz
    # theta = ParameterVector('theta', 2)
    # qc = QuantumCircuit(2)
    # qc.rx(theta[0], 0)
    # qc.cz(0, 1)
    # qc.ry(theta[1], 1)
    # template = AnsatzTemplate(qc, type='Qiskit')
    # template.visualize()
    #
    #
    # ##### Test construction of pennylane ansatz
    # def my_quantum_function(x):
    #     qml.RZ(x[0], wires=0)
    #     qml.CNOT(wires=[0, 1])
    #     qml.RY(x[1], wires=1)
    #     return qml.expval(qml.PauliZ(wires=0))
    #
    #
    # dev = qml.device("default.qubit", wires=2)
    # circuit = qml.QNode(my_quantum_function, dev)
    # circuit([0, 1])
    #
    # template = AnsatzTemplate(circuit, type='Pennylane')
    # template.visualize()
    # print(template.PQC.parameters)
