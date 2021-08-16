import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import pennylane as qml
import qiskit


class AnsatzTemplate():

    def __init__(self, PQC=None, type='Qiskit'):
        if PQC:

            if type == 'Qiskit':
                self.PQC = PQC
                self.num_qubits = PQC.num_qubits

            elif type == 'Pennylane':
                assert isinstance(PQC, qml.QNode), "PQC must be defined as a QNode"

                self.num_qubits = len(PQC.device.wires)

                func = PQC.func
                dev = qml.device("qiskit.aer", wires=self.num_qubits)  ## switch device to qiskit.aer
                placeholder_param = np.zeros(10000)  ## a big array just to make sure it suffice for every param

                qnode = qml.QNode(func, dev)
                qnode(placeholder_param)  ## initiate qnode
                assert qnode.specs['num_device_wires'] == qnode.specs[
                    'num_used_wires'], "Number of used qubits must be equal to number of device qubits"

                temp_qc = dev._circuit
                temp_qc.remove_final_measurements()

                param_dim = qnode.specs['num_trainable_params']
                theta = ParameterVector('θ', param_dim)

                qc = QuantumCircuit(self.num_qubits)

                i = 0

                for op in temp_qc.data:

                    if len(op[0].params) > 1:
                        raise Exception(
                            'Pennylane quantum function must contain non-repeating single parameter gates only.')
                    elif len(op[0].params) == 1:
                        gate = op[0].__deepcopy__()
                        gate.__init__(theta[i])
                        qc.append(gate, op[1])
                        i += 1
                    else:
                        gate = op[0].__deepcopy__()
                        qc.append(gate, op[1])

                self.PQC = qc

        else:
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
        twolocal = qiskit.circuit.library.TwoLocal(self.num_qubits, ['rx', 'ry'], 'cx', 'linear', reps=num_layers)
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
