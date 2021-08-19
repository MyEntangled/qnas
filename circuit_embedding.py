from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import pennylane as qml
import numpy as np

GATE_TYPE = {'Identity':0, 'RX':1, 'RY':2, 'RZ':3, 'CNOT_U':4, 'CNOT_D':5}
UNITARY = {0: qml.Identity, 1: qml.RX, 2: qml.RY, 3: qml.RZ, 4: qml.CNOT, 5: qml.CNOT}
NUM_param_PER_GATE = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0}
TWO_QUBIT_GATE_TYPE = ['CNOT']


def PQC_function(wires, param):

    for wire in range(wires):
        qml.Hadamard(wires = wire)

    param = iter(param)
    for wire in range(wires):
        qml.RX(next(param), wires=wire)
        qml.RZ(next(param), wires=wire)

    for wire in range(wires-1):
        qml.CNOT(wires=[wire,wire+1])
        qml.CNOT(wires=[wire+1,wire])

    for wire in range(wires):
        qml.RX(next(param), wires=wire)
        qml.RZ(next(param), wires=wire)

def circuit_to_tensor(template):
    wires = 2
    param_dim = wires * 4
    placeholder_param = np.zeros(param_dim)

    with qml.tape.QuantumTape() as tape:
        template(wires, placeholder_param)

    n_operations = len(tape.operations)

    embedding = np.zeros((len(tape.operations), wires, len(GATE_TYPE)))
    #embedding = np.zeros(embedding_shape)

    for time_idx, opr in enumerate(tape.operations):
        #print(opr.name)
        if opr.name in TWO_QUBIT_GATE_TYPE:
            ctrl, target = opr.wires.toarray()
            if ctrl < target:
                gate_type_idx = GATE_TYPE[opr.name + '_U']
                embedding[time_idx, ctrl, gate_type_idx] = 1
            elif ctrl > target:
                gate_type_idx = GATE_TYPE[opr.name + '_D']
                embedding[time_idx, ctrl, gate_type_idx] = 1

        elif opr.name in GATE_TYPE:
            qubit = opr.wires.toarray()[0]
            gate_type_idx = GATE_TYPE[opr.name]
            embedding[time_idx, qubit, gate_type_idx] = 1

    return embedding


def tensor_to_circuit(embedding):
    '''

    :param embedding: 3-D tensor: moment x qubit x gate_type
    :return:
    '''

    n_operations, wires, _ = embedding.shape
    operations = []

    for time_idx in range(n_operations):

        for qubit in range(wires):

            gate_type_idx = np.argmax(embedding[time_idx, qubit])

            if gate_type_idx != 0:
                num_param_per_gate = NUM_param_PER_GATE[gate_type_idx]
                operations.append((qubit, gate_type_idx, num_param_per_gate))

            else:
                pass

    def operations_to_circuit(param, operations):
        param_dim = np.sum(np.array(operations)[:,2])
        param = np.zeros(param_dim)
        param_idx = 0

        for qubit, gate_type_idx, num_param_per_gate in operations:
            #print(qubit, int(gate_type_idx), num_param_per_gate)

            gate_type_idx_int = int(gate_type_idx)
            if num_param_per_gate == 0:
                if gate_type_idx < 4:
                    UNITARY[gate_type_idx_int](wires = qubit)
                elif gate_type_idx == 4:
                    UNITARY[gate_type_idx_int](wires = [qubit, qubit + 1])
                elif gate_type_idx == 5:
                    UNITARY[gate_type_idx_int](wires = [qubit, qubit - 1])

            elif num_param_per_gate == 1:
                gate_param = param[param_idx]
                param_idx += 1
                if gate_type_idx < 4:
                    UNITARY[gate_type_idx_int](gate_param, wires = qubit)
                elif gate_type_idx == 4:
                    UNITARY[gate_type_idx_int](gate_param, wires = [qubit, qubit + 1])
                elif gate_type_idx == 5:
                    UNITARY[gate_type_idx_int](gate_param, wires = [qubit, qubit - 1])

            else:
                gate_param = tuple(param[param_idx: param_idx + num_param_per_gate])
                param_idx += num_param_per_gate
                if gate_type_idx < 4:
                    UNITARY[gate_type_idx_int](gate_param, wires = qubit)
                elif gate_type_idx == 4:
                    UNITARY[gate_type_idx_int](gate_param, wires = [qubit, qubit + 1])
                elif gate_type_idx == 5:
                    UNITARY[gate_type_idx_int](gate_param, wires = [qubit, qubit - 1])
            #print('DONE', qubit, int(gate_type_idx), num_param_per_gate)

        return qml.sample(qml.PauliZ(0))

    param_dim = np.sum(np.array(operations)[:, 2])
    param = np.zeros(param_dim)

    dev = qml.device("default.qubit", wires=wires, shots=1)
    qnode = qml.QNode(operations_to_circuit, dev)
    print(qnode(param, operations = operations))
    print(qnode.draw())

test_tensor = circuit_to_tensor(PQC_function)
reconstructed_circuit = tensor_to_circuit(test_tensor)
print(test_tensor)
reconstructed_circuit
