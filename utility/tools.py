import qiskit
from qiskit.opflow import I, X, Y, Z

def prepare_multiprocesses(num_trials, num_processes):
    indices = []
    start = 0
    end = 0
    size = num_trials // num_processes

    for i in range(num_processes - 1):
        end = start + size
        indices += [list(range(start, end))]
        start = end

    indices += [list(range(end, num_trials))]

    return indices

def compose_param_dict(variables, values):
    circ_param_dict = dict(zip(variables, values))
    return circ_param_dict


def get_measurement_operator(observable):

    if isinstance(observable, qiskit.opflow.primitive_ops.pauli_sum_op.PauliSumOp):
        return observable
    elif isinstance(observable, qiskit.opflow.primitive_ops.pauli_op.PauliOp):
        return observable

    else:
        for idx, bit in enumerate(observable):
            if bit == '0':
                op = (I + Z) / 2
            elif bit == '1':
                op = (I - Z) / 2
            if idx == 0:
                operator = op
            else:
                operator = operator ^ op
        return operator

def generate_bitstrings(length):
    bitstrings = []

    def genbin(n, bs=''):
        if len(bs) == n:
            bitstrings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')

    genbin(length)
    return bitstrings

def get_parity(bitstring):
    parity = 0
    for bit in bitstring:
        parity += int(bit)
    parity = parity % 2
    return parity

def get_parity_observables(num_qubits):

    observables = generate_bitstrings(num_qubits)
    basis_ops = [get_measurement_operator(observable) for observable in observables]

    even_observable = sum(basis_ops[0::2])
    odd_observable = sum(basis_ops[1::2])

    #print(observables[0::2], observables[1::2])

    return [even_observable, odd_observable]



if __name__ == '__main__':
    a = get_parity_observables(3)
    #print(even.to_matrix(), odd.to_matrix())
    s = sum(a)
    iden = I^I^I
    #print((s.to_matrix() == iden.to_matrix()).all())