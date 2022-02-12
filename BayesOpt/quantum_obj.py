import numpy as np
from scipy.optimize import minimize, basinhopping

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator

from gate_distance import MUBs
from utility.quantum_nn import QuantumNeuralNetwork
from utility.ansatz_template import AnsatzTemplate
from utility.data_encoding import FeatureMap

from surfer.gradient import ReverseGradient
from qiskit.opflow import Gradient, StateFn, OperatorStateFn

from qiskit.compiler import transpile
from qiskit.transpiler.passes import RemoveResetInZeroState


def maximize_QFT_fidelity(PQC, input_states=None):
    num_qubits = PQC.num_qubits
    qft_circ = QFT(num_qubits=num_qubits, approximation_degree=0, do_swaps=False, inverse=False, insert_barriers=False,name=None)

    input_states = input_states or [Statevector(state) for state in MUBs.get_anchor_states(num_qubits)]
    remove_reset_inst = RemoveResetInZeroState()
    stateprep_circs = []
    for idx, state in enumerate(input_states):
        circ = QuantumCircuit(num_qubits)
        circ.initialize(state)
        stateprep_circs.append(remove_reset_inst(circ.decompose()))

    outcome_states = [state.evolve(qft_circ) for state in input_states]
    observables = [state.to_operator() for state in outcome_states]
    rev_grad = ReverseGradient()
    gradient = Gradient()

    print(len(outcome_states))

    def fidelity_obj(x):
        #placeholder_featuremap_input = np.zeros(model.PQC.num_parameters)
        fid = 0
        # for idx, state in enumerate(input_states):
            # _, output_exps = model.forward(inputs=[state], params=[x], observables=[observables[idx]])
            # fid += output_exps[0][0]

        U = Operator(PQC.bind_parameters(x))
        output_states = [state.evolve(U) for state in input_states]
        for idx, state in enumerate(output_states):
            # fid += state.data.conj() @ observables[idx].data @ state.data
            #fid += np.trace(observables[idx].data @ state.to_operator().data)
            fid += state.expectation_value(observables[idx])
        print(np.real(fid / len(input_states)))
        return np.real(-fid / len(input_states))

    def get_gradient(x):
        grad = np.zeros(len(x))
        for idx,state in enumerate(input_states):
            #input_circ = QuantumCircuit(num_qubits)
            #input_circ.initialize(state)
            #input_circ = transpile(input_circ, optimization_level=3)
            grad += rev_grad.compute(observables[idx], stateprep_circs[idx].compose(PQC), x)
            print(grad)
            #expectation = ~OperatorStateFn(observables[idx]) @ StateFn(input_circ.compose(PQC))
            #grad += np.real(gradient.convert(expectation).bind_parameters(dict(zip(PQC.parameters, x))).eval())
        print(np.real(-grad))
        return np.real(-grad / len(input_states))

    initial_guess = np.random.rand(model.param_dim)
    result = minimize(fidelity_obj, initial_guess, method='L-BFGS-B', jac=None)
    #minimizer_kwargs = dict(method="L-BFGS-B")
    #result = basinhopping(fidelity_obj, initial_guess, minimizer_kwargs=minimizer_kwargs)
    print(result)
    print('Final circuit')
    print(PQC.bind_parameters(result.x).draw())


if __name__ == '__main__':
    feature_map = FeatureMap('AmplitudeEmbedding', 2**3)

    template = AnsatzTemplate()
    template.construct_simple_template(num_qubits=3, num_layers=5)

    model = QuantumNeuralNetwork(None, template)
    maximize_QFT_fidelity(model.PQC)





