import networkx as nx
import numpy as np
from scipy.optimize import minimize, basinhopping

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator
import qiskit.quantum_info as qi
from qiskit.opflow import StateFn

from gate_distance import MUBs
# from utility.quantum_nn import QuantumNeuralNetwork
# from utility.ansatz_template import AnsatzTemplate
# from utility.data_encoding import FeatureMap

# from surfer.gradient import ReverseGradient
# from qiskit.opflow import Gradient, StateFn, OperatorStateFn

from qiskit.compiler import transpile
from qiskit.transpiler.passes import RemoveResetInZeroState

def get_QFT_states(num_qubits, input_states=None):
    qft_circ = QFT(num_qubits=num_qubits, approximation_degree=0, do_swaps=False, inverse=False, insert_barriers=False,name=None)

    if input_states is None:
        input_states = [Statevector(state) for state in MUBs.get_anchor_states(num_qubits)]

    remove_reset_inst = RemoveResetInZeroState()
    stateprep_circs = []
    for idx, state in enumerate(input_states):
        circ = QuantumCircuit(num_qubits)
        circ.initialize(state)
        stateprep_circs.append(remove_reset_inst(circ.decompose()))

    outcome_states = [state.evolve(qft_circ) for state in input_states]
    return outcome_states

def maximize_QFT_fidelity(PQC, input_states=None, outcome_states=None):
    #num_qubits = PQC.num_qubits

    if input_states is None:
        input_states = [Statevector(state) for state in MUBs.get_anchor_states(PQC.num_qubits)]
    if outcome_states is None:
        outcome_states = get_QFT_states(num_qubits=PQC.num_qubits, input_states=input_states)
    assert len(input_states) == len(outcome_states)

    observables = [state.to_operator() for state in outcome_states]
    # rev_grad = ReverseGradient()
    # gradient = Gradient()


    def fidelity_obj(x):
        fid = 0
        #print(x)
        U = Operator(PQC.bind_parameters(x))
        output_states = [state.evolve(U) for state in input_states]
        for idx, state in enumerate(output_states):
            # fid += state.data.conj() @ observables[idx].data @ state.data
            #fid += np.trace(observables[idx].data @ state.to_operator().data)
            fid += state.expectation_value(observables[idx])
        #print(np.real(fid / len(input_states)))
        return np.real(-fid / len(input_states))

    # def get_gradient(x):
    #     grad = np.zeros(len(x))
    #     for idx,state in enumerate(input_states):
    #         #input_circ = QuantumCircuit(num_qubits)
    #         #input_circ.initialize(state)
    #         #input_circ = transpile(input_circ, optimization_level=3)
    #         grad += rev_grad.compute(observables[idx], stateprep_circs[idx].compose(PQC), x)
    #         print(grad)
    #         #expectation = ~OperatorStateFn(observables[idx]) @ StateFn(input_circ.compose(PQC))
    #         #grad += np.real(gradient.convert(expectation).bind_parameters(dict(zip(PQC.parameters, x))).eval())
    #     print(np.real(-grad))
    #     return np.real(-grad / len(input_states))

    if PQC.num_parameters > 0:
        initial_guess = np.random.rand(PQC.num_parameters)
        result = minimize(fidelity_obj, initial_guess)
        #minimizer_kwargs = dict(method="L-BFGS-B")
        #result = basinhopping(fidelity_obj, initial_guess, minimizer_kwargs=minimizer_kwargs)

        return result.x, -result.fun
    else:
        return [], -fidelity_obj([])

def define_maxcut_problem():
    n = 3
    V = np.arange(0, n, 1)
    #E = [(0, 1, 10.0), (0, 2, 5.0), (1, 2, 3.0), (1, 3, 10.0),(2, 3, 7.0)]
    E = [(0, 1, 10.0), (0, 2, 10.0), (1, 2, 9)]
    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    def define_problem_hamiltonian(G):
        H_c = 0
        for i in list(G.edges()):
            qc = QuantumCircuit(n)
            for j in i:
                qc.z(j)
            op = qi.Operator(qc)
            matrix = op.data
            H_c += 1/2*G.edges[i]['weight']*(np.eye(2**n)-matrix)
        return H_c
    return define_problem_hamiltonian(G)

def maximize_maxcut_fidelity(PQC):
    H_c = define_maxcut_problem()
    def obj_func(x):

        qc_withparam = PQC.bind_parameters(x)
        psi = StateFn(qc_withparam).to_matrix()
        expectation_value = (psi.conj().T) @ H_c @ psi
        return np.real(-expectation_value)
    if PQC.num_parameters > 0:
        initial_guess = np.random.rand(PQC.num_parameters)
        result = minimize(obj_func, initial_guess)
        return result.x, -result.fun
    else:
        return [], -obj_func([])

def check_maxcut(qc,opt_param,n):
    qc = qc.bind_parameters(opt_param)
    result_psi = np.abs(StateFn(qc).to_matrix())**2
    print(result_psi)
    state = format(np.argmax(result_psi),'0'+str(n)+'b')
    return state



if __name__ == '__main__':
    # feature_map = FeatureMap('AmplitudeEmbedding', 2**3)
    #
    # template = AnsatzTemplate()
    # template.construct_simple_template(num_qubits=3, num_layers=5)
    #
    #
    # model = QuantumNeuralNetwork(None, template)
    #maximize_QFT_fidelity(model.PQC)
# *************** Test QFT *********************
    # from qiskit import QuantumCircuit
    # from qiskit.circuit import ParameterVector
    # theta = ParameterVector('t',3)
    # qc = QuantumCircuit(4)
    # qc.h(0); qc.rx(theta[0],0); qc.cry(theta[1],1,2); qc.rzz(theta[2],2,3)
    # opt_param, opt_val = maximize_QFT_fidelity(qc)
    # print(opt_param, opt_val)
    # print('Final circuit')
    # print(qc.bind_parameters(opt_param).draw())
    #
    #
    # qc = QuantumCircuit(2)
    # qc.h(0)
    # qc.crx(0.2,0,1)
    # opt_param, opt_val = maximize_QFT_fidelity(qc)
    # print(opt_param, opt_val)
    # print('Final circuit')
    # print(qc.bind_parameters(opt_param).draw())

#*************** Test max cut *********************
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    theta = ParameterVector('t', 4)
    qc = QuantumCircuit(3)
    qc.h(0);qc.h(1);qc.h(2); qc.rzz(theta[0],0,1);  qc.rzz(theta[1],0,2); qc.rzz(theta[2],1,2); qc.rx(theta[3],0);qc.rx(theta[3],1);qc.rx(theta[3],2)
    opt_param, opt_val = maximize_maxcut_fidelity(qc)
    state = check_maxcut(qc,opt_param,3)
    print('Max cut: ' + state)
    print(opt_param, opt_val)
    print('Final circuit')
    print(qc.bind_parameters(opt_param).draw())
