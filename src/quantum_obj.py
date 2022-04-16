import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator

from gate_distance import MUBs

# from surfer.gradient import ReverseGradient
# from qiskit.opflow import Gradient, StateFn, OperatorStateFn

from qiskit.compiler import transpile
from qiskit.transpiler.passes import RemoveResetInZeroState

from qiskit.opflow import StateFn, OperatorStateFn, CircuitStateFn,I,X,Y,Z

import networkx as nx

from qiskit import QuantumCircuit, BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import PyTorchDiscriminator, QGAN

from torch import nn, optim
from qiskit.algorithms.optimizers import ADAM

import matplotlib.pyplot as plt

class QFT_objective():
    def __init__(self, num_qubits, input_states=None):
        self.num_qubits = num_qubits
        if input_states is None:
            self.input_states = [Statevector(state) for state in MUBs.get_anchor_states(num_qubits)]
        else:
            self.input_states = input_states

        self.outcome_states = self.get_QFT_states(num_qubits=self.num_qubits, input_states=self.input_states)
        #print(self.outcome_states[:4])

    def get_QFT_states(self, num_qubits, input_states):
        qft_circ = QFT(num_qubits=num_qubits, approximation_degree=0, do_swaps=False, inverse=False, insert_barriers=False,name=None)

        # remove_reset_inst = RemoveResetInZeroState()
        # stateprep_circs = []
        # for idx, state in enumerate(input_states):
        #     circ = QuantumCircuit(num_qubits)
        #     circ.initialize(state)
        #     stateprep_circs.append(remove_reset_inst(circ.decompose()))

        outcome_states = [state.evolve(qft_circ) for state in input_states]
        return outcome_states

    def maximize_QFT_fidelity(self, PQC, input_states=None, outcome_states=None):
        #num_qubits = PQC.num_qubits

        input_states = input_states or self.input_states
        outcome_states = outcome_states or self.outcome_states

        assert len(input_states) == len(outcome_states)

        observables = [state.to_operator() for state in outcome_states]

        def fidelity_obj(x):
            fid = 0
            #print(x)
            U = Operator(PQC.bind_parameters(x))
            output_states = [state.evolve(U) for state in input_states]
            for idx, state in enumerate(output_states):
                fid += state.expectation_value(observables[idx])
            #print(np.real(fid) / len(input_states))
            return np.real(-fid) / len(input_states)

        if PQC.num_parameters > 0:
            initial_guess = np.random.rand(PQC.num_parameters)
            result = minimize(fidelity_obj, initial_guess)

            #return result.x, -result.fun
            return -result.fun
        else:
            #return [], -fidelity_obj([])
            return -fidelity_obj([])


class MAXCUT_objective():
    def __init__(self, graphs=None, num_graphs=None, num_nodes=None, num_edges=None):
        self.graphs = graphs or self.generate_random_graphs(num_graphs, num_nodes, num_edges=num_edges)
        self.hamiltonians = [self.maxcut_hamiltonian(G) for G in self.graphs]
        self.opt_cut_vals = [self.classical_maxcut(G)[0] for G in self.graphs]
        self.sum_opt_cut_val = sum(self.opt_cut_vals)

    # def define_maxcut_problem(self):
    #     n = 3
    #     V = np.arange(0, n, 1)
    #     #E = [(0, 1, 10.0), (0, 2, 5.0), (1, 2, 3.0), (1, 3, 10.0),(2, 3, 7.0)]
    #     E = [(0, 1, 10.0), (0, 2, 10.0), (1, 2, 9)]
    #     G = nx.Graph()
    #     G.add_nodes_from(V)
    #     G.add_weighted_edges_from(E)
    #     return self.maxcut_hamiltonian(G)
    def compute_cut_value(self, G, partitions):
        """
        Return list of cut weights corresponding to input partitions

        Args:
            G : Graph
            A weighted graph

            partitions : list(str)
            List of bitstring
        """

        vertices = list(G.nodes().keys())
        edges = G.edges()
        cut_vals = []

        for partition in partitions:
            cut = 0

            idx_A = [idx for idx, char in enumerate(partition) if char == '0']
            idx_B = [idx for idx, char in enumerate(partition) if char == '1']

            half_A = [vertices[i] for i in idx_A]
            half_B = [vertices[i] for i in idx_B]

            for v1 in half_A:
                for v2 in half_B:
                    if (v1, v2) in edges:
                        cut += G[v1][v2]['weight']
            cut_vals.append(cut)

        return cut_vals

    def classical_maxcut(self, G):
        """
        Return the maximum cut value and all cuts that produce the maximum value.

        Args:
            G : nx.Graph
            A weighted graph
        """
        n = len(G.nodes())
        partitions = []

        # Generate all bitstrings of length n
        for i in range(2 ** n):
            partitions.append(bin(i)[2:].zfill(n))

        cut_vals = self.compute_cut_value(G, partitions)
        max_cut_val = max(cut_vals)

        max_cuts = [p for idx, p in enumerate(partitions) if cut_vals[idx] == max_cut_val]

        return max_cut_val, max_cuts

    def generate_random_graphs(self,num_graphs:int, num_nodes:int, num_edges:int=None):
        """
        Generate graphs with same topology but different weights
        """
        graphs = []
        if num_edges is None:
            G_form = nx.complete_graph(num_nodes)
        else:
            G_form = nx.gnm_random_graph(n=num_nodes, m=num_edges, seed=np.random)

        for i in range(num_graphs):
            G = G_form.copy()
            for (u, v) in G.edges():
                #G.edges[u, v]['weight'] = np.random.randint(0, 10)
                G.edges[u, v]['weight'] = np.random.rand()
            graphs.append(G)
        return graphs

    def maxcut_hamiltonian(self,G):
        n = len(G.nodes)
        H_c = 0
        for edge in list(G.edges()):
            qc = QuantumCircuit(n)
            for end_node in edge:
                qc.z(end_node)
            op = Operator(qc)
            matrix = op.data
            H_c += 1/2*G.edges[edge]['weight']*(np.eye(2**n)-matrix)
        #return Operator(H_c)
        return H_c

    def maximize_maxcut_hamiltonian(self, PQC, graphs:list=None, hamiltonians:list=None, opt_cut_vals:list=None):
        graphs = graphs or self.graphs
        hamiltonians = hamiltonians or self.hamiltonians
        if opt_cut_vals is None:
            opt_cut_vals =  self.opt_cut_vals
            sum_opt_cut_val = self.sum_opt_cut_val
        else:
            sum_opt_cut_val = self.sum_opt_cut_val

        assert len(graphs) == len(hamiltonians) and len(graphs) == len(opt_cut_vals)

        # def maxcut_obj(x):
        #     # qc = PQC.bind_parameters(x)
        #     # psi = CircuitStateFn(qc).to_matrix()
        #     # expectation_value = sum( [(psi.conj().T @ H @ psi) for H in hamiltonians] )
        #     # return np.real(-expectation_value) / sum_opt_cut_val
        #
        #     if sum_opt_cut_val == 0:
        #         return -1.
        #
        #     U = Operator(PQC.bind_parameters(x))
        #     output_state = Statevector.from_label('0'*PQC.num_qubits).evolve(U)
        #     exp_vals = [output_state.expectation_value(Operator(H)).real for H in hamiltonians]
        #     return np.real(-sum(exp_vals)) / sum_opt_cut_val
        #
        # if PQC.num_parameters > 0:
        #     initial_guess = np.random.uniform(0,2*np.pi,PQC.num_parameters)
        #     result = minimize(maxcut_obj, initial_guess)
        #     return result.x, -result.fun
        # else:
        #     return [], -maxcut_obj([])

        def maxcut_obj_single(x, idx):
            if opt_cut_vals[idx] == 0:
                return -1.

            U = Operator(PQC.bind_parameters(x))
            output_state = Statevector.from_label('0'*PQC.num_qubits).evolve(U)
            exp_val = output_state.expectation_value(Operator(hamiltonians[idx])).real
            return -np.real(exp_val)

        if sum_opt_cut_val == 0:
            return 1.

        if PQC.num_parameters > 0:
            sum_exp_val = 0.
            for idx in range(len(graphs)):
                initial_guess = np.random.uniform(0,2*np.pi,PQC.num_parameters)
                result = minimize(maxcut_obj_single, initial_guess, args=(idx))
                sum_exp_val += result.fun
            return -sum_exp_val / sum_opt_cut_val
        else:
            sum_exp_val = 0
            for idx in range(len(graphs)):
                sum_exp_val += maxcut_obj_single([],idx)
            return -sum_exp_val / sum_opt_cut_val





    # def check_maxcut(self, PQC ,param, n):
    #     qc = PQC.bind_parameters(param)
    #     result_psi = np.abs(StateFn(qc).to_matrix())**2
    #     print(result_psi)
    #     state = format(np.argmax(result_psi),'0'+str(n)+'b')
    #     return state

    def get_max_cut_value(self, G, PQC, param):
        qc = PQC.bind_parameters(param)
        n = qc.num_qubits
        output_state = Statevector.from_label('0'*n).evolve(qc)
        prob_dict = output_state.probabilities_dict()

        max_prob = max(prob_dict.values())
        cuts = [key[::-1] for key,prob in prob_dict.items() if prob == max_prob]
        #print(cuts)
        cuts_value = self.compute_cut_value(G,partitions=cuts)
        #print(cuts_value)
        return max(cuts_value)

    def all_maxcuts_stat(self, PQC, param, graphs=None):
        if graphs is None:
            graphs = self.graphs
            sum_opt_cut_val = self.sum_opt_cut_val
        else:
            opt_cut_vals =  [self.classical_maxcut(G)[0] for G in graphs]
            sum_opt_cut_val = sum(opt_cut_vals)

        max_cut_values = [self.get_max_cut_value(G, PQC, param) for G in graphs]
        sum_max_cut_val = sum(max_cut_values)

        ## The former is from the circuit, the latter is the sum of (global) max cuts.
        return sum_max_cut_val, sum_opt_cut_val

class QGAN_objective():
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qubit_set = [self.num_qubits]
        self.bounds = np.array([0., 2**self.num_qubits-1.])

    @staticmethod
    def generate_data(distribution_type, mu, sigma, sample_size=10000):
        if distribution_type == 'lognormal':
            data = np.random.lognormal(mean=mu,sigma=sigma,size=sample_size)
        elif distribution_type == 'normal':
            data = np.random.normal(loc=mu, scale=sigma, size=sample_size)
        else: ## 'mixnormal'
            assert len(mu) == len(sigma)
            num_modes = len(mu)
            mixture = np.random.randint(low=0, high=num_modes, size=sample_size)

            data = np.zeros(sample_size)
            for i in range(num_modes):
                data[mixture == i] = np.random.normal(mu[i], sigma[i], sum(mixture == i))
        return data

    def set_true_distribution(self, distribution_type, mu, sigma, sample_size=10000):
        assert distribution_type in ['lognormal', 'normal', 'mixnormal']
        self.distribution_type = distribution_type
        self.mu = mu
        self.sigma = sigma
        self.sample_size = sample_size
        self.real_data = self.__class__.generate_data(self.distribution_type, self.mu, self.sigma, self.sample_size)


    def optimize_qgan(self, PQC, num_epochs, batch_size, seed = 27112021):
        algorithm_globals.random_seed = seed

        # Note: The algorithm's runtime can be shortened by reducing the number of training epochs.
        num_epochs = 200
        # Batch size
        batch_size = 100

        # Initialize qGAN
        qgan = QGAN(self.real_data, self.bounds, self.qubit_set, batch_size, num_epochs, snapshot_dir=None)
        qgan.seed = 1
        # Set quantum instance to run the quantum generator
        quantum_instance = QuantumInstance(
            backend=BasicAer.get_backend("statevector_simulator"), seed_transpiler=seed, seed_simulator=seed
        )


        # init_dist = QuantumCircuit(sum(self.qubit_set))
        # init_dist.h(init_dist.qubits)

        # Set generator's initial parameters - in order to reduce the training time
        init_params = np.random.uniform(-0.1, 0.1, PQC.num_parameters)

        #g_circuit = PQC.compose(init_dist, front=True)

        qgan.set_generator(generator_circuit=PQC, generator_init_params=init_params)
        qgan._generator._free_parameters = sorted(PQC.parameters, key=lambda p: p.name)
        qgan._generator._optimizer = ADAM(
            maxiter=1,
            tol=1e-6,
            lr=1e-3,
            beta_1=0.7,
            beta_2=0.99,
            noise_factor=1e-6,
            eps=1e-6,
            amsgrad=True
        )

        discriminator = PyTorchDiscriminator(len(self.qubit_set))
        discriminator._optimizer = optim.Adam(discriminator._discriminator.parameters(), lr=1e-3, amsgrad=True)
        qgan.set_discriminator(discriminator)
        
        result = qgan.run(quantum_instance)
        return -result['rel_entr']

    # @staticmethod
    # def plot_loss_functions(qgan):
    #     # Plot progress w.r.t the generator's and the discriminator's loss function
    #     t_steps = np.arange(len(qgan.g_loss))
    #
    #     fig, ax1 = plt.subplots(figsize=(7, 5))
    #     ax2 = ax1.twinx()
    #     plt.title("Progress in the loss function")
    #     lns1 = ax1.plot(
    #         t_steps, qgan.g_loss, label="Generator loss function", color="mediumvioletred", linewidth=2
    #     )
    #     lns2 = ax1.plot(
    #         t_steps, qgan.d_loss, label="Discriminator loss function", color="rebeccapurple", linewidth=2
    #     )
    #     lns3 = ax2.plot(
    #         t_steps, qgan.rel_entr, label='Relative Entropy', color='mediumblue', lw=4, ls=':'
    #     )
    #
    #     plt.grid()
    #     ax1.set_xlabel("time steps")
    #     ax1.set_ylabel("loss")
    #
    #     ax2.set_ylabel("relative entropy")
    #
    #     lns = lns1+lns2+lns3
    #     labs = [l.get_label() for l in lns]
    #     ax1.legend(lns, labs, loc='best')
    #
    #     plt.show()

    # def plot_distribution(self, real_distribution, qgan):
    #
    #     real_distribution = np.round(real_distribution)
    #     real_distribution = real_distribution[(real_distribution >= self.bounds[0]) & (real_distribution <= self.bounds[1])]
    #     temp = []
    #     for i in range(int(self.bounds[1] + 1)):
    #         temp += [np.sum(real_distribution == i)]
    #     real_distribution = np.array(temp / sum(temp))
    #
    #     plt.figure(figsize=(6, 5))
    #     plt.title("PDF")
    #     samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
    #     samples_g = np.array(samples_g)
    #     samples_g = samples_g.flatten()
    #
    #     plt.bar(samples_g, prob_g, color="royalblue", width=0.8, label="simulation")
    #     plt.plot(
    #         real_distribution, "-o", label="log-normal", color="deepskyblue", linewidth=4, markersize=6
    #     )
    #     plt.xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
    #     plt.grid()
    #     plt.xlabel("x")
    #     plt.ylabel("p(x)")
    #     plt.legend(loc="best")
    #     plt.show()




if __name__ == '__main__':


    from qiskit.circuit import ParameterVector

    # theta = ParameterVector('t',3)
    # qc = QuantumCircuit(4)
    # qc.h(0); qc.rx(theta[0],0); qc.cry(theta[1],1,2); qc.rzz(theta[2],2,3)
    # qft_obj = QFT_objective(num_qubits=4)
    # opt_param, opt_val = qft_obj.maximize_QFT_fidelity(qc)
    # print(opt_param, opt_val)
    # print('Final circuit')
    # print(qc.bind_parameters(opt_param).draw())
    #
    #
    # qc = QuantumCircuit(2)
    # qc.h(0)
    # qc.crx(0.2,0,1)
    # qft_obj = QFT_objective(num_qubits=2)
    # opt_param, opt_val = qft_obj.maximize_QFT_fidelity(qc)
    # print(opt_param, opt_val)
    # print('Final circuit')
    # print(qc.bind_parameters(opt_param).draw())

    qc = QuantumCircuit(4)
    theta = ParameterVector('t', 6)
    qc.h(3)
    qc.crz(theta[0],2,3); qc.crz(theta[1],1,3); qc.crz(theta[2],0,3);
    qc.h(2)
    qc.crz(theta[3], 1, 2); qc.crz(theta[4],0,2);
    qc.h(1)
    qc.crz(theta[5], 0, 1);
    qc.h(0)
    qft_obj = QFT_objective(num_qubits=4)
    opt_param, opt_val = qft_obj.maximize_QFT_fidelity(qc)
    print(opt_param, opt_val)
    print('Final circuit')
    print(qc.bind_parameters(opt_param).draw())


    import matplotlib.pyplot as plt

    theta = ParameterVector('t',3)
    qc = QuantumCircuit(4)
    qc.h(0); qc.rx(theta[0],[0,1,2,3]); qc.cry(theta[1],1,2); qc.rzz(theta[2],2,3)

    n = 4
    V = np.arange(0, n, 1)
    E = [(0, 1, 3.0), (0, 2, 5.0), (0, 3, 2.0), (1, 2, 5.0), (1, 3, 5.0), (2, 3, 2.0)]
    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)

    maxcut_obj = MAXCUT_objective(graphs=[G],num_graphs=1,num_nodes=qc.num_qubits)
    opt_param, opt_val = maxcut_obj.maximize_maxcut_hamiltonian(qc)
    sum_max_cut_val, sum_opt_cut_val = maxcut_obj.all_maxcuts_stat(qc, opt_param)
    print(opt_param, opt_val)
    print(sum_max_cut_val, ' / ', sum_opt_cut_val)
    print('Final circuit')
    print(qc.bind_parameters(opt_param).draw())
    for G in maxcut_obj.graphs:
        #pos = nx.get_node_attributes(G, 'pos')
        pos = nx.spring_layout(G, k=4)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        #plt.show()

