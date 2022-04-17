

def latent_func(circuit, objective, num_qubits):
    if objective.objective_type == 'qft':
        # opt_param, opt_val = self.objective.maximize_QFT_fidelity(PQC=circuit)
        opt_val = objective.maximize_QFT_fidelity(PQC=circuit)
    elif objective.objective_type == 'maxcut':
        # opt_param, opt_val = self.objective.maximize_maxcut_hamiltonian(PQC=circuit)
        opt_val = objective.maximize_maxcut_hamiltonian(PQC=circuit)
    else:  # 'qgan'
        num_epoch = num_qubits * 50
        if num_qubits == 1:
            objective.set_true_distribution(distribution_type='lognormal', mu=1., sigma=1., sample_size=1000)
        elif num_qubits == 2:
            objective.set_true_distribution(distribution_type='lognormal', mu=1., sigma=1., sample_size=1000)
        elif num_qubits == 3:
            objective.set_true_distribution(distribution_type='mixnormal', mu=[0.5, 3.5], sigma=[1., 0.5],
                                                 sample_size=1000)

        opt_val = objective.optimize_qgan(PQC=circuit, num_epochs=num_epoch, batch_size=100)
    return opt_val