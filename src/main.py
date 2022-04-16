import numpy as np
import torch
import gpytorch

from embedding import qc_embedding
from QuOTMANN import optimal_transport, structural_cost
from quantum_obj import QFT_objective, MAXCUT_objective, QGAN_objective

from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP

from botorch import fit_gpytorch_model
import botorch.optim.fit

from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import ticker

import pickle

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class CircuitDistKernel(gpytorch.kernels.Kernel):

    # We will register the parameter when initializing the kernel
    def __init__(self, encoder, decoder, num_qubits, MAX_OP_NODES, nu_list,
                 device=None, dtype=None, priors=None, constraints=None,
                 distance_dict={}, structral_paths_dict={}, **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.num_qubits = num_qubits
        self.MAX_OP_NODES = MAX_OP_NODES
        self.nu_list = nu_list
        self.num_str_weights = len(nu_list)

        self.distance_dict = distance_dict
        self.structural_paths_dict = structral_paths_dict

        # register the raw parameter
        self.register_parameter(
            name='raw_alpha', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1), requires_grad=True)
        )
        self.register_parameter(
            name='raw_alphanorm', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1), requires_grad=True)
        )

        # for i in range(self.num_str_weights):
        #     self.register_parameter(
        #         name='raw_beta_'+str(i), parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        #     )
        #     self.register_parameter(
        #         name='raw_betanorm_'+str(i), parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        #     )
        self.register_parameter(
            name='raw_beta', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, len(nu_list)), requires_grad=True)
        )
        self.register_parameter(
            name='raw_betanorm', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, len(nu_list)), requires_grad=True)
        )

        # set the parameter constraint to be positive, when nothing is specified
        if constraints is None:
            alpha_constraint = gpytorch.constraints.Positive()
            alphanorm_constraint = gpytorch.constraints.Positive()
            beta_constraints = gpytorch.constraints.Positive()
            betanorm_constraints = gpytorch.constraints.Positive()
        else:
            alpha_constraint = constraints[0]
            alphanorm_constraint = constraints[1]
            beta_constraints, betanorm_constraints = zip(*constraints[2])


        # register the constraints
        self.register_constraint("raw_alpha", alpha_constraint)
        self.register_constraint("raw_alphanorm", alphanorm_constraint)
        # for i in range(self.num_str_weights):
        #     self.register_constraint("raw_beta_"+str(i), beta_constraints[i])
        #     self.register_constraint("raw_betanorm_"+str(i), betanorm_constraints[i])
        self.register_constraint('raw_beta', beta_constraints)
        self.register_constraint('raw_betanorm', betanorm_constraints)


        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if priors is None:
            alpha_prior = None
            alphanorm_prior = None
            #beta_priors = [None] * self.num_str_weights
            #betanorm_priors = [None] * self.num_str_weights
            beta_priors = None
            betanorm_priors = None
        else:
            alpha_prior = priors[0]
            alphanorm_prior = priors[1]
            beta_priors, betanorm_priors = zip(*priors[2])


        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior, lambda m: m.alpha, lambda m,v: m._set_alpha)
        if alphanorm_prior is not None:
            self.register_prior("alphanorm_prior", alphanorm_prior, lambda m: m.alphanorm, lambda m,v: m._set_alphanorm)
        # for i in range(self.num_str_weights):
        #     if beta_priors[i] is not None:
        #         self.register_prior("beta_prior_"+str(i), beta_priors[i], lambda m: m.beta[i], lambda m, v: m._set_beta(idx=i,val=v))
        #     if betanorm_priors is not None:
        #         self.register_prior("betanorm_prior_"+str(i), betanorm_priors[i], lambda m: m.betanorm[i], lambda m, v: m._set_betanorm(idx=i,val=v))
        if beta_priors is not None:
            self.register_prior('beta_priors', beta_priors, lambda m: m.beta, lambda m,v: m._set_beta)
        if betanorm_priors is not None:
            self.register_prior('betanorm_priors', betanorm_priors, lambda m: m.betanorm, lambda m,v: m._set_betanorm)

    # now set up the 'actual' parameter
    @property
    def alpha(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_alpha_constraint.transform(self.raw_alpha)
    @property
    def alphanorm(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_alphanorm_constraint.transform(self.raw_alphanorm)

    # def beta(self,idx):
    #     # when accessing the parameter, apply the constraint transform
    #     return getattr(self, f"raw_beta_{idx}_constraint").transform(getattr(self, f"raw_betanorm_{idx}"))
    #
    # def betanorm(self,idx):
    #     # when accessing the parameter, apply the constraint transform
    #     return getattr(self, f"raw_betanorm_{idx}_constraint").transform(getattr(self, f"raw_betanorm_{idx}"))
    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)
    @property
    def betanorm(self):
        return self.raw_betanorm_constraint.transform(self.raw_betanorm)

    @alpha.setter
    def alpha(self, value):
        return self._set_alpha(value)
    @alphanorm.setter
    def alphanorm(self, value):
        return self._set_alphanorm(value)
    @beta.setter
    def beta(self, value):
        return self._set_beta(value)
    @betanorm.setter
    def betanorm(self, value):
        return self._set_betanorm(value)

    def _set_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def _set_alphanorm(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alphanorm)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_alphanorm=self.raw_alphanorm_constraint.inverse_transform(value))

    # def _set_beta(self, idx, value):
    #     if not torch.is_tensor(value):
    #         value = torch.as_tensor(value).to(getattr(self, f"raw_beta_{idx}"))
    #     kwargs = {f'raw_beta_{idx}': getattr(self, "raw_beta_{idx}_constraint").inverse_transform(value)}
    #     getattr(self, 'initialize')(**kwargs)
    #
    # def _set_betanorm(self, idx, value):
    #     if not torch.is_tensor(value):
    #         value = torch.as_tensor(value).to(getattr(self, f"raw_betanorm_{idx}"))
    #     kwargs = {f'raw_betanorm_{idx}': getattr(self, "raw_betanorm_{idx}_constraint").inverse_transform(value)}
    #     getattr(self, 'initialize')(**kwargs)
    def _set_beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))
    def _set_betanorm(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_betanorm)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_betanorm=self.raw_betanorm_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, diag=False, **params):
        # # calculate the distance between inputs

        if len(x1.shape) == 4: ## (n_batchs, q, n_samples, n_features)
            is_batched = True
            has_q = True

            weighted_dist = torch.zeros(size=(x1.shape[0], x1.shape[1], x1.shape[2], x2.shape[2])).to(x1)
            weighted_distnorm_square = torch.zeros_like(weighted_dist)

            all_circuits_1 = []
            all_circuits_2 = []
            for k in range(weighted_dist.shape[0]): # iterate through batches
                batch_circs_1 = []
                batch_circs_2 = []
                for q in range(weighted_dist.shape[1]):
                    q_circs_1 = []
                    q_circs_2 = []

                    for i in range(weighted_dist.shape[1]):
                        qc1 = self.decoder(vec=x1[k, q, i].detach().numpy())
                        qc1.vec_repr = self.encoder(qc=qc1)
                        qc1.vec_repr_bytes = qc1.vec_repr.tobytes()
                        try:
                            qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes = self.structural_paths_dict[qc1.vec_repr_bytes]
                        except:
                            qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes = structural_cost.structural_path_lengths_circ(qc1)
                            self.structural_paths_dict[qc1.vec_repr_bytes] = (qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes)
                        else:
                            #print("Structural path dict: ", len(self.structural_paths_dict))
                            pass

                        q_circs_1.append(qc1)

                    for j in range(weighted_dist.shape[2]):
                        qc2 = self.decoder(vec=x2[k, q, j].detach().numpy())
                        qc2.vec_repr = self.encoder(qc=qc2)
                        qc2.vec_repr_bytes = qc2.vec_repr.tobytes()
                        try:
                            qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes = self.structural_paths_dict[qc2.vec_repr_bytes]
                        except:
                            qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes = structural_cost.structural_path_lengths_circ(qc2)
                            self.structural_paths_dict[qc2.vec_repr_bytes] = (qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes)
                        else:
                            #print("Structural path dict: ", len(self.structural_paths_dict))
                            pass

                        q_circs_2.append(qc2)

                    batch_circs_1.append(q_circs_1)
                    batch_circs_2.append(q_circs_2)


                all_circuits_1.append(batch_circs_1)
                all_circuits_2.append(batch_circs_2)

            for k in range(weighted_dist.shape[0]): # iterate through batchs
                for q in range(weighted_dist.shape[1]):
                    for i in range(weighted_dist.shape[1]):
                        for j in range(weighted_dist.shape[2]):
                            qc1 = all_circuits_1[k][q][i]
                            qc2 = all_circuits_2[k][q][j]

                            try:
                                one_two_repr = np.stack([qc1.vec_repr, qc2.vec_repr])
                                two_one_repr = np.stack([qc2.vec_repr, qc1.vec_repr])
                                one_two_repr_bytes = one_two_repr.tobytes()
                                two_one_repr_bytes = two_one_repr.tobytes()

                                all_dist, all_distnorm = self.distance_dict[one_two_repr_bytes]
                            except:
                                all_dist, all_distnorm = self.circuit_distance(circ1=qc1, circ2=qc2, nu_list=self.nu_list)
                                self.distance_dict[one_two_repr_bytes] = (all_dist, all_distnorm)
                                self.distance_dict[two_one_repr_bytes] = (all_dist, all_distnorm)

                            else:
                                #print("Distance dict", len(self.distance_dict))
                                pass

                            #weighted_dist[k,q,i,j] =  sum([ self.beta(idx=idx) * dist for idx,dist in enumerate(all_dist) ])
                            #weighted_distnorm_square[k,q,i,j] = sum([ self.betanorm(idx=idx) * (distnorm**2) for idx,distnorm in enumerate(all_distnorm) ])
                            weighted_dist[k,q,i,j] = torch.inner(self.beta, torch.tensor(all_dist).to(self.beta))
                            weighted_distnorm_square[k,q,i,j] = torch.inner(self.betanorm, torch.tensor(all_distnorm).to(self.betanorm) ** 2)

        elif len(x1.shape) == 3: ## (n_batchs, n_samples, n_features)
            is_batched = True

            weighted_dist = torch.zeros(size=(x1.shape[0], x1.shape[1], x2.shape[1])).to(x1)
            weighted_distnorm_square = torch.zeros_like(weighted_dist)

            all_circuits_1 = []
            all_circuits_2 = []
            for k in range(weighted_dist.shape[0]): # iterate through batchs
                batch_circs_1 = []
                batch_circs_2 = []
                for i in range(weighted_dist.shape[1]):
                    qc1 = self.decoder(vec=x1[k, i].detach().numpy())
                    qc1.vec_repr = self.encoder(qc=qc1)
                    qc1.vec_repr_bytes = qc1.vec_repr.tobytes()
                    try:
                        qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes = self.structural_paths_dict[qc1.vec_repr_bytes]
                    except:
                        qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes = structural_cost.structural_path_lengths_circ(qc1)
                        self.structural_paths_dict[qc1.vec_repr_bytes] = (qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes)
                    else:
                        #print("Structural path dict: ", len(self.structural_paths_dict))
                        pass

                    batch_circs_1.append(qc1)
                for j in range(weighted_dist.shape[2]):
                    qc2 = self.decoder(vec=x2[k, j].detach().numpy())
                    qc2.vec_repr = self.encoder(qc=qc2)
                    qc2.vec_repr_bytes = qc2.vec_repr.tobytes()
                    try:
                        qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes = self.structural_paths_dict[qc2.vec_repr_bytes]
                    except:
                        qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes = structural_cost.structural_path_lengths_circ(qc2)
                        self.structural_paths_dict[qc2.vec_repr_bytes] = (qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes)
                    else:
                        #print("Structural path dict: ", len(self.structural_paths_dict))
                        pass

                    batch_circs_2.append(qc2)


                all_circuits_1.append(batch_circs_1)
                all_circuits_2.append(batch_circs_2)

            for k in range(weighted_dist.shape[0]): # iterate through batchs
                for i in range(weighted_dist.shape[1]):
                    for j in range(weighted_dist.shape[2]):
                        qc1 = all_circuits_1[k][i]
                        qc2 = all_circuits_2[k][j]
                        try:
                            one_two_repr = np.stack([qc1.vec_repr, qc2.vec_repr])
                            two_one_repr = np.stack([qc2.vec_repr, qc1.vec_repr])
                            one_two_repr_bytes = one_two_repr.tobytes()
                            two_one_repr_bytes = two_one_repr.tobytes()

                            all_dist, all_distnorm = self.distance_dict[one_two_repr_bytes]
                        except:
                            all_dist, all_distnorm = self.circuit_distance(circ1=qc1, circ2=qc2, nu_list=self.nu_list)
                            self.distance_dict[one_two_repr_bytes] = (all_dist, all_distnorm)
                            self.distance_dict[two_one_repr_bytes] = (all_dist, all_distnorm)
                        else:
                            #print("Distance dict", len(self.distance_dict))
                            pass

                        #weighted_dist[k,i,j] =  sum([ self.beta(idx=idx) * dist for idx,dist in enumerate(all_dist) ])
                        #weighted_distnorm_square[k,i,j] = sum([ self.betanorm(idx=idx) * (distnorm**2) for idx,distnorm in enumerate(all_distnorm) ])
                        weighted_dist[k,i,j] = torch.inner(self.beta, torch.tensor(all_dist).to(self.beta))
                        weighted_distnorm_square[k,i,j] = torch.inner(self.betanorm, torch.tensor(all_distnorm).to(self.betanorm) ** 2)


        elif len(x1.shape) == 2: ## (n_samples, n_features)
            is_batched = False

            weighted_dist = torch.zeros(size=(x1.shape[0], x2.shape[0])).to(x1)
            weighted_distnorm_square = torch.zeros_like(weighted_dist)

            all_circuits_1 = []
            all_circuits_2 = []
            for i in range(weighted_dist.shape[0]):
                qc1 = self.decoder(vec=x1[i].detach().numpy())
                qc1.vec_repr = self.encoder(qc=qc1)
                qc1.vec_repr_bytes = qc1.vec_repr.tobytes()
                try:
                    qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes = self.structural_paths_dict[qc1.vec_repr_bytes]
                except:
                    qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes = structural_cost.structural_path_lengths_circ(qc1)
                    self.structural_paths_dict[qc1.vec_repr_bytes] = (qc1.lengths_to_in_nodes, qc1.lengths_to_out_nodes)
                else:
                    #print("Structural path dict: ", len(self.structural_paths_dict))
                    pass

                all_circuits_1.append(qc1)

            for j in range(weighted_dist.shape[1]):
                qc2 = self.decoder(vec=x2[j].detach().numpy())
                qc2.vec_repr = self.encoder(qc=qc2)
                qc2.vec_repr_bytes = qc2.vec_repr.tobytes()
                try:
                    qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes = self.structural_paths_dict[qc2.vec_repr_bytes]
                except:
                    qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes = structural_cost.structural_path_lengths_circ(qc2)
                    self.structural_paths_dict[qc2.vec_repr_bytes] = (qc2.lengths_to_in_nodes, qc2.lengths_to_out_nodes)
                else:
                    #print("Structural path dict: ", len(self.structural_paths_dict))
                    pass

                all_circuits_2.append(qc2)

            for i in range(weighted_dist.shape[0]):
                for j in range(weighted_dist.shape[1]):
                    qc1 = all_circuits_1[i]
                    qc2 = all_circuits_2[j]
                    try:
                        one_two_repr = np.stack([qc1.vec_repr, qc2.vec_repr])
                        two_one_repr = np.stack([qc2.vec_repr, qc1.vec_repr])
                        one_two_repr_bytes = one_two_repr.tobytes()
                        two_one_repr_bytes = two_one_repr.tobytes()

                        all_dist, all_distnorm = self.distance_dict[one_two_repr_bytes]
                    except:
                        all_dist, all_distnorm = self.circuit_distance(circ1=qc1, circ2=qc2, nu_list=self.nu_list)
                        self.distance_dict[one_two_repr_bytes] = (all_dist, all_distnorm)
                        self.distance_dict[two_one_repr_bytes] = (all_dist, all_distnorm)
                    else:
                        #print("Distance dict", len(self.distance_dict))
                        pass

                    #weighted_dist[i,j] =  sum([ self.beta(idx=idx) * dist for idx,dist in enumerate(all_dist) ])
                    #weighted_distnorm_square[i,j] = sum([ self.betanorm(idx=idx) * (distnorm**2) for idx,distnorm in enumerate(all_distnorm) ])
                    # print(len(self.beta), self.beta)
                    # print(len(all_dist), all_dist)
                    weighted_dist[i,j] = torch.inner(self.beta, torch.tensor(all_dist).to(self.beta))
                    weighted_distnorm_square[i,j] = torch.inner(self.betanorm, torch.tensor(all_distnorm).to(self.betanorm)**2)

        #print(torch.mean(weighted_dist), torch.mean(weighted_distnorm_square))
        K = self.alpha * torch.exp(-weighted_dist) + self.alphanorm * torch.exp(-weighted_distnorm_square)
        if diag:
            K = torch.diagonal(K, dim1=-2, dim2=-1)

        #print('covar module: ', x1.shape, x2.shape, K.shape, type(K))

        return K
        #return gpytorch.lazify(K)

    def circuit_distance(self, circ1, circ2, nas_cost=1, nu_list=[0.1]):
        return optimal_transport.circuit_distance_POT(PQC_1=circ1, PQC_2=circ2, nas_cost=nas_cost, nu_list=nu_list)

class GPModel(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, covar_module):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = covar_module
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class QNN_BO():
    def __init__(self, objective_type, num_qubits, MAX_OP_NODES, N_TRIALS, N_BATCH, BATCH_SIZE, MC_SAMPLES, device=None, dtype=None):
        assert objective_type in ['qft','maxcut','qgan']
        self.objective_type = objective_type
        if self.objective_type == 'qft':
            self.objective = QFT_objective(num_qubits=num_qubits)
        elif self.objective_type == 'maxcut':
            self.objective = MAXCUT_objective(num_graphs=50,num_nodes=num_qubits)
        else: #'qgan'
            assert num_qubits <= 3, "QGAN objective only supports <= 3 qubits; otherwise it takes a lot of time."
            self.objective = QGAN_objective(num_qubits=num_qubits)

        self.num_qubits = num_qubits
        self.MAX_OP_NODES = MAX_OP_NODES
        self.encoding_length = (self.num_qubits + 1) * self.MAX_OP_NODES

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.double

        self.N_TRIALS = N_TRIALS
        self.N_BATCH = N_BATCH
        self.BATCH_SIZE = BATCH_SIZE
        self.MC_SAMPLES = MC_SAMPLES

        self.distance_dict = {}
        self.structural_paths_dict = {}


    ## PROBLEM SETUP
    def obj_func(self, X):
        """
        Objective function that decode X into circuit and pass it to a training task.
        """
        latent_func_values = []
        for enc in X.detach().numpy():
            qc = self.vec_to_circuit(vec=enc)
            latent_func_values.append(self.latent_func(qc))
        #return latent_func_values
        return torch.as_tensor(latent_func_values, device=self.device, dtype=self.dtype).unsqueeze(-1)

    def latent_func(self,circuit):
        if self.objective_type == 'qft':
            #opt_param, opt_val = self.objective.maximize_QFT_fidelity(PQC=circuit)
            opt_val = self.objective.maximize_QFT_fidelity(PQC=circuit)
        elif self.objective_type == 'maxcut':
            #opt_param, opt_val = self.objective.maximize_maxcut_hamiltonian(PQC=circuit)
            opt_val = self.objective.maximize_maxcut_hamiltonian(PQC=circuit)
        else: # 'qgan'
            num_epoch = self.num_qubits * 50
            if self.num_qubits == 1:
                self.objective.set_true_distribution(distribution_type='lognormal', mu=1., sigma=1., sample_size=1000)
            elif self.num_qubits == 2:
                self.objective.set_true_distribution(distribution_type='lognormal',mu=1.,sigma=1.,sample_size=1000)
            elif self.num_qubits == 3:
                self.objective.set_true_distribution(distribution_type='mixnormal',mu=[0.5,3.5],sigma=[1.,0.5],sample_size=1000)

            opt_val = self.objective.optimize_qgan(PQC=circuit, num_epochs=num_epoch, batch_size=100)
        return torch.as_tensor(opt_val, device=self.device, dtype=self.dtype)

    def vec_to_circuit(self,vec):
        return qc_embedding.enc_to_qc(num_qubits=self.num_qubits, encoding=vec)

    def circuit_to_vec(self,qc):
        return qc_embedding.qc_to_enc(qc=qc,MAX_OP_NODES=self.MAX_OP_NODES)

    ## MODEL INITIALIZATION
    def generate_initial_data(self, n, bounds):
        # generate training data

        #train_x = torch.rand(n, self.encoding_length, device=self.device, dtype=self.dtype)
        train_x = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()).squeeze(1)

        train_obj = self.obj_func(X=train_x)
        #train_obj = torch.as_tensor(train_obj, device=self.device, dtype=self.dtype).unsqueeze(-1)

        best_observed_value = train_obj.max().item()
        best_observed_x = train_x[torch.nonzero(torch.isclose(train_obj, train_obj.max()).ravel()).ravel()].tolist()
        return train_x, train_obj, best_observed_x, best_observed_value

    def initialize_model(self, train_x, train_obj, covar_module=None, input_transform=None, state_dict=None):
        # define models for objective

        if covar_module is None:
            covar_module = CircuitDistKernel(encoder=self.circuit_to_vec,
                                             decoder=self.vec_to_circuit,
                                             num_qubits=self.num_qubits,
                                             MAX_OP_NODES=MAX_OP_NODES,
                                             nu_list=[0.1, 0.2, 0.4, 0.8],
                                             device=self.device,
                                             dtype=self.dtype,
                                             distance_dict=self.distance_dict,
                                             structral_paths_dict=self.structural_paths_dict
                                             )

        model = SingleTaskGP(train_x, train_obj, covar_module=covar_module, input_transform=input_transform).to(train_x)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def lbfgsb_optimize_acqf(self, acq_func, bounds):
        def neg_acq_func(x):
            X = torch.from_numpy(x).to(device=self.device, dtype=self.dtype).unsqueeze(-2)
            Y = -acq_func(X)
            y = Y.view(-1).double().numpy()
            return y

        candidates = torch.empty(size=(self.BATCH_SIZE, self.encoding_length), device=self.device, dtype=self.dtype)

        for i in range(self.BATCH_SIZE):
            with torch.no_grad():
                # get initial condition for L-BFGS-B in numpy form
                # note that L-BFGS-B expects a different shape (no explicit q-batch dimension)
                #x0 = np.random.normal(loc=0.5,scale=0.4,size=self.encoding_length).clip(bounds[0].numpy(), bounds[1].numpy())
                x0 = np.random.rand(self.encoding_length).clip(bounds[0].numpy(), bounds[1].numpy())
                res = minimize(fun=neg_acq_func, x0=x0, method='L-BFGS-B', bounds=np.array(list(zip(bounds[0].numpy(), bounds[1].numpy()))))
                candidates[i] = torch.from_numpy(res.x).to(candidates)
        return candidates

    ## Helper function that performs essential BO steps
    def optimize_acqf_and_get_observation(self, acq_func, bounds):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates = self.lbfgsb_optimize_acqf(acq_func=acq_func, bounds=bounds)

        #print(candidates.shape)

        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=bounds)

        train_obj = self.obj_func(X=new_x)
        #train_obj = torch.as_tensor(train_obj, device=self.device, dtype=self.dtype).unsqueeze(-1)

        return new_x, train_obj

    # def update_random_observations(self, best_random, num_random_points=1):
    #     """Simulates a random policy by taking a the current list of best values observed randomly,
    #     drawing a new random point, observing its value, and updating the list.
    #     """
    #     #rand_x = torch.rand(BATCH_SIZE, self.encoding_length)
    #     rand_x = draw_sobol_samples(bounds=bounds, n=num_random_points, q=1).squeeze(1)
    #     next_random_best = self.obj_func(X=rand_x)
    #     next_random_best = torch.as_tensor(next_random_best, device=self.device, dtype=self.dtype).max().item()
    #     best_random.append(max(best_random[-1], next_random_best))
    #     return best_random

    def update_random_observations(self, best_random_x, best_random_value, num_random_points=1):
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        rand_x = draw_sobol_samples(bounds=bounds, n=num_random_points, q=1).squeeze(1)
        rand_obj = self.obj_func(X=rand_x)

        next_random_best_value = rand_obj.max().item()
        next_random_best_x = rand_x[torch.nonzero(torch.isclose(rand_obj, rand_obj.max()).ravel()).ravel()].tolist()
        # print(rand_x.shape, rand_obj.shape)
        # print(rand_x[torch.nonzero(torch.isclose(rand_obj, rand_obj.max())).ravel()].shape)
        # print("best y", best_random_value)
        # print("next y", next_random_best_value)
        # print("best x", best_random_x)
        # print("next x", next_random_best_x)

        if best_random_value[-1] > next_random_best_value:
            best_random_value.append(best_random_value[-1])

        elif best_random_value[-1] < next_random_best_value:
            best_random_value.append(next_random_best_value)
            best_random_x = next_random_best_x

        else:
            best_random_value.append(next_random_best_value)
            best_random_x += next_random_best_x

        return best_random_x, best_random_value

    def bayesopt_trial(self, model, mll, train_x, train_obj, best_observed_x=[], best_observed_value=[], acqf_choice='random', candidate_set_size=10, torch_optimizer=True):
        print('Choice of acquisition function: ', acqf_choice)

        is_random_acqf = acqf_choice == 'random'
        # run n_batch rounds of BayesOpt
        for iteration in range(1, self.N_BATCH + 1):
            print('iteration: ', iteration)

            if acqf_choice == 'qEI':
                qmc_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)
                acqf = qExpectedImprovement(
                    model=model,
                    best_f=standardize(train_obj).max(),
                    sampler=qmc_sampler
                )
            elif acqf_choice == 'EI':
                acqf = ExpectedImprovement(
                    model=model,
                    best_f=standardize(train_obj).max()
                )

            elif acqf_choice == 'qUCB':
                qmc_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)
                acqf = qUpperConfidenceBound(
                    model=model,
                    beta=0.1,
                    sampler=qmc_sampler
                )
            elif acqf_choice == 'UCB':
                acqf = UpperConfidenceBound(
                    model=model,
                    beta=0.1
                )

            elif acqf_choice == 'GIBBON':
                candidate_set = torch.rand(candidate_set_size, bounds.size(1), device=self.device, dtype=self.dtype)
                candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
                acqf = qLowerBoundMaxValueEntropy(model, candidate_set)


            if is_random_acqf:
                print('update random')
                best_observed_x, best_observed_value = self.update_random_observations(best_observed_x, best_observed_value)

            else: #optimize and get new observation


                # print('Model parameters BEFORE fitting:',  model.likelihood.noise, model.covar_module.alpha, model.covar_module.alphanorm, '\n',
                #       model.covar_module.beta,'\n', model.covar_module.betanorm)
                # for name, param in model.named_parameters():
                #     print(name, param)

                if torch_optimizer:
                    fit_gpytorch_model(mll=mll, optimizer=botorch.optim.fit.fit_gpytorch_torch, max_retries=10)
                else:
                    fit_gpytorch_model(mll=mll, max_retries=10)


                # print('Model parameters AFTER fitting:', model.likelihood.noise, model.covar_module.alpha, model.covar_module.alphanorm, '\n',
                #       model.covar_module.beta,'\n', model.covar_module.betanorm)
                # for name, param in model.named_parameters():
                #     print(name, param)

                print('optimize acquisition function')
                new_x, new_obj = self.optimize_acqf_and_get_observation(acq_func=acqf, bounds=bounds)
                #print("New candidates", new_obj.shape)

                # update training points
                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj])

                # update progress

                print('update acqf best value')
                best_value = train_obj.max().item()
                best_x = train_x[torch.nonzero(torch.isclose(train_obj, train_obj.max()).ravel()).ravel()].tolist()
                best_observed_value.append(best_value)
                #best_observed_x.append(best_x)
                best_observed_x = best_x

                print('end of batch: ', train_x.shape, train_obj.shape, best_observed_value)

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                mll, model = self.initialize_model(
                    normalize(train_x, bounds=bounds),
                    standardize(train_obj),
                    state_dict=model.state_dict(),
                )

        return best_observed_x, best_observed_value


    def optimize(self, bounds, acqf_choices, num_init_points, optimizer):
        verbose = False
        assert optimizer in ['torch','scipy']
        torch_optimizer = True if optimizer == 'torch' else False

        #best_observed_all_ei, best_observed_all_mes, best_random_all = [], [], []
        list_of_best_observed_value_all = [[] for _ in range(len(acqf_choices))]
        list_of_best_observed_x_all = [[] for _ in range(len(acqf_choices))]

        # average over multiple trials
        for trial in range(1, self.N_TRIALS + 1):

            print(f"\nTrial {trial:>2} of {self.N_TRIALS} ", end="")

            # call helper functions to generate initial training data and initialize model
            train_x_init, train_obj_init, best_observed_x_init, best_observed_value_init = self.generate_initial_data(n=num_init_points, bounds=bounds)

            print('data initialization: ', train_x_init.shape, train_obj_init.shape, best_observed_x_init, best_observed_value_init)

            # run n_batch rounds of BayesOpt after the initial random batch
            for idx,choice in enumerate(acqf_choices):
                mll, model = self.initialize_model(normalize(train_x_init, bounds=bounds), standardize(train_obj_init))
                ## best_observed_value stores the optimal obj over batchs
                ## best_observed_x only stores the final optimal circuit(s)
                best_observed_x, best_observed_value = self.bayesopt_trial(model, mll, train_x_init.clone(), train_obj_init.clone(),
                                                                 best_observed_x=best_observed_x_init,
                                                                 best_observed_value=[best_observed_value_init],
                                                                 acqf_choice=choice,
                                                                 candidate_set_size=50,
                                                                 torch_optimizer=torch_optimizer)
                if self.objective_type == 'qgan':
                    list_of_best_observed_value_all[idx].append([-val for val in best_observed_value])
                else:
                    list_of_best_observed_value_all[idx].append(best_observed_value)
                list_of_best_observed_x_all[idx].append(best_observed_x)
                print(f'trial {trial}, {choice}:', list_of_best_observed_value_all[idx][-1])

        return list_of_best_observed_x_all, list_of_best_observed_value_all


    def plot(self, to_plot, filename):
        def ci(y):
            ## Confidence interval
            return 1.96 * y.std(axis=0) / np.sqrt(self.N_TRIALS)

        iters = np.arange(self.N_BATCH + 1) * self.BATCH_SIZE

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for label, best_observed_all in to_plot.items():
            y = np.asarray(best_observed_all)
            mean_y = y.mean(axis=0)
            conf = ci(y)

            ax.errorbar(iters, mean_y, yerr=conf, errorevery=self.N_BATCH*self.BATCH_SIZE // 5, label=label, alpha=.75, fmt=':', capsize=3, capthick=1, linewidth=2)

            #ax.plot(iters, mean_y, linewidth=1.5, label=label)
            ax.fill_between(iters, (mean_y - conf), (mean_y + conf), alpha=.05)
            print(label, mean_y)

        if self.objective_type != 'qgan':
            plt.axhline(y=1., color='k', linestyle='--', linewidth=2)
            ax.set_ylim(top=1.05, bottom=0.45)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.suptitle(self.objective_type.upper())

        ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
        ax.legend(loc="lower right")
        plt.grid()
        plt.savefig(filename, bbox_inches='tight')
        plt.show()

    def plot_ansatz(self, to_plot_ansatz):
        if self.objective_type == 'maxcut':
            import networkx as nx
            G = self.objective.graphs[0]
            pos = nx.spring_layout(G)  # pos = nx.nx_agraph.graphviz_layout(G)
            nx.draw_networkx(G, pos)
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        print(to_plot_ansatz)
        for label, best_observed_x_trials in to_plot_ansatz.items():
            for trial,best_observed_x in enumerate(best_observed_x_trials):
                print(label, trial)


                for vec in best_observed_x:
                    qc = self.vec_to_circuit(np.array(vec))
                    print(qc.draw())
                    if self.objective_type == 'qgan':
                        print(-self.latent_func(qc))
                    else:
                        print(self.latent_func(qc))
                    print('----------------------------------')


if __name__ == '__main__':
    import sys
    sys.path.append('/Users/erio/Dropbox/URP project/Code/PQC_composer')

    seed = 27112021
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(precision=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    objective_type = 'qgan'  # ['qft', 'maxcut', 'qgan']
    num_qubits = 3
    MAX_OP_NODES = 10  # Maximum number of gates
    num_init_points = 5  # Number of points sampled randomly at the beginning

    N_TRIALS = 10  # Number of times the experiments run
    N_BATCH = 25 # Number of batch per trial
    BATCH_SIZE = 1  # Number of new points being sampled in a batch

    MC_SAMPLES = 2048  # Number of points sampled in optimization of acquisition functions

    qnnbo = QNN_BO(
        objective_type = objective_type,
        num_qubits = num_qubits,
        MAX_OP_NODES = MAX_OP_NODES,
        N_TRIALS = N_TRIALS,
        N_BATCH = N_BATCH,
        BATCH_SIZE = BATCH_SIZE,
        MC_SAMPLES = MC_SAMPLES
    )

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = torch.tensor([[0.] * encoding_length, [1.0] * encoding_length], device=device, dtype=dtype)


    acqf_choices = ['random', 'EI', 'GIBBON']
    optimizer = 'torch' ## 'torch' or 'scipy'

    list_of_best_observed_x_all, list_of_best_observed_value_all = qnnbo.optimize(bounds=bounds, acqf_choices=acqf_choices, num_init_points=num_init_points, optimizer=optimizer)

    to_plot = dict(zip(acqf_choices, list_of_best_observed_value_all))
    to_plot_ansatz = dict(zip(acqf_choices, list_of_best_observed_x_all))
    qnnbo.plot_ansatz(to_plot_ansatz)

    imgname = '_'.join([objective_type, str(num_qubits), str(MAX_OP_NODES), str(num_init_points), str(BATCH_SIZE), str(N_BATCH), str(N_TRIALS), *acqf_choices, optimizer, str(seed)])
    filename = './output/' + imgname
    qnnbo.plot(to_plot, filename)

    pkl_filename = './output/' + imgname + '.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump({'QNN':to_plot_ansatz, 'obj':to_plot}, f)