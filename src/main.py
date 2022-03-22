import botorch.optim.fit
import numpy as np
import torch

from embedding import qc_embedding
from QuOTMANN import optimal_transport, structural_cost
from quantum_obj import get_QFT_states, maximize_QFT_fidelity

import gpytorch

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from botorch.optim import optimize_acqf

#import cma
from scipy.optimize import minimize

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy, qLowerBoundMaxValueEntropy

from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples


#### Save resources when computing covariance kernel by reusing previous kernel (of smaller size)
#### Find a way to explain why all beta[i] = 0
#https://github.com/kirthevasank/nasbot/blob/3c745dc986be30e3721087c8fa768099032a0802/nn/nn_gp.py#L120
#https://github.com/kirthevasank/nasbot/blob/3c745dc986be30e3721087c8fa768099032a0802/nn/unittest_nn_gp.py#L56

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
    def forward(self, x1, x2, **params):
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

        #print('covar module: ', x1.shape, x2.shape, K.shape, type(K))
        #print('kernel: ', K)

        #return K
        return gpytorch.lazify(K)

    def circuit_distance(self, circ1, circ2, nas_cost=1, nu_list=[0.1]):
        return optimal_transport.circuit_distance_POT(PQC_1=circ1, PQC_2=circ2, nas_cost=nas_cost, nu_list=nu_list)

class FirstSincKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    has_lengthscale = False

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        #x1 = x1.div(self.lengthscale)
        #x2 = x2.div(self.lengthscale)
        diff = self.covar_dist(x1, x2, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20).to(x1))
        K = torch.sin(diff).div(diff)
        print('covar module', x1.shape, x2.shape, K.shape, type(K))
        #print('kernel: ', K)

        return gpytorch.lazify(K)


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
    def __init__(self, num_qubits, MAX_OP_NODES, N_TRIALS, N_BATCH, BATCH_SIZE, MC_SAMPLES, device=None, dtype=None):
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
        """Feasibility weighted objective"""
        #print(X)

        latent_func_values = []
        for enc in X.detach().numpy():
            qc = self.vec_to_circuit(vec=enc)
            latent_func_values.append(self.latent_func(qc))
        return latent_func_values

    def latent_func(self,circuit):
        #f = circuit.num_parameters #/ self.MAX_OP_NODES
        opt_param, opt_val = maximize_QFT_fidelity(PQC=circuit)
        return torch.as_tensor(opt_val, device=self.device, dtype=self.dtype)

    def vec_to_circuit(self,vec):
        return qc_embedding.enc_to_qc(num_qubits=self.num_qubits, encoding=vec)

    def circuit_to_vec(self,qc):
        return qc_embedding.qc_to_enc(qc=qc,MAX_OP_NODES=self.MAX_OP_NODES)

    ## MODEL INITIALIZATION
    def generate_initial_data(self, n):
        # generate training data

        #train_x = torch.rand(n, self.encoding_length, device=self.device, dtype=self.dtype)
        train_x = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()).squeeze(1)

        #train_obj = [self.latent_func(self.vec_to_circuit(vec=vec)) for vec in train_x.numpy()]
        train_obj = self.obj_func(X=train_x)
        train_obj = torch.as_tensor(train_obj, device=self.device, dtype=self.dtype).unsqueeze(-1)

        best_observed_value = train_obj.max().item()
        return train_x, train_obj, best_observed_value

    def initialize_model(self, train_x, train_obj, covar_module=None, input_transform=None, state_dict=None):
        # define models for objective

        #covar_module = covar_module or gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        #covar_module = covar_module or FirstSincKernel()


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

    def fit_gp_model(self, model, mll):
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=0.1)
        NUM_EPOCHS = 150

        mll.model.train()

        for epoch in range(NUM_EPOCHS):
            # clear gradients
            optimizer.zero_grad()
            # forward pass through the model to obtain the output MultivariateNormal
            #print(model.train_inputs.shape, model.train_inputs)
            output = model(model.train_inputs)
            # Compute negative marginal log likelihood
            loss = - mll(output, model.train_targets)
            # back prop gradients
            loss.backward()
            # print every 10 iterations
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
                    f"noise: {model.likelihood.noise.item():>4.3f}"
                )
            optimizer.step()
        return mll.model.eval()

    ## Zero-th order optimizer of acqf
    def cmaes_optimize_acqf(self, acq_func, bounds):
        """
        Return solution candidates for the acquisition function being maximized
        """
        print(bounds)
        candidates = torch.empty(size=(self.BATCH_SIZE, self.encoding_length), device=self.device, dtype=self.dtype)

        for i in range(self.BATCH_SIZE):
            # get initial condition for CMAES in numpy form
            # note that CMAES expects a different shape (no explicit q-batch dimension)
            x0 = np.random.normal(loc=0.5,scale=0.4,size=self.encoding_length).clip(bounds[0].numpy(), bounds[1].numpy())

            # create the CMA-ES optimizer
            es = cma.CMAEvolutionStrategy(
                x0=x0,
                sigma0=0.2,
                inopts={'bounds': bounds.tolist(), 'popsize': 64, 'verbose':-1},
            )


            # speed up things by telling pytorch not to generate a compute graph in the background
            with torch.no_grad():
                # Run the optimization loop using the ask/tell interface -- this uses
                # PyCMA's default settings, see the PyCMA documentation for how to modify these
                while not es.stop():
                    xs = es.ask()  # as for new points to evaluate
                    # convert to Tensor for evaluating the acquisition function
                    X = torch.tensor(xs, device=self.device, dtype=self.dtype)
                    # evaluate the acquisition function (optimizer assumes we're minimizing)
                    Y = - acq_func(X.unsqueeze(-2))  # acquisition functions require an explicit q-batch dimension
                    y = Y.view(-1).double().numpy()  # convert result to numpy array
                    #print(X.shape, Y.shape, y.shape)
                    es.tell(xs, y)  # return the result to the optimizer

            # convert result back to a torch tensor
            candidates[i] = torch.from_numpy(es.best.x).to(candidates)
        return candidates

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

        # candidates, _ = optimize_acqf(
        #     acq_function=acq_func,
        #     bounds=bounds,
        #     q=self.BATCH_SIZE,
        #     num_restarts=10,
        #     raw_samples=512,  # used for intialization heuristic
        #     options={
        #         "batch_limit": 3,
        #         "maxiter": 200,
        #     }
        # )

        #candidates = self.cmaes_optimize_acqf(acq_func=acq_func, bounds=bounds)
        candidates = self.lbfgsb_optimize_acqf(acq_func=acq_func, bounds=bounds)

        #print(candidates.shape)

        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=bounds)
        #print(new_x)

        #train_obj = [self.latent_func(self.vec_to_circuit(vec=vec)) for vec in new_x.numpy()]
        train_obj = self.obj_func(X=new_x)
        train_obj = torch.as_tensor(train_obj, device=self.device, dtype=self.dtype).unsqueeze(-1)

        return new_x, train_obj

    def update_random_observations(self, best_random, num_random_points=1):
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        #rand_x = torch.rand(BATCH_SIZE, self.encoding_length)
        rand_x = draw_sobol_samples(bounds=bounds, n=num_random_points, q=1).squeeze(1)
        next_random_best = self.obj_func(X=rand_x)
        next_random_best = torch.as_tensor(next_random_best, device=self.device, dtype=self.dtype).max().item()
        best_random.append(max(best_random[-1], next_random_best))
        return best_random

    def bayesopt_trial(self, model, mll, train_x, train_obj, best_observed=[], acqf_choice='qEI', candidate_set_size=10):
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
            elif acqf_choice == 'qMES':
                candidate_set = torch.rand(candidate_set_size, bounds.size(1), device=self.device, dtype=self.dtype)
                candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
                acqf = qMaxValueEntropy(
                    model=model,
                    candidate_set=candidate_set)

            elif acqf_choice == 'GIBBON':
                candidate_set = torch.rand(candidate_set_size, bounds.size(1), device=self.device, dtype=self.dtype)
                candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
                acqf = qLowerBoundMaxValueEntropy(model, candidate_set)


            if is_random_acqf:
                print('update random')
                best_observed = self.update_random_observations(best_observed)

            else: #optimize and get new observation


                print('Model parameters BEFORE fitting:',  model.likelihood.noise, model.covar_module.alpha, model.covar_module.alphanorm, '\n',
                      model.covar_module.beta,'\n', model.covar_module.betanorm)
                for name, param in model.named_parameters():
                    print(name, param)

                fit_gpytorch_model(mll=mll, optimizer=botorch.optim.fit.fit_gpytorch_torch, max_retries=10)
                #fit_gpytorch_model(mll=mll, max_retries=10)
                #self.fit_gp_model(model=model, mll=mll)

                print('Model parameters AFTER fitting:', model.likelihood.noise, model.covar_module.alpha, model.covar_module.alphanorm, '\n',
                      model.covar_module.beta,'\n', model.covar_module.betanorm)
                for name, param in model.named_parameters():
                    print(name, param)

                print('optimize acquisition function')
                new_x, new_obj = self.optimize_acqf_and_get_observation(acq_func=acqf, bounds=bounds)
                print("New candidates", new_obj.shape)

                # update training points
                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj])

                # update progress

                print('update acqf best value')
                best_value = train_obj.max().item()
                best_observed.append(best_value)

                print('end of batch: ', train_x.shape, train_obj.shape, best_observed)

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                mll, model = self.initialize_model(
                    normalize(train_x, bounds=bounds),
                    standardize(train_obj),
                    state_dict=model.state_dict(),
                )

        return best_observed


    def optimize_new(self, bounds, acqf_choices, num_init_points):
        verbose = False

        #best_observed_all_ei, best_observed_all_mes, best_random_all = [], [], []
        list_of_best_observed_all = [[] for _ in range(len(acqf_choices))]

        # average over multiple trials
        for trial in range(1, self.N_TRIALS + 1):

            print(f"\nTrial {trial:>2} of {self.N_TRIALS} ", end="")

            # call helper functions to generate initial training data and initialize model
            train_x_init, train_obj_init, best_observed_value_init = self.generate_initial_data(n=num_init_points)


            print('data initialization: ', train_x_init.shape, train_obj_init.shape, best_observed_value_init)

            # run n_batch rounds of BayesOpt after the initial random batch
            for idx,choice in enumerate(acqf_choices):
                mll, model = self.initialize_model(normalize(train_x_init, bounds=bounds), standardize(train_obj_init))
                best_observed = self.bayesopt_trial(model, mll, train_x_init.clone(), train_obj_init.clone(),
                                                                 best_observed=[best_observed_value_init],
                                                                 acqf_choice=choice,candidate_set_size=50)
                list_of_best_observed_all[idx].append(best_observed)

        return list_of_best_observed_all



    def optimize(self, bounds, num_init_points):
        verbose = False

        best_observed_all_ei, best_observed_all_mes, best_random_all = [], [], []

        # average over multiple trials
        for trial in range(1, self.N_TRIALS + 1):

            print(f"\nTrial {trial:>2} of {self.N_TRIALS} ", end="")
            best_observed_ei, best_observed_mes, best_random = [], [], []

            # call helper functions to generate initial training data and initialize model
            train_x_init, train_obj_init, best_observed_value_init = self.generate_initial_data(n=num_init_points)
            mll_ei, model_ei = self.initialize_model(normalize(train_x_init, bounds=bounds), standardize(train_obj_init))
            mll_mes, model_mes = self.initialize_model(normalize(train_x_init, bounds=bounds), standardize(train_obj_init))

            best_observed_ei.append(best_observed_value_init)
            best_observed_mes.append(best_observed_value_init)
            best_random.append(best_observed_value_init)

            train_x_ei = train_x_init.clone()
            train_obj_ei = train_obj_init.clone()
            train_x_mes = train_x_init.clone()
            train_obj_mes = train_obj_init.clone()

            print('data initialization: ', train_x_init.shape, train_obj_init.shape, best_observed_ei)#, best_observed_mes)

            # run n_batch rounds of BayesOpt after the initial random batch
            for iteration in range(1, self.N_BATCH + 1):
                print('iteration: ', iteration)
                t0 = time.time()

                # fit the models
                # for name, param in model_ei.named_parameters():
                #     print(name, param)
                print('Model parameters BEFORE fitting:', model_ei.covar_module.alpha, model_ei.covar_module.alphanorm,
                      model_ei.covar_module.beta, model_ei.covar_module.betanorm, model_ei.likelihood.noise)

                print('fit the model')
                #fit_gpytorch_model(mll=mll_ei, max_retries=10)
                #fit_gpytorch_model(mll=mll_mes, max_retries=10)
                fit_gpytorch_model(mll=mll_ei, optimizer=botorch.optim.fit.fit_gpytorch_torch, max_retries=10)
                fit_gpytorch_model(mll=mll_mes, optimizer=botorch.optim.fit.fit_gpytorch_torch, max_retries=10)


                print('Model parameters AFTER fitting:', model_ei.covar_module.alpha, model_ei.covar_module.alphanorm,
                      model_ei.covar_module.beta, model_ei.covar_module.betanorm, model_ei.likelihood.noise)
                # define the qEI and qNEI acquisition modules using a QMC sampler
                qmc_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)

                # for best_f, we use the best observed noisy values as an approximation
                qEI = qExpectedImprovement(
                    model=model_ei,
                    best_f=standardize(train_obj_ei).max(),
                    sampler=qmc_sampler
                )

                candidate_set = torch.rand(25, bounds.size(1), device=self.device, dtype=self.dtype)
                candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set

                qMES = qMaxValueEntropy(
                    model=model_ei,
                    candidate_set=candidate_set)

                print('optimize acquisition function')
                # optimize and get new observation
                new_x_ei, new_obj_ei = self.optimize_acqf_and_get_observation(acq_func=qEI, bounds=bounds)
                new_x_mes, new_obj_mes = self.optimize_acqf_and_get_observation(acq_func=qMES, bounds=bounds)
                print("New candidates", new_obj_ei.shape, new_obj_mes.shape)


                # update training points
                train_x_ei = torch.cat([train_x_ei, new_x_ei])
                train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

                train_x_mes = torch.cat([train_x_mes, new_x_mes])
                train_obj_mes = torch.cat([train_obj_mes, new_obj_mes])

                # update progress
                print('update random')
                best_random = self.update_random_observations(best_random)
                print('update qEI best value')
                best_value_ei = train_obj_ei.max().item()
                best_observed_ei.append(best_value_ei)
                print('update qMES best value')
                best_value_mes = train_obj_mes.max().item()
                best_observed_mes.append(best_value_mes)

                print('end of batch: ', train_x_ei.shape, train_obj_ei.shape, best_observed_ei, best_observed_mes)

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                mll_ei, model_ei = self.initialize_model(
                    normalize(train_x_ei, bounds=bounds),
                    standardize(train_obj_ei),
                    state_dict=model_ei.state_dict(),
                )
                mll_mes, model_mes = self.initialize_model(
                    normalize(train_x_mes, bounds=bounds),
                    standardize(train_obj_mes),
                    state_dict=model_mes.state_dict(),
                )
                t1 = time.time()

                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: best_value (random, qEI) = "
                        f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}), "
                        f"time = {t1 - t0:>4.2f}.", end=""
                    )
                else:
                    print(".", end="")

            best_observed_all_ei.append(best_observed_ei)
            best_observed_all_mes.append(best_observed_mes)
            best_random_all.append(best_random)

        return best_observed_all_ei, best_observed_all_mes, best_random_all
        #return best_observed_all_ei, best_random_all


    def plot(self, **kwargs):
        import numpy as np
        import matplotlib.pyplot as plt
        #plt.interactive(False)

        def ci(y):
            ## Confidence interval
            return 1.96 * y.std(axis=0) / np.sqrt(self.N_TRIALS)

        iters = np.arange(self.N_BATCH + 1) * self.BATCH_SIZE

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for label, best_observed_all in kwargs.items():
            y = np.asarray(best_observed_all)
            ax.errorbar(iters, y.mean(axis=0), yerr=ci(y), label=label, linewidth=1.5)
            print(label, y.mean(axis=0))


        plt.plot([0, self.N_BATCH * self.BATCH_SIZE], [1] * 2, 'k', label="true best bjective", linewidth=2)
        ax.set_ylim(bottom=0.)
        ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
        ax.legend(loc="lower right")
        plt.savefig('large_scale_test.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    import sys
    sys.path.append('/Users/erio/Dropbox/URP project/Code/PQC_composer')
    np.random.seed(20)
    torch.manual_seed(20)

    torch.set_printoptions(precision=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    BATCH_SIZE = 5
    num_qubits = 4
    MAX_OP_NODES = 30

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = torch.tensor([[0.] * encoding_length, [1.0] * encoding_length], device=device, dtype=dtype)

    N_TRIALS = 1
    N_BATCH = 10
    MC_SAMPLES = 2048

    qnnbo = QNN_BO(
        num_qubits = num_qubits,
        MAX_OP_NODES = MAX_OP_NODES,
        N_TRIALS = N_TRIALS,
        N_BATCH = N_BATCH,
        BATCH_SIZE = BATCH_SIZE,
        MC_SAMPLES = MC_SAMPLES
    )

    # best_observed_all_ei, best_observed_all_mes, best_random_all = qnnbo.optimize(bounds=bounds, num_init_points=5)
    # #best_observed_all_ei, best_random_all = qnnbo.optimize(bounds=bounds, num_init_points=30)
    #
    # #qnnbo.plot(best_observed_all_ei, best_observed_all_nei, best_random_all)
    # to_plot = {'qEI': best_observed_all_ei, 'qMES': best_observed_all_mes, 'random': best_random_all}
    # #to_plot = {'qEI': best_observed_all_ei, 'random': best_random_all}
    #
    # qnnbo.plot(to_plot)



    acqf_choices = ['random', 'qEI', 'GIBBON']
    list_of_best_observed_all = qnnbo.optimize_new(bounds=bounds,acqf_choices=acqf_choices,num_init_points=5)
    to_plot = dict(zip(acqf_choices, list_of_best_observed_all))
    qnnbo.plot(to_plot)