import gpytorch
import torch
import numpy as np

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from QuOTMANN import optimal_transport, structural_cost

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
                        #qc1 = self.decoder(vec=x1[k, q, i].cpu().detach().numpy())
                        qc1 = self.decoder(vec=x1[k, q, i])
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
                        #qc2 = self.decoder(vec=x2[k, q, j].cpu().detach().numpy())
                        qc2 = self.decoder(vec=x2[k, q, j])
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
                    #qc1 = self.decoder(vec=x1[k, i].cpu().detach().numpy())
                    qc1 = self.decoder(vec=x1[k, i])
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
                    #qc2 = self.decoder(vec=x2[k, j].cpu().detach().numpy())
                    qc2 = self.decoder(vec=x2[k, j])
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
                #qc1 = self.decoder(vec=x1[i].cpu().detach().numpy())
                qc1 = self.decoder(vec=x1[i])
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
                #qc2 = self.decoder(vec=x2[j].cpu().detach().numpy())
                qc2 = self.decoder(vec=x2[j])
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

        if not diag and K.shape[-1] == K.shape[-2]:
            print(torch.linalg.eigvalsh(K))

        return K
        #return gpytorch.lazify(K)

    def circuit_distance(self, circ1, circ2, nas_cost=1, nu_list=[0.1]):
        return optimal_transport.circuit_distance_POT(PQC_1=circ1, PQC_2=circ2, nas_cost=nas_cost, nu_list=nu_list)