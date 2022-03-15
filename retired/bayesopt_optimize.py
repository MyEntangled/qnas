import numpy as np
import torch

from embedding import qc_embedding
from QuOTMANN import optimal_transport
import gpytorch

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

import cma

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from botorch.utils.transforms import standardize, normalize, unnormalize


class CircuitKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, decoder, num_qubits, MAX_OP_NODES, device, dtype, alpha=1, beta=0.1, **kwargs):
        self.decoder = decoder
        self.num_qubits = num_qubits
        self.MAX_OP_NODES = MAX_OP_NODES
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        super().__init__(**kwargs)

    def forward(self, x1, x2, **params):

        if len(x1.shape) == 3: ## (n_batchs, n_samples, n_features)
            is_batched = True

            dist = torch.zeros(size=(x1.shape[0], x1.shape[1], x2.shape[1])).to(x1)
            for k in range(dist.shape[0]): # iterate through batchs
                for i in range(dist.shape[1]):
                    for j in range(dist.shape[2]):
                        qc1 = self.decoder(vec=x1[k,i].detach().numpy())
                        qc2 = self.decoder(vec=x2[k,j].detach().numpy())
                        dist[k,i,j] = self.circuit_distance(circ1=qc1, circ2=qc2)

        elif len(x1.shape) == 2: ## (n_samples, n_features)
            is_batched = False
            #x1 = x1.view(1,x1.shape[0],x1.shape[1])
            #x2 = x2.view(1,x2.shape[0],x2.shape[1])

            dist = torch.zeros(size=(x1.shape[0], x2.shape[0])).to(x1)
            for i in range(dist.shape[0]):
                for j in range(dist.shape[1]):
                    qc1 = self.decoder(vec=x1[i].detach().numpy())
                    qc2 = self.decoder(vec=x2[j].detach().numpy())
                    dist[i,j] = self.circuit_distance(circ1=qc1, circ2=qc2)

        # if not is_batched:
        #     dist = dist[0]
        #     x1 = x1.view(x1.shape[1],x1.shape[2])
        #     x2 = x2.view(x2.shape[1], x2.shape[2])

        K = self.alpha * torch.exp(-self.beta * dist)
        #K = gpytorch.lazy.lazify(K)
        print(x1.shape, x2.shape, K.shape, type(K))
        return K

        # dist = np.zeros((len(vec_list1), len(vec_list2)))
        # for i, vec1 in enumerate(vec_list1):
        #     for j, vec2 in enumerate(vec_list2):
        #         qc1 = self.decoder(num_qubits=self.num_qubits, encoding=vec1.detach().numpy())
        #         qc2 = self.decoder(num_qubits=self.num_qubits, encoding=vec2.detach().numpy())
        #         #dist.append(self.circuit_distance(circ1=qc1, circ2=qc2))
        #         dist[i,j] = self.circuit_distance(circ1=qc1, circ2=qc2)
        # #dist = torch.stack(dist).view(len(vec_list1), len(vec_list2))
        # dist = torch.tensor(dist, device=device, dtype=dtype)
        # K = self.alpha * torch.exp(-self.beta * dist)
        # #K = gpytorch.lazy.LazyTensor(K)
        # K = gpytorch.lazy.lazify(K)
        # print(dist.shape, type(K))
        # return K

    def circuit_distance(self, circ1, circ2):
        return optimal_transport.circuit_distance(PQC_1=circ1, PQC_2=circ2)

class FirstSincKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20).to(x1))
        K = torch.sin(diff).div(diff)
        trash1 = x1.detach().numpy()
        trash2 = x2.detach().numpy()
        print(x1.shape, x2.shape, K.shape, type(K))
        return K

class QNN_BO():
    def __init__(self, num_qubits, MAX_OP_NODES, N_TRIALS, N_BATCH, BATCH_SIZE, MC_SAMPLES, NOISE_SE, device=None, dtype=None):
        self.num_qubits = num_qubits
        self.MAX_OP_NODES = MAX_OP_NODES
        self.encoding_length = (self.num_qubits + 1) * self.MAX_OP_NODES

        self.NOISE_SE = NOISE_SE

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.double

        self.N_TRIALS = N_TRIALS
        self.N_BATCH = N_BATCH
        self.BATCH_SIZE = BATCH_SIZE
        self.MC_SAMPLES = MC_SAMPLES

        # define a feasibility-weighted objective for optimization
        # self.constrained_obj = ConstrainedMCObjective(
        #     objective=self.obj_callable,
        #     constraints=[self.constraint_callable],
        # )

    ## PROBLEM SETUP
    def outcome_constraint(self,X):
        """L1 constraint; feasible if less than or equal to zero."""
        #print(torch.all(X>=-1), torch.all(X<=1))
        return X.sum(dim=-1)
        #return -100*torch.ones_like(X[:,0])
        #print(X.shape, torch.any(X[:,-self.MAX_OP_NODES:] > 0, dim=1).type(dtype).shape)
        #return 1000*torch.any(X[:,-self.MAX_OP_NODES:] > 0, dim=1).type(dtype).to(X)

    def weighted_obj(self,X):
        """Feasibility weighted objective; zero if not feasible."""
        #return neg_hartmann6(X) * (outcome_constraint(X) <= 0).type_as(X)
        latent_func_values = []
        for enc in X.detach().numpy():
            #qc = vec_to_circuit(vec=enc, num_qubits=num_qubits, MAX_OP_NODES=MAX_OP_NODES)
            #qc = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=enc)
            qc = self.vec_to_circuit(vec=enc)
            latent_func_values.append(self.latent_func(qc))
        return torch.tensor(latent_func_values)
        r#eturn torch.tensor(latent_func_values) * (self.outcome_constraint(X) <= 0).type_as(X)

    def latent_func(self,circuit):
        f = circuit.num_parameters / self.MAX_OP_NODES
        return torch.as_tensor(f, device=self.device, dtype=self.dtype)

    # def vec_to_circuit(vec, num_qubits=2, MAX_OP_NODES=6):
    #     qc = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=vec)
    #     return qc
    def vec_to_circuit(self,vec):
        qc = qc_embedding.enc_to_qc(num_qubits=self.num_qubits, encoding=vec)
        #print(qc.draw())
        return qc

    ## MODEL INITIALIZATION
    def generate_initial_data(self, n):
        # generate training data

        train_x = torch.rand(n, self.encoding_length, device=self.device, dtype=self.dtype)
        exact_obj = [self.latent_func(self.vec_to_circuit(vec=vec)) for vec in train_x.numpy()]
        #print(exact_obj)

        exact_obj = torch.as_tensor(exact_obj, device=self.device, dtype=self.dtype).unsqueeze(-1)
        exact_con = self.outcome_constraint(train_x).unsqueeze(-1)
        train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
        train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
        best_observed_value = self.weighted_obj(X=train_x).max().item()
        return train_x, train_obj, train_con, best_observed_value

    # def initialize_model(self, train_x, train_obj, train_con, covar_module=None, input_transform=None, state_dict=None):
    #     # define models for objective and constraint
    #     #model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj), covar_module, None, input_transform).to(train_x)
    #     #model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con), covar_module, None, input_transform).to(train_x)
    #     #likelihood_obj = gpytorch.likelihoods.GaussianLikelihood()
    #     #likelihood_con = gpytorch.likelihoods.GaussianLikelihood()
    #
    #     covar_module = covar_module or gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel)
    #     #covar_module = covar_module or FirstSincKernel()
    #     # if covar_module is None:
    #     #     covar_module = CircuitKernel(
    #     #         decoder=self.vec_to_circuit,
    #     #         num_qubits=self.num_qubits,
    #     #         MAX_OP_NODES=MAX_OP_NODES,
    #     #         alpha=1,
    #     #         beta=0.1,
    #     #         device=self.device,
    #     #         dtype=self.dtype
    #     #     )
    #
    #     model_obj = SingleTaskGP(train_x, train_obj, covar_module=covar_module, input_transform=input_transform).to(train_x)
    #     model_con = SingleTaskGP(train_x, train_con, covar_module=covar_module, input_transform=input_transform).to(train_x)
    #
    #     # combine into a multi-output GP model
    #     model = ModelListGP(model_obj, model_con)
    #     mll = SumMarginalLogLikelihood(model.likelihood, model)
    #     # load state dict if it is passed
    #     if state_dict is not None:
    #         model.load_state_dict(state_dict)
    #     return mll, model

    def initialize_model_2(self, train_x, train_obj, covar_module=None, input_transform=None, state_dict=None):
        # define models for objective

        covar_module = covar_module or gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        #covar_module = covar_module or FirstSincKernel()

        # if covar_module is None:
        #     covar_module = CircuitKernel(
        #         decoder=self.vec_to_circuit,
        #         num_qubits=self.num_qubits,
        #         MAX_OP_NODES=MAX_OP_NODES,
        #         alpha=1,
        #         beta=0.1,
        #         device=self.device,
        #         dtype=self.dtype
        #     )

        model = SingleTaskGP(normalize(train_x, bounds=bounds), standardize(train_obj), covar_module=covar_module, input_transform=input_transform).to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    ## EXTRACTION OF OBJECTIVE AND CONSTRAINT FROM THE GP
    def obj_callable(self, Z):
        return Z[..., 0]
    def constraint_callable(self, Z):
        return Z[..., 1]


    ## Zero-th order optimizer of acqf
    def custom_optimize_acqf(self, acq_func, bounds):
        """
        Return solution candidates for the acquisition function being maximized
        """

        candidates = torch.empty(size=(self.BATCH_SIZE, self.encoding_length), device=self.device, dtype=self.dtype)

        for i in range(self.BATCH_SIZE):
            # get initial condition for CMAES in numpy form
            # note that CMAES expects a different shape (no explicit q-batch dimension)
            x0 = np.random.normal(loc=0.5,scale=0.4,size=self.encoding_length).clip(bounds[0].numpy(), bounds[1].numpy())

            # create the CMA-ES optimizer
            es = cma.CMAEvolutionStrategy(
                x0=x0,
                sigma0=0.2,
                inopts={'bounds': bounds.tolist(), 'popsize': 50}
                #inopts={'popsize':50, 'transformation': [np.tanh, None]}
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
            candidates[i] = torch.from_numpy(es.best.x).to(X)
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
        candidates = self.custom_optimize_acqf(acq_func=acq_func, bounds=bounds)

        #print(candidates.shape)
        # observe new values
        new_x = candidates.detach()
        new_x = unnormalize(new_x, bounds=bounds)

        exact_obj = [self.latent_func(self.vec_to_circuit(vec=vec)) for vec in new_x.numpy()]
        exact_obj = torch.as_tensor(exact_obj, device=self.device, dtype=self.dtype).unsqueeze(-1)
        exact_con = self.outcome_constraint(new_x).unsqueeze(-1)
        new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
        new_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
        return new_x, new_obj, new_con
    def update_random_observations(self, best_random):
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        rand_x = torch.rand(BATCH_SIZE, self.encoding_length)
        next_random_best = self.weighted_obj(X=rand_x).max().item()
        best_random.append(max(best_random[-1], next_random_best))
        return best_random

    def optimize(self, bounds):
        verbose = False

        best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []

        # average over multiple trials
        for trial in range(1, self.N_TRIALS + 1):

            print(f"\nTrial {trial:>2} of {self.N_TRIALS} ", end="")
            best_observed_ei, best_observed_nei, best_random = [], [], []

            # call helper functions to generate initial training data and initialize model
            train_x_ei, train_obj_ei, train_con_ei, best_observed_value_ei = self.generate_initial_data(n=5)
            #mll_ei, model_ei = self.initialize_model(train_x_ei, train_obj_ei, train_con_ei)
            mll_ei, model_ei = self.initialize_model_2(train_x_ei, train_obj_ei)

            #train_x_nei, train_obj_nei, train_con_nei = train_x_ei, train_obj_ei, train_con_ei
            #best_observed_value_nei = best_observed_value_ei
            #mll_nei, model_nei = self.initialize_model(train_x_nei, train_obj_nei, train_con_nei)

            best_observed_ei.append(best_observed_value_ei)
            #best_observed_nei.append(best_observed_value_nei)
            best_random.append(best_observed_value_ei)
            print(train_x_ei.shape, train_obj_ei.shape, best_observed_ei)

            # run n_batch rounds of BayesOpt after the initial random batch
            for iteration in range(1, self.N_BATCH + 1):

                t0 = time.time()

                # fit the models
                #with gpytorch.settings.cholesky_jitter(10e-8):
                #fit_gpytorch_model(mll_ei, fit_gpytorch_torch)
                #fit_gpytorch_model(mll_nei, fit_gpytorch_torch)
                fit_gpytorch_model(mll_ei,max_retries=10)
                #fit_gpytorch_model(mll_nei)

                # define the qEI and qNEI acquisition modules using a QMC sampler
                qmc_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)

                # for best_f, we use the best observed noisy values as an approximation
                # qEI = qExpectedImprovement(
                #     model=model_ei,
                #     best_f=(train_obj_ei * (train_con_ei <= 0).to(train_obj_ei)).max(),
                #     sampler=qmc_sampler,
                #     objective=self.constrained_obj,
                # )

                qEI = qExpectedImprovement(
                    model=model_ei,
                    best_f=standardize(train_obj_ei).max(),
                    sampler=qmc_sampler
                )

                # qNEI = qNoisyExpectedImprovement(
                #     model=model_nei,
                #     X_baseline=train_x_nei,
                #     sampler=qmc_sampler,
                #     objective=self.constrained_obj,
                # )

                # optimize and get new observation
                new_x_ei, new_obj_ei, new_con_ei = self.optimize_acqf_and_get_observation(acq_func=qEI, bounds=bounds)
                #new_x_nei, new_obj_nei, new_con_nei = self.optimize_acqf_and_get_observation(acq_func=qNEI, bounds=bounds)

                # update training points
                train_x_ei = torch.cat([train_x_ei, new_x_ei])
                train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
                train_con_ei = torch.cat([train_con_ei, new_con_ei])

                #train_x_nei = torch.cat([train_x_nei, new_x_nei])
                #train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
                #train_con_nei = torch.cat([train_con_nei, new_con_nei])

                # update progress
                best_random = self.update_random_observations(best_random)
                best_value_ei = self.weighted_obj(train_x_ei).max().item()
                #best_value_nei = self.weighted_obj(train_x_nei).max().item()
                best_observed_ei.append(best_value_ei)
                #best_observed_nei.append(best_value_nei)
                print(train_x_ei.shape, train_obj_ei.shape, train_con_ei.shape, best_observed_ei)

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                # mll_ei, model_ei = self.initialize_model(
                #     train_x_ei,
                #     train_obj_ei,
                #     train_con_ei,
                #     state_dict=model_ei.state_dict(),
                # )
                mll_ei, model_ei = self.initialize_model_2(
                    train_x_ei,
                    train_obj_ei,
                    state_dict=model_ei.state_dict(),
                )
                # mll_nei, model_nei = self.initialize_model(
                #     train_x_nei,
                #     train_obj_nei,
                #     train_con_nei,
                #     state_dict=model_nei.state_dict(),
                # )

                t1 = time.time()

                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI) = "
                        f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_nei:>4.2f}), "
                        f"time = {t1 - t0:>4.2f}.", end=""
                    )
                else:
                    print(".", end="")

            best_observed_all_ei.append(best_observed_ei)
            #best_observed_all_nei.append(best_observed_nei)
            best_random_all.append(best_random)

        return best_observed_all_ei, best_observed_all_nei, best_random_all

    # def plot(self, best_observed_all_ei, best_observed_all_nei, best_random_all):
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     plt.interactive(False)
    #
    #     def ci(y):
    #         ## Confidence interval
    #         return 1.96 * y.std(axis=0) / np.sqrt(self.N_TRIALS)
    #
    #     iters = np.arange(self.N_BATCH + 1) * self.BATCH_SIZE
    #     y_ei = np.asarray(best_observed_all_ei)
    #     y_nei = np.asarray(best_observed_all_nei)
    #     y_rnd = np.asarray(best_random_all)
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #     ax.errorbar(iters, y_rnd.mean(axis=0), yerr=ci(y_rnd), label="random", linewidth=1.5)
    #     ax.errorbar(iters, y_ei.mean(axis=0), yerr=ci(y_ei), label="qEI", linewidth=1.5)
    #     ax.errorbar(iters, y_nei.mean(axis=0), yerr=ci(y_nei), label="qNEI", linewidth=1.5)
    #     plt.plot([0, self.N_BATCH * self.BATCH_SIZE], [1] * 2, 'k', label="true best bjective", linewidth=2)
    #     ax.set_ylim(bottom=0.5)
    #     ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
    #     ax.legend(loc="lower right")
    #     plt.plot()
    #
    #     print(y_rnd.mean(axis=0))
    #     print(y_ei.mean(axis=0))
    #     print(y_nei.mean(axis=0))

    def plot(self, kwargs):
        import numpy as np
        import matplotlib.pyplot as plt
        #plt.interactive(False)

        def ci(y):
            ## Confidence interval
            return 1.96 * y.std(axis=0) / np.sqrt(self.N_TRIALS)

        iters = np.arange(self.N_BATCH + 1) * self.BATCH_SIZE
        # y_ei = np.asarray(best_observed_all_ei)
        # y_nei = np.asarray(best_observed_all_nei)
        # y_rnd = np.asarray(best_random_all)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for label, best_observed_all in kwargs.items():
            y = np.asarray(best_observed_all)
            ax.errorbar(iters, y.mean(axis=0), yerr=ci(y), label=label, linewidth=1.5)
            print(label, y.mean(axis=0))

        # ax.errorbar(iters, y_rnd.mean(axis=0), yerr=ci(y_rnd), label="random", linewidth=1.5)
        # ax.errorbar(iters, y_ei.mean(axis=0), yerr=ci(y_ei), label="qEI", linewidth=1.5)
        # ax.errorbar(iters, y_nei.mean(axis=0), yerr=ci(y_nei), label="qNEI", linewidth=1.5)

        plt.plot([0, self.N_BATCH * self.BATCH_SIZE], [1] * 2, 'k', label="true best bjective", linewidth=2)
        ax.set_ylim(bottom=0.5)
        ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
        ax.legend(loc="lower right")
        plt.plot()

        # print(y_rnd.mean(axis=0))
        # print(y_ei.mean(axis=0))
        # print(y_nei.mean(axis=0))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    NOISE_SE = 0.5

    BATCH_SIZE = 4
    num_qubits = 2
    MAX_OP_NODES = 50

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = torch.tensor([[-1.0] * encoding_length, [1.0] * encoding_length], device=device, dtype=dtype)

    N_TRIALS = 1
    N_BATCH = 5
    MC_SAMPLES = 30

    qnnbo = QNN_BO(
        num_qubits = num_qubits,
        MAX_OP_NODES = MAX_OP_NODES,
        N_TRIALS = N_TRIALS,
        N_BATCH = N_BATCH,
        BATCH_SIZE = BATCH_SIZE,
        MC_SAMPLES = MC_SAMPLES,
        NOISE_SE = NOISE_SE
    )

    best_observed_all_ei, best_observed_all_nei, best_random_all = qnnbo.optimize(bounds=bounds)

    #qnnbo.plot(best_observed_all_ei, best_observed_all_nei, best_random_all)
    to_plot = {'qEI': best_observed_all_ei,'random': best_random_all}
    qnnbo.plot(to_plot)