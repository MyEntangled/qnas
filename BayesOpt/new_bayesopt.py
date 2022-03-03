import numpy as np
import torch

from embedding import qc_embedding
from QuOTMANN import optimal_transport
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
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples


class CustomSincGPKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, num_str_weights, priors=None, constraints=None, **kwargs):
        super().__init__(**kwargs)

        # register the raw parameter
        self.register_parameter(
            name='raw_alpha', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_alphanorm', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        for i in range(num_str_weights):
            self.register_parameter(
                name='raw_beta_'+str(i), parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
            )
            self.register_parameter(
                name='raw_betanorm_'+str(i), parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
            )

        # set the parameter constraint to be positive, when nothing is specified
        if constraints is None:
            alpha_constraint = gpytorch.constraints.Positive()
            alphanorm_constraint = gpytorch.constraints.Positive()
            beta_constraints = [gpytorch.constraints.Positive()]*num_str_weights
            betanorm_constraints = [gpytorch.constraints.Positive()]*num_str_weights
        else:
            alpha_constraint = constraints[0]
            alphanorm_constraint = constraints[1]
            beta_constraints, betanorm_constraints = zip(*constraints[2])


        # register the constraints
        self.register_constraint("raw_alpha", alpha_constraint)
        self.register_constraint("raw_alphanorm", alphanorm_constraint)
        for i in range(num_str_weights):
            self.register_constraint("raw_beta_"+str(i), beta_constraints[i])
            self.register_constraint("raw_betanorm_"+str(i), betanorm_constraints[i])


        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if priors is None:
            alpha_prior = None
            alphanorm_prior = None
            beta_priors = [None] * num_str_weights
            betanorm_priors = [None] * num_str_weights
        else:
            alpha_prior = priors[0]
            alphanorm_prior = priors[1]
            beta_priors, betanorm_priors = zip(*priors[2])


        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior, lambda m: m.alpha, lambda m,v: m._set_alpha)
        if alphanorm_prior is not None:
            self.register_prior("alphanorm_prior", alphanorm_prior, lambda m: m.alphanorm, lambda m,v: m._set_alphanorm)
        for i in range(num_str_weights):
            if beta_priors[i] is not None:
                self.register_prior(name="beta_prior_"+str(i), prior=beta_priors[i], lambda m: m.beta[i], lambda m, v: m._set_beta(idx=i,val=v))
            if betanorm_priors is not None:
                self.register_prior(name="betanorm_prior_"+str(i), prior=betanorm_priors[i], lambda m: m.betanorm[i], lambda m, v: m._set_betanorm(idx=i,val=v))

    # now set up the 'actual' paramter
    @property
    def alpha(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_alpha_constraint.transform(self.raw_alpha)
    @property
    def alphanorm(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_alphanorm_constraint.transform(self.raw_alphanorm)
    @property
    def beta(self,idx):
        # when accessing the parameter, apply the constraint transform
        return getattr(self, f"raw_beta_{idx}_constraint")
    @property
    def betanorm(self,idx):
        # when accessing the parameter, apply the constraint transform
        return getattr(self, f"raw_betanorm_{idx}_constraint")

    def _set_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length_1)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length_1=self.raw_length_constraint.inverse_transform(value))

    def _set_length_2(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length_2)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length_2=self.raw_length_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        print(self.length_1, self.length_2)
        x1 = x1.div(self.length_1)
        x2 = x2.div(self.length_2)
        diff = self.covar_dist(x1, x2, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20).to(x1))
        K = torch.sin(diff).div(diff)
        print(x1.shape, x2.shape, K.shape, type(K))
        print('kernel: ', K)

        return gpytorch.lazify(K)

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
            dist_norm = torch.zeros_like(dist)

            for k in range(dist.shape[0]): # iterate through batchs
                for i in range(dist.shape[1]):
                    for j in range(dist.shape[2]):
                        qc1 = self.decoder(vec=x1[k,i].detach().numpy())
                        qc2 = self.decoder(vec=x2[k,j].detach().numpy())
                        dist[k,i,j], dist_norm[k,i,j] = self.circuit_distance(circ1=qc1, circ2=qc2)

        elif len(x1.shape) == 2: ## (n_samples, n_features)
            is_batched = False
            #x1 = x1.view(1,x1.shape[0],x1.shape[1])
            #x2 = x2.view(1,x2.shape[0],x2.shape[1])

            dist = torch.zeros(size=(x1.shape[0], x2.shape[0])).to(x1)
            dist_norm = torch.zeros_like(dist)

            for i in range(dist.shape[0]):
                for j in range(dist.shape[1]):
                    qc1 = self.decoder(vec=x1[i].detach().numpy())
                    qc2 = self.decoder(vec=x2[j].detach().numpy())
                    dist[i,j], dist_norm[i,j] = self.circuit_distance(circ1=qc1, circ2=qc2)

        K = self.alpha * torch.exp(-self.beta * dist) + self.alpha * torch.exp(-self.beta * dist_norm)
        #K = gpytorch.lazy.lazify(K)
        #print(x1[0], x2[0])
        print(x1.shape, x2.shape, K.shape, type(K))
        print('kernel: ', K)
        return K


    def circuit_distance(self, circ1, circ2):
        return optimal_transport.circuit_distance(PQC_1=circ1, PQC_2=circ2)

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
        print(x1.shape, x2.shape, K.shape, type(K))
        print('kernel: ', K)

        return gpytorch.lazify(K)


class CustomGPModel(ExactGP, GPyTorchModel):
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
        f = circuit.num_parameters #/ self.MAX_OP_NODES
        return torch.as_tensor(f, device=self.device, dtype=self.dtype)

    def vec_to_circuit(self,vec):
        qc = qc_embedding.enc_to_qc(num_qubits=self.num_qubits, encoding=vec)
        #print(qc.draw())
        return qc

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

        model = SingleTaskGP(train_x, train_obj, covar_module=covar_module, input_transform=input_transform).to(train_x)
        print("NOISE LEVEL = ", model.likelihood.noise.item())
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model


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
                x0 = np.random.normal(loc=0.5,scale=0.4,size=self.encoding_length).clip(bounds[0].numpy(), bounds[1].numpy())
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

    def update_random_observations(self, best_random):
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        #rand_x = torch.rand(BATCH_SIZE, self.encoding_length)
        rand_x = draw_sobol_samples(bounds=bounds, n=1, q=1).squeeze(1)
        next_random_best = self.obj_func(X=rand_x)
        next_random_best = torch.as_tensor(next_random_best, device=self.device, dtype=self.dtype).max().item()
        best_random.append(max(best_random[-1], next_random_best))
        return best_random

    def optimize(self, bounds):
        verbose = False

        best_observed_all_ei, best_random_all = [], []

        # average over multiple trials
        for trial in range(1, self.N_TRIALS + 1):

            print(f"\nTrial {trial:>2} of {self.N_TRIALS} ", end="")
            best_observed_ei, best_random = [], []

            # call helper functions to generate initial training data and initialize model
            train_x_ei, train_obj_ei, best_observed_value_ei = self.generate_initial_data(n=5)
            mll_ei, model_ei = self.initialize_model(normalize(train_x_ei, bounds=bounds), standardize(train_obj_ei))

            best_observed_ei.append(best_observed_value_ei)
            best_random.append(best_observed_value_ei)
            print('data initialization: ', train_x_ei.shape, train_obj_ei.shape, best_observed_ei)

            # run n_batch rounds of BayesOpt after the initial random batch
            for iteration in range(1, self.N_BATCH + 1):
                print('iteration: ', iteration)
                t0 = time.time()

                # fit the models
                #with gpytorch.settings.cholesky_jitter(10e-8):
                #fit_gpytorch_model(mll_ei, fit_gpytorch_torch)
                for name, param in model_ei.named_parameters():
                    print(name, param)

                print('fit the model')
                fit_gpytorch_model(mll_ei,max_retries=10)

                # define the qEI and qNEI acquisition modules using a QMC sampler
                qmc_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)

                # for best_f, we use the best observed noisy values as an approximation
                qEI = qExpectedImprovement(
                    model=model_ei,
                    best_f=standardize(train_obj_ei).max(),
                    sampler=qmc_sampler
                )

                print('optimize acquisition function')
                # optimize and get new observation
                new_x_ei, new_obj_ei = self.optimize_acqf_and_get_observation(acq_func=qEI, bounds=bounds)

                # update training points
                train_x_ei = torch.cat([train_x_ei, new_x_ei])
                train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
                print('temp')

                # update progress
                print('update random')
                best_random = self.update_random_observations(best_random)
                print('update qEI best value')
                best_value_ei = train_obj_ei.max().item()
                best_observed_ei.append(best_value_ei)
                print('end of batch: ', train_x_ei.shape, train_obj_ei.shape, best_observed_ei)

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                mll_ei, model_ei = self.initialize_model(
                    normalize(train_x_ei, bounds=bounds),
                    standardize(train_obj_ei),
                    state_dict=model_ei.state_dict(),
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
            best_random_all.append(best_random)

        return best_observed_all_ei, best_random_all


    def plot(self, kwargs):
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
        ax.set_ylim(bottom=0.5)
        ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
        ax.legend(loc="lower right")
        plt.plot()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    BATCH_SIZE = 3
    num_qubits = 2
    MAX_OP_NODES = 12

    encoding_length = (num_qubits + 1) * MAX_OP_NODES
    bounds = torch.tensor([[0.] * encoding_length, [1.0] * encoding_length], device=device, dtype=dtype)

    N_TRIALS = 1
    N_BATCH = 20
    MC_SAMPLES = 2048

    qnnbo = QNN_BO(
        num_qubits = num_qubits,
        MAX_OP_NODES = MAX_OP_NODES,
        N_TRIALS = N_TRIALS,
        N_BATCH = N_BATCH,
        BATCH_SIZE = BATCH_SIZE,
        MC_SAMPLES = MC_SAMPLES
    )

    best_observed_all_ei, best_random_all = qnnbo.optimize(bounds=bounds)

    #qnnbo.plot(best_observed_all_ei, best_observed_all_nei, best_random_all)
    to_plot = {'qEI': best_observed_all_ei,'random': best_random_all}
    qnnbo.plot(to_plot)