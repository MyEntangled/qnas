import blackbox
from embedding import qc_embedding
from quantum_obj import QFT_objective, MAXCUT_objective, QGAN_objective
from kernel import CircuitDistKernel

import numpy as np, torch, pickle, argparse, warnings, sys, os, matplotlib.pyplot as plt
from matplotlib import ticker

import cma

import botorch, botorch.optim.fit
botorch.settings.propagate_grads(state=True)

from botorch.models.gpytorch import GPyTorchModel
from botorch import fit_gpytorch_model

from botorch.optim.optimize import optimize_acqf
from botorch.optim.initializers import initialize_q_batch_nonneg

from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy, qMaxValueEntropy

from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.containers import TrainingData

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.append(os.getcwd())

# passing
parser = argparse.ArgumentParser(description='Quantum Neural Architecture Search')
parser.add_argument('-obj', '--objective_type', type=str, metavar='', required=True, help='Objective type')
parser.add_argument('-n', '--num_qubits', type=int, metavar='', required=True, help='Number of qubits')
parser.add_argument('-no', '--max_op_nodes', type=int, metavar='', required=True, help='Maximum number of gates')
parser.add_argument('-init', '--num_init_points', type=int, metavar='', required=True, help='Number of initial points')
parser.add_argument('-T', '--N_TRIALS', type=int, metavar='', required=True, help='Number of trials')
parser.add_argument('-B', '--N_BATCH', type=int, metavar='', required=True, help='Number of batches per trial')
parser.add_argument('-S', '--BATCH_SIZE', type=int, metavar='', required=True, help='Batch size')
parser.add_argument('-s', '--seed', type=int, metavar='', required=True, help='Seed')
parser.add_argument('-dir', '--output_dir', type=str, metavar='', required=True, help='Output directory name')
parser.add_argument('-gpuid', '--gpuid', type=int, metavar='', help='GPU ID')
args = parser.parse_args()

objective_type = args.objective_type
num_qubits = args.num_qubits
max_op_nodes = args.max_op_nodes
num_init_points = args.num_init_points
N_TRIALS = args.N_TRIALS
N_BATCH = args.N_BATCH
BATCH_SIZE = args.BATCH_SIZE
MC_SAMPLES = 2048  # Number of points sampled in optimization of acquisition functions
seed = args.seed
output_dir = args.output_dir
device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# seeding
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_printoptions(precision=4)

# objective_type = 'qft'  # ['qft', 'maxcut', 'qgan']
# num_qubits = 2
# max_op_nodes = 6  # Maximum number of gates
# num_init_points = 5  # Number of points sampled randomly at the beginning
# 
# N_TRIALS = 2  # Number of times the experiments run
# N_BATCH = 5 # Number of batch per trial
# BATCH_SIZE = 1  # Number of new points being sampled in a batch


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

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs):
        r"""Construct kwargs for the `SimpleCustomGP` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}

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

        self.device = device
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
        #for enc in X.cpu().detach().numpy():
        for enc in X:
            qc = self.vec_to_circuit(vec=enc)
            f = blackbox.latent_func(qc, self.objective, self.num_qubits)
            latent_func_values.append(f)
        return torch.as_tensor(latent_func_values, device=self.device, dtype=self.dtype).unsqueeze(-1)

    def vec_to_circuit(self,vec):
        return qc_embedding.enc_to_qc_torch(num_qubits=self.num_qubits, encoding=vec)

    def circuit_to_vec(self,qc):
        return qc_embedding.qc_to_enc(qc=qc,MAX_OP_NODES=self.MAX_OP_NODES)

    ## MODEL INITIALIZATION
    def generate_initial_data(self, n, bounds):
        # generate training data

        #train_x = torch.rand(n, self.encoding_length, device=self.device, dtype=self.dtype)
        train_x = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()).squeeze(1)

        train_obj = self.obj_func(X=train_x)

        best_observed_value = train_obj.max().item()
        best_observed_x = train_x[torch.nonzero(torch.isclose(train_obj, train_obj.max()).ravel()).ravel()].tolist()
        return train_x, train_obj, best_observed_x, best_observed_value

    def initialize_model(self, train_x, train_obj, covar_module=None, input_transform=None, state_dict=None):
        # define models for objective

        if covar_module is None:
            covar_module = CircuitDistKernel(encoder=self.circuit_to_vec,
                                             decoder=self.vec_to_circuit,
                                             num_qubits=self.num_qubits,
                                             MAX_OP_NODES=max_op_nodes,
                                             nu_list=[0.1, 0.2, 0.4, 0.8],
                                             device=self.device,
                                             dtype=self.dtype,
                                             distance_dict=self.distance_dict,
                                             structral_paths_dict=self.structural_paths_dict
                                             )
        #model = SingleTaskGP(train_x, train_obj, covar_module=covar_module, input_transform=input_transform).to(train_x)
        model = GPModel(train_x, train_obj, covar_module)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def optimize_acq_func(self, acq_func, bounds, num_restarts=5, raw_samples=100):
        def get_numerical_gradient(func, X, eps=1e-6):
            g = torch.zeros_like(X)
            for i in range(X.shape[-1]):
                e = torch.zeros(X.shape[-1])
                e[i] = 1.0
                res = (func(X + eps * e[None, None, :]) - func(X - eps * e[None, None, :])) / (2 * eps)
                g[:, 0, i] = res
            return -g

        min_loss = 1000
        # generate a large number of random q-batches
        for _ in range(num_restarts):
            breakpoint()
            Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(raw_samples * self.BATCH_SIZE, 1, self.encoding_length, device=self.device, dtype=self.dtype)
            Yraw = acq_func(Xraw)  # evaluate the acquisition function on these q-batches

            # apply the heuristic for sampling promising initial conditions
            X = initialize_q_batch_nonneg(Xraw, Yraw, self.BATCH_SIZE)
            #X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(num_inits,1, self.encoding_length)
            X.to(device=self.device, dtype=self.dtype)
            X.requires_grad_(True)

            optimizer = torch.optim.Adam([X], lr=0.1)

            # run a basic optimization loop
            with torch.no_grad():
                for i in range(75):
                    optimizer.zero_grad()
                    # this performs batch evaluation, so this is an N-dim tensor
                    loss = - acq_func(X).sum()  # torch.optim minimizes

                    #loss.backward()  # perform backward pass
                    numerical_grad = get_numerical_gradient(acq_func, X, 1e-6)
                    X.grad = numerical_grad
                    optimizer.step()  # take a step

                    # clamp values to the feasible set
                    for j, (lb, ub) in enumerate(zip(*bounds)):
                        X.data[..., j].clamp_(lb, ub)  # need to do this on the data not X itself

                if loss < min_loss:
                    candidates = X
                    min_loss = loss
            #         print(min_loss)
            # print('-----------')
        return candidates.squeeze(1)


    ## Zero-th order optimizer of acqf
    def cmaes_optimize_acqf(self, acq_func, bounds):
        """
        Return solution candidates for the acquisition function being maximized
        """
        candidates = torch.empty(size=(self.BATCH_SIZE, self.encoding_length), device=self.device, dtype=self.dtype)

        for i in range(self.BATCH_SIZE):
            # get initial condition for CMAES in numpy form
            # note that CMAES expects a different shape (no explicit q-batch dimension)
            #x0 = np.random.normal(loc=0.5,scale=0.4,size=self.encoding_length).clip(bounds[0].numpy(), bounds[1].numpy())
            x0 = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(self.encoding_length)

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
                    X = torch.tensor(np.array(xs), device=self.device, dtype=self.dtype)
                    # evaluate the acquisition function (optimizer assumes we're minimizing)
                    Y = - acq_func(X.unsqueeze(-2))  # acquisition functions require an explicit q-batch dimension
                    y = Y.view(-1).double().numpy()  # convert result to numpy array
                    #print(X.shape, Y.shape, y.shape)
                    es.tell(xs, y)  # return the result to the optimizer
                    for i in range(len(y)):
                        print(xs[i], y[i])

            # convert result back to a torch tensor
            candidates[i] = torch.from_numpy(es.best.x).to(candidates)
            best_y = - acq_func(torch.tensor(es.best.x, device=self.device, dtype=self.dtype).unsqueeze(-2))
            print(es.best.x, best_y)
        return candidates

    ## Helper function that performs essential BO steps
    def optimize_acqf_and_get_observation(self, model, acq_func, bounds):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates = self.optimize_acq_func(acq_func=acq_func, bounds=bounds)
        #candidates,_ = optimize_acqf(acq_function=acq_func,bounds=bounds,q=1,num_restarts=20,raw_samples=100)
        #candidates = self.cmaes_optimize_acqf(acq_func=acq_func, bounds=bounds)

        # observe new values
        new_x = unnormalize(candidates, bounds=bounds)
        train_obj = self.obj_func(X=new_x)

        return new_x, train_obj

    def update_random_observations(self, best_random_x, best_random_value, num_random_points=1):
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        rand_x = draw_sobol_samples(bounds=bounds, n=num_random_points, q=1).squeeze(1)
        rand_obj = self.obj_func(X=rand_x)

        next_random_best_value = rand_obj.max().item()
        next_random_best_x = rand_x[torch.nonzero(torch.isclose(rand_obj, rand_obj.max()).ravel()).ravel()].tolist()


        if best_random_value[-1] > next_random_best_value:
            best_random_value.append(best_random_value[-1])

        elif best_random_value[-1] < next_random_best_value:
            best_random_value.append(next_random_best_value)
            best_random_x = next_random_best_x

        else:
            best_random_value.append(next_random_best_value)
            best_random_x += next_random_best_x

        return best_random_x, best_random_value

    def bayesopt_trial(self, model, mll, train_x, train_obj, bounds, best_observed_x=[], best_observed_value=[], acqf_choice='random', candidate_set_size=10, torch_optimizer=True):
        print('Choice of acquisition function: ', acqf_choice)
        #print(train_obj)
        is_random_acqf = acqf_choice == 'random'
        # run n_batch rounds of BayesOpt
        for iteration in range(1, self.N_BATCH + 1):
            print('iteration: ', iteration)

            if acqf_choice != 'random':
                if torch_optimizer:
                    fit_gpytorch_model(mll=mll, optimizer=botorch.optim.fit.fit_gpytorch_torch, max_retries=10)
                else:
                    fit_gpytorch_model(mll=mll, max_retries=10)

            if acqf_choice == 'qEI':
                qmc_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)
                acqf = qExpectedImprovement(
                    model=model,
                    best_f = standardize(train_obj).max(),
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
            # elif acqf_choice == 'MES':
            #     candidate_set = torch.rand(candidate_set_size, bounds.size(1), device=self.device, dtype=self.dtype)
            #     candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
            #     acqf = qMaxValueEntropy(model, candidate_set)

            if is_random_acqf:
                print('update random')
                best_observed_x, best_observed_value = self.update_random_observations(best_observed_x, best_observed_value)
            else: #optimize and get new observation

                print('optimize acquisition function')
                new_x, new_obj = self.optimize_acqf_and_get_observation(model=model, acq_func=acqf, bounds=bounds)

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

            print('data initialization: x shape =', train_x_init.shape, ',best init value =', best_observed_value_init)

            # run n_batch rounds of BayesOpt after the initial random batch
            for idx,choice in enumerate(acqf_choices):
                mll, model = self.initialize_model(normalize(train_x_init, bounds=bounds), standardize(train_obj_init))
                ## best_observed_value stores the optimal obj over batchs
                ## best_observed_x only stores the final optimal circuit(s)
                best_observed_x, best_observed_value = self.bayesopt_trial(model, mll, train_x_init.clone(), train_obj_init.clone(),
                                                                 bounds=bounds,
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
        def std(y):
            return y.std(axis=0) / np.sqrt(self.N_TRIALS)

        iters = np.arange(self.N_BATCH + 1) * self.BATCH_SIZE

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for label, best_observed_all in to_plot.items():
            y = np.asarray(best_observed_all)
            mean_y = y.mean(axis=0)
            error = std(y)

            ax.errorbar(iters, mean_y, yerr=error, errorevery=self.N_BATCH*self.BATCH_SIZE // 5, label=label, alpha=.75, fmt=':', capsize=3, capthick=1, linewidth=2)

            #ax.plot(iters, mean_y, linewidth=1.5, label=label)
            ax.fill_between(iters, (mean_y - error), (mean_y + error), alpha=.05)
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
                    qc = self.vec_to_circuit(torch.tensor(vec))
                    print(qc.draw())
                    if self.objective_type == 'qgan':
                        print(-blackbox.latent_func(qc, self.objective, self.num_qubits))
                    else:
                        print(blackbox.latent_func(qc, self.objective, self.num_qubits))
                    print('----------------------------------')


qnnbo = QNN_BO(
    objective_type = objective_type,
    num_qubits = num_qubits,
    MAX_OP_NODES = max_op_nodes,
    N_TRIALS = N_TRIALS,
    N_BATCH = N_BATCH,
    BATCH_SIZE = BATCH_SIZE,
    MC_SAMPLES = MC_SAMPLES
)

encoding_length = (num_qubits + 1) * max_op_nodes
bounds = torch.tensor([[0.] * encoding_length, [1.0] * encoding_length], device=device, dtype=dtype)

acqf_choices = ['random', 'EI', 'GIBBON']
optimizer = 'torch' ## 'torch' or 'scipy'

list_of_best_observed_x_all, list_of_best_observed_value_all = qnnbo.optimize(bounds=bounds, acqf_choices=acqf_choices, num_init_points=num_init_points, optimizer=optimizer)

to_plot = dict(zip(acqf_choices, list_of_best_observed_value_all))
to_plot_ansatz = dict(zip(acqf_choices, list_of_best_observed_x_all))

imgname = '_'.join(
    [objective_type, str(num_qubits), str(max_op_nodes), str(num_init_points), str(BATCH_SIZE), str(N_BATCH),
        str(N_TRIALS), *acqf_choices, optimizer, str(seed)])
pkl_filename = './' + output_dir + '/' + imgname + '.pkl'

with open(pkl_filename, 'wb') as f:
    pickle.dump({'QNN':to_plot_ansatz, 'obj':to_plot}, f)

qnnbo.plot_ansatz(to_plot_ansatz)

filename = './' + output_dir + '/' + imgname
qnnbo.plot(to_plot, filename)
