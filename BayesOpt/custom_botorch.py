import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


from botorch.test_functions import Hartmann
neg_hartmann6 = Hartmann(dim=6, negate=True)
def outcome_constraint(X):
    """L1 constraint; feasible if less than or equal to zero."""
    return X.sum(dim=-1) - 3
def weighted_obj(X,num_qubits=2,MAX_OP_NODES=12):
    """Feasibility weighted objective; zero if not feasible."""
    #return neg_hartmann6(X) * (outcome_constraint(X) <= 0).type_as(X)
    latent_func_values = []
    for enc in X.detach().numpy():
        qc = vec_to_circuit(vec=enc, num_qubits=num_qubits, MAX_OP_NODES=MAX_OP_NODES)
        latent_func_values.append(latent_func(qc))
    return torch.tensor(latent_func_values)
def latent_func(circuit):
    #return sum(circuit.count_ops().values())
    return len(circuit.parameters)


# from botorch.models import FixedNoiseGP, ModelListGP
# from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
# NOISE_SE = 0.5
# train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)


from embedding import qc_embedding
from QuOTMANN import optimal_transport
import gpytorch
def vec_to_circuit(vec, num_qubits=2, MAX_OP_NODES=6):
    #_, qc, _ = dagcircuit_embedding.enc_to_qc(num_qubits=num_qubits, adj_encoding=vec, MAX_OP_NODES=MAX_OP_NODES)
    qc = qc_embedding.enc_to_qc(num_qubits=num_qubits, encoding=vec)
    return qc
class CircuitKernel(gpytorch.kernels.Kernel):
    is_stationary = True
    def forward(self, vec_list1, vec_list2, num_qubits, MAX_OP_NODES, alpha=1, beta=1, **params):
        dist = []
        for i,vec1 in enumerate(vec_list1):
            for j,vec2 in enumerate(vec_list2):
                circ1 = vec_to_circuit(vec1.detach().numpy(), num_qubits, MAX_OP_NODES)
                circ2 = vec_to_circuit(vec2.detach().numpy(), num_qubits, MAX_OP_NODES)
                dist.append(self.circuit_distance(circ1, circ2))
        dist = torch.stack(dist).view(len(vec_list1), len(vec_list2))
        K = alpha * torch.exp(-beta * dist)
        return K
    def circuit_distance(self, circ1, circ2):
        return optimal_transport.circuit_distance(PQC_1=circ1, PQC_2=circ2)


from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
NOISE_SE = 0.5
def generate_initial_data(n, num_qubits=2, MAX_OP_NODES=12):
    # generate training data
    #train_x = torch.rand(10, 6, device=device, dtype=dtype)
    #exact_obj = neg_hartmann6(train_x).unsqueeze(-1)  # add output dimension
    #exact_con = outcome_constraint(train_x).unsqueeze(-1)  # add output dimension
    #train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    #train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)

    #encoding_length = int((MAX_OP_NODES ** 2 + (2 * num_qubits + 1) * MAX_OP_NODES) // 2)
    encoding_length = (num_qubits+1)*MAX_OP_NODES
    train_x = torch.rand(n, encoding_length, device=device, dtype=dtype)
    exact_obj = [latent_func(vec_to_circuit(vec,num_qubits=num_qubits,MAX_OP_NODES=MAX_OP_NODES)) for vec in train_x.numpy()]
    #print(exact_obj)
    exact_obj = torch.as_tensor(exact_obj, device=device, dtype=dtype).unsqueeze(-1)
    exact_con = outcome_constraint(train_x).unsqueeze(-1)
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    best_observed_value = weighted_obj(train_x).max().item()
    return train_x, train_obj, train_con, best_observed_value
def initialize_model(train_x, train_obj, train_con, covar_module=None, input_transform=None, state_dict=None):
    # define models for objective and constraint
    #model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj), covar_module, None, input_transform).to(train_x)
    #model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con), covar_module, None, input_transform).to(train_x)
    #likelihood_obj = gpytorch.likelihoods.GaussianLikelihood()
    #likelihood_con = gpytorch.likelihoods.GaussianLikelihood()
    model_obj = SingleTaskGP(train_x, train_obj,
                             covar_module=covar_module, outcome_transform=None, input_transform=input_transform).to(train_x)
    model_con = SingleTaskGP(train_x, train_con,
                             covar_module=covar_module, outcome_transform=None, input_transform=input_transform).to(train_x)
    # combine into a multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model



from botorch.acquisition.objective import ConstrainedMCObjective
def obj_callable(Z):
    return Z[..., 0]
def constraint_callable(Z):
    return Z[..., 1]
# define a feasibility-weighted objective for optimization
constrained_obj = ConstrainedMCObjective(
    objective=obj_callable,
    constraints=[constraint_callable],
)


from botorch.optim import optimize_acqf
BATCH_SIZE = 3
num_qubits = 2; MAX_OP_NODES = 12
#encoding_length = int((MAX_OP_NODES ** 2 + (2 * num_qubits + 1) * MAX_OP_NODES) // 2)
encoding_length = (num_qubits+1) * MAX_OP_NODES
bounds = torch.tensor([[0.0] * encoding_length, [1.0] * encoding_length], device=device, dtype=dtype)
def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=500,  # used for intialization heuristic
        options={
            "batch_limit": 5,
            "max_iter": 200,
        }
    )
    # observe new values
    new_x = candidates.detach()
    # exact_obj = neg_hartmann6(new_x).unsqueeze(-1)  # add output dimension
    # exact_con = outcome_constraint(new_x).unsqueeze(-1)  # add output dimension
    # new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    # new_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    exact_obj = [latent_func(vec_to_circuit(vec,num_qubits=num_qubits,MAX_OP_NODES=MAX_OP_NODES)) for vec in new_x.numpy()]
    exact_obj = torch.as_tensor(exact_obj, device=device, dtype=dtype).unsqueeze(-1)
    exact_con = outcome_constraint(new_x).unsqueeze(-1)
    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    new_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    return new_x, new_obj, new_con
def update_random_observations(best_random):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = torch.rand(BATCH_SIZE, encoding_length)
    next_random_best = weighted_obj(rand_x).max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 2
N_BATCH = 3
MC_SAMPLES = 30

verbose = False

best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []

# average over multiple trials
for trial in range(1, N_TRIALS + 1):

    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    best_observed_ei, best_observed_nei, best_random = [], [], []

    # call helper functions to generate initial training data and initialize model
    train_x_ei, train_obj_ei, train_con_ei, best_observed_value_ei = generate_initial_data(n=10)
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei, train_con_ei)

    train_x_nei, train_obj_nei, train_con_nei = train_x_ei, train_obj_ei, train_con_ei
    best_observed_value_nei = best_observed_value_ei
    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)

    best_observed_ei.append(best_observed_value_ei)
    best_observed_nei.append(best_observed_value_nei)
    best_random.append(best_observed_value_ei)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):

        t0 = time.time()

        # fit the models
        fit_gpytorch_model(mll_ei)
        fit_gpytorch_model(mll_nei)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=model_ei,
            best_f=(train_obj_ei * (train_con_ei <= 0).to(train_obj_ei)).max(),
            sampler=qmc_sampler,
            objective=constrained_obj,
        )

        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
            objective=constrained_obj,
        )

        # optimize and get new observation
        new_x_ei, new_obj_ei, new_con_ei = optimize_acqf_and_get_observation(qEI)
        new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(qNEI)

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
        train_con_ei = torch.cat([train_con_ei, new_con_ei])

        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
        train_con_nei = torch.cat([train_con_nei, new_con_nei])

        # update progress
        best_random = update_random_observations(best_random)
        best_value_ei = weighted_obj(train_x_ei).max().item()
        best_value_nei = weighted_obj(train_x_nei).max().item()
        best_observed_ei.append(best_value_ei)
        best_observed_nei.append(best_value_nei)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei,
            train_obj_ei,
            train_con_ei,
            state_dict=model_ei.state_dict(),
        )
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            train_con_nei,
            state_dict=model_nei.state_dict(),
        )

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
    best_observed_all_nei.append(best_observed_nei)
    best_random_all.append(best_random)


import numpy as np
#rom botorch.test_functions.hartmann6 import GLOBAL_MAXIMUM

from matplotlib import pyplot as plt
#%matplotlib inline

def ci(y):
    ## Confidence interval
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

iters = np.arange(N_BATCH + 1) * BATCH_SIZE
y_ei = np.asarray(best_observed_all_ei)
y_nei = np.asarray(best_observed_all_nei)
y_rnd = np.asarray(best_random_all)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.errorbar(iters, y_rnd.mean(axis=0), yerr=ci(y_rnd), label="random", linewidth=1.5)
ax.errorbar(iters, y_ei.mean(axis=0), yerr=ci(y_ei), label="qEI", linewidth=1.5)
ax.errorbar(iters, y_nei.mean(axis=0), yerr=ci(y_nei), label="qNEI", linewidth=1.5)
plt.plot([0, N_BATCH * BATCH_SIZE], [neg_hartmann6.optimal_value] * 2, 'k', label="true best bjective", linewidth=2)
ax.set_ylim(bottom=0.5)
ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
ax.legend(loc="lower right")
plt.plot()

print(y_rnd.mean(axis=0))
print(y_ei.mean(axis=0))
print(y_nei.mean(axis=0))