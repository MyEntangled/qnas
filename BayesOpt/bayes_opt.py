import torch
import gpytorch
import math
from retired import dagcircuit_embedding
from QuOTMANN import optimal_transport
import botorch
from botorch.models.gpytorch import GPyTorchModel

class FirstSincKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        # return sinc(diff) = sin(diff) / diff
        print(x1.shape)
        return torch.sin(diff).div(diff)

# def RBF(X,Y,params=[1,1]):
#     pairwise_dist = pairwise_QuOTMANN(X,Y)
#     K = var * np.exp(-gamma * pairwise_dist))
#     return K
#

class CircuitKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    def forward(self, vec_list1, vec_list2, num_qubits, MAX_OP_NODES, alpha=1, beta=1, **params):
        #K = torch.zeros(size=(vec_list1.shape[0],vec_list2.shape[0]))
        dist = []
        for i,vec1 in enumerate(vec_list1):
            for j,vec2 in enumerate(vec_list2):
                circ1 = self.vec_to_circuit(vec1.numpy(), num_qubits, MAX_OP_NODES)
                circ2 = self.vec_to_circuit(vec2.numpy(), num_qubits, MAX_OP_NODES)
                dist.append(self.circuit_distance(circ1, circ2))
        dist = torch.stack(dist).view(len(vec_list1), len(vec_list2))
        K = alpha * torch.exp(-beta * dist)
        return K

    def vec_to_circuit(self, vec, num_qubits, MAX_OP_NODES):
        _, qc, _ = dagcircuit_embedding.enc_to_qc(num_qubits=num_qubits, adj_encoding=vec, MAX_OP_NODES=MAX_OP_NODES)
        return qc

    def circuit_distance(self, circ1, circ2):
        return optimal_transport.circuit_distance(PQC_1=circ1, PQC_2=circ2)


class CustomGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1 # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y.squeeze(), gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = FirstSincKernel() #CircuitKernel()
        self.to(train_X)

    def forward(self, x, num_qubits, MAX_OP_NODES):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, num_qubits=num_qubits, MAX_OP_NODES=MAX_OP_NODES)
        print(x.shape, mean_x.shape, covar_x.shape)
        print(covar_x.evaluate())
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = CustomGPModel(Xs[0], Ys[0])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    botorch.fit.fit_gpytorch_model(mll)
    return model

if __name__ == '__main__':
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 100)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    qc1 = QuantumCircuit(4)
    theta1 = ParameterVector('theta', 8)
    for i in range(4):
        qc1.rx(theta1[i],i)
    for i in range(4):
        qc1.ry(theta1[i+4],i)
    _,vec1 = dagcircuit_embedding.circuit_to_vector(qc1, 12)

    qc2 = QuantumCircuit(4)
    theta2 = ParameterVector('theta', 12)
    #theta.resize(len(theta) + 1)
    for i in range(4):
        qc2.rx(theta2[i],i)
    for i in range(4):
        qc2.cry(theta2[i+4],i%4,(i+1)%4)
    for i in range(4):
        qc2.rz(theta2[i+8],i)
    _,vec2 = dagcircuit_embedding.circuit_to_vector(qc2, 12)

    print(qc1.draw(), qc2.draw())
    print(vec1.shape, vec2.shape)

    train_x = torch.tensor([vec1, vec2])
    train_y = torch.tensor([0.6, 0.8])
    #train_x = torch.linspace(0,1,100)
    #train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
    print(train_x.shape, train_y.shape)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x, num_qubits=4, MAX_OP_NODES=12)
        print(output)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    # test_x = torch.randn(size=(10))
    # f_preds = model(test_x)
    # y_preds = likelihood(model(test_x))
    #
    # f_mean = f_preds.mean
    # f_var = f_preds.variance
    # f_covar = f_preds.covariance_matrix
    # f_samples = f_preds.sample(sample_shape=torch.Size(1000, ))



    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        observed_pred = likelihood(model(test_x))

    # with torch.no_grad():
    #     # Initialize plot
    #     f, ax = plt.subplots(1, 1, figsize=(4, 3))
    #
    #     # Get upper and lower confidence bounds
    #     lower, upper = observed_pred.confidence_region()
    #     # Plot training data as black stars
    #     ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    #     # Plot predictive means as blue line
    #     ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    #     # Shade between the lower and upper confidence bounds
    #     ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #     ax.set_ylim([-3, 3])
    #     ax.legend(['Observed Data', 'Mean', 'Confidence'])
    #     plt.show()