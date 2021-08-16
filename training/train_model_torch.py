import numpy as np
import matplotlib.pyplot as plt
import torch.nn

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit  import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector

# Additional torch-related imports
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.nn import (Module, Conv2d, Linear, Dropout2d, NLLLoss,
                     MaxPool2d, Flatten, Sequential, ReLU)
import torch.nn.functional as F

from qiskit.opflow import StateFn

import matplotlib.pyplot as plt

from utility.quantum_nn import  QuantumNeuralNetwork
from utility.ansatz_template import AnsatzTemplate
from utility.data_encoding import FeatureMap
from utility.tools import *
from utility.dataset_tools import *

class PQCNet(Module):

    def __init__(self, model,
                 qi=QuantumInstance(Aer.get_backend('aer_simulator_statevector'))):
        super().__init__()
        self.qi = qi
        self.qnn = TwoLayerQNN(feature_map=model.feature_map_circ, ansatz=model.PQC,
                               input_gradients=True, exp_val=AerPauliExpectation(), quantum_instance=qi)

        # 1-dimensional input to QNN
        self.qnn = TorchConnector(self.qnn)  # Apply torch connector, weights chosen
                                         # uniformly at random from interval [-1,1].

    def forward(self, x):
        x = self.qnn(x)  # apply QNN
        #return cat((x, 1 - x), -1)
        return x

def train_PQC(net_model, train_loader, loss_func, lr, epochs):
    # Define model, optimizer, and loss function
    optimizer = optim.Adam(net_model.parameters(), lr=lr)
    #loss_func = NLLLoss()

    # Start training
    loss_list = []  # Store loss history
    net_model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = []
        for batch_idx, batch_samples in enumerate(train_loader):

            data = batch_samples['Data sample']
            target = batch_samples['Class']

            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = net_model(data)  # Forward pass

            loss = loss_func(output, target)  # Calculate loss

            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights

            total_loss.append(loss.item())  # Store loss

        loss_list.append(sum(total_loss) / len(total_loss))
        print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
            100. * (epoch + 1) / epochs, loss_list[-1]))

    return loss_list


def evaluate_PQC(net_model, test_loader, batch_size, loss_func):
    net_model.eval()  # set model to evaluation mode
    total_loss = []
    with no_grad():

        correct = 0
        for batch_idx, batch_data in enumerate(test_loader):

            data = batch_data['Data sample']
            target = batch_data['Class']
            output = net_model(data)

            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)

            #pred = output.argmax(dim=1, keepdim=True)
            pred = output > 0
            pred = 2*pred - 1
            #print(output, target, pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss = loss_func(output, target)
            total_loss.append(loss.item())

        print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'
              .format(sum(total_loss) / len(total_loss),
                      correct / len(test_loader) / batch_size * 100)
              )


if __name__ == '__main__':
    np.random.seed(0)

    feature_map = FeatureMap('PauliFeatureMap', feature_dim=2, reps=1)

    template = AnsatzTemplate()
    template.construct_simple_template(num_qubits=2, num_layers=3)

    model = QuantumNeuralNetwork(feature_map, template, platform='Qiskit')
    
    model.visualize()

    net = PQCNet(model)
    #loss_func = NLLLoss()
    loss_func = torch.nn.MSELoss()

##################

    torch.manual_seed(0)

    input_dim = 2
    num_samples = 40

    # Generate random input coordinates (X) and binary labels (y)
    X = 2 * np.random.rand(num_samples, input_dim) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}, y01 will be used for CircuitQNN example
    y = 2 * y01 - 1  # in {-1, +1}, y will be used for OplowQNN example

    # Convert to torch Tensors
    X_ = Tensor(X)
    y01_ = Tensor(y01).reshape(len(y)).long()
    y_ = Tensor(y).reshape(len(y), 1)


    num_train_data = int(0.8 * len(X_))
    data_train = CustomDataset(X_[:num_train_data], y_[:num_train_data])#, transform=transforms.Compose([ToTensor()]))
    data_test = CustomDataset(X_[num_train_data:], y_[num_train_data:])#, transform=transforms.Compose([ToTensor()]))

    batch_size = 4
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=4)


    lr = 0.01
    epochs = 100
    # Train the PQC model
    loss_list = train_PQC(net, train_loader, loss_func, lr, epochs)

    # Plot loss convergence
    plt.plot(loss_list)
    plt.title('Hybrid NN Training Convergence')
    plt.xlabel('Training Iterations')
    plt.ylabel('MSE Loss')
    plt.show()

    # Evaluate the model
    evaluate_PQC(net, test_loader, batch_size, loss_func)