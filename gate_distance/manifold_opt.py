import torch
import geoopt

#torch.manual_seed(0)

class UnitaryLinear(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = geoopt.ManifoldParameter(torch.empty(self.dim, self.dim, dtype=torch.cfloat),
                                               manifold=geoopt.EuclideanStiefel())
        self.reset_parameters()

    def forward(self, input):
        return unitary_linear(input, weight=self.weight)

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight)


def unitary_linear(input, weight):
    output = torch.matmul(input, weight.T)
    return output


def fidelity_loss(output, target):
    '''
    Loss = - mean( |x[i].T * y[i]|^2 ), where x,y are output and target
    :param output:
    :param target:
    :return:
    '''
    fidelity = 0
    unit_norm_diff = 0
    # for i in range(len(output)):
    #     inner_prod = torch.dot(output[i].T.conj(), target[i])
    #     fidelity += torch.abs(inner_prod) ** 2 / len(output)
    #     unit_norm_diff += (output[i].norm() - 1)**2
    batch_size = output.shape[0]
    dim = output.shape[1]
    inner_prods = torch.bmm(output.conj().view(batch_size,1,dim), target.view(batch_size,dim,1))
    #print(output, target, inner_prods)
    fidelity = inner_prods.abs()**2

    #U = list(model.parameters())[0].detach()
    #orth_loss = (torch.matmul(U, U.T.conj()) - torch.eye(U.shape[0])).abs().sum()
    return 1-fidelity.mean()

class StateCloudDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def match_state_clouds(cloud1,cloud2,batch_size=None):
    x = torch.tensor(cloud1, dtype=torch.cfloat, requires_grad=True)
    y = torch.tensor(cloud2, dtype=torch.cfloat)
    #print(x,y)

    model = UnitaryLinear(dim=cloud1.shape[-1])
    model.train()

    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=100)
    min_distance = 1
    NUM_EPOCHS = 500

    dataset = StateCloudDataset(x,y)
    if batch_size is None:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for i in range(NUM_EPOCHS):
        batch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = fidelity_loss(output, target)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()*data.size(0)
        batch_loss = batch_loss / len(train_loader.sampler)
        #print(batch_loss)
        if batch_loss < min_distance:
            min_distance = batch_loss

        # optimizer.zero_grad()
        # output = model(x)
        # loss = fidelity_loss(output, y)
        # if -loss > max_fidelity:
        #     max_fidelity = -loss.detach().numpy()
        # loss.backward()
        # #print(loss)
        # optimizer.step()

    #print(list(model.parameters()))
    #U = list(model.parameters())[0].detach()
    #orth_loss = (torch.matmul(U, U.T.conj()) - torch.eye(U.shape[0])).abs().sum()
    #print(orth_loss, U)
    return min_distance

if __name__ == '__main__':
    import numpy as np
    dummy_x = np.array([[0+1j, 0+0j],[0+0j, 1+0j]])
    dummy_y = np.array([[1+0j, 0+1j],[1+0j, 0-1j]]) / np.sqrt(2)
    print(match_state_clouds(dummy_x, dummy_y))