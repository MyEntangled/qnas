import torch
from torch.nn.functional import softmax
from botorch.optim.initializers import initialize_q_batch_nonneg, initialize_q_batch

import gpytorch

from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, \
    SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, \
    TWO_QUBIT_VARIATIONAL_GATES, \
    ADMISSIBLE_GATES, \
    DIRECTED_GATES, \
    UNITARY, \
    OP_NODE_DICT

OP_VALUES_TORCH = torch.tensor(list(OP_NODE_DICT.values()))
GATE_SELECT_PROB = torch.tensor([1]*(len(SINGLE_QUBIT_DETERMINISTIC_GATES) + len(TWO_QUBIT_DETERMINISTIC_GATES)) + [10]*(len(SINGLE_QUBIT_VARIATIONAL_GATES) + len(TWO_QUBIT_VARIATIONAL_GATES))).to(dtype=float)

def warm_init(acq_func, bounds, encoding_length, batch_size, raw_samples):
    Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(raw_samples * batch_size, 1, encoding_length).to(bounds)
    Yraw = acq_func(Xraw)  # evaluate the acquisition function on these q-batches

    # apply the heuristic for sampling promising initial conditions
    X = initialize_q_batch(Xraw, Yraw, 5)
    return X.to(bounds)


def EA_optimize(acq_func, X0, num_qubits, num_gates, num_iters, num_offsprings, k=5, num_candidates=1):
    X = X0.squeeze(1).view(-1, num_qubits+1, num_gates)

    for iter in range(num_iters):

        n = X.shape[0]
        X = torch.tile(X, [num_offsprings + 1, 1, 1]).to(X0) ## Total of (num_offsprings+1)*n clones of X
        num_mut_probs = torch.tensor([0.4, 0.3, 0.2, 0.1])

        num_mut_for_offsprings = torch.multinomial(num_mut_probs, n*num_offsprings, replacement=True) # maximum 4 mutations
        num_mut_for_offsprings = torch.cat((num_mut_for_offsprings, -torch.ones(n))).to(X)

        for i in torch.arange(len(num_mut_probs)):
            to_be_mutated = (num_mut_for_offsprings >= i)
            num_offs = torch.sum(to_be_mutated)

            # gate mutation vs wire mutation
            #mut_type = torch.randint(0, 2, (num_offs,))
            mut_type_slice = torch.randint(0, 2, (num_offs,)).to(X)
            mut_type = -torch.ones(len(num_mut_for_offsprings)).to(X)
            mut_type[to_be_mutated] = mut_type_slice

            num_gate_mut = torch.sum(mut_type == 0.)
            num_wire_mut = torch.sum(mut_type == 1.)

            # mutate gate
            if num_gate_mut > 0:
                to_change_gate_idx = torch.randint(0,num_gates,(num_gate_mut,))
                new_gate_idx = torch.multinomial(GATE_SELECT_PROB, num_gate_mut, replacement=True)
                new_gate_code = OP_VALUES_TORCH[new_gate_idx].to(X)
                #print('before gate update', X[to_be_mutated & (mut_type == 0), -1, to_change_gate_idx])
                X[to_be_mutated & (mut_type == 0), -1, to_change_gate_idx] = new_gate_code


                #print('after gate update', X[to_be_mutated & (mut_type == 0), -1, to_change_gate_idx])

            if num_wire_mut > 0:
                # permute wire (by permuting each column of X[:-1,:] independently
                temp = X[to_be_mutated & (mut_type == 1), :-1, :]
                indices = torch.argsort(torch.rand(*temp.shape), dim=1)
                X[to_be_mutated & (mut_type == 1), :-1, :] = temp[torch.arange(temp.shape[0]).view(-1, 1, 1), indices, torch.arange(temp.shape[2]).view(1, 1, -1)]

                # print('before wire update', temp.shape, temp[0])
                # print('after wire update', X[to_be_mutated & (mut_type == 1), :-1, :][0])
                # print('indices', indices[0])

        with gpytorch.settings.cholesky_jitter(1e-1, 1e-1, 1e-1):
            if iter == 0:
                Y = acq_func(X.view(X.shape[0],1,(num_qubits+1)*num_gates))
            else:
                Y_new = acq_func(X[:-n].view(X.shape[0]-n, 1, (num_qubits+1)*num_gates))
                Y = torch.cat((Y_new, Y))

        if iter < num_iters - 1:
            _, top_idx = torch.topk(Y,k=k-k//2)
            X2 = X[top_idx]
            Y2 = Y[top_idx]

            not_top_idx = ~torch.isin(torch.arange(len(Y)).to(top_idx), top_idx)
            prob = softmax(Y[not_top_idx], dim=0)
            good_idx = torch.multinomial(prob, num_samples=k//2, replacement=False)
            X1 = X[good_idx]
            Y1 = Y[good_idx]

            X = torch.cat((X1,X2))
            Y = torch.cat((Y1,Y2))

            #print(f'iter {iter}:', Y)
        else:
            _, top_idx = torch.topk(Y,k=num_candidates)
            X = X[top_idx]
            Y = Y[top_idx]
            #print(f'iter {iter}:', Y)

    X = X.view(num_candidates, (num_qubits+1)*num_gates)
    return X,Y

