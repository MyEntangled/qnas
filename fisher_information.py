import pennylane as qml
import numpy as np
import time
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

np.random.seed(0)

def PQC_function(wires, layers, param):

    for wire in range(wires):
        qml.Hadamard(wires = wire)

    param = iter(param)

    for _ in range(layers):
        for wire in range(wires):
            qml.RX(next(param), wires=wire)
            qml.RY(next(param), wires=wire)

        for wire in range(wires-1):
            qml.CNOT(wires=[wire,wire+1])

    for wire in range(wires):
        qml.RX(next(param), wires=wire)
        qml.RY(next(param), wires=wire)


def find_output_state(template, dev, wires, layers, *args, **kwargs):
    @qml.qnode(dev)
    def circuit(wires, layers, *args, **kwargs):
        template(wires, layers, *args, **kwargs)
        return qml.state()
    return circuit(wires, layers, *args, **kwargs)


def compute_fisher_matrix(template, wires, param, layers=1):
    dev = qml.device('default.qubit', wires=wires)

    dim = 2**wires
    param_dim = len(param)

    fisher_matrix = np.zeros((param_dim, param_dim))

    state = find_output_state(template, dev, wires, layers, param)
    bra_state = state.conj().T

    for i in range(param_dim):
        for j in range(param_dim):
            basis_i = np.zeros(param_dim)
            basis_i[i] = 1.
            basis_j = np.zeros(param_dim)
            basis_j[j] = 1.

            param_pp = param + (np.pi / 2) * (basis_i + basis_j)
            param_pm = param + (np.pi / 2) * (basis_i - basis_j)
            param_mp = param + (np.pi / 2) * (-basis_i + basis_j)
            param_mm = param + (np.pi / 2) * (-basis_i - basis_j)

            state_pp = find_output_state(template, dev, wires, layers, param_pp)
            state_pm = find_output_state(template, dev, wires, layers, param_pm)
            state_mp = find_output_state(template, dev, wires, layers, param_mp)
            state_mm = find_output_state(template, dev, wires, layers, param_mm)

            fisher_matrix[i,j] =  abs(bra_state @ state_pp)**2 \
                                    - abs(bra_state @ state_pm)**2 \
                                    - abs(bra_state @ state_mp)**2 \
                                    + abs(bra_state @ state_mm)**2

    return (-1./2) * fisher_matrix


def compute_fisher_matrix_spsa(template, wires, param, shots, layers=1, error_log=False):
    '''
    Compute the quantum Fisher information matrix 
    using SPSA. If error_log, return also Frobenius norm 
    difference between true QFI and approximated QFI.
    '''

    dev = qml.device('default.qubit', wires=wires)

    # parameter dimension
    param_dim = len(param)

    # output state
    state       = find_output_state(template, dev, wires, layers, param)
    bra_state   = state.conj().T

    if error_log:
        fisher_array = [] # store error log
        shot_list = [] # shots at which error is stored
        # true Fisher matrix
        fisher_info = compute_fisher_matrix(template, wires, param, layers)

    fisher_sum = np.zeros((param_dim, param_dim))

    eps = .001 # very small number

    for shot in range(shots):
        
        # random pertubations
        del1, del2  = np.random.choice(a=[1,-1], size=(2,param_dim), p=[.5,.5])
        combined    = np.outer(del1, del2) + np.outer(del2, del1)

        # small pertubation of output state
        state_pp    = find_output_state(template, dev, wires, layers, param+eps*del1+eps*del2)
        state_p     = find_output_state(template, dev, wires, layers, param+eps*del1)
        state_mp    = find_output_state(template, dev, wires, layers, param-eps*del1+eps*del2)
        state_m     = find_output_state(template, dev, wires, layers, param-eps*del1)

        # overlap
        dF =  abs(bra_state @ state_pp)**2  \
            - abs(bra_state @ state_p)**2   \
            - abs(bra_state @ state_mp)**2  \
            + abs(bra_state @ state_m)**2

        fisher_sum += -dF*combined
        
        if error_log:
            fisher_info_spsa = fisher_sum / ((shot+1) * (2*eps**2))
            fisher_array.append(np.linalg.norm(fisher_info_spsa-fisher_info))

    if error_log:
        return fisher_info_spsa, fisher_array

    return fisher_sum/(shots * 2 * eps**2)

def effective_dimension_naive(fisher_info, z=0):
    if z == 0:
        z = 10e-9

    f = lambda x: x/(x+z)
    f = np.vectorize(f)

    eigvals = np.linalg.eigvals(fisher_info)
    print(np.sort(eigvals))
    eff_dim = sum(f(eigvals))
    return eff_dim












# convergence visualization

'''
wires = 2
param_dim = wires * 4
param = np.random.uniform(0, 2 * np.pi, param_dim)

fig = plt.figure(figsize=(10,8))

plt.xlabel('Number of shots')
plt.ylabel('Frobenius norm difference between true QFI and QFI SPSA')

err = 0
for trial in range(10):
    fisher_array    = compute_fisher_matrix_spsa(PQC_function, 
        wires, param, shots=300, error_log=True)[1]
    plt.plot(range(1,301), fisher_array, color='r', alpha=.3)
    err += fisher_array[-1]
    print(f'trial {trial+1} done!')

plt.title(f'{wires} wire(s) with 300 shot approximation, mean error {round(err/10,5)}')

plt.savefig('convergence.jpg')
'''

# spectral analysis
#rank = np.linalg.matrix_rank(fisher_info)
#eigvals = np.linalg.eigvals(fisher_info)
#print(rank)

if __name__ == '__main__':
    wires = 2
    n_layers = 1
    param_dim = n_layers * wires * 4
    param = np.random.uniform(0, 2 * np.pi, param_dim)
    fisher_info = compute_fisher_matrix(PQC_function, wires, param, n_layers)
    fisher_info_spsa = compute_fisher_matrix_spsa(PQC_function, wires, param, 2000, n_layers, error_log=False)
    print(fisher_info.shape, fisher_info_spsa.shape)
    print(effective_dimension_naive(fisher_info, 0), effective_dimension_naive(fisher_info_spsa, 0))