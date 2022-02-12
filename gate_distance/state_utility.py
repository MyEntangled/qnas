import numpy as np

def reformat_statevector(vec):
    idx = next(i for i,ampl in enumerate(vec) if abs(ampl) > 10e-9)
    unit_core = vec[idx] / abs(vec[idx]) # the exp(i*theta) part of the first nonzero element
    state = vec * unit_core.conj() # cancel the core in the first nonzero element
    return state

def reformat_statedata(arr_of_states):
    '''
    Reformat an array/tensor of states by global phase. Each state should be read along the last axis.
    :param arr_of_states:
    :return:
    '''
    shape = arr_of_states.shape
    formatted_states = arr_of_states.reshape(-1,shape[-1])
    formatted_states = np.apply_along_axis(reformat_statevector, 1, formatted_states)
    formatted_states = formatted_states.reshape(shape)
    return formatted_states


def state_to_cartesian(state):
    '''
    Get the cartesian representation (on a Bloch sphere) for one-qubit state
    :param state:
    :return:
    '''
    assert len(state) == 2, "Only one-qubit states can be converted to Cartesian coordinate."
    ket = state
    bra = ket.conj()
    rho = np.outer(ket, bra)

    z = (2 * rho[0, 0] - 1)
    x = rho[0, 1] + rho[1, 0]
    y = (rho[1, 0] - rho[0, 1]) / 1j
    return [x.real, y.real, z.real]

if __name__ == '__main__':

    print("TEST REFORMAT_STATEVECTOR()")
    dim = 4
    z = np.random.rand(dim) + np.random.rand(dim) * 1j
    z[0] = 0
    z = z / np.linalg.norm(z)
    print(z)
    new_z = reformat_statevector(z)
    print(new_z, np.linalg.norm(new_z))

    print("TEST REFORMAT_STATEDATA()")
    dim = 4
    r = 6
    c = 200
    lst_z = np.random.rand(r,c,dim) + np.random.rand(r,c,dim)*1j
    normalizer = lambda vec: vec/np.linalg.norm(vec)
    lst_z = np.apply_along_axis(normalizer, 2, lst_z)
    print(reformat_statedata(lst_z))