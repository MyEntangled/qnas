import galois
import numpy as np
import itertools

import pickle
import os

'''
Construction of the MUBs using Galois field GF(p^n) (p is odd prime) and Galois ring GR(2^2,n) 
References: https://arxiv.org/pdf/quant-ph/0309120.pdf,
https://reader.elsevier.com/reader/sd/pii/B978178548235950004X?token=5C6F8E22F480889694B06062D87EF8861B4989EA5DC9638FCB0A3566F61A7E819DF55B2438850A5DD787926F0EDC4915&originRegion=us-east-1&originCreation=20220105064911,
https://uwaterloo.ca/combinatorics-and-optimization/sites/ca.combinatorics-and-optimization/files/uploads/files/mohammad-d.pdf
'''

def generate_MUBs(p: int, n: int):
    '''
    Compute a set of mutually unbiased bases for dimension q = p^n.
    :param p: a prime number
    :param n: power of p
    :return:
    '''
    if p % 2 == 1:
        return _odd_char_MUBs(p,n)
    else:
        return _even_char_MUBs(2,n)

def _odd_char_MUBs(p: int, n: int):
    assert p % 2 == 1, "p should be odd for odd characteristic Galois ring."

    q = p**n
    omega = np.exp(1j * 2*np.pi / p)
    bases = []

    standard_basis = np.hsplit(np.eye(q), q)
    standard_basis = [vector.squeeze() for vector in standard_basis]
    bases.append(standard_basis)

    GF = galois.GF(q)
    x = GF.Elements()

    for a in range(q):
        basis = []
        for b in range(q):
            y = a*(x**2) + b*x
            if n > 1:
                basis.append(1./np.sqrt(q) * omega ** np.array(y.field_trace(), dtype=float))
            elif n == 1:
                basis.append(1./np.sqrt(q) * omega ** np.array(y, dtype=float))

        bases.append(basis)
    return np.around(np.array(bases), decimals=6)

def _even_char_MUBs(p: int, n: int):
    assert p == 2, "p must be 2 for even characteristic."

    f = galois.conway_poly(characteristic=2, degree=n) ## generating function
    #print(f)
    f = np.array(f.coeffs, dtype=int)

    ## 4^n Elements of Galois ring GR(4,n) written as i_{n-1}*\xi^n + ... + i_{0}*\xi^0, or [i_{n-1], i_{n-2}, ..., i_{0}]
    ## where \xi is the primitive root of f
    GR_elements = np.array(list(map(list, itertools.product([0, 1, 2, 3], repeat=n))))
    #print(GR_elements)

    for e in GR_elements[1:]:
        order = _compute_order(e,f)
        if order == 2**n-1:
            xi = e
            break


    zero = [0]*(n)
    one = [0]*(n-1) + [1]
    #xi = [0]*(n-2) + [1] + [0]
    #xi = list(f)
    #xi[-2] = (xi[-2] + 1) % 4

    #print('xi', xi)
    teichmuller_set = []
    teichmuller_set.append(zero)

    for i in range(2**n - 1):
        #np.polynomial.polynomial.polypow([], i)
        if i == 0:
            xi_to_the_i = one
        else:
            xi_to_the_i = _polymul_modulo(teichmuller_set[-1], xi, f)
        teichmuller_set.append(xi_to_the_i)

    teichmuller_set = np.array(teichmuller_set, dtype=int)
    #print('teimuller', teichmuller_set)
    q = 2**n
    coeff = 1. / np.sqrt(q)
    bases = []

    standard_basis = np.hsplit(np.eye(q, dtype=complex), q)
    standard_basis = [vector.squeeze() for vector in standard_basis]
    #print(standard_basis)
    bases.append(standard_basis)

    for a in teichmuller_set:
        basis = []
        for b in teichmuller_set:
            basis_vector = []
            for x in teichmuller_set:
                y = _polymul_modulo(a+2*b, x, f)

                omega = 2*1j*np.pi/4
                vec = coeff * np.exp(omega * _GR_trace(y,f,teichmuller_set))
                basis_vector.append(vec)
            basis.append(basis_vector)
        bases.append(basis)
    return np.array(bases)

def _compute_order(e ,f):
    #print('compute order of', e)
    power = 0
    e_to_power = [0]*len(e)
    e_to_power[-1] = 1
    while True:
        e_to_power = _polymul_modulo(e_to_power, e, f)
        power += 1
        #print(e_to_power)
        if all([term == 0 for term in e_to_power[:-1]]):
            if e_to_power[-1] == 1:
                return power
            if e_to_power[-1] == 0:
                return 'inf'


def _polymul_modulo(u,v,f):
    '''
    Compute u * v (mod f), where u,v,f are polynomials
    :param u:
    :param v:
    :param f:
    :return:
    '''
    q, r = np.polydiv(np.polymul(u,v), f)
    ## update
    r = [ri % 4 for ri in r]
    #print('r_bef',r)
    r = np.pad(r, (len(f)-len(r)-1,0), 'constant').astype(int)
    #print('r_aft',r)
    return r

def _are_polys_equal(u,v,field_cardinal):
    for entry in u-v:
        if entry % field_cardinal != 0:
            return False
    #print(u-v, 'True')
    return True

def _frobenius_isomorphism(e, f, teichmuller_set, degree=1):
    '''
    Compute the Frobenius map σ:GR(4,n) --> Z_4 that σ(e) = a^2 + 2b^2,
    since any element e in GR(4,n) can be uniquely written as e=a+2b, where a,b belong to the Teichmuller set
    :param e: element of GR(4,n), given as a list [[i_{n-1], i_{n-2}, ..., i_{0}]]
    :return:
    '''

    def _two_adic_repr(e, teichmuller_set):
        #print('e', e)
        for i in range(0,len(teichmuller_set)):
            for j in range(0,len(teichmuller_set)):
                # if (e == teichmuller_set[i] + 2*teichmuller_set[j]).all():
                if _are_polys_equal(e, teichmuller_set[i] + 2 * teichmuller_set[j], field_cardinal=4):
                    #print(i,j, teichmuller_set[i], 2*teichmuller_set[j])
                    return (i,j)


    ## e = teichmuller_set[i] + 2*teichmuller_set[j]
    #print('teichmuller', teichmuller_set)
    i,j = _two_adic_repr(e, teichmuller_set)
    a = teichmuller_set[i]
    b = teichmuller_set[j]
    #print(i,j, e - (a+2*b))

    for i in range(degree):
        a = _polymul_modulo(a,a,f)
        b = _polymul_modulo(b,b,f)
    #print('fro of deg ', degree, 'is: ', a,b)
    return a+2*b

def _GR_trace(e, f, teichmuller_set):
    '''
    Compute the trace map Tr: GR(4,n) --> Z_4 sigven by Tr(e) = \sum_{k=0}^{n-1} σ^k(e),
    where σ is the Frobenius map
    :param e:
    :param f:
    :param teichmuller_set:
    :return:
    '''
    e = np.array(e, dtype=int)
    f = np.array(f, dtype=int)
    teichmuller_set = np.array(teichmuller_set, dtype=int)
    n = teichmuller_set.shape[1]

    trace = np.zeros(shape=(n))
    for k in range(n):
        trace += _frobenius_isomorphism(e,f,teichmuller_set,degree=k)
    #print('trace', trace)
    return trace[-1]

def generate_anchor_states(num_qubit_list):
    '''
    Generate all states in MUBs as anchor states
    :param num_qubits:
    :return:
    '''
    ALL_MUBs = {}
    for n in num_qubit_list:
        ALL_MUBs[n] = generate_MUBs(2,n).reshape(-1,2**n)
    #print(len(ALL_MUBs))
    #print(ALL_MUBs)

    with open('all_mubs.pkl', 'wb') as f:
        pickle.dump(ALL_MUBs, f)

    with open('all_mubs.pkl', 'rb') as f:
        loaded_MUBs = pickle.load(f)
    return

def get_anchor_states(num_qubits):
    CURRENT_DIR = os.path.dirname(__file__)
    file_path = os.path.join(CURRENT_DIR, 'all_mubs.pkl')
    with open(file_path, 'rb') as f:
        loaded_MUBs = pickle.load(f)
    if num_qubits in loaded_MUBs:
        return loaded_MUBs[num_qubits]
    else:
        return generate_MUBs(2,num_qubits)



if __name__ == '__main__':
    # n = 4
    # MUBs = _even_char_MUBs(2,n)
    # MUBs = MUBs.reshape(-1,2**n)
    # print(MUBs.shape)
    # d = MUBs.shape[1]
    # bases = np.split(MUBs, d + 1)
    # for m, basis1 in enumerate(bases):
    #     # check intra basis
    #     for i, vec1 in enumerate(basis1):
    #         for j, vec2 in enumerate(basis1):
    #             f = abs(vec1.conj().T @ vec2) ** 2
    #             # print(f)
    #             assert np.isclose(f, 0.) or np.isclose(f, 1.)
    #
    #     # check inter basis
    #     for n, basis2 in enumerate(bases):
    #         if n > m:  # different basis
    #             for i, vec1 in enumerate(basis1):
    #                 for j, vec2 in enumerate(basis2):
    #                     f = abs(vec1.conj().T @ vec2) ** 2
    #                     #print(f)
    #                     assert np.isclose(f, 1. / d)
    # print(MUBs)

    generate_anchor_states([1,2,3,4,5])
    print(get_anchor_states(1))
    print(get_anchor_states(2))


