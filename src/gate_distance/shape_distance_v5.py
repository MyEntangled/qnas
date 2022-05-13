import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary, Operator

from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
import scipy

import sys
sys.path.append('/Users/erio/Dropbox/URP project/Code/PQC_composer/src')
from QuOTMANN.gate_info import SINGLE_QUBIT_DETERMINISTIC_GATES, SINGLE_QUBIT_VARIATIONAL_GATES, \
    TWO_QUBIT_DETERMINISTIC_GATES, TWO_QUBIT_VARIATIONAL_GATES, ADMISSIBLE_GATES, DIRECTED_GATES, UNITARY
import gate_positioning
import MUBs


def optimize_phases(X:np.ndarray, Y:np.ndarray):
    """
    Compute phases δ = (δ_1,...,δ_{N} such that ||X @ diag(exp(1j*δ)) - Y||^2 is minimized, which equals
    \sum_{i=1}^{N} (2 - 2*abs(⟨X_i|Y_i⟩))

        *** In practice N = d(d+1)T, where T is the number of theta's values.

    :param X: np.ndarray of size d x KT
    :param Y: np.ndarray of size d x KT
    :return: np.ndarray of size KT
    """
    assert X.shape == Y.shape and len(X.shape) == 2
    d,KT = X.shape

    U = np.sum(Y.conj() * X, axis=0)
    phases = -np.angle(U)

    #res = np.linalg.norm(X * np.exp(1j*phases)[:,None] - Y)**2
    #res = np.linalg.norm(X @ np.diag(np.exp(1j*phases)) - Y)**2
    res = np.linalg.norm(X * np.exp(1j*phases) - Y)**2

    exact = 2*KT - 2*np.sum(np.abs(U))
    assert abs(res-exact) < 1e-6, print('lalal', res, exact)

    ###
    return phases, res

def optimize_phases_right(X:np.ndarray, Y:np.ndarray):
    """
    Compute phases δ = (δ_1,...,δ_{N} such that ||X  - Y @ diag(exp(1j*δ))||^2 is minimized, which equals
    \sum_{i=1}^{N} (2 - 2*abs(⟨X_i|Y_i⟩))

        *** In practice N = d(d+1)T, where T is the number of theta's values.

    :param X: np.ndarray of size d x KT
    :param Y: np.ndarray of size d x KT
    :return: np.ndarray of size KT
    """
    assert X.shape == Y.shape and len(X.shape) == 2
    d,KT = X.shape

    U = np.sum(Y.conj() * X, axis=0)
    phases = np.angle(U)

    #res = np.linalg.norm(X * np.exp(1j*phases)[:,None] - Y)**2
    #res = np.linalg.norm(X @ np.diag(np.exp(1j*phases)) - Y)**2
    res = np.linalg.norm(X - Y * np.exp(1j*phases))**2

    exact = 2*KT - 2*np.sum(np.abs(U))
    assert abs(res-exact) < 1e-6, print('lalal', res, exact)

    ###
    return phases, res

def optimize_V(X:np.ndarray, Y:np.ndarray):
    """
    Find a unitary matrix V to minimize ||VX - Y||^2 (Complex orthogonal Procrustes problem)

    :param X: np.ndarray of size d x KT
    :param Y: np.ndarray of size d x KT
    :return: np.ndarray of size d x d
    """
    assert X.shape == Y.shape and len(X.shape) == 2
    d, KT = X.shape
    #print("before procrustes: ", np.linalg.norm(X-Y)**2)

    R, Sigma, W_dag = scipy.linalg.svd(X @ Y.conj().T)
    W = W_dag.conj().T

    V = W @ R.conj().T
    res = np.linalg.norm(V @ X - Y) ** 2
    ###
    #print("after procrustes: ", res)
    return V, res

def optimize_M(E:np.ndarray, S:np.ndarray):
    assert len(S.shape) == 2 and S.shape[1] == E.shape[1]
    d,KT = S.shape
    K,KT = E.shape
    T = KT // K
    #print('K,T,d', K,T,d)
    M = np.zeros((d,K), dtype=np.complex128)
    for k in range(K):
        all_yt = S[:,k*T:(k+1)*T]
        y = np.sum(all_yt, axis=1)
        #print(all_yt)
        #print(y)
        re_x = np.real(y)
        im_x = np.imag(y)
        x = re_x + 1j * im_x
        #print(f'at k={k}', x/ np.linalg.norm(x))
        M[:,k] = x / np.linalg.norm(x)
    #print(M.shape)
    #print(np.linalg.norm(M, axis=0))

    res = np.linalg.norm(M @ E - S) ** 2
    return M, res


def get_state_spectrum(num_qubits, V:str, qargs, thetas, anchor_states):
    '''
    Get V(theta)|anchor⟩ for various thetas and anchor states
    :param num_qubits:
    :param V:
    :param qargs:
    :param thetas:
    :param anchor_states:
    :return:
    '''
    assert V in ADMISSIBLE_GATES, f"V({V}) must belong to ADMISSIBLE_GATES({ADMISSIBLE_GATES})"

    output_states = np.zeros(shape=(len(anchor_states), len(thetas), 2 ** num_qubits), dtype=np.complex128)

    for i, anchor_state in enumerate(anchor_states):

        anchor = Statevector(anchor_state)  # initialize an anchor state

        for j, theta in enumerate(thetas):
            var_V_circ = QuantumCircuit(num_qubits)
            if V in SINGLE_QUBIT_DETERMINISTIC_GATES:  # one-qubit deterministic
                args = (*qargs,)
            elif V in SINGLE_QUBIT_VARIATIONAL_GATES:  # one-qubit variational
                args = (theta, *qargs)
            elif V in TWO_QUBIT_DETERMINISTIC_GATES:  # two-qubit deterministic
                args = (*qargs,)
            elif V in TWO_QUBIT_VARIATIONAL_GATES:  # two-qubit variational
                args = (theta, *qargs)

            getattr(var_V_circ, V)(*args)
            # print(var_V_circ.draw())
            output_states[i, j] = anchor.evolve(var_V_circ).data

    return np.array(output_states)

def multiply_U2(U2_thetas, spectrum, is_dagger):
    T = len(U2_thetas)

    output_states = spectrum.copy()

    for t in range(T):
        if is_dagger:
            U2_dag = U2_thetas[t].conj().T
            output_states[:,t] = (U2_dag @ output_states[:,t].T).T
        else:
            U2 = U2_thetas[t]
            output_states[:, t] = (U2 @ output_states[:, t].T).T

    return output_states

def optimization_routine(num_qubits, V:np.ndarray, M:np.ndarray, V2:str, qargs2,
                         thetas, spectrum_V1, tol=1e-5, max_turn=200):
    K,T,d = spectrum_V1.shape
    val_min = 1e9
    val_prev = 1e9
    tail_length = 0
    E = np.zeros((K,K*T), dtype=complex)
    for k in range(K):
        E[k, k*T : (k+1)*T] = 1.

    U2_thetas = []
    for t,theta in enumerate(thetas):
        var_V2_circ = QuantumCircuit(num_qubits)
        if V2 in SINGLE_QUBIT_DETERMINISTIC_GATES:  # one-qubit deterministic
            args = (*qargs2,)
        elif V2 in SINGLE_QUBIT_VARIATIONAL_GATES:  # one-qubit variational
            args = (theta, *qargs2)
        elif V2 in TWO_QUBIT_DETERMINISTIC_GATES:  # two-qubit deterministic
            args = (*qargs2,)
        elif V2 in TWO_QUBIT_VARIATIONAL_GATES:  # two-qubit variational
            args = (theta, *qargs2)

        getattr(var_V2_circ, V2)(*args)
        U2_thetas.append(Operator(var_V2_circ).data)
    U2_thetas = np.array(U2_thetas)

    L1 = V @ spectrum_V1.reshape(K * T, d).T
    L1 = multiply_U2(U2_thetas=U2_thetas,spectrum=L1.T.reshape(K,T,d), is_dagger=True).reshape(K*T,d).T
    L2 = M @ E

    turn = 0
    while True:
        turn += 1
        L1 = multiply_U2(U2_thetas=U2_thetas, spectrum=L1.T.reshape(K,T,d), is_dagger=False).reshape(K*T,d).T
        L2 = multiply_U2(U2_thetas=U2_thetas, spectrum=L2.T.reshape(K,T,d), is_dagger=False).reshape(K*T,d).T
        #print('pre alpha_V', np.linalg.norm(L1-L2)**2)
        phases, val = optimize_phases_right(X=L1, Y=L2)

        E = E * np.exp(1j * phases)
        L2 = L2 * np.exp(1j * phases)
        #print('phase1', val)

        # Update V
        V, val = optimize_V(L1, L2)
        L1 = V @ L1
        #print('V', val)

        ########################
        L1 = multiply_U2(U2_thetas=U2_thetas, spectrum=L1.T.reshape(K,T,d), is_dagger=True).reshape(K*T,d).T
        L2 = multiply_U2(U2_thetas=U2_thetas, spectrum=L2.T.reshape(K,T,d), is_dagger=True).reshape(K*T,d).T

        phases, val = optimize_phases_right(X=L1, Y=L2)
        E = E * np.exp(1j * phases)
        L2 = L2 * np.exp(1j * phases)
        #print('phase2', val)

        # Update M
        phases, val = optimize_phases_right(X=L1, Y=M@E)
        E = E * np.exp(1j * phases)
        L2 = L2 * np.exp(1j * phases)
        #print('pre M', val)

        M, val = optimize_M(E=E, S=L1)
        #print('M', val)
        L2 = M @ E
        phases, val = optimize_phases_right(X=L1, Y=L2)
        # L1 = L1 * np.exp(1j * phases)
        E = E * np.exp(1j * phases)
        L2 = M @ E
        #print('post M', val)

        if val < val_min:
            val_min = val
        if turn == max_turn:
            opt_sum_fid = (2 * K * T - val_min) / 2.
            return opt_sum_fid

        #print('------')
        if abs(val-val_prev) < tol:
            tail_length += 1
            val_prev = val
            if tail_length == 3:
                opt_sum_fid = (2*K*T - val) / 2.
                return opt_sum_fid
        else:
            val_prev = val
            tail_length = 0


def _shape_distance_with_config(num_qubits, V1, V2, qargs1, qargs2, num_theta_samples=4, num_trials=500):
    '''
    Return the shape distance between two quantum gates
    :param V1:
    :param V2:
    :return:
    '''

    assert V1 in ADMISSIBLE_GATES and V2 in ADMISSIBLE_GATES, "Input gates are not admissible."
    d = 2**num_qubits
    K = d*(d+1)
    T = num_theta_samples

    # Get the list of d(d+1) anchor states
    anchor_states = MUBs.get_anchor_states(num_qubits)

    # Rotation angles
    lo_bound = -np.pi
    up_bound = np.pi
    thetas = np.linspace(lo_bound, up_bound, num_theta_samples, endpoint=False)

    # Generate necessary clusters
    spectrum_V1 = get_state_spectrum(num_qubits, V1, qargs1, thetas, anchor_states)
    #spectrum_V2M = np.zeros_like(spectrum_V1) # M-dependent
    #spectrum_mix = np.zeros_like(spectrum_V1) # V-dependent


    opt_val_list = []
    for i in range(num_trials):
        #print(f'Trial {i+1}: ')

        # random initial M
        rand_r = np.random.rand(d, K).astype(complex)
        rand_phase = np.random.uniform(0, 2 * np.pi, (d, K)).astype(complex)
        M = rand_r * np.exp(1j * rand_phase)
        M = M / np.linalg.norm(M, axis=0)

        # random initial V
        V = random_unitary(d).data

        opt_val = optimization_routine(num_qubits=num_qubits, V=V, M=M, V2=V2, qargs2=qargs2,
                                       thetas=thetas, spectrum_V1=spectrum_V1, tol=1e-3, max_turn=500)
        opt_val_list.append(opt_val)

        if np.isclose(opt_val, K*T):
            break

    opt_val_list = np.array(opt_val_list)

    print(f'best = {opt_val_list.max()}, mean = {opt_val_list.mean()}, sample variance = {opt_val_list.var()}')

    shape_distance = 1. - opt_val_list.max() / (K*T)

    if shape_distance < 10e-5:
        return 0
    else:
        return shape_distance

def compute_shape_distance(V1: str, V2: str, num_qubits: int, num_theta_samples:int, num_trials:int=500) -> dict:
    all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    all_distances = {}

    for pos_tag, qargs in all_positions.items():
        if (V1 in SINGLE_QUBIT_VARIATIONAL_GATES or V1 in TWO_QUBIT_VARIATIONAL_GATES) and \
            (V2 in SINGLE_QUBIT_VARIATIONAL_GATES or V2 in TWO_QUBIT_VARIATIONAL_GATES):

            all_distances[pos_tag] = _shape_distance_with_config(num_qubits=num_qubits, V1=V1, V2=V2, num_theta_samples=num_theta_samples,
                                                             qargs1=qargs[0], qargs2=qargs[1], num_trials=num_trials)
        elif (V1 in SINGLE_QUBIT_DETERMINISTIC_GATES or V1 in TWO_QUBIT_DETERMINISTIC_GATES) and \
            (V2 in SINGLE_QUBIT_DETERMINISTIC_GATES or V2 in TWO_QUBIT_DETERMINISTIC_GATES):

            all_distances[pos_tag] = 0

        else:
            all_distances[pos_tag] = np.inf

    return all_distances

def compute_shape_distance_no_qubit_order(V1: str, V2: str, num_qubits: int, num_theta_samples:int, num_trials:int=10) -> dict:
    all_positions = gate_positioning.all_relative_positions(V1=V1, V2=V2, num_qubits=num_qubits)
    all_distances = {}

    qargs = list(all_positions.values())[0]
    print(V1, V2)
    if (V1 in SINGLE_QUBIT_VARIATIONAL_GATES or V1 in TWO_QUBIT_VARIATIONAL_GATES) and \
            (V2 in SINGLE_QUBIT_VARIATIONAL_GATES or V2 in TWO_QUBIT_VARIATIONAL_GATES):
        dist = _shape_distance_with_config(num_qubits=num_qubits, V1=V1, V2=V2,
                                                             num_theta_samples=num_theta_samples,
                                                             qargs1=qargs[0], qargs2=qargs[1], num_trials=num_trials)
    elif (V1 in SINGLE_QUBIT_DETERMINISTIC_GATES or V1 in TWO_QUBIT_DETERMINISTIC_GATES) and \
            (V2 in SINGLE_QUBIT_DETERMINISTIC_GATES or V2 in TWO_QUBIT_DETERMINISTIC_GATES):
        dist = 0

    else:
        dist = np.inf


    for pos_tag, qargs in all_positions.items():
            all_distances[pos_tag] = dist

    return all_distances

# def refine_shape_dist_dict(max_qubits, all_shape_distances):
#     for pairname, dist in all_shape_distances.items():
#         if dist == np.inf:
#             all_shape_distances[pairname] = 10e4
#
#     r_cr = [1.]*max_qubits
#     r_rr = [1.]*max_qubits
#     cr_rr = [1.]*max_qubits
#
#     for pairname, dist in all_shape_distances.items():
#         V1,V2,num_qubits,pos_tag = pairname.split('_')
#         num_qubits = int(num_qubits)
#         all_positioning = gate_positioning.all_relative_positions(V1=V1,V2=V2,num_qubits=num_qubits)
#
#         if pairname in all_positioning:
#             qargs = all_positioning[pairname]
#             rev_qargs = [qargs[1], qargs[0]]
#
#             rev_pos = gate_positioning.get_pos_from_gate_name(V1=V2,V2=V1,qargs=rev_qargs)
#
#             rev_pairname = '_'.join([V2,V1,str(num_qubits),rev_pos])
#
#             rev_dist = all_shape_distances[rev_pairname]
#             #print(pairname, rev_pairname, qargs, rev_qargs, dist, rev_dist)
#             #if not np.isclose(dist, rev_dist):
#             if ADMISSIBLE_GATES.index(V1) <= ADMISSIBLE_GATES.index(V2) and abs(dist - rev_dist) >= 0.001:
#                 print(pairname, rev_pairname, qargs, rev_qargs, dist, rev_dist)
#
#             #all_shape_distances[pairname] = min(all_shape_distances[pairname], all_shape_distances[rev_pairname])
#
#             if V1 in ['rx','ry','rz']:
#                 if V2 in ['crx','cry','crz']:
#                     r_cr[num_qubits-1] = min(r_cr[num_qubits-1] , all_shape_distances[pairname])
#
#                 elif V2 in ['rxx','ryy','rzz']:
#                     r_rr[num_qubits-1]  = min(r_rr[num_qubits-1] , all_shape_distances[pairname])
#
#             elif V1 in ['crx','cry','crz']:
#                 if V2 in ['rx','ry','rz']:
#                     r_cr[num_qubits-1]  = min(r_cr[num_qubits-1] , all_shape_distances[pairname])
#
#                 elif V2 in ['rxx','ryy','rzz']:
#                     cr_rr[num_qubits-1]  = min(cr_rr[num_qubits-1] , all_shape_distances[pairname])
#
#             elif V1 in ['rxx', 'ryy', 'rzz']:
#                 if V2 in ['rx','ry','rz']:
#                     r_rr[num_qubits-1]  = min(r_rr[num_qubits-1] , all_shape_distances[pairname])
#                 elif V2 in ['crx','cry','crz']:
#                     cr_rr[num_qubits-1]  = min(cr_rr[num_qubits-1] , all_shape_distances[pairname])
#     print(r_rr, r_cr, cr_rr)
#     return all_shape_distances


import sys
import pickle
sys.path.append('/Users/erio/Dropbox/URP project/Code/PQC_composer')
np.random.seed(20)

num_theta_samples = 12

#print(compute_shape_distance('rz', 'crz', num_qubits=4, num_theta_samples=12, num_trials=10))

filename = './' + str(num_theta_samples) + 'newnew_theta_raw_all_shape_distances.pkl'


# ALL_SHAPE_DISTANCES = {}
# with open(filename, 'rb') as f:
#     ALL_SHAPE_DISTANCES = pickle.load(f)
#
# for q in [1,2,3,4]:
#     for i,V1 in enumerate(ADMISSIBLE_GATES):
#         for j,V2 in enumerate(ADMISSIBLE_GATES):
#             if q == 1:
#                 if V1 in TWO_QUBIT_DETERMINISTIC_GATES or V1 in TWO_QUBIT_VARIATIONAL_GATES \
#                     or V2 in TWO_QUBIT_DETERMINISTIC_GATES or V2 in TWO_QUBIT_VARIATIONAL_GATES:
#                         continue ## not enough qubit
#
#             shape_distance = compute_shape_distance_no_qubit_order(V1,V2, num_qubits=q,
#                                                                    num_theta_samples=num_theta_samples,
#                                                                    num_trials=20)
#
#             ALL_SHAPE_DISTANCES.update(shape_distance)
# with open(filename, 'wb') as f:
#     pickle.dump(ALL_SHAPE_DISTANCES, f)


with open(filename, 'rb') as f:
    ALL_SHAPE_DISTANCES = pickle.load(f)



