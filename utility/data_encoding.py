from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit import QuantumCircuit

FM_DICT = {
    'PauliFeatureMap': [PauliFeatureMap, 'n'],
    'ZFeatureMap': [ZFeatureMap, 'n'],
    'ZZFeatureMap': [ZZFeatureMap, 'n'],
    'AmplitudeEmbedding': [RawFeatureVector, '2**n']
 }

class FeatureMap():

    def __init__(self, fm_type, feature_dim, reps=1, data_map_func=None):

        assert fm_type in FM_DICT, "Feature map is not defined."

        if fm_type == 'AmplitudeEmbedding':
            self.circ = FM_DICT[fm_type][0](feature_dim)
        else:
            self.circ = FM_DICT[fm_type][0](feature_dim, reps=reps, data_map_func=data_map_func).decompose() # feature_dim typically equal num_qubits
        self.num_qubits = self.circ.num_qubits

        if FM_DICT[fm_type][1] == 'n':
            self.input_dim = self.num_qubits
        elif FM_DICT[fm_type][1] == '2**n':
            self.input_dim = 2 ** self.num_qubits

    def visualize(self,output=None):
        print(self.circ.draw(output))

if __name__ == '__main__':
    feature_map = FeatureMap('PauliFeatureMap', 4, 1)
    feature_map.visualize()

    ######
    feature_map = FeatureMap('AmplitudeEmbedding', 8)
    feature_map.visualize()