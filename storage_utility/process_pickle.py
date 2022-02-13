import pickle
from utility.data_encoding import FeatureMap
from utility.quantum_nn import QuantumNeuralNetwork
from utility.ansatz_template import AnsatzTemplate

def save_qnn(model, filename):
    with open(filename, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load_qnn(filename):
    with open(filename, 'rb') as input:
        model = pickle.load(input)
    return model



if __name__ == '__main__':
    feature_map = FeatureMap('PauliFeatureMap', feature_dim=4, reps=5)

    template = AnsatzTemplate()
    template.construct_simple_template(num_qubits=4, num_layers=2)

    model = QuantumNeuralNetwork(feature_map, template, platform='Qiskit')
    #model.visualize()

    print(model.num_qubits, model.input_dim, model.param_dim)

    #############
    filename = '../retired/storage/test_save_pickle.pkl'

    save_qnn(model, filename)

    model_reload = load_qnn(filename)
    model_reload.visualize()


