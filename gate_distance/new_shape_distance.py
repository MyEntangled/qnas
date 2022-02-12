import MUBs
import fubini_distance
import gate_positioning

def compute_shape_distance(V1,V2, num_qubits):
    all_positions = gate_positioning.all_relative_positions(V1=V1,V2=V2,num_qubits=num_qubits)

