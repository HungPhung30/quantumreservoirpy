import numpy as np
from quantumreservoirpy.reservoirs import Static
from qiskit.quantum_info import random_unitary

# Encoder maps 0->|0>, 1->|1>
encoder = {
    0: '0',
    1: '1',
}

class RandomUnitary(Static):
    def __init__(self, n_qubits, memory=np.inf, backend=None):
        super().__init__(n_qubits, memory, backend)
        self.operator = random_unitary(2 ** n_qubits)

    def before(self, circuit):
        circuit.h(circuit.qubits)

    def during(self, circuit, timestep, reservoirnumber):
        circuit.measure(0)                              # measure qubit 0
        circuit.initialize(encoder[timestep], 0)       # re-encode input
        circuit.append(self.operator, circuit.qubits)  # apply random unitary

# Binary target sequence: alternating 0s and 1s
n_qubits = 4
shots    = 10000
target   = [1, 0] * 30   # 60 timesteps of alternating 1, 0

res  = RandomUnitary(n_qubits=n_qubits)
mean = res.run(target, shots=shots)

print("Reservoir state shape:", mean.shape)  # should be (60, 1)