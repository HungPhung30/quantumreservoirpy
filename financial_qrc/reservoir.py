import numpy as np
from quantumreservoirpy.reservoirs import Static
from qiskit.quantum_info import random_unitary

encoder = {
    0: '0',
    1: '1',
}

class FinancialQRC(Static):
    """
    Quantum Reservoir for financial binary time-series.
    Uses a larger random unitary to produce richer reservoir states.
    """

    def __init__(self, n_qubits, memory=np.inf, backend=None):
        super().__init__(n_qubits, memory, backend)
        self.operator = random_unitary(2 ** n_qubits)

    def before(self, circuit):
        circuit.h(circuit.qubits)

    def during(self, circuit, timestep, reservoirnumber):
        circuit.measure(0)
        circuit.initialize(encoder[timestep], 0)
        circuit.append(self.operator, circuit.qubits)