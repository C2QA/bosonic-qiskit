import c2qa
import qiskit
import numpy as np

### Initialize the oscillators to zero (spin 1) and the qubit to a superposition
# Two modes and 1 qubit
qmr = c2qa.QumodeRegister(num_qumodes=4)
qbr = qiskit.QuantumRegister(size=1)
circuit = c2qa.CVCircuit(qmr, qbr)

# Initialize qubit to superposition
circuit.initialize((1 / np.sqrt(2)) * np.array([1, 1]), qbr[0])

# Initialize both qumodes to a zero spin 1 state (Fock state 1)
for i in range(qmr.num_qumodes):
    circuit.cv_initialize(1, qmr[i])

state0, _ = c2qa.util.simulate(circuit)
# print(state0)
print("normalised initial state ", np.conj(state0.data).T.dot(state0))

# circuit.cv_bs2m1q(qmr[0], qmr[1], qbr[0])
circuit.cv_aklt(qmr[0], qmr[1], qbr[0])
#
# state, _ = c2qa.util.simulate(circuit)
# print(state)
# print("normalised final state ",np.conj(state.data).T.dot(state))
