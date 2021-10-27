import c2qa
import qiskit
import numpy as np
import scipy
import itertools
import projectors, gatetesting, stateReadout
import numpy as np
# Import Qiskit
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import CircuitOp, CircuitStateFn
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import MatrixExpectation, CircuitSampler, StateFn
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.providers.aer import AerSimulator

### Initialize the oscillators to zero (spin 1) and the qubit to a superposition
# Two modes and 1 qubit
numberofmodes=2
qmr = c2qa.QumodeRegister(num_qumodes=numberofmodes)
qbr = qiskit.QuantumRegister(size=3)
circuit = c2qa.CVCircuit(qmr, qbr)
zeroQB=np.array([1,0]) #211012 agrees with Kevin's notation
oneQB=np.array([0,1]) #211012 agrees with Kevin's notation
three=np.array([0,0,0,1])
two=np.array([0,0,1,0])
one=np.array([0,1,0,0])
zero=np.array([1,0,0,0])
projtwo=np.outer(two,two)

# Choose initial state
qbinist=0
samestallmodes=1
diffstallmodes=[0,1,2,3,2,1]

# Initialize qubit
# circuit.initialize((1 / np.sqrt(2)) * np.array([1, 1]), qbr[0])
qubitinitialstate=[[zeroQB,"0"],[oneQB,"1"]]
circuit.initialize(qubitinitialstate[qbinist][0], qbr[0])
# Initialize both qumodes to a zero spin 1 state (Fock state 1)
for i in range(qmr.num_qumodes):
    circuit.cv_initialize(samestallmodes, qmr[i])
# Check the input state is normalised
# state0, _ = c2qa.util.simulate(circuit)
# print("normalised initial state ", np.conj(state0.data).T.dot(state0))
# print(arg)


circuit.x(qbr[0])
circuit.x(qbr[1])
circuit.h(qbr[0])
circuit.cnot(qbr[0],qbr[1])
circuit.barrier()
# Native gates circuit
for i in range(numberofmodes-1):
    if (i % 2) == 0:
        circuit.cv_bs(np.arctan(1/np.sqrt(2)), qmr[i+1], qmr[i])
        circuit.cv_snap2(qmr[i + 1])
        circuit.cv_controlledparity(qmr[i],qbr[0])
        circuit.cv_bs(np.pi/4, qmr[i+1], qmr[i])
        circuit.x(qbr[0])
        circuit.cv_snap1X(qmr[i],qbr[0])
        # circuit.cv_snap2(qmr[i + 1])
        circuit.x(qbr[0])
        circuit.z(qbr[0])
        circuit.x(qbr[0])
# circuit.barrier()
# circuit.h(qbr[2])
# circuit.cswap(qbr[2], qbr[0], qbr[1])
# circuit.h(qbr[2])
# circuit.barrier()

# # you can define your operator as circuit
# operatorcirc = c2qa.CVCircuit(qmr, qbr)
# operatorcirc.z(0)
# op = CircuitOp(operatorcirc)  # and convert to an operator
#
# # convert to a state
# psi = CircuitStateFn(circuit)
#
# # define your backend or quantum instance (where you can add settings)
# backend = Aer.get_backend('qasm_simulator')
# q_instance = QuantumInstance(backend, shots=1024)
#
# # define the state to sample
# measurable_expression = StateFn(op, is_measurement=True).compose(psi)
#
# expectation = MatrixExpectation().convert(measurable_expression)
# sampler = CircuitSampler(backend).convert(expectation)
# print('Matrix:', sampler.eval().real)


# print('Math:', psi.adjoint().compose(op).compose(psi).eval().real)


# stateAKLT, _ = c2qa.util.simulate(circuit)
# circuit.cv_RSzSB(qmr[1],qmr[0])
# print(circuit)

# diffstallmodes=[1,1]
# gatetesting.differentThetaInitialisation(qmr, circuit, numberofmodes, qbinist, samestallmodes, diffstallmodes)

#simulate circuit and see if it's normalised
stateop, _ = c2qa.util.simulate(circuit)
# print(stateop)
# print("normalised final state ",np.conj(state.data).T.dot(state))
print("Reading out")
# stateop, _ = c2qa.util.simulate(circuit)
stateReadout.stateread(stateop, qbr.size, numberofmodes, qbinist, samestallmodes, diffstallmodes, "samestallmodes", 4)
# print("Projectors")
# projectors.overlap(stateop, numberofmodes, qbinist, samestallmodes, diffstallmodes, "samestallmodes" ,"all")

# # Construct an ideal simulator
# aersim = AerSimulator()
# result_ideal = qiskit.execute(circuit, aersim).result()
# counts_ideal = result_ideal.get_counts(0)
# print('Counts(ideal):', counts_ideal)
# print(plot_histogram(counts_ideal, title='AKLT').show())

circuit.draw(output='mpl', filename='/Users/ecrane/Dropbox/Qiskit c2qa/my_circuit.png')


# circuit.measure_all()
circuit.measure(-1,0)

# Construct an ideal simulator
aersim = AerSimulator()
result_ideal = qiskit.execute(circuit, aersim).result()
counts_ideal = result_ideal.get_counts(0)
print('Counts(ideal):', counts_ideal)
print(plot_histogram(counts_ideal, title='AKLT').show())

circuit.draw(output='mpl', filename='my_circuit.png')

# # Transpile for simulator
# simulator = Aer.get_backend('aer_simulator')
# circ = transpile(circuit, simulator)
#
# # Run and get counts
# result = simulator.run(circ).result()
# counts = result.get_counts(circ)
# print(counts)
# print(plot_histogram(counts, title='AKLT').show())
