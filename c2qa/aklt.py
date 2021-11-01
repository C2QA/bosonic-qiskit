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

### Initialize the oscillators to zero (spin 1) and the qubit to a superposition
# Two modes and 1 qubit
numberofmodes=3
qmr = c2qa.QumodeRegister(num_qumodes=numberofmodes)
qbr = qiskit.QuantumRegister(size=3)
cbr = qiskit.ClassicalRegister(size=1)
circuit = c2qa.CVCircuit(qmr, qbr, cbr)
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
diffstallmodes=[1,1,1]
# Initialize qubit
qubitinitialstate=[[zeroQB,"0"],[oneQB,"1"]]
# circuit.initialize(qubitinitialstate[qbinist][0], qbr[0])
# Initialize both qumodes to a zero spin 1 state (Fock state 1)
for i in range(qmr.num_qumodes):
    circuit.cv_initialize(samestallmodes, qmr[i]) #diffstallmodes[i] or samestallmodes
word="samestallmodes" #should correspond to the above line

circuit.x(qbr[0])
circuit.x(qbr[1])
circuit.h(qbr[0])
circuit.cnot(qbr[0],qbr[1])
# circuit.barrier()
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
circuit.h(qbr[2])
circuit.cswap(qbr[2], qbr[0], qbr[1])
circuit.h(qbr[2])
circuit.measure(-1,0)
circuit.x(qbr[0]).c_if(cbr, 0)
circuit.x(qbr[1]).c_if(cbr, 0)
circuit.z(qbr[0]).c_if(cbr, 0)
circuit.z(qbr[1]).c_if(cbr, 0)
# circuit.barrier()
circuit.measure_all()
# # print("Measurement")
#
#
stateop, _ = c2qa.util.simulate(circuit)
print("Simulated the circuit with rectification")
stateReadout.stateread(stateop, qbr.size, numberofmodes, qbinist, samestallmodes, diffstallmodes, word, 4)
circuit.draw(output='mpl', filename='/Users/ecrane/Dropbox/Qiskit c2qa/my_circuit.png')
#
# Construct an ideal simulator
aersim = AerSimulator()
result_ideal = qiskit.execute(circuit, aersim, memory=True).result()
counts_ideal = result_ideal.get_counts(0)
print('Counts(ideal):', counts_ideal)
chain=stateReadout.interpretmeasurementresult(list(counts_ideal.keys()), numberofmodes)
dict=stateReadout.makedictionnary(chain, list(counts_ideal.values()))
print(list(counts_ideal.values()))
stateReadout.stringoperator(chain, list(counts_ideal.values()))
plt=plot_histogram(dict, title='AKLT')
plt.tight_layout()
print(plt.show())


# from qiskit import IBMQ
# # IBMQ.save_account('74e12532dbce8c34a6e9c9a058822a5ef6a56142c323c9d964837b4ffee47408a560b45350eef82fb5dc061ba6dd818c2bbc6a884316a98947914b4161b08afb')
# # print(IBMQ.load_account())
# provider = IBMQ.get_provider()
# print(provider)
# backend = provider.get_backend('simulator_statevector')
# print(backend)


# # From IBM documentation
# from qiskit import IBMQ, transpile
# from qiskit.providers.ibmq.managed import IBMQJobManager
# # from qiskit.circuit.random import random_circuit
# provider = IBMQ.load_account()
# backend = provider.get_backend('simulator_statevector')
# # # Build a thousand circuits.
# # circs = []
# # for _ in range(1000):
# #     circs.append(random_circuit(num_qubits=5, depth=4, measure=True))
# # Need to transpile the circuits first.
# circs = transpile([circuit], backend=backend)
# # Use Job Manager to break the circuits into multiple jobs.
# job_manager = IBMQJobManager()
# shotnb=1024
# job_8_1024 = job_manager.run(circs, backend=backend, shots=shotnb, ame='aklt'+str(numberofmodes)+str(shotnb)) #, shots=8190





# diffstallmodes=[1,1]
# gatetesting.differentThetaInitialisation(qmr, circuit, numberofmodes, qbinist, samestallmodes, diffstallmodes)
# projectors.overlap(stateop, numberofmodes, qbinist, samestallmodes, diffstallmodes, "samestallmodes" ,"all")
# projectors.overlap(stateop, numberofmodes, qbinist, samestallmodes, diffstallmodes, "samestallmodes" ,"all")



# # Construct an ideal simulator
# aersim = AerSimulator()
# result_ideal = qiskit.execute(circuit, aersim).result()
# counts_ideal = result_ideal.get_counts(0)
# print("Mid-circuit measurement")
# print(' Mid-circuit measurement Counts 0:', counts_ideal)
# print(plot_histogram(counts_ideal, title='mid-circuit'))

# if counts_ideal['0']>1000:
#     print('Was measured to be in triplet so rectifying')
#     circuit.barrier()
#     circuit.x(qbr[0])
#     circuit.x(qbr[1])
#     circuit.z(qbr[0])
#     circuit.z(qbr[1])
# else:
#     print("singlet")

# mapped_circuit = transpile(circuit, backend=backend)
# qobj = assemble(mapped_circuit, backend=backend, shots=1024)
# job = backend.run(qobj)

# # Stackoverflow
# # Need to transpile the circuits first.
# qclist = transpile(qclist, backend=backend)
# # Use Job Manager to break the circuits into multiple jobs.
# job_manager = IBMQJobManager()
# job_set = job_manager.run(qclist, backend=backend, name='L_3_vqe_qc')
# result_qc = job_set.results()
# result_qc = [ result_qc.get_counts(ind) for ind in range(len(qclist)) ]
# print( result_qc )
#
# # Previous code from internet following simulation
# # Transpile for simulator
# simulator = Aer.get_backend('aer_simulator')
# circ = transpile(circuit, simulator)
# # Run and get counts from simulator
# result = simulator.run(circ).result()
# counts = result.get_counts(circ)
# print(counts)
# print(plot_histogram(counts, title='AKLT').show())
