import c2qa
import qiskit
import numpy as np
import scipy
import itertools
import projectors, gatetesting, stateReadout
import numpy as np
import matplotlib.pyplot as plt
# Import Qiskit
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.providers.aer import AerSimulator
from qiskit import IBMQ


# ### Initialize the oscillators to zero (spin 1) and the qubit to a superposition
# # Two modes and 1 qubit
# numberofmodes=8
# qmr = c2qa.QumodeRegister(num_qumodes=numberofmodes)
# qbr = qiskit.QuantumRegister(size=3)
# cbr = qiskit.ClassicalRegister(size=1)
# circuit = c2qa.CVCircuit(qmr, qbr, cbr)
# zeroQB=np.array([1,0]) #211012 agrees with Kevin's notation
# oneQB=np.array([0,1]) #211012 agrees with Kevin's notation
# three=np.array([0,0,0,1])
# two=np.array([0,0,1,0])
# one=np.array([0,1,0,0])
# zero=np.array([1,0,0,0])
# projtwo=np.outer(two,two)
#
# # Choose initial state
# qbinist=0
# samestallmodes=1
# diffstallmodes=[1,1,1]
# # Initialize qubit
# qubitinitialstate=[[zeroQB,"0"],[oneQB,"1"]]
# # circuit.initialize(qubitinitialstate[qbinist][0], qbr[0])
# # Initialize both qumodes to a zero spin 1 state (Fock state 1)
# # for i in range(qmr.num_qumodes):
# #     circuit.cv_initialize(samestallmodes, qmr[i]) #diffstallmodes[i] or samestallmodes
# word="samestallmodes" #should correspond to the above line
#
# circuit.x(qbr[0])
# circuit.x(qbr[1])
# circuit.h(qbr[0])
# circuit.cnot(qbr[0],qbr[1])
# # circuit.barrier()
# # Native gates circuit
# for i in range(numberofmodes-1):
#     if (i % 2) == 0:
#         circuit.cv_bs(np.arctan(1/np.sqrt(2)), qmr[i+1], qmr[i])
#         circuit.cv_snap2(qmr[i + 1])
#         circuit.cv_controlledparity(qmr[i],qbr[0])
#         circuit.cv_bs(np.pi/4, qmr[i+1], qmr[i])
#         circuit.x(qbr[0])
#         circuit.cv_snap1X(qmr[i],qbr[0])
#         # circuit.cv_snap2(qmr[i + 1])
#         circuit.x(qbr[0])
#         circuit.z(qbr[0])
#         circuit.x(qbr[0])
# # circuit.barrier()
# circuit.h(qbr[2])
# circuit.cswap(qbr[2], qbr[0], qbr[1])
# circuit.h(qbr[2])
# circuit.measure(-1,0)
# circuit.x(qbr[0]).c_if(cbr, 0)
# circuit.x(qbr[1]).c_if(cbr, 0)
# circuit.z(qbr[0]).c_if(cbr, 0)
# circuit.z(qbr[1]).c_if(cbr, 0)
# circuit.measure_all()
#
#
# stateop, _ = c2qa.util.simulate(circuit)
# print("Simulated the circuit with rectification")
# stateReadout.stateread(stateop, qbr.size, numberofmodes, qbinist, samestallmodes, diffstallmodes, word, 4)
# list=stateReadout.statelist(stateop, qbr.size, numberofmodes, qbinist, samestallmodes, diffstallmodes, word, 4)
# chain=list[0]
# weights=list[1]
# print("Main ",chain, weights)
# stateReadout.stringoperator(chain, weights)
# dict=stateReadout.makedictionnary(chain, weights)
# plt=plot_histogram(dict, title='AKLT')
# plt.tight_layout()
# print(plt.show())
# circuit.draw(output='mpl', filename='/Users/ecrane/Dropbox/Qiskit c2qa/my_circuit.png')
#
# # Construct an ideal simulator
# aersim = AerSimulator()
# result_ideal = qiskit.execute(circuit, aersim, memory=True).result()
# counts = result_ideal.get_counts(0)
# print('Counts(ideal):', counts)


# IBMQ.save_account('74e12532dbce8c34a6e9c9a058822a5ef6a56142c323c9d964837b4ffee47408a560b45350eef82fb5dc061ba6dd818c2bbc6a884316a98947914b4161b08afb')
print(IBMQ.load_account())
provider = IBMQ.get_provider()
print(provider)
# backend = provider.get_backend('simulator_statevector')
backend = provider.get_backend('simulator_mps')
#
# circuit_sys = transpile(circuit, backend)
# shnb=8190
# job = backend.run(circuit_sys, shots=shnb, job_name="AKLT_MPS_"+str(numberofmodes)+"_"+str(shnb))
#
print(backend)
job = backend.retrieve_job('6182ac2507a4a353ad7c2416')
res = job.result()
counts = res.get_counts()



print(counts)
dict0=stateReadout.changeBasis(counts, 8, splitup=1)
print(dict0[0])
tripletdict=dict0[0]
weights=list(tripletdict.values())
chain=list(tripletdict.keys())
print(chain,weights)
print(dict0[2])
tripletcounts=list(dict0[2].values())[1]
print(tripletcounts)
# print(len(chain[0]))
for d in range(2,8):
    # stateReadout.stringoperator(chain,weights,tripletcounts)
    stateReadout.stringoperator_variable(chain,weights,tripletcounts,d)














# print(dict0)
# print(counts)
# chain=stateReadout.interpretmeasurementresult(list(counts.keys()), numberofmodes)
# weights = list(counts.values())
# print("Main raw",chain, weights)
# plt.bar(chain, weights, color='g')
# plt.yticks(rotation='vertical')
# # plt.tight_layout()
# plt.show()
# list=stateReadout.clean(chain, weights)
# chain=list[0]
# weights=list[1]
# print("Main after cleaning ",chain, weights)

# dict=stateReadout.makedictionnary(chain, weights)
# print(list(counts.values()))
# stateReadout.stringoperator(chain, list(counts.values()))


# from qiskit import IBMQ
# # IBMQ.save_account('74e12532dbce8c34a6e9c9a058822a5ef6a56142c323c9d964837b4ffee47408a560b45350eef82fb5dc061ba6dd818c2bbc6a884316a98947914b4161b08afb')
# print(IBMQ.load_account())
# provider = IBMQ.get_provider()
# print(provider)
# # backend = provider.get_backend('simulator_statevector')
# backend = provider.get_backend('simulator_mps')
#
# circuit_sys = transpile(circuit, backend)
# shnb=8190
# job = backend.run(circuit_sys, shots=shnb, job_name="AKLT_MPS_"+str(numberofmodes)+"_"+str(shnb))
#
# print(backend)
# job = backend.retrieve_job('617ea4b09c7dc5cc4facab7d')
# res = job.result()
# counts = res.get_counts()
#
#
# print('Counts :', counts)
# chain=stateReadout.interpretmeasurementresult(list(counts.keys()), (len(list(counts.keys())[0])-5)/2)
# print(chain)
# print(list(counts.values()))
# stateReadout.stringoperator(chain, list(counts.values()))
# dict=stateReadout.makedictionnary(chain, list(counts.values()))
# plt=plot_histogram(dict, title='AKLT')
# plt.tight_layout()
# print(plt.show())


# projectors.overlap(stateop, numberofmodes, qbinist, samestallmodes, diffstallmodes, "samestallmodes" ,"all")




# job_manager = IBMQJobManager()
# job_set = job_manager.run(qclist, backend=backend, name='L_3_vqe_qc')
# result_qc = job_set.results()
# result_qc = [ result_qc.get_counts(ind) for ind in range(len(qclist)) ]
# print( result_qc )

# result = simulator.run(circ).result()
# counts = result.get_counts(circ)
# print(counts)
# print(plot_histogram(counts, title='AKLT').show())
