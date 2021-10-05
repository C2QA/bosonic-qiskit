import c2qa
import qiskit
import numpy as np
import scipy
import itertools
import projectors

### Initialize the oscillators to zero (spin 1) and the qubit to a superposition
# Two modes and 1 qubit
numberofmodes=2
qmr = c2qa.QumodeRegister(num_qumodes=numberofmodes)
qbr = qiskit.QuantumRegister(size=1)
circuit = c2qa.CVCircuit(qmr, qbr)
zeroQB=np.array([0,1]) #here: Kevin says for physicists should be one but I have algorithms definition I think (but usually zero) <- check
oneQB=np.array([1,0]) #here: zero (but usually one) <- check
three=np.array([0,0,0,1])
two=np.array([0,0,1,0])
one=np.array([0,1,0,0])
zero=np.array([1,0,0,0])
projtwo=np.outer(two,two.T)

# Initialize qubit to superposition
# circuit.initialize((1 / np.sqrt(2)) * np.array([1, 1]), qbr[0])
qbinist=1
qubitinitialstate=[[zeroQB,"0"],[oneQB,"1"]]
circuit.initialize(qubitinitialstate[qbinist][0], qbr[0])
# Initialize both qumodes to a zero spin 1 state (Fock state 1)
fockstate=2
for i in range(qmr.num_qumodes):
    circuit.cv_initialize(fockstate, qmr[i])
# Check the input state is normalised
state0, _ = c2qa.util.simulate(circuit)
# print("normalised initial state ", np.conj(state0.data).T.dot(state0))
# print(arg)

# Apply circuit
# for i in range(numberofmodes-1):
#     if (i % 2) == 0:
#         circuit.cv_aklt(qmr[i], qmr[i+1], qbr[0])
#         circuit.cv_snap2(qmr[i+1])

# # Native gates circuit
# for i in range(numberofmodes-1):
#     if (i % 2) == 0:
#         circuit.z(qbr[0])
#         circuit.h(qbr[0])
#         circuit.cv_cpbs(np.arctan(1/np.sqrt(2)), qmr[i], qmr[i+1], qbr[0])
#         circuit.h(qbr[0])
#         circuit.cv_cpbs(np.pi/2, qmr[i], qmr[i+1], qbr[0])
#         circuit.cv_bs(-np.pi/2, qmr[i], qmr[i+1])
#         circuit.h(qbr[0])
#         circuit.cv_snap2(qmr[i+1])
#         circuit.cv_cpbs(np.pi/4, qmr[i], qmr[i+1], qbr[0])
#         circuit.h(qbr[0])
#         circuit.cv_snap2(qmr[i+1])

circuit.cv_snap2(qmr[0])
# circuit.cv_snap2(qmr[1])

#simulate circuit and see if it's normalised
state, _ = c2qa.util.simulate(circuit)
# print(state)
# print("normalised final state ",np.conj(state.data).T.dot(state))

projectors.overlap(state, numberofmodes, qubitinitialstate, qbinist, fockstate, "all")

