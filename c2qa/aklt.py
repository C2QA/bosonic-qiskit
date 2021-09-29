import c2qa
import qiskit
import numpy as np
import scipy
import itertools

### Initialize the oscillators to zero (spin 1) and the qubit to a superposition
# Two modes and 1 qubit
numberofmodes=2
qmr = c2qa.QumodeRegister(num_qumodes=numberofmodes)
qbr = qiskit.QuantumRegister(size=1)
circuit = c2qa.CVCircuit(qmr, qbr)
oneQB=np.array([0,1]) #here: one (but usually zero) <- check
zeroQB=np.array([1,0]) #here: zero (but usually one) <- check
three=np.array([0,0,0,1])
two=np.array([0,0,1,0])
one=np.array([0,1,0,0])
zero=np.array([1,0,0,0])
modestates=[[zero,0],[one,1],[two,2],[three,3]]
# modestates=[[zero,"0"],[zero,"0"],[zero,"0"],[zero,"0"],[one,"1"],[one,"1"],[one,"1"],[one,"1"],[two,"2"],[two,"2"],[two,"2"],[two,"2"],[three,"3"],[three,"3"],[three,"3"],[three,"3"]]


# Initialize qubit to superposition
circuit.initialize((1 / np.sqrt(2)) * np.array([1, 1]), qbr[0])
circuit.initialize(zeroQB, qbr[0])
# Initialize both qumodes to a zero spin 1 state (Fock state 1)
for i in range(qmr.num_qumodes):
    circuit.cv_initialize(1, qmr[i])
# Check the input state is normalised
state0, _ = c2qa.util.simulate(circuit)
# print("normalised initial state ", np.conj(state0.data).T.dot(state0))

# Apply circuit
for i in range(numberofmodes-1):
    if (i % 2) == 0:
        print(i)
        circuit.cv_aklt(qmr[i], qmr[i+1], qbr[0])
        # circuit.cv_snap(qmr[i+1], qbr[0])
#simulate circuit and see if it's normalised
state, _ = c2qa.util.simulate(circuit)
# print(state)
# print("normalised final state ",np.conj(state.data).T.dot(state))

# Create all Schwinger-Boson permutations of 'numberofmodes' of mode states
sbstates=[]
list=list(itertools.permutations(modestates, r=numberofmodes))
for i in range(len(list)):
    if np.array(list[i][0][1])+np.array(list[i][1][1]) == 2:
        sbstates.append([scipy.sparse.kron(np.array(list[i][0][0]),np.array(list[i][1][0])),np.array(list[i][0][1]),np.array(list[i][1][1])])
for i in range(len(modestates)):
    if modestates[i][1] == 1:
        sbstates.append([scipy.sparse.kron(modestates[i][0], modestates[i][0]), modestates[i][1],modestates[i][1]])

# # Generates all possible permutations (not just SB ones)
# for i in range(len(list)):
#     sbstates.append([scipy.sparse.kron(np.array(list[i][0][0]),np.array(list[i][1][0])),np.array(list[i][0][1]),np.array(list[i][1][1])])
# for i in range(len(modestates)):
#     sbstates.append([scipy.sparse.kron(modestates[i][0], modestates[i][0]), modestates[i][1],modestates[i][1]])

# list=list(itertools.permutations(modestates, r=4))
# for i in range(len(list)):
#     sbstates.append([scipy.sparse.kron(np.array(list[i][0][0]),scipy.sparse.kron(np.array(list[i][0][0]),scipy.sparse.kron(np.array(list[i][0][0]),np.array(list[i][1][0])))),np.array(list[i][0][1]),np.array(list[i][1][1]),np.array(list[i][2][1]),np.array(list[i][3][1])])

# Create the final states which contain also the qubit values
fstates=[]
for i in range(len(sbstates)):
    fstates.append([scipy.sparse.kron(oneQB,sbstates[i][0]),1, sbstates[i][1], sbstates[i][2]])
    fstates.append([scipy.sparse.kron(zeroQB, sbstates[i][0]),0, sbstates[i][1], sbstates[i][2]])

# fstates=[]
# for i in range(len(sbstates)):
#     fstates.append([scipy.sparse.kron(sbstates[i][0],upQB), sbstates[i][1], sbstates[i][2], sbstates[i][3], sbstates[i][4]])
#     fstates.append([scipy.sparse.kron(sbstates[i][0], downQB), sbstates[i][1], sbstates[i][2], sbstates[i][3], sbstates[i][4]])

# Take the overlap and calculate probablility of final state occuring in prepared state
probs=0
amp=[]
for i in range(len(fstates)):
    res=np.conj(fstates[i][0]).dot(state)
    prob=np.abs(res)**2
    # print("Probability to get ",fstates[i][1],fstates[i][2],fstates[i][3]," is: ", prob)
    print("Overlap with ", fstates[i][1], fstates[i][2], fstates[i][3], " is: ", res)
    probs+=prob
    amp.append(res)
print("probs ",probs )










# print(list)
# print(list[1][0][0])

# states=[scipy.sparse.kron(sbp1,upQB),scipy.sparse.kron(sbp1,downQB),scipy.sparse.kron(sb0,upQB),scipy.sparse.kron(sb0,downQB),scipy.sparse.kron(sbm1,upQB),scipy.sparse.kron(sbm1,downQB)]


    # sbstates.append([scipy.sparse.kron(zero,zero),"0","0"])
    # sbstates.append([scipy.sparse.kron(one,one),"1","1"])
    # sbstates.append([scipy.sparse.kron(two,two),"2","2"])
    # sbstates.append([scipy.sparse.kron(three,three),"3","3"])

# print("sbstates: ", sbstates[5][0], sbstates[5][1], sbstates[5][2])


# data = np.sqrt(range(4))
# a = scipy.sparse.spdiags(data=data, diags=[1], m=len(data), n=len(data))
# a_dag = a.conjugate().transpose()
# N = a_dag * a
# eyeQB = scipy.sparse.eye(2)
# eye = scipy.sparse.eye(4)



# upQB=np.array([0,1])
# downQB=np.array([1,0])
#
# three=np.array([0,0,0,1])
# two=np.array([0,0,1,0])
# one=np.array([0,1,0,0])
# zero=np.array([1,0,0,0])
# modestates=[zero,one,two,three]
#
# sbstates=[]
# list=list(itertools.permutations(modestates, r=2))
# # print(list)
# # print(list[1][0])
# for i in range(len(list)):
#     sbstates.append(scipy.sparse.kron(np.array(list[i][0]),np.array(list[i][1])))
#
# print("sbstates: ", np.array(sbstates[0]))
#
#     # sbstates=scipy.sparse.kron(two,zero)
#     # sb0=scipy.sparse.kron(one,one)
#     # sbm1=scipy.sparse.kron(zero,two)
#
# fstates=[]
# for i in range(len(sbstates)):
#     fstates.append(scipy.sparse.kron(sbstates[i],upQB))
#     fstates.append(scipy.sparse.kron(sbstates[i], downQB))
#
# # states=[scipy.sparse.kron(sbp1,upQB),scipy.sparse.kron(sbp1,downQB),scipy.sparse.kron(sb0,upQB),scipy.sparse.kron(sb0,downQB),scipy.sparse.kron(sbm1,upQB),scipy.sparse.kron(sbm1,downQB)]
#
# amp=0
# for i in range(len(fstates)):
#     val=np.abs(np.conj(fstates[i]).dot(state))**2
#     print("i ",i, " overlap with +1: ", val)
#     amp+=val
#
# print(amp)

# circuit.cv_bs2m1q(qmr[0], qmr[1], qbr[0])
