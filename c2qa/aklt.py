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
upQB=np.array([0,1])
downQB=np.array([1,0])
three=np.array([0,0,0,1])
two=np.array([0,0,1,0])
one=np.array([0,1,0,0])
zero=np.array([1,0,0,0])
modestates=[[zero,"0"],[one,"1"],[two,"2"],[three,"3"]]

# circuit.cv_initialize(2, qmr[0])
# # circuit.initialize((1 / np.sqrt(2)) * np.array([1, 1]), qbr[0])
# circuit.initialize(downQB, qbr[0])
# state, _ = c2qa.util.simulate(circuit)
# print("initial state ", state)
#
# fstates=[]
# for i in range(len(modestates)):
#     fstates.append([scipy.sparse.kron(upQB,modestates[i][0]), modestates[i][1],"up"])
#     fstates.append([scipy.sparse.kron(downQB,modestates[i][0]), modestates[i][1],"down"])
#
# amp=0
# for i in range(len(fstates)):
#     val=np.abs(np.conj(fstates[i][0]).dot(state))**2
#     print("i ",i, " overlap with ",fstates[i][1]," ",fstates[i][2]," is: ", val)
#     amp+=val
#
# print(amp)

# Initialize qubit to superposition
circuit.initialize((1 / np.sqrt(2)) * np.array([1, 1]), qbr[0])
# circuit.initialize(np.array([0, 1]), qbr[0])

# Initialize both qumodes to a zero spin 1 state (Fock state 1)
for i in range(qmr.num_qumodes):
    circuit.cv_initialize(1, qmr[i])

state0, _ = c2qa.util.simulate(circuit)
print("normalised initial state ", np.conj(state0.data).T.dot(state0))

for i in range(numberofmodes-1):
    if (i % 2) == 0:
        print(i)
        circuit.cv_aklt(qmr[i], qmr[i+1], qbr[0])
        # circuit.cv_snap(qmr[i+1], qbr[0])
#
state, _ = c2qa.util.simulate(circuit)
print(state)
print("normalised final state ",np.conj(state.data).T.dot(state))

upQB=np.array([0,1])
downQB=np.array([1,0])

three=np.array([0,0,0,1])
two=np.array([0,0,1,0])
one=np.array([0,1,0,0])
zero=np.array([1,0,0,0])
modestates=[[zero,"0"],[one,"1"],[two,"2"],[three,"3"]]
# modestates=[[zero,"0"],[zero,"0"],[zero,"0"],[zero,"0"],[one,"1"],[one,"1"],[one,"1"],[one,"1"],[two,"2"],[two,"2"],[two,"2"],[two,"2"],[three,"3"],[three,"3"],[three,"3"],[three,"3"]]

sbstates=[]
list=list(itertools.permutations(modestates, r=2))

for i in range(len(list)):
    sbstates.append([scipy.sparse.kron(np.array(list[i][0][0]),np.array(list[i][1][0])),np.array(list[i][0][1]),np.array(list[i][1][1])])
for i in range(len(modestates)):
    sbstates.append([scipy.sparse.kron(modestates[i][0], modestates[i][0]), modestates[i][1],modestates[i][1]])

# list=list(itertools.permutations(modestates, r=4))
# for i in range(len(list)):
#     sbstates.append([scipy.sparse.kron(np.array(list[i][0][0]),scipy.sparse.kron(np.array(list[i][0][0]),scipy.sparse.kron(np.array(list[i][0][0]),np.array(list[i][1][0])))),np.array(list[i][0][1]),np.array(list[i][1][1]),np.array(list[i][2][1]),np.array(list[i][3][1])])

fstates=[]
for i in range(len(sbstates)):
    fstates.append([scipy.sparse.kron(upQB,sbstates[i][0]), sbstates[i][1], sbstates[i][2]])
    fstates.append([scipy.sparse.kron(downQB, sbstates[i][0]), sbstates[i][1], sbstates[i][2]])

# fstates=[]
# for i in range(len(sbstates)):
#     fstates.append([scipy.sparse.kron(sbstates[i][0],upQB), sbstates[i][1], sbstates[i][2], sbstates[i][3], sbstates[i][4]])
#     fstates.append([scipy.sparse.kron(sbstates[i][0], downQB), sbstates[i][1], sbstates[i][2], sbstates[i][3], sbstates[i][4]])

amp=0
for i in range(len(fstates)):
    val=np.abs(np.conj(fstates[i][0]).dot(state))**2
    print("i ",i, " overlap with ",fstates[i][1],fstates[i][2]," up is: ", val)
    # print("i ",i, " overlap with ",fstates[i][1],fstates[i][2],fstates[i][3],fstates[i][4]," up is: ", val)
    amp+=val

print(amp)









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
