import c2qa
import qiskit
import numpy as np
import scipy
import itertools
from scipy.sparse import csr_matrix

zeroQB=np.array([0,1]) #here: Kevin says for physicists should be one but I have algorithms definition I think (but usually zero) <- check
oneQB=np.array([1,0]) #here: zero (but usually one) <- check
three=np.array([0,0,0,1])
two=np.array([0,0,1,0])
one=np.array([0,1,0,0])
zero=np.array([1,0,0,0])
projtwo=np.outer(two,two.T)

def singlemodestates(numberofmodes):
    modestates = []
    for i in range(int(numberofmodes / 2)):
        modestates.append([zero, 0])
        modestates.append([one, 1])
        modestates.append([two, 2])
        modestates.append([three, 3])
    return modestates

def allkroneckermodestates(numberofmodes):
    sbstates = []
    modestates=singlemodestates(numberofmodes)
    list1 = list(itertools.permutations(modestates, r=numberofmodes))
    for i in range(len(list1)):
        # if sum(x[1] for x in list[i]) == numberofmodes:
        inside = np.array(list1[i][0][0])
        line = []
        line.append(np.array(list1[i][0][1]))
        for j in range(1, len(list1[i])):
            # print(j)
            inside = np.kron(inside, np.array(list1[i][j][0]))
            # print(inside)
            line.append(np.array(list1[i][j][1]))
            # print(line)
        # print("line ", line)
        line.append(inside)
        # print("line ", line)
        sbstates.append(line)

    for i in range(len(modestates)):
        # if modestates[i][1] == 1:
        sbstates.append([modestates[i][1],modestates[i][1], scipy.sparse.kron(modestates[i][0], modestates[i][0])])

    return sbstates

def sbkroneckermodestates(numberofmodes):
    sbstates = []
    modestates=singlemodestates(numberofmodes)
    list1 = list(itertools.permutations(modestates, r=numberofmodes))
    for i in range(len(list1)):
        if sum(x[1] for x in list1[i]) == numberofmodes:
            inside = np.array(list1[i][0][0])
            line = []
            line.append(np.array(list1[i][0][1]))
            for j in range(1, len(list1[i])):
                # print(j)
                inside = np.kron(inside, np.array(list1[i][j][0]))
                # print(inside)
                line.append(np.array(list1[i][j][1]))
                # print(line)
            # print("line ", line)
            line.append(inside)
            # print("line ", line)
            sbstates.append(line)

    for i in range(len(modestates)):
        if modestates[i][1] == 1:
            sbstates.append([modestates[i][1],modestates[i][1], scipy.sparse.kron(modestates[i][0], modestates[i][0])])

    return sbstates

# Create all permutations of 'numberofmodes' of mode states
def overlap(state, numberofmodes, qbinist, samestallmodes, diffstallmodes, modeinichoice, choice):
    if choice == "all":
        sbstates=allkroneckermodestates(numberofmodes)
    else:
        sbstates = sbkroneckermodestates(numberofmodes)
    # Create the final states which contain also the qubit values
    fstates=[]
    for i in range(len(sbstates)):
        fstates.append([scipy.sparse.kron(oneQB,sbstates[i][-1]), 1, np.stack(sbstates[i][:-1])])
        fstates.append([scipy.sparse.kron(zeroQB, sbstates[i][-1]), 0, np.stack(sbstates[i][:-1])])

    # Take the overlap and calculate probablility of final state occuring in prepared state
    probs=0
    amp=[]
    if modeinichoice=="samestallmodes":
        inim=[samestallmodes]*numberofmodes
        modesini=str(qbinist)+" "
        for i in range(len(inim)):
            modesini=modesini+str(inim[i])
    else:
        modesini = str(qbinist) + " "
        for i in range(len(diffstallmodes)):
            modesini = modesini + str(diffstallmodes[i])

    # print("\n")
    for i in range(len(fstates)):
        res=np.conj(fstates[i][0]).dot(state)
        prob=np.abs(res)**2
        # print("Probability to get ",fstates[i][1],fstates[i][2],fstates[i][3]," is: ", prob)
        sbstr=["".join(item) for item in fstates[i][2].astype(str)]
        finalres=(np.real(res)*(np.abs(np.real(res))>1e-10)[0] + 1j*np.imag(res)*(np.abs(np.imag(res))>1e-10)[0])[0]
        if finalres != 0:
            print(modesini, " overlap with ", fstates[i][1], ''.join(sbstr), " is: ", finalres )
        probs+=prob
        amp.append(res)
    # print("probs ",probs )


# print("sbstates ", sbstates)
# sbstates.append([scipy.sparse.kron(np.array(list[i][0][0]),np.array(list[i][1][0])),np.array(list[i][0][1]),np.array(list[i][1][1])])

# for i in range(len(modestates)):
#     if modestates[i][1] == 1:
#         sbstates.append([scipy.sparse.kron(modestates[i][0], modestates[i][0]), modestates[i][1],modestates[i][1]])

# # Generates all possible permutations (not just SB ones)
# for i in range(len(list)):
#     sbstates.append([scipy.sparse.kron(np.array(list[i][0][0]),np.array(list[i][1][0])),np.array(list[i][0][1]),np.array(list[i][1][1])])
# for i in range(len(modestates)):
#     sbstates.append([scipy.sparse.kron(modestates[i][0], modestates[i][0]), modestates[i][1],modestates[i][1]])

# print(np.stack(sbstates[1][:-1]))
