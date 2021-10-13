import copy

import c2qa
import qiskit
import numpy as np
import scipy
import itertools
from scipy.sparse import csr_matrix

# a = "0123012301230123"
# p = itertools.permutations(a,4)
# for j in list(p):
#     print(j)

zeroQB=np.array([1,0]) #211012 agrees with Kevin's notation
oneQB=np.array([0,1]) #211012 agrees with Kevin's notation
three=np.array([0,0,0,1])
two=np.array([0,0,1,0])
one=np.array([0,1,0,0])
zero=np.array([1,0,0,0])
projtwo=np.outer(two,two.T)

def singlemodestates(numberofmodes):
    modestates = []
    for i in range(int(numberofmodes)):
        modestates.append([zero, 0])
        modestates.append([one, 1])
        modestates.append([two, 2])
        modestates.append([three, 3])
    return modestates

def singlemodestatevalues(numberofmodes):
    modestates = []
    for i in range(int(numberofmodes)):
        modestates.append(0)
        modestates.append(1)
        modestates.append(2)
        modestates.append(3)
    return modestates

def reduceduplicates(numberofmodes):
    sbstates = []
    modestates=singlemodestatevalues(numberofmodes)
    list0 = list(itertools.permutations(modestates, r=numberofmodes))
    # print(len(list0), " and ", len(set(list0)))
    list1=[]
    for i in set(list0):
        # print(i)
        inside=[]
        for j in range(numberofmodes):
            # print(i[j])
            if i[j]==0:
                inside.append([zero, 0])
            elif i[j]==1:
                inside.append([one, 1])
            elif i[j]==2:
                inside.append([two, 2])
            elif i[j]==3:
                inside.append([three, 3])
            else:
                print("Cutoff above 4?")
        # print(inside)
        list1.append(inside)
    # print("list1 ", list1)
    # print("len " ,len(list1))
    return list1

def allkroneckermodestates(numberofmodes):
    sbstates = []
    list1 = reduceduplicates(numberofmodes)
    # print(list1)
    for i in range(len(list1)):
        # if sum(x[1] for x in list[i]) == numberofmodes:
        inside = np.array(list1[i][0][0])
        line = []
        line.append(np.array(list1[i][0][1]))
        for j in range(1, len(list1[i])):
            # print(j)
            inside = np.kron(np.array(list1[i][j][0]), inside)
            # print(inside)
            line.append(np.array(list1[i][j][1]))
            # print(line)
        # print("line ", line)
        line.append(inside)
        # print("line ", line)
        sbstates.append(line)

    return sbstates

def sbkroneckermodestates(numberofmodes):
    sbstates = []
    list1 = reduceduplicates(numberofmodes)
    for i in range(len(list1)):
        if sum(x[1] for x in list1[i]) == numberofmodes:
            inside = np.array(list1[i][0][0])
            line = []
            line.append(np.array(list1[i][0][1]))
            for j in range(1, len(list1[i])):
                # print(j)
                inside = np.kron(np.array(list1[i][j][0]), inside)
                # print(inside)
                line.append(np.array(list1[i][j][1]))
                # print(line)
            # print("line ", line)
            line.append(inside)
            # print("line ", line)
            sbstates.append(line)
    #
    # for i in range(len(modestates)):
    #     inside=modestates[i][0]
    #     line=[]
    #     line.append(np.array(modestates[i][1]))
    #     for j in range(numberofmodes-1):
    #         line.append(np.array(modestates[i][1]))
    #         inside = np.kron(modestates[i][0], inside)
    #     line.append(inside)
    #     # print(line)
    #     sbstates.append(line)

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
        # print("fstates[i][0] i ",i," fstates[i] ", fstates[i], fstates[i][0].shape)
        # print("state " ,np.array(state).shape)
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
