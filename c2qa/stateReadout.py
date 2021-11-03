import copy
import c2qa
import qiskit
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csr_matrix, block_diag
import math

zeroQB = np.array([1, 0])  # 211012 agrees with Kevin's notation
oneQB = np.array([0, 1])  # 211012 agrees with Kevin's notation
three = np.array([0, 0, 0, 1])
two = np.array([0, 0, 1, 0])
one = np.array([0, 1, 0, 0])
zero = np.array([1, 0, 0, 0])
projone = np.outer(one, one.T)
projtwo = np.outer(two, two.T)

def stateread(stateop, numberofqubits, numberofmodes, qbinist, samestallmodes, diffstallmodes, modeinichoice, cutoff):
    st = np.array(stateop) #convert state to np.array
    amp = []

    # What states the qubits and modes are initialised to - if they are initialised with an x-gate in the circuit it won't show up here, this is only what goes into circuit.cv_initialize()
    iniq = [qbinist] * numberofqubits # list of length number of qubits initialised to qbinist
    modesini = ""
    for i in range(len(iniq)):
        modesini = modesini + str(iniq[i]) #create a string from it
    modesini = modesini + " "
    if modeinichoice == "samestallmodes":
        inim = [samestallmodes] * numberofmodes
        for i in range(len(inim)):
            modesini = modesini + str(inim[i])
    else:
        for i in range(len(diffstallmodes)):
            modesini = modesini + str(diffstallmodes[i])

    # print("beginning", numberofmodes)
    for i in range(len(st)):
        res = st[i]
        if (np.abs(np.real(res)) > 1e-10):
            pos=i
            # print("position of non-zero real: ", pos, " res = ", res)
            qbst=np.empty(numberofqubits, dtype='int')
            iqb=0
            sln=len(st)
            while(iqb<numberofqubits):
                if pos<sln/2:
                    qbst[iqb]=int(0)
                else:
                    qbst[iqb]=int(1)
                    pos=pos-(sln/2)
                    # print("pos (sln/2)", pos, "sln ",sln)
                sln=sln/2
                iqb=iqb+1
            qbstr = ["".join(item) for item in qbst.astype(str)]
            # print("which half of the kronecker, ie. state of qubit: ", qbst)
            # print(modesini, " overlap with ",qbst[0], " is: ", np.real(res))
            qbsitestr = "measurement qubit spin-1/2:    " + str(qbst[0]) + "   ancilla qubits spin-1 and modes spin-1 chain:    "
            if qbst[1]==0:
                if qbst[2]==0:
                    qbsitestr=qbsitestr + "+"
                else:
                    qbsitestr = qbsitestr + "0"
            elif qbst[1]==1:
                if qbst[2]==1:
                    qbsitestr=qbsitestr + "-"
                else:
                    qbsitestr=qbsitestr + "0"

            # print("Qmode detector")
            qmst=np.empty(numberofmodes, dtype='int')
            # print("qmst starting in ", qmst)
            iqm=0
            # print("position is now: ",pos)
            while(iqm<numberofmodes):
                # print("mode counter iqm ", iqm)
                # print("cutoff ", cutoff)
                # print("length of vector left to search: sln ", sln)
                lendiv=sln/cutoff
                # print("lendiv (sln/cutoff)", lendiv)
                val=pos/lendiv
                # print("rough estimate of the position of the non-zero element: val (pos/lendiv) ", val)
                fock = math.floor(val)
                # print("Fock st/ resulting position in Kronecker product (math.floor(val)) ", fock)
                qmst[-iqm-1]=int(fock)
                # print("Storing that fock state: qmst ", qmst)
                # if val==math.ceil(val):
                #     print("value is val = ceil.val ",val)
                #     rdval=val-1
                # else:
                # print("remove a number of divisions corresponding to fock")
                pos=pos-(fock*lendiv)
                # print("new position for next order of depth of Kronecker product/pos: (pos-(rdiv*lendiv)) ",pos)
                sln=sln-((cutoff-1)*lendiv)
                # print("New length of vector left to search: sln (sln-((cutoff-1)*lendiv))", sln)
                iqm=iqm+1
            # print("qumode states at the end of one number's worth of searching: ", qmst)

            sbstr = ["".join(item) for item in qmst.astype(str)]
            sitestr = ""
            for site in range(numberofmodes):
                if (site % 2 == 0):
                    if qmst[site]==0 :
                        sitestr=sitestr+"-"
                    elif qmst[site]==2 :
                        sitestr=sitestr+"+"
                    elif qmst[site] == 1 & qmst[site + 1] == 1:
                        sitestr=sitestr+"0"

            print(qbsitestr, sitestr, "     is: ", np.real(res))#, "\n",''.join(qbstr), ''.join(sbstr))
            # print(modesini, " overlap with ", ''.join(qbsitestr), ''.join(sitestr), "     is: ", np.real(res))


    # print("end")

    # if (np.abs(np.imag(res)) > 1e-10):
    #     print(modesini, " overlap with ", " is: ", 1j * np.imag(res))

def interpretmeasurementresult(list, numberofmodes):
    finallist=[]
    for i in range(len(list)):
        qbsitestr = ""
        if int(list[i][1]) == 0:
            if int(list[i][2]) == 0:
                qbsitestr = "".join("+")
            else:
                qbsitestr = "".join("0")
        elif int(list[i][1]) == 1:
            if int(list[i][2]) == 1:
                qbsitestr = "".join("-")
            else:
                qbsitestr = "".join("0")

        sitestr = ""
        numberofsites=numberofmodes/2
        site = 3
        while site < (numberofsites*4+3):
            # print(site, " ", int(list[i][site]))
            if int(list[i][site]) == 1: #Qumode b comes before qumode a so if the 1st qubit of the 4 qubits showing qumodeBqumodeA is in 1 then qumodeB=10 which is fock=2 which is a=0,b=2 which is spin1=-1
                sitestr = sitestr + "-"
            elif int(list[i][site+1]) == 1:
                sitestr = sitestr + "0"
            else:
                sitestr = sitestr + "+"
            site = site + 4

        # print(''.join(qbsitestr), ''.join(sitestr))
        finallist.append(qbsitestr+sitestr)
    return finallist


def stringoperator(chain, weights):
    fval = 1
    print("str order param func ", chain, weights)
    weights = np.array(weights)
    #     weights=weights/np.sum(weights)
    #     print(np.sum(weights))
    finalres = 0
    for i in range(len(chain)):
        fval = 1

        if chain[i][0] == "-":
            fval = -1
        elif chain[i][0] == "0":
            fval = 0

        if chain[i][-1] == "-":
            fval = fval * (-1)
        elif chain[i][-1] == "0":
            fval = 0

        if fval != 0:
            for j in range(len(chain[i]) - 1):
                if j != 0:
                    if chain[i][j] == "-":
                        res = -1
                    elif chain[i][j] == "0":
                        res = 0
                    else:
                        res = 1
                    fval = fval * np.exp((1j) * np.pi * res)

            finalres = finalres + (fval * weights[i] ** 2)

    print("str len: ",len(chain[0])," string order param: ", finalres)
    return finalres


def makedictionnary(test_keys, test_values):
    res = {}
    for key in test_keys:
        for value in test_values:
            res[key] = value
            test_values.remove(value)
            break
    return res


def statelist(stateop, numberofqubits, numberofmodes, qbinist, samestallmodes, diffstallmodes, modeinichoice, cutoff):
    st = np.array(stateop) #convert state to np.array
    amp = []

    # What states the qubits and modes are initialised to - if they are initialised with an x-gate in the circuit it won't show up here, this is only what goes into circuit.cv_initialize()
    iniq = [qbinist] * numberofqubits # list of length number of qubits initialised to qbinist
    modesini = ""
    for i in range(len(iniq)):
        modesini = modesini + str(iniq[i]) #create a string from it
    modesini = modesini + " "
    if modeinichoice == "samestallmodes":
        inim = [samestallmodes] * numberofmodes
        for i in range(len(inim)):
            modesini = modesini + str(inim[i])
    else:
        for i in range(len(diffstallmodes)):
            modesini = modesini + str(diffstallmodes[i])

    # print("beginning", numberofmodes)
    chain=[]
    weights=[]
    for i in range(len(st)):
        res = st[i]
        if (np.abs(np.real(res)) > 1e-10):
            pos=i
            # print("position of non-zero real: ", pos, " res = ", res)
            qbst=np.empty(numberofqubits, dtype='int')
            iqb=0
            sln=len(st)
            while(iqb<numberofqubits):
                if pos<sln/2:
                    qbst[iqb]=int(0)
                else:
                    qbst[iqb]=int(1)
                    pos=pos-(sln/2)
                    # print("pos (sln/2)", pos, "sln ",sln)
                sln=sln/2
                iqb=iqb+1
            qbstr = ["".join(item) for item in qbst.astype(str)]
            # print("which half of the kronecker, ie. state of qubit: ", qbst)
            # print(modesini, " overlap with ",qbst[0], " is: ", np.real(res))
            qbsitestr = ""
            if qbst[1]==0:
                if qbst[2]==0:
                    qbsitestr=qbsitestr + "+"
                else:
                    qbsitestr = qbsitestr + "0"
            elif qbst[1]==1:
                if qbst[2]==1:
                    qbsitestr=qbsitestr + "-"
                else:
                    qbsitestr=qbsitestr + "0"

            # print("Qmode detector")
            qmst=np.empty(numberofmodes, dtype='int')
            # print("qmst starting in ", qmst)
            iqm=0
            # print("position is now: ",pos)
            while(iqm<numberofmodes):
                # print("mode counter iqm ", iqm)
                # print("cutoff ", cutoff)
                # print("length of vector left to search: sln ", sln)
                lendiv=sln/cutoff
                # print("lendiv (sln/cutoff)", lendiv)
                val=pos/lendiv
                # print("rough estimate of the position of the non-zero element: val (pos/lendiv) ", val)
                fock = math.floor(val)
                # print("Fock st/ resulting position in Kronecker product (math.floor(val)) ", fock)
                qmst[-iqm-1]=int(fock)
                # print("Storing that fock state: qmst ", qmst)
                # if val==math.ceil(val):
                #     print("value is val = ceil.val ",val)
                #     rdval=val-1
                # else:
                # print("remove a number of divisions corresponding to fock")
                pos=pos-(fock*lendiv)
                # print("new position for next order of depth of Kronecker product/pos: (pos-(rdiv*lendiv)) ",pos)
                sln=sln-((cutoff-1)*lendiv)
                # print("New length of vector left to search: sln (sln-((cutoff-1)*lendiv))", sln)
                iqm=iqm+1
            # print("qumode states at the end of one number's worth of searching: ", qmst)

            # sbstr = ["".join(item) for item in qmst.astype(str)]
            sitestr = qbsitestr
            for site in range(numberofmodes):
                if (site % 2 == 0):
                    if qmst[site]==0 :
                        sitestr=sitestr+"-"
                    elif qmst[site]==2 :
                        sitestr=sitestr+"+"
                    elif qmst[site] == 1 & qmst[site + 1] == 1:
                        sitestr=sitestr+"0"

            # print(qbsitestr, sitestr, "     is: ", np.real(res))#, "\n",''.join(qbstr), ''.join(sbstr))
            chain.append(sitestr)
            weights.append(np.real(res))

    remember=[]
    for i in range(len(chain)):
        for j in range(i+1,len(chain)):
            if chain[i]==chain[j]:
                # print("i ",i,"j",j,chain[i], chain[j], weights[i], weights[j])
                weights[i]=(1/np.sqrt(2))*(weights[i]+weights[j])
                remember.append(j)

    for j in range(len(remember)):
        # print("j",j,"remember[j]",remember[j])
        # print("remove",remember[j]-j)
        weights.remove(weights[remember[j]-j])
        chain.remove(chain[remember[j]-j])

    return [chain,weights]
