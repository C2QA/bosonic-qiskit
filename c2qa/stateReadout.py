import copy
# import c2qa
import qiskit
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csr_matrix, block_diag
import math
import itertools

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
    print("Qubits and modes initialised in: ",modesini,"\n")

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
            print("which half of the kronecker, ie. state of qubit: ", qbst)
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


def getDuplicatesWithInfo(listOfElems):
    ''' Get duplicate element in a list along with thier indices in list
     and frequency count'''
    dictOfElems = dict()
    index = 0
    # Iterate over each element in list and keep track of index
    for elem in listOfElems:
        # If element exists in dict then keep its index in lisr & increment its frequency
        if elem in dictOfElems:
            dictOfElems[elem][0] += 1
            dictOfElems[elem][1].append(index)
        else:
            # Add a new entry in dictionary
            dictOfElems[elem] = [1, [index]]
        index += 1

    dictOfElems = {key: value for key, value in dictOfElems.items() if value[0] > 1}
    return dictOfElems

def clean(chain, weights):
    duplicateinfo=getDuplicatesWithInfo(chain)
    remember = list(duplicateinfo.values())
    print("new func ", remember)

    # remember=[]
    # for i in range(len(chain)):
    #     for j in range(i+1,len(chain)):
    #         if chain[i]==chain[j]:
    #             print("i ",i,"j",j,chain[i], chain[j], weights[i], weights[j])
    #             weights[i]=(1/np.sqrt(2))*(weights[i]+weights[j])
    #             remember.append(j)

    for i in range(len(remember)):
        print("i: ", i," remember: ", remember[i])
        ind=int(remember[i][1][0])
        print("ind",ind)
        j=1
        indj=int(remember[i][1][j])
        print("indj",indj)
        print("weights[ind] is now: ",weights[ind])
        print("weights at indices indj: ", weights[indj])
        weights[indj] = weights[indj] + weights[ind]
        print("weights[indj] is now: ",weights[indj])
        weights.remove(weights[ind])
        chain.remove(chain[ind])

        print(chain, weights)

        if len(remember[i][1])>2:
            print("bigger longer")
            ind=int(remember[i][1][2])
            print("ind", ind)
            j=3
            indj=int(remember[i][1][j])
            print("indj", indj)
            print("weights[ind] is now: ", weights[ind])
            print("weights at indices indj: ", weights[indj])
            weights[indj] = weights[indj] + weights[ind]
            print("weights[indj] is now: ", weights[indj])
            weights.remove(weights[ind])
            chain.remove(chain[ind])
            print(chain, weights)

    # for i in range(len(remember)):
    #     print(i, " ", remember[i])
    #     for j in range(1,len(remember[i])):
    #         print(j, " ", remember[i][1][j])
    #         print("j",j,"remember[j]",remember[i][1][j])
    #         print("remove",remember[i][1][j]-j)
    #         weights.remove(weights[remember[i][1][j]-j])
    #         chain.remove(chain[remember[i][1][j]-j])

    return [chain,weights]

def stringoperator(chain, weights,tripletcounts):
    fval = 1
    # print("str order param func ", chain, weights)
    weights = np.array(weights)/tripletcounts
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
            for j in range(1,len(chain[0]) - 1):
                if chain[i][j] == "-":
                    res = -1
                elif chain[i][j] == "0":
                    res = 0
                else:
                    res = 1
                fval = fval * np.exp((1j) * np.pi * res)

            finalres = finalres + (fval * weights[i])# ** 2)

    print("ch. len: ",len(chain[0])," str. order param.: ", finalres)
    return finalres


def stringoperator_variable(chain, weights,tripletcounts,d=0):
    if d == 0:
        d = len(chain[0])-1

    fval = 1
    # print("str order param func ", chain, weights)
    weights = np.array(weights)/tripletcounts
    finalres = 0
    for i in range(len(chain)):
        fval = 1

        if chain[i][0] == "-":
            fval = -1
        elif chain[i][0] == "0":
            fval = 0

        if chain[i][d] == "-":
            fval = fval * (-1)
        elif chain[i][d] == "0":
            fval = 0

        if fval != 0:
            for j in range(1,d):
                if chain[i][j] == "-":
                    res = -1
                elif chain[i][j] == "0":
                    res = 0
                else:
                    res = 1
                fval = fval * np.exp((1j) * np.pi * res)

            finalres = finalres + (fval * weights[i])# ** 2)

    print("ch. len: ",len(chain[0])," str. len: ",d+1," str. order param.: ", finalres)
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

def changeBasis(exp_counts, Nsites, splitup=0):
    strList = ['+', '-', '0', 's']
    if splitup:
        tripdict = {}
        singdict = {}
        measdict = {}
    else:
        fulldict = {}
    encBasis = {'+': [[1], ['0010']], '-': [[1], ['1000']], '0': [[1], ['0101']], 's': [[1], ['0000']]}
    natBasis = {'+': [[1], ['00']], '-': [[1], ['11']], '0': [[1 / np.sqrt(2), 1 / np.sqrt(2)], ['01', '10']],
                's': [[1 / np.sqrt(2), -1 / np.sqrt(2)], ['01', '10']]}
    totStates = 4 ** Nsites
    for i in np.arange(totStates):
        spinlist = [int(p) for p in np.base_repr(i, base=4)]
        while len(spinlist) < Nsites:
            spinlist.insert(0, 0)
        coeff_str = [1]
        bit_str = ['']
        spin1_str = ''
        for n in np.arange(Nsites):
            if n == 0:  # Use natBasis
                state = strList[spinlist[n]]
                spin1_str = spin1_str + state
                coeff_str = ([i * j for i, j in itertools.product(coeff_str, natBasis[state][0])])
                bit_str = ([i + j for i, j in itertools.product(bit_str, natBasis[state][1])])
            else:  # use encBasis
                state = strList[spinlist[n]]
                spin1_str = spin1_str + state
                coeff_str = ([i * j for i, j in itertools.product(coeff_str, encBasis[state][0])])
                bit_str = ([i + j for i, j in itertools.product(bit_str, encBasis[state][1])])
        # Compute counts -- uses first bit for measurement of singlet or triplet.
        # print(spin1_str)
        # print(coeff_str)
        # print(bit_str)
        if spin1_str[0] == 's':
            if np.all(['1' + bits + ' 1' in exp_counts.keys() for bits in bit_str]):
                if splitup:
                    singdict[spin1_str] = sum([exp_counts['1' + bits + ' 1'] for bits, c in zip(bit_str, coeff_str)])
                else:
                    fulldict[spin1_str] = sum([exp_counts['1' + bits + ' 1'] for bits, c in zip(bit_str, coeff_str)])
            # else:
                # print('key ' + str(['1' + bits + ' 1' in exp_counts.keys() for bits in bit_str]) + ' ' + str(['1' + bits for bits in bit_str]))
        else:
            if np.all(['0' + bits + ' 0' in exp_counts.keys() for bits in bit_str]):
                if splitup:
                    tripdict[spin1_str] = sum([exp_counts['0' + bits + ' 0'] for bits, c in zip(bit_str, coeff_str)])
                else:
                    fulldict[spin1_str] = sum([exp_counts['0' + bits + ' 0'] for bits, c in zip(bit_str, coeff_str)])
            # else:
                # print('key ' + str(['0' + bits + ' 0' in exp_counts.keys() for bits in bit_str]) + ' ' + str(['1' + bits + ' 1' for bits in bit_str]))
        # print('-----')
    if splitup:
        measdict['singlet'] = sum(singdict.values())
        measdict['triplet'] = sum(tripdict.values())
        return [tripdict, singdict, measdict]
    else:
        return fulldict

def inistateread(numberofqubits, numberofmodes, qbinist, samestallmodes, diffstallmodes, modeinichoice):

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
    print("Qubits and modes initialised in: ",modesini,"\n")

def statereadbasic(stateop, numberofqubits, numberofmodes, qbinist, samestallmodes, diffstallmodes, modeinichoice, cutoff):
    st = np.array(stateop) #convert state to np.array
    amp = []

    for i in range(len(st)):
        res = st[i] # go through vector element by element and find the positions of the non-zero elements with next if clause
        if (np.abs(np.real(res)) > 1e-10):
            pos=i # position of amplitude (non-zero real)
            # print("position of non-zero real amplitude: ", pos, " res = ", res)
            sln = len(st)  # length of the state vector

            ## Find the qubit states
            qbst=np.empty(numberofqubits, dtype='int') # stores the qubit state
            iqb=0 # counts up until the total number of qubits is reached
            # which half of the vector the amplitude is in is the state of the first qubit because of how the kronecker product is made
            while(iqb<numberofqubits):
                if pos<sln/2: # if the amplitude is in the first half of the state vector or remaining statevector
                    qbst[iqb]=int(0) # then the qubit is in 0
                else:
                    qbst[iqb]=int(1) # if the amplitude is in the second half then it is in 1
                    pos=pos-(sln/2) # if the amplitude is in the second half of the statevector, then to find out the state of the other qubits and cavities then we remove the first half of the statevector for simplicity because it corresponds to the qubit being in 0 which isn't the case.
                    # print("pos (sln/2)", pos, "sln ",sln)
                sln=sln/2 # only consider the part of the statevector corresponding to the qubit state which has just been discovered
                iqb=iqb+1 # up the qubit counter to start finding out the state of the next qubit
            qbstr = ["".join(item) for item in qbst.astype(str)]

            ## Find the qumode states
            qmst=np.empty(numberofmodes, dtype='int') # will contain the Fock state of each mode
            # print("qmst starting in ", qmst)
            iqm=0 # counts up the number of modes
            # print("position is now: ",pos)
            while(iqm<numberofmodes):
                # print("mode counter iqm ", iqm)
                # print("cutoff ", cutoff)
                # print("length of vector left to search: sln ", sln)
                lendiv=sln/cutoff # length of a division is the length of the statevector divided by the cutoff of the hilbert space (which corresponds to the number of fock states which a mode can have)
                # print("lendiv (sln/cutoff)", lendiv)
                val=pos/lendiv
                # print("rough estimate of the position of the non-zero element: val (pos/lendiv) ", val)
                fock = math.floor(val)
                # print("Fock st resulting position in Kronecker product (math.floor(val)) ", fock)
                qmst[iqm]=fock
                pos=pos-(fock*lendiv) #remove a number of divisions to then search a subsection of the Kronecker product
                # print("new position for next order of depth of Kronecker product/pos: (pos-(fock*lendiv)) ",pos)
                sln=sln-((cutoff-1)*lendiv) # New length of vector left to search
                # print("New length of vector left to search: sln (sln-((cutoff-1)*lendiv))", sln)
                iqm=iqm+1
            qmstr = ["".join(item) for item in qmst.astype(str)]

            print("qubits: ",''.join(qbstr), " qumodes: ",''.join(qmstr), "    with amplitude: ", np.real(res))

    if (np.abs(np.imag(res)) > 1e-10):
        print("\n imaginary amplitude: ", 1j * np.imag(res))