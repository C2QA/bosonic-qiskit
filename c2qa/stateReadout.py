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
    st = np.array(stateop)
    amp = []
    if modeinichoice == "samestallmodes":
        inim = [samestallmodes] * numberofmodes
        modesini = str(qbinist) + " "
        for i in range(len(inim)):
            modesini = modesini + str(inim[i])
    else:
        modesini = str(qbinist) + " "
        for i in range(len(diffstallmodes)):
            modesini = modesini + str(diffstallmodes[i])

    # print("beginning", numberofmodes)
    for i in range(len(st)):
        res = st[i]
        if (np.abs(np.real(res)) > 1e-10):
            pos=i
            # print("position of non-zero real: ", pos, " res = ", res)
            qbst=np.zeros([numberofqubits])
            iqb=0
            sln=len(st)
            while(iqb<numberofqubits):
                if pos<sln/2:
                    qbst[iqb]="0"
                else:
                    qbst[iqb]="1"
                    pos=pos-(sln/2)
                    # print("pos (sln/2)", pos, "sln ",sln)
                sln=sln/2
                iqb=iqb+1
            # print("which half of the kronecker, ie. state of qubit: ", qbst)
            # print(modesini, " overlap with ",qbst[0], " is: ", np.real(res))

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

            print(modesini, " overlap with ", int(qbst[0]), ''.join(sbstr), "  is: ", np.real(res))

    # print("end")

    # if (np.abs(np.imag(res)) > 1e-10):
    #     print(modesini, " overlap with ", " is: ", 1j * np.imag(res))
