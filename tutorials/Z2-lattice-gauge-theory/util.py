from qutip import *
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from quspin.operators import hamiltonian  # operators
from quspin.basis import boson_basis_1d  # Hilbert space boson basis
from quspin.basis import tensor_basis, spinless_fermion_basis_1d  # Hilbert spaces
from quspin.basis import spin_basis_1d  # Hilbert space spin basis

def build_H(hopping_strength, field_strength, L_modes, L_spin, P_sparse, basis):
    hop = [[hopping_strength, i, i, i + 1] for i in range(L_modes - 1)]
    # hop+=[[-1.0,L_modes-1,L_modes-1,0]]
    field = [[field_strength, i] for i in range(L_spin)]
    static = [["z|+-", hop], ["z|-+", hop], ["x|", field]]
    ###### setting up operators
    # set up hamiltonian dictionary and observable (imbalance I)
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    H = hamiltonian(static, [], basis=basis, **no_checks)
    H_sparse = H.tocsr()
    Hgaugefixed = P_sparse @ H_sparse @ P_sparse.T.conj()

    return Hgaugefixed

def state_checks(psi0, P_sparse):
    # Relative phases for a two site system
    print(np.angle(psi0[0])-np.angle(psi0[1]),np.angle(psi0[2])-np.angle(psi0[1]),np.angle(psi0[0])-np.angle(psi0[2]))
    # Normalised
    print(abs(psi0.T.conj()@psi0)**2)
    # Real?
    np.allclose(P_sparse.T.conj() @ psi0, P_sparse.T @ psi0)

def build_state_manually(spins_index, bosons_index, basis_spin, basis_boson, ):
    ##### define initial state #####
    #Spin - find index of spin state |01>
    spins_index = "01"
    bosons_index = "02"
    ispin = basis_spin.index(spins_index)
    #Boson - find index of Fock state |20>
    iboson = basis_boson.index(bosons_index)
    # Ns is the size of the Hilbert space
    psispin = np.zeros(basis_spin.Ns,dtype=np.float64) # for 2 bosons in 2 modes Ns=3 ("20","11","02")
    psispin[ispin] = 1.0
    # Or
    # psispin = (1 / np.sqrt(2)) * np.array([1, 1])
    psiboson = np.zeros(basis_boson.Ns,dtype=np.float64)
    psiboson[iboson] = 1.0
    psi=np.kron(psispin,psiboson)

    return psi

def check_mode_occupation(psi, which_mode_to_check):
    field = [[1.0,which_mode_to_check]]  # second index chooses which spin or mode to check (ie. 0 is the 1st mode, 1 is the second and same for spins)
    static = [["|n", field]]  # |n checks boson number in modes
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    H_check = hamiltonian(static, [], basis=basis, **no_checks)
    res = np.dot(psi.conj().T, H_check.dot(psi))
    print(res)

    return res


def check_spin_value(psi, which_spin_to_check):
    field = [[1.0,
              which_spin_to_check]]  # second index chooses which spin or mode to check (ie. 0 is the 1st mode, 1 is the second and same for spins)
    static = [["z|", field]]  # z| checks magnetization of spins
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    H_check = hamiltonian(static, [], basis=basis, **no_checks)
    res = np.dot(psi.conj().T, H_check.dot(psi))
    print(res)

    return res



def flip(s):
    if s == '+':
        return '-'
    elif s == '-':
        return '+'


def isodd(n):
    return int(n) % 2 == True


def binom(n, k):
    return np.math.factorial(n) // np.math.factorial(k) // np.math.factorial(n - k)


def gaugeFixedBasis(Nsites, Nbosons):
    bosonStates = []
    gaugefixedStates = []
    # Let's first list out all possible boson occupations.
    # We can do this by looping through all numbers and putting it into base Nbosons
    for number in np.arange((Nbosons + 1) ** Nsites):
        bosonString = np.base_repr(number, base=Nbosons + 1)
        # print(bosonString)
        bosonString = '0' * (Nsites - len(bosonString)) + bosonString

        # check total boson number
        if sum([int(c) for c in bosonString]) == Nbosons:
            bosonStates.append(bosonString)

    # Now loop through each state and insert appropriate qubit state which fixes the gauge condition to +1
    for state in bosonStates:
        gaugefixedstate = ''
        for site in np.arange(len(state) - 1):
            thisn = state[site]
            gaugefixedstate += thisn
            if site == 0:  # For the first site
                thislink = '-' * (isodd(thisn)) + '+' * (not isodd(thisn))
            else:  # For all other sites
                if isodd(thisn):
                    thislink = flip(lastlink)
                else:
                    thislink = lastlink
            gaugefixedstate += thislink
            lastlink = thislink
        gaugefixedstate += state[-1]
        gaugefixedStates.append(gaugefixedstate)

    print(gaugefixedStates)
    return gaugefixedStates


# Now that we have the gauge fixed basis vectors, we could proceed in a few different ways. The harder
# thing would be to build the Hamiltonian and all operators explicitly in this basis. While probably
# more efficient for very large systems, we could also just build projectors that take us from
# the full Hilbert space down to the gauge fixed Hilbert space. Let's do that here in Qutip:

def siteState(c, Nbosons):
    # print("site state ", basis(Nbosons + 1, int(c)))
    return basis(Nbosons + 1, np.abs(Nbosons-int(c)))


def linkState(c):
    if c == '+':
        # print("link state ", (basis(2, 0) + basis(2, 1)).unit())
        return (basis(2, 0) + basis(2, 1)).unit()
    elif c == '-':
        return (basis(2, 0) - basis(2, 1)).unit()


def Projector_to_gauge_peserving_basis(Nsites, Nbosons):
    basisStatesList = gaugeFixedBasis(Nsites, Nbosons)
    # print(basisStatesList)
    # Build basis vectors in full Hilbert space
    fullBasis = []
    for state in basisStatesList:  # Loop through each basis state
        basisVector = []
        for ind in np.arange(len(state)):  # Loop through each site/link from left to right
            c = state[ind]
            # print("c ",c)
            if isodd(ind):
                basisVector.append(linkState(c))
        for ind in np.arange(len(state)):  # Loop through each site/link from left to right
            c = state[ind]
            if ind % 2 == 0:
                basisVector.append(siteState(c, Nbosons))

        # Now take tensor product to get the full basisVector
        fullBasis.append(tensor(basisVector))

    # Now build projectors onto the gauge fixed Hilbert space
    P_gaugefixed = 0
    for i in np.arange(len(fullBasis)):
        P_gaugefixed += basis(len(fullBasis), i) * fullBasis[i].dag()

    P_sparse = P_gaugefixed.data

    return P_sparse