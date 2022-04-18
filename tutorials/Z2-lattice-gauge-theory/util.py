from qutip import *
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from quspin.operators import hamiltonian  # operators
from quspin.basis import boson_basis_1d  # Hilbert space boson basis
from quspin.basis import tensor_basis, spinless_fermion_basis_1d  # Hilbert spaces
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
import matplotlib.pyplot as plt
from quspin.tools.measurements import obs_vs_time  # t_dep measurements


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

def gauge_invariant_correlator(hopping_strength, field_strength, Nsites, psi0_notgaugefixed, Nbosons, basis):
    resRe = np.empty([Nsites, Nsites])
    resRe.fill(-1)
    resIm = np.empty([Nsites, Nsites])
    for l in range(Nsites):
        for i in range(0, Nsites - l):
            hop = [1.0]
            for add in range(i):
                hop.append(l + add)
            hop.append(l)
            hop.append(l + i)
            static = [["z" * i + "|+-", [hop]], ["z" * i + "|+-", [hop]]]
            no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
            correlator = hamiltonian(static, [], basis=basis, **no_checks).tocsr()
            # correlator = P_sparse@correlator@P_sparse.T.conj()
            H_expt = np.dot(psi0_notgaugefixed.T.conj(), correlator @ psi0_notgaugefixed)
            # print(H_expt[0,0],E)
            resRe[l][i] = np.real(H_expt[0, 0])
            resIm[l][i] = np.imag(H_expt[0, 0])

    # from matplotlib import cm
    print(resRe.T)
    plt.imshow(resRe.T)  # ,cmap=cm.Reds)#np.flip(resRe,0))
    plt.colorbar()
    plt.clim(0, resRe.max())
    plt.xlabel("position (i)")
    plt.ylabel("length (j-i)")
    plt.axis([-0.5, Nbosons - 0.5, -0.5, Nbosons - 0.5])
    plt.title("$a^{\dagger}_iZ_i...Z_{j-1}\:a_j$+h.c., $J =$ " + str(hopping_strength) + ", $\lambda =$ " + str(
        field_strength) + ", " + str(Nbosons) + " bosons")
    plt.show()
    # plt.imshow(np.flip(resIm,0))
    # plt.colorbar()
    # plt.show()


def pairing_correlator(hopping_strength, field_strength, Nsites, psi0_notgaugefixed, Nbosons, basis):
    resRe = np.zeros([Nsites, Nsites])
    resIm = np.zeros([Nsites, Nsites])
    for i in range(Nsites):
        for j in range(Nsites):
            pairing = [1.0]
            pairing.append(i)
            pairing.append(i)
            pairing.append(j)
            pairing.append(j)
            static = [["|--++", [pairing]], ["|++--", [pairing]]]
            no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
            correlator = hamiltonian(static, [], basis=basis, **no_checks).tocsr()
            # correlator = P_sparse@correlator@P_sparse.T.conj()
            # H_expt = np.dot(psi0.T.conj(),correlator@psi0)
            H_expt = np.dot(psi0_notgaugefixed.T.conj(), correlator @ psi0_notgaugefixed)
            resRe[i][j] = np.real(H_expt[0, 0])
            resIm[i][j] = np.imag(H_expt[0, 0])

    plt.imshow(resRe)
    plt.colorbar()
    plt.xlabel("position (i)")
    plt.ylabel("position (j)")
    plt.clim(0, resRe.max())
    # plt.axis([-0.5, Nbosons-0.5, -0.5, Nbosons-0.5])
    plt.title("$a_ia_ia^{\dagger}_ja^{\dagger}_j$+h.c., $J =$ " + str(hopping_strength) + ", $\lambda =$ " + str(
        field_strength) + ", " + str(Nbosons) + " bosons")
    plt.show()


def pairing_length(hopping_strength, field_strength, Nsites, psi0_notgaugefixed, Nbosons, basis):
    resRe = np.empty([Nsites, Nsites])
    resRe.fill(-1)
    resIm = np.empty([Nsites, Nsites])
    for i in range(Nsites):
        for l in range(Nsites - i):
            pairing = [1.0]
            pairing.append(i)
            pairing.append(i)
            pairing.append(i + l)
            pairing.append(i + l)
            static = [["|--++", [pairing]], ["|++--", [pairing]]]
            no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
            correlator = hamiltonian(static, [], basis=basis, **no_checks).tocsr()
            # correlator = P_sparse@correlator@P_sparse.T.conj()
            # H_expt = np.dot(psi0.T.conj(),correlator@psi0)
            H_expt = np.dot(psi0_notgaugefixed.T.conj(), correlator @ psi0_notgaugefixed)
            resRe[l][i] = np.real(H_expt[0, 0])
            resIm[l][i] = np.imag(H_expt[0, 0])

    plt.imshow(resRe)
    plt.colorbar()
    plt.xlabel("position (i)")
    plt.ylabel("length (j-i)")
    plt.clim(0, resRe.max())
    plt.axis([-0.5, Nbosons - 0.5, -0.5, Nbosons - 0.5])
    plt.title("$a_ia_ia^{\dagger}_ja^{\dagger}_j$+h.c., $\lambda \longrightarrow 0$, " + str(Nbosons) + " bosons")
    plt.show()


def energy_gap(set, lab, min, max, L_modes, L_spin, P_sparse):
    numberofvalues = 50
    vals=np.linspace(min,max,numberofvalues)

    # Different system sizes and number of bosons - choose number of bosons to be equal to the system size
    deltas=np.zeros((2,len(vals)))
    energy0=np.zeros((2,len(vals)))
    energy1=np.zeros((2,len(vals)))
    for i in range(len(vals)):
        val=vals[i]
        ##### create model
        hop=[[-0.1,i,i,i+1] for i in range(L_modes-1)]
        # density = [[0,i,i] for i in range(L_modes)]
        field = [[val,i] for i in range(L_spin)]
        static=[["z|+-",hop],["z|-+",hop],["x|",field]]#,["|nn",density]]
        ###### setting up operators
        # set up hamiltonian dictionary and observable (imbalance I)
        no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
        H = hamiltonian(static,[],basis=basis,**no_checks)
        H_sparse = H.tocsr()

        Hgaugefixed=P_sparse@H_sparse@P_sparse.T.conj()
        print("done ")
        E,V = eigsh(Hgaugefixed,k=2,which='SA')
        delta=np.abs(E[1]-E[0])
        if val==0:
            print("E[0] ",E[0]," E[1] ",E[1]," delta ",delta)
        deltas[0][i]=np.abs(val)
        deltas[1][i]=np.abs(delta)

        energy0[0][i]=np.abs(val)
        energy0[1][i]=E[0]

        energy1[0][i]=np.abs(val)
        energy1[1][i]=E[1]

    set.append(deltas)

    return [set, energy0, energy1]

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

def occupations_densities(Nbosons, psi0_notgaugefixed, P_sparse, psi0):
    for i in range(Nbosons):
        n = [[1.0,i]]  # second index chooses which spin or mode to check (ie. 0 is the 1st mode, 1 is the second and same for spins)
        static = [["|n", n]]  # z| checks magnetization of spins, |n checks boson number in modes
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        n_check = hamiltonian(static, [], basis=basis, **no_checks)
        n_sparse = n_check.tocsr()
        n_gf = P_sparse @ n_sparse @ P_sparse.T.conj()
        # O_n = np.dot(psi0[:, 6].conj().T, n_gf.dot(psi0[:, 6])) # Just checking the 6th eigenstate or so
        O_n = np.dot(psi0.conj().T, n_gf.dot(psi0))
        print("occupation of mode ",i,":",O_n)
        n = [[1.0, i, i]]
        static = [["|nn", n]]
        n_check = hamiltonian(static, [], basis=basis, **no_checks)
        n_sparse = n_check.tocsr()
        n_gf = P_sparse @ n_sparse @ P_sparse.T.conj()
        O_n2 = np.dot(psi0.conj().T, n_gf.dot(psi0))
        print(O_n2)
        print("density of mode ",i,":", 1 + (2 * np.abs(O_n)) + (np.abs(O_n2)))

    # for i in range(Nbosons):
    #     obs_args = {"basis": basis, "check_herm": False, "check_symm": False}
    #     n = hamiltonian([["|n", [[1.0, i]]]], [], dtype=np.float64, **obs_args)
    #     Obs_t = obs_vs_time(psi0_notgaugefixed, t, {"n": n})
    #     O_n = Obs_t["n"]
    #     print("mode number: ", i, ", occupation: ", np.real(O_n))

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