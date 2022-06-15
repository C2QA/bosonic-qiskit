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
from quspin.tools.evolution import evolve  # ODE evolve tool



def build_H(hopping_strength, field_strength, density_strength, L_modes, L_spin, P_sparse, basis, periodicBC=True):
    hop = [[hopping_strength, i, i, (i + 1)%L_modes] for i in range(L_modes)]
    density = [[density_strength, i, i] for i in range(L_modes)]
    #if periodicBC==True:
    #    hop+=[[hopping_strength,L_modes,L_modes,0]]
    field = [[field_strength, i] for i in range(L_spin)]
    static = [["z|+-", hop], ["z|-+", hop], ["x|", field], ["|nn", density]]
    ###### setting up operators
    # set up hamiltonian dictionary and observable (imbalance I)
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    H = hamiltonian(static, [], basis=basis, **no_checks)
    H_sparse = H.tocsr()
    Hgaugefixed = P_sparse @ H_sparse @ P_sparse.T.conj()

    return Hgaugefixed

def free_boson_limit(L_modes, L_spin, P_sparse, Nsites, basis):
    # ED Z2 LGT
    elist = []
    hopping_strength = -1
    field_strength = 0
    Hgaugefixed = build_H(hopping_strength, field_strength, L_modes, L_spin, P_sparse, basis, True)
    # E, psi0 = eigsh(Hgaugefixed, k=20, which='SA')
    E, psi0 = np.linalg.eigh(Hgaugefixed.todense())
    elist.append(E)

    # Analytical free bosons
    elist_freebosons = []
    E_k = -2 * hopping_strength * np.cos(2 * np.pi * np.arange(0, L_modes, 1) / L_modes)
    elist_freebosons.append(E_k)

    # print(np.sort(E))
    # print(np.sort(E_k))
    # print(E[1]-E[0])
    # print(psi0)
    # psi0[:,6]

    for i in range(len(elist)):
        plt.plot(range(len(elist[i])), elist[i], ".", label="ED Z2LGT no field, 1 boson")
        plt.plot(range(len(elist_freebosons[i])), np.flip(elist_freebosons[i]), "x", label="Analytical, 1 free boson")
    # plt.xlabel("ED: Eigenenergy number/ Analytical: Momentum ($2\pi/L$)")
    plt.ylabel("Eigenenergy")
    plt.xlabel("ED: eigenenergy number/ analytical: momentum ($2\pi/L$)")
    plt.title(r"$J \greater \lambda$, " + str(Nsites) + " sites")
    plt.legend(loc="lower right")
    plt.show()

def paired_bosons(L_modes, cutoff, Nbosons):
    basis_boson = boson_basis_1d(L=L_modes, sps=cutoff, Nb=Nbosons)
    print(basis_boson)
    hop = [[-1, i, i, (i + 1) % L_modes, (i + 1) % L_modes] for i in range(L_modes)]
    static = [["++--", hop], ["--++", hop]]
    H = hamiltonian(static, [], basis=basis_boson, dtype=np.float64)
    print(H)
    E, psi0 = H.eigh()
    print("energy ", E)
    _, psi0_s = H.eigsh(k=1, which='SA')
    # same (there is global minus sign, but that doesn't matter)
    print("compare ", psi0[:, 0], " with ", np.round(psi0_s, 10)[:, 0])
    columns_states = np.round(psi0, 10)
    print(columns_states)

def paired_bosons_limit(L_modes, L_spin, P_sparse, Nsites, basis, cutoff, Nbosons, hopping_strength, field_strength, paired_hopping_only_strength):
    # ED Z2 LGT
    elist = []
    Hgaugefixed = build_H(hopping_strength, field_strength, L_modes, L_spin, P_sparse, basis, True)
    # E, psi0 = eigsh(Hgaugefixed, k=20, which='SA')
    E, psi0 = np.linalg.eigh(Hgaugefixed.todense())
    elist.append(E)

    # Analytical free bosons
    elist_freebosons = []
    E_k = -2 * 1 * np.cos(2 * np.pi * np.arange(0, L_modes, 1) / (2 * L_modes))
    elist_freebosons.append(E_k)

    # ED paired bosons
    elist_pairedbosons = []
    basis_boson2 = boson_basis_1d(L=L_modes, sps=cutoff, Nb=Nbosons)
    hop = [[paired_hopping_only_strength, i, i, (i + 1) % L_modes, (i + 1) % L_modes] for i in range(L_modes)]
    static = [["++--", hop], ["--++", hop]]
    hamiltonian_paired_hopping = hamiltonian(static, [], basis=basis_boson2, dtype=np.float64)
    print(hamiltonian_paired_hopping)
    E, psi0 = hamiltonian_paired_hopping.eigh()
    elist_pairedbosons.append(E)

    elist_pairedbosons = []
    basis_boson2 = boson_basis_1d(L=L_modes, sps=cutoff, Nb=Nbosons)
    print(basis_boson2)
    hop = [[paired_hopping_only_strength, i, i, (i + 1) % L_modes, (i + 1) % L_modes] for i in range(L_modes)]
    static = [["++--", hop], ["--++", hop]]
    hamiltonian_paired_hopping = hamiltonian(static, [], basis=basis_boson2, dtype=np.float64)
    E2, psi2 = hamiltonian_paired_hopping.eigh()
    print(psi2)  # [:,0])#, hamiltonian_paired_hopping.eigsh(k=1,which='SA'))
    elist_pairedbosons.append(E2)

    for i in range(len(elist)):
        plt.plot(range(len(elist[i])), elist[i], ".", label="ED Z2LGT, 2 bosons")
        plt.plot(range(len(elist_pairedbosons[i])), elist_pairedbosons[i], ".", label="ED paired bosons, 2 bosons")
        plt.plot(range(len(elist_freebosons[i])), np.flip(elist_freebosons[i]), "x", label="Analytical, 1 free boson")
    # plt.xlabel("ED: Eigenenergy number/ Analytical: Momentum ($2\pi/L$)")
    plt.ylabel("Eigenenergy")
    plt.xlabel("ED: eigenenergy number/ analytical: momentum ($2\pi/L$)")
    plt.title(r"$J \less \lambda$, " + str(Nsites) + " sites")
    plt.legend()
    plt.show()

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


def energy_scaling():
    Nsites = 4
    Nbosons = 4
    ###### parameters
    L_spin = Nsites
    L_modes = Nsites  # system size
    cutoff = Nbosons + 1  # sites+2
    h = 1  # field strength
    t = 1

    P_sparse = gaussProjector(Nsites, Nbosons)
    basis_spin = spin_basis_1d(L=L_spin)
    basis_boson = boson_basis_1d(L=L_modes, sps=cutoff)
    basis = tensor_basis(basis_spin, basis_boson)


    set = []
    lab = []
    lab.append(str(L_modes) + " sites, " + str(Nbosons) + " bosons")

    ###### parameters
    min = -5
    max = 0
    numberofvalues = 100
    vals = np.linspace(min, max, numberofvalues)
    # vals=np.logspace(-2,2,numberofvalues)

    # Different system sizes and number of bosons - choose number of bosons to be equal to the system size

    deltas = np.zeros((2, len(vals)))
    energy0 = np.zeros((2, len(vals)))
    energy1 = np.zeros((2, len(vals)))
    for i in range(len(vals)):
        val = vals[i]
        hopping_strength = val
        field_strength = 1
        Hgaugefixed = build_H(hopping_strength, field_strength, L_modes, L_spin, P_sparse, basis, True)
        E, V = eigsh(Hgaugefixed, k=2, which='SA')
        delta = np.abs(E[1] - E[0])
        if val == 0:
            print("E[0] ", E[0], " E[1] ", E[1], " delta ", delta)
        deltas[0][i] = val
        deltas[1][i] = delta

        energy0[0][i] = val
        energy0[1][i] = E[0]

        energy1[0][i] = val
        energy1[1][i] = E[1]

    set.append(deltas)

    xlabelplot = r"$|J/\lambda|$"
    plt.title("Z2LGT Energy gaps")
    for i in range(len(set)):
        plt.plot(energy0[0], energy0[1], label="E0 " + lab[i])
    for i in range(len(set)):
        plt.plot(energy1[0], energy1[1], label="E1 " + lab[i])
    plt.xlabel(xlabelplot)
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.show()

    plt.title("Z2LGT Energy gap")
    for i in range(len(set)):
        plt.plot(np.abs(set[i][0]), np.abs(set[i][1]), label=lab[i])
        plt.plot(np.abs(set[i][0]), np.abs(set[i][0] ** (2)), label="$x^{2}$")
        plt.plot(np.abs(set[i][0]), np.abs(set[i][0] ** (1)), label="$x$")
    plt.xlabel(xlabelplot)
    plt.ylabel("Energy gap")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(10 ** (-3))
    plt.xlim(0.05)
    plt.legend()
    plt.show()


def energy_gap(set, lab, min, max, L_modes, L_spin, P_sparse, basis, Nbosons):
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
    lab.append(str(L_modes) + " sites, " + str(Nbosons) + " bosons")

    return [set, energy0, energy1, lab]

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


def evolve_occupations(L_modes, L_spin, psi1):
    res = []
    N_timesteps = 20
    t0s = np.logspace(-3, 1, N_timesteps)
    for t0 in t0s:
        hop = [[-1.0, i, i, i + 1] for i in range(L_modes - 1)]
        field = [[-1, i] for i in range(L_spin)]
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        static = [["x|", field], ["z|+-", hop], ["z|-+", hop]]
        dynamic = []
        H = hamiltonian(static, dynamic, basis=basis, **no_checks)
        H = H.tocsr()

        def adiabatic(time, phi):
            phi_dot = -1j * H @ phi
            return phi_dot

        # psi_t=H.evolve(psi,0.0,[t0],iterate=False,rtol=1E-9,atol=1E-9)
        psi_t = evolve(psi1, 0.0, [t0], adiabatic, iterate=False, rtol=1E-9, atol=1E-9)

        obs_args = {"basis": basis, "check_herm": False, "check_symm": False}
        n = hamiltonian([["|n", [[1.0, 1]]]], [], dtype=np.float64, **obs_args)
        Obs_t = obs_vs_time(psi_t, t0s, {"n": n})
        O_n = Obs_t["n"]
        print(O_n)

        ##### plot results #####
        str_n = "$\\langle n\\rangle,$"
        fig = plt.figure()
        plt.plot(t0s, np.real(O_n), "k", linewidth=1, label=str_n)
        plt.xlabel("$t/T$", fontsize=18)
        # plt.ylim([-1.1,1.4])
        plt.legend(loc="upper right", ncol=5, columnspacing=0.6, numpoints=4)
        plt.tick_params(labelsize=16)
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig('example3.pdf', bbox_inches='tight')
        plt.show()
        # plt.close()

    # plt.pcolormesh(np.arange(Nsites + 1) - Nsites // 2 - 0.5, np.arange(N_timesteps + 1) * dt, occupations[0],
    #                cmap=matplotlib.cm.Blues, linewidth=0, rasterized=True)
    # plt.title("Mode occupation: gauge field flipping dominates")
    # plt.xlabel("Modes")
    # plt.ylabel("Time")
    # plt.colorbar()

def testing_functions(Nsites):
    for l in range(Nsites):
        for i in range(Nsites-l):
            hop=[1.0]
            for add in range(i):
                hop.append(l+add)
            hop.append(l)
            hop.append(l+i)
            print(hop)

    resli = np.zeros([Nsites, Nsites])
    for l in range(Nsites):
        for i in range(Nsites - l):
            hop = [1.0]
            for add in range(i):
                hop.append(l + add)
            hop.append(l)
            hop.append(l + i)
            resli[l][i] = i

    resli = np.delete(resli, 0, 1)

    resli = np.delete(resli, -1, 0)

    plt.imshow(np.flip(resli, 0))
    plt.colorbar()
    plt.show()

    for i in range(Nsites):
        for j in range(Nsites):
            pairing = [1.0]
            pairing.append(i)
            pairing.append(i)
            pairing.append(j)
            pairing.append(j)
            print(pairing)

def flip(s):
    if s == '+':
        return '-'
    elif s == '-':
        return '+'


def up(s):
    sint = int(s)
    return np.sqrt(sint + 1), str(sint + 1)


def down(s):
    sint = int(s)
    if sint <= 0:
        return 0, '0'
    else:
        return np.sqrt(sint), str(sint - 1)


def isodd(n):
    return int(n) % 2 == True


def binom(n, k):
    return np.math.factorial(n) // np.math.factorial(k) // np.math.factorial(n - k)


def gaugeFixedBasis(Nsites, Nbosons, periodicBC):
    import copy
    bosonStates = []
    gaugefixedStates = []
    # Let's first list out all possible boson occupations.
    # We can do this by looping through all numbers and putting it into base Nbosons
    for number in np.arange((Nbosons + 1) ** Nsites):
        bosonString = np.base_repr(number, base=Nbosons + 1)
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
        #         if periodicBC:
        # add final link
        thisn = state[-1]
        if isodd(thisn):
            thislink = flip(lastlink)
        else:
            thislink = lastlink
        gaugefixedstate += thislink
        gaugefixedStates.append(gaugefixedstate)
    # Above I made a choice. For OBC, this is fine, but for PBC there is a symmetry where you flip every single link. So we have to add those as well
    if periodicBC:
        subset_states = copy.deepcopy(gaugefixedStates)
        for state in subset_states:
            mirror = ''
            for c, ind in zip(state, np.arange(len(state))):
                if ind % 2:  # if odd
                    mirror += flip(c)
                else:
                    mirror += c
            gaugefixedStates.append(mirror)
    return gaugefixedStates

#         # START CHANGE ELLA
def siteState(c, Nbosons):
    # print("site state ", basis(Nbosons + 1, int(c)))
    return basis(Nbosons + 1, np.abs(Nbosons-int(c)))


def linkState(c):
    if c == '+':
        # print("link state ", (basis(2, 0) + basis(2, 1)).unit())
        return (basis(2, 0) + basis(2, 1)).unit()
    elif c == '-':
        return (basis(2, 0) - basis(2, 1)).unit()
#         # END CHANGE ELLA

def gaussProjector(Nsites, Nbosons):
    basisStates = gaugeFixedBasis(Nsites, Nbosons, periodicBC=True)
    # Build basis vectors in full Hilbert space
    fullBasis = []
    for state in basisStates:  # Loop through each basis state
        basisVector = []
        for ind in np.arange(len(state)):  # Loop through each site/link from left to right
            c = state[ind]
            # print("c ",c)
            if isodd(ind):
                basisVector.append(linkState(c))
        # START CHANGE ELLA
        for ind in np.arange(len(state)):  # Loop through each site/link from left to right
            c = state[ind]
            if ind % 2 == 0:
                basisVector.append(siteState(c, Nbosons))
        # END CHANGE ELLA
        # Now take tensor product to get the full basisVector
        fullBasis.append(tensor(basisVector))

    # Now build projectors onto the gauge fixed Hilbert space
    P_gaugefixed = 0
    for i in np.arange(len(fullBasis)):
        P_gaugefixed += basis(len(fullBasis), i) * fullBasis[i].dag()

    P_sparse = P_gaugefixed.data

    return P_sparse

# Kevin old code

# def flip(s):
#     if s == '+':
#         return '-'
#     elif s == '-':
#         return '+'
#
#
# def isodd(n):
#     return int(n) % 2 == True
#
#
# def binom(n, k):
#     return np.math.factorial(n) // np.math.factorial(k) // np.math.factorial(n - k)
#
#
# def gaugeFixedBasis(Nsites, Nbosons):
#     bosonStates = []
#     gaugefixedStates = []
#     # Let's first list out all possible boson occupations.
#     # We can do this by looping through all numbers and putting it into base Nbosons
#     for number in np.arange((Nbosons + 1) ** Nsites):
#         bosonString = np.base_repr(number, base=Nbosons + 1)
#         # print(bosonString)
#         bosonString = '0' * (Nsites - len(bosonString)) + bosonString
#
#         # check total boson number
#         if sum([int(c) for c in bosonString]) == Nbosons:
#             bosonStates.append(bosonString)
#
#     # Now loop through each state and insert appropriate qubit state which fixes the gauge condition to +1
#     for state in bosonStates:
#         gaugefixedstate = ''
#         for site in np.arange(len(state) - 1):
#             thisn = state[site]
#             gaugefixedstate += thisn
#             if site == 0:  # For the first site
#                 thislink = '-' * (isodd(thisn)) + '+' * (not isodd(thisn))
#             else:  # For all other sites
#                 if isodd(thisn):
#                     thislink = flip(lastlink)
#                 else:
#                     thislink = lastlink
#             gaugefixedstate += thislink
#             lastlink = thislink
#         gaugefixedstate += state[-1]
#         gaugefixedStates.append(gaugefixedstate)
#
#     print(gaugefixedStates)
#     return gaugefixedStates
#
#
# # Now that we have the gauge fixed basis vectors, we could proceed in a few different ways. The harder
# # thing would be to build the Hamiltonian and all operators explicitly in this basis. While probably
# # more efficient for very large systems, we could also just build projectors that take us from
# # the full Hilbert space down to the gauge fixed Hilbert space. Let's do that here in Qutip:
#
# def siteState(c, Nbosons):
#     # print("site state ", basis(Nbosons + 1, int(c)))
#     return basis(Nbosons + 1, np.abs(Nbosons-int(c)))
#
#
# def linkState(c):
#     if c == '+':
#         # print("link state ", (basis(2, 0) + basis(2, 1)).unit())
#         return (basis(2, 0) + basis(2, 1)).unit()
#     elif c == '-':
#         return (basis(2, 0) - basis(2, 1)).unit()
#
#
# def gaussProjector(Nsites, Nbosons):
#     basisStatesList = gaugeFixedBasis(Nsites, Nbosons)
#     # print(basisStatesList)
#     # Build basis vectors in full Hilbert space
#     fullBasis = []
#     for state in basisStatesList:  # Loop through each basis state
#         basisVector = []
#         for ind in np.arange(len(state)):  # Loop through each site/link from left to right
#             c = state[ind]
#             # print("c ",c)
#             if isodd(ind):
#                 basisVector.append(linkState(c))
#         for ind in np.arange(len(state)):  # Loop through each site/link from left to right
#             c = state[ind]
#             if ind % 2 == 0:
#                 basisVector.append(siteState(c, Nbosons))
#
#         # Now take tensor product to get the full basisVector
#         fullBasis.append(tensor(basisVector))
#
#     # Now build projectors onto the gauge fixed Hilbert space
#     P_gaugefixed = 0
#     for i in np.arange(len(fullBasis)):
#         P_gaugefixed += basis(len(fullBasis), i) * fullBasis[i].dag()
#
#     P_sparse = P_gaugefixed.data
#
#     return P_sparse