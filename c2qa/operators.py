import numpy
import scipy.sparse
import scipy.sparse.linalg


xQB = numpy.array([[0, 1], [1, 0]])
yQB = numpy.array([[0, -1j], [1j, 0]])
zQB = numpy.array([[1, 0], [0, -1]])
idQB = numpy.array([[1, 0], [0, 1]])
sigma_plus = numpy.array([[0, 1], [0, 0]])
sigma_minus = numpy.array([[0, 0], [1, 0]])


class CVOperators:
    """Build operator matrices for continuously variable bosonic gates."""

    def __init__(self, cutoff: int, num_qumodes: int):
        """Initialize shared matrices used in building operators.

        Args:
            cutoff (int): qumode cutoff level
            num_qumodes (int): number of qumodes being represented
        """
        # Annihilation operator
        data = numpy.sqrt(range(cutoff))
        self.a = scipy.sparse.spdiags(
            data=data, diags=[1], m=len(data), n=len(data)
        ).tocsc()

        # Creation operator
        self.a_dag = self.a.conjugate().transpose().tocsc()

        # Number operator for a single qumode.
        # self.N = scipy.sparse.matmul(self.a_dag, self.a)
        self.N = self.a_dag * self.a

        self.eye = scipy.sparse.eye(cutoff)

        self.cutoff_value = cutoff

    def id(self):
        """Identity gate (used by cv_delay)

        Args:
            None

        Returns:
            dia_matrix: operator matrix
        """

        return self.eye

    def r(self, theta):
        """Phase space rotation operator

        Args:
            theta (real): rotation

        Returns:
            csc_matrix: operator matrix
        """
        arg = 1j * theta * self.N

        return scipy.sparse.linalg.expm(arg)

    def d(self, alpha):
        """Displacement operator

        Args:
            alpha (real): displacement

        Returns:
            csc_matrix: operator matrix
        """
        arg = (alpha * self.a_dag) - (numpy.conjugate(alpha) * self.a)

        return scipy.sparse.linalg.expm(arg)

    def s(self, theta):
        """Single-mode squeezing operator

        Args:
            theta (real): squeeze

        Returns:
            csc_matrix: operator matrix
        """
        a_sqr = self.a * self.a
        a_dag_sqr = self.a_dag * self.a_dag
        arg = 0.5 * ((numpy.conjugate(theta) * a_sqr) - (theta * a_dag_sqr))

        return scipy.sparse.linalg.expm(arg)

    def s2(self, theta):
        """Two-mode squeezing operator

        Args:
            g (real): multiplied by 1j to yield imaginary phase

        Returns:
            csc_matrix: operator matrix
        """

        self.a1 = scipy.sparse.kron(self.a, self.eye).tocsc()
        self.a2 = scipy.sparse.kron(self.eye, self.a).tocsc()
        self.a1_dag = self.a1.conjugate().transpose().tocsc()
        self.a2_dag = self.a2.conjugate().transpose().tocsc()

        a12_dag = self.a1_dag * self.a2_dag
        a12 = self.a1 * self.a2

        arg = (numpy.conjugate(theta * 1j) * a12_dag) - (theta * 1j * a12)

        return scipy.sparse.linalg.expm(arg)

    def bs(self, theta):
        """Two-mode beam splitter operator

        Args:
            theta: phase

        Returns:
            csc_matrix: operator matrix
        """

        self.a1 = scipy.sparse.kron(self.a, self.eye).tocsc()
        self.a2 = scipy.sparse.kron(self.eye, self.a).tocsc()
        self.a1_dag = self.a1.conjugate().transpose().tocsc()
        self.a2_dag = self.a2.conjugate().transpose().tocsc()

        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        arg = theta * a1dag2 - numpy.conj(theta) * a12dag

        return scipy.sparse.linalg.expm(arg)

    def cr(self, theta):
        """Controlled phase space rotation operator

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(zQB, self.N).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def crx(self, theta):
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(xQB, self.N).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def cry(self, theta):
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(yQB, self.N).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def cd(self, theta, beta=None):
        """Controlled displacement operator

        Args:
            theta (real): displacement for qubit state 0
            beta (real): displacement for qubit state 1. If None, use -alpha.

        Returns:
            bsr_matrix: operator matrix
        """
        displace0 = (theta * self.a_dag) - (numpy.conjugate(theta) * self.a)
        if beta is None:
            beta = -theta
        displace1 = (beta * self.a_dag) - (numpy.conjugate(beta) * self.a)

        return scipy.sparse.kron(
            (idQB + zQB) / 2, scipy.sparse.linalg.expm(displace0)
        ) + scipy.sparse.kron((idQB - zQB) / 2, scipy.sparse.linalg.expm(displace1))

    def ecd(self, theta):
        """Echoed controlled displacement operator

        Args:
            theta (real): displacement

        Returns:
            csr_matrix: operator matrix
        """
        argm = (theta * self.a_dag) - (numpy.conjugate(theta) * self.a)
        arg = scipy.sparse.kron(zQB, argm)

        return scipy.sparse.linalg.expm(arg)

    def cbs(self, theta):
        """Controlled phase two-mode beam splitter operator

        Args:
            theta (real): real phase

        Returns:
            csc_matrix: operator matrix
        """
        self.a1 = scipy.sparse.kron(self.a, self.eye).tocsc()
        self.a2 = scipy.sparse.kron(self.eye, self.a).tocsc()
        self.a1_dag = self.a1.conjugate().transpose().tocsc()
        self.a2_dag = self.a2.conjugate().transpose().tocsc()

        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        argm = theta*a1dag2 - numpy.conjugate(theta)*a12dag
        arg = scipy.sparse.kron(zQB, argm).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def cschwinger(self, beta, theta_1, phi_1, theta_2, phi_2):
        """General form of a controlled Schwinger gate

        Args:
            params (real): [beta, theta_1, phi_1, theta_2, phi_2]

        Returns:
            csc_matrix: operator matrix
        """
        self.a1 = scipy.sparse.kron(self.a, self.eye).tocsc()
        self.a2 = scipy.sparse.kron(self.eye, self.a).tocsc()
        self.a1dag = self.a1.conjugate().transpose().tocsc()
        self.a2dag = self.a2.conjugate().transpose().tocsc()

        a12dag = self.a1 * self.a2dag
        a1dag2 = self.a1dag * self.a2

        Sx = (self.a1 * self.a2dag + self.a1dag * self.a2)/2
        Sy = (self.a1 * self.a2dag - self.a1dag * self.a2)/(2*1j)
        Sz = (self.a2dag * self.a2 - self.a1dag * self.a1)/2

        sigma = numpy.sin(theta_1)*numpy.cos(phi_1)*xQB + numpy.sin(theta_1)*numpy.sin(phi_1)*yQB + numpy.cos(theta_1)*zQB
        S = numpy.sin(theta_2)*numpy.cos(phi_2)*Sx + numpy.sin(theta_2)*numpy.sin(phi_2)*Sy + numpy.cos(theta_2)*Sz
        arg = scipy.sparse.kron(sigma, S).tocsc()

        return scipy.sparse.linalg.expm(-1j*beta*arg)

    def snap(self, theta, n):
        """SNAP (Selective Number-dependent Arbitrary Phase) operator

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            csc_matrix: operator matrix
        """

        ket_n = numpy.zeros(self.cutoff_value)
        ket_n[n] = 1
        projector = numpy.outer(ket_n, ket_n)
        sparse_projector = scipy.sparse.csr_matrix(projector)
        arg = theta * 1j * sparse_projector.tocsc()
        return scipy.sparse.linalg.expm(arg)

    def csnap(self, theta, n):
        """SNAP (Selective Number-dependent Arbitrary Phase) operator,
        with explicit sigma_z in exponential. Can be used to generate
        fock-number selective qubit rotations.

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            csc_matrix: operator matrix
        """

        ket_n = numpy.zeros(self.cutoff_value)
        ket_n[n] = 1
        projector = numpy.outer(ket_n, ket_n)
        # sparse_projector = scipy.sparse.csc_matrix(projector)
        arg = theta * 1j * scipy.sparse.kron(zQB, projector).tocsc()
        return scipy.sparse.linalg.expm(arg)
        
    def multisnap(self, *args):
        """SNAP (Selective Number-dependent Arbitrary Phase) operator for multiple Fock states.
        Generates an arbitrary number of fock-number selective qubit rotations.
        
        Args:
            args (List[reals, integers]): [List of phases, List of Fock states in which the mode should acquire the associated phase]
        
        Returns:
            csr_matrix: operator matrix
        """
        # Divide list in two because thetas and ns must be sent in as a single list
        thetas = args[:len(args) // 2] # arguments
        ns = args[len(args) // 2:] # Fock states on which they are applied
        if len(thetas)!=len(ns): # one theta per Fock state to apply it to
            raise Exception("len(theta) must be equal to len(n)")

        id = numpy.eye(self.cutoff_value)
        gate = scipy.sparse.csr_matrix(id)
        for i in range(len(ns)):
            ket_n = numpy.zeros(self.cutoff_value)
            ket_n[ns[i]] = 1
            projector = numpy.outer(ket_n, ket_n)
            coeff = numpy.exp(1j * thetas[i]) - 1
            mat = scipy.sparse.csr_matrix(coeff * projector)
            gate = numpy.add(gate, mat)
        return scipy.sparse.csr_matrix(gate)

    def multicsnap(self, *args):
        """SNAP (Selective Number-dependent Arbitrary Phase) operator for multiple Fock states.
        Generates an arbitrary number of fock-number selective qubit rotations, with the qubit that accrues the geometric phase explicit.
        
        Args:
            args (List[reals, integers]): [List of phases, List of Fock states in which the mode should acquire the associated phase]
        
        Returns:
            csr_matrix: operator matrix
        """
        # Divide list in two because thetas and ns must be sent in as a single list
        thetas = args[:len(args) // 2] # arguments
        ns = args[len(args) // 2:] # Fock states on which they are applied
        if len(thetas)!=len(ns): # one theta per Fock state to apply it to
            raise Exception("len(theta) must be equal to len(n)")

        id = numpy.eye(self.cutoff_value)
        gate = scipy.sparse.csr_matrix(scipy.sparse.kron(idQB,id))
        for i in range(len(ns)):
            ket_n = numpy.zeros(self.cutoff_value)
            ket_n[ns[i]] = 1
            projector = numpy.outer(ket_n, ket_n)
            coeff = scipy.sparse.linalg.expm(1j * thetas[i] * zQB) - idQB
            mat = scipy.sparse.kron(coeff,projector).tocsr()
            gate = numpy.add(gate, mat)
        return scipy.sparse.csr_matrix(gate)

    
    def pnr(self, max):
        """Support gate for photon number readout (see Curtis et al., PRA (2021) and Wang et al., PRX (2020))
        
        Args:
            max (int): the period of the mapping
        
        Returns:
            csc_matrix: operator matrix
        """
        ket_n = numpy.zeros(self.cutoff_value)
        projector = numpy.outer(ket_n, ket_n)
        # binary search
        for j in range(int(max / 2)):
            for i in range(j, self.cutoff_value, max):
                ket_n = numpy.zeros(self.cutoff_value)
                # fill from right to left
                ket_n[-(i+1)] = 1
                projector += numpy.outer(ket_n, ket_n)

        # Flip qubit if there is a boson present in any of the modes addressed by the projector
        arg = 1j * (-numpy.pi/2) * scipy.sparse.kron(xQB, projector).tocsc()
        return scipy.sparse.linalg.expm(arg)

    def eswap(self, theta):
        """Exponential SWAP operator

        Args:
            theta (real): rotation

        Returns:
            csc_matrix: operator matrix
        """

        self.mat = numpy.zeros([self.cutoff_value * self.cutoff_value, self.cutoff_value * self.cutoff_value])
        for j in range(self.cutoff_value):
            for i in range(self.cutoff_value):
                self.mat[i + (j * self.cutoff_value)][i * self.cutoff_value + j] = 1
        self.sparse_mat = scipy.sparse.csr_matrix(self.mat).tocsc()

        arg = 1j * theta * self.sparse_mat

        return scipy.sparse.linalg.expm(arg)


    def csq(self, theta):
        """Single-mode squeezing operator

        Args:
            theta (real): squeeze

        Returns:
            csc_matrix: operator matrix
        """
        a_sqr = self.a * self.a
        a_dag_sqr = self.a_dag * self.a_dag
        arg = scipy.sparse.kron(zQB, 0.5 * ((numpy.conjugate(theta) * a_sqr) - (theta * a_dag_sqr))).tocsc()

        return scipy.sparse.linalg.expm(arg)


    def testqubitorderf(self, phi):

        arg = 1j * phi * scipy.sparse.kron(xQB, idQB)
        return scipy.sparse.linalg.expm(arg)

    def c_multiboson_sampling(self, max):
        """SNAP gate creation for multiboson sampling purposes.
        
        Args:
            max (int): the period of the mapping
        
        Returns:
            dia_matrix: operator matrix
        """
        print(max)

        return self.eye

    def gate_from_matrix(self, matrix):
        """Converts matrix into gate. Called using ParameterizedUnitaryGate.
        Args:
            matrix (list): the (unitary) matrix that you wish to convert into a gate
        Returns:
            csc_matrix: operator matrix
        """ 
        return scipy.sparse.csc_matrix(matrix)