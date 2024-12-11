import numpy
import scipy.sparse
import scipy.sparse.linalg
import pdb


xQB = numpy.array([[0, 1], [1, 0]])
yQB = numpy.array([[0, -1j], [1j, 0]])
zQB = numpy.array([[1, 0], [0, -1]])
idQB = numpy.array([[1, 0], [0, 1]])
sigma_plus = numpy.array([[0, 1], [0, 0]])
sigma_minus = numpy.array([[0, 0], [1, 0]])


class CVOperators:
    """Build operator matrices for continuously variable bosonic gates."""

    def get_a(self, cutoff: int):
        """Annihilation operator"""
        data = numpy.sqrt(range(cutoff))
        return scipy.sparse.spdiags(
            data=data, diags=[1], m=len(data), n=len(data)
        ).tocsc()

    def get_a1(self, cutoff_a: int, cutoff_b: int):
        return scipy.sparse.kron(self.get_a(cutoff_a), self.get_eye(cutoff_b)).tocsc()

    def get_a2(self, cutoff_a: int, cutoff_b: int):
        return scipy.sparse.kron(self.get_eye(cutoff_a), self.get_a(cutoff_b)).tocsc()

    def get_b1(self, cutoff_a: int, cutoff_b: int, cutoff_c: int):
        kron_ab = scipy.sparse.kron(self.get_a(cutoff_a), self.get_eye(cutoff_b))
        return scipy.sparse.kron(kron_ab, self.get_eye(cutoff_c)).tocsc()

    def get_b2(self, cutoff_a: int, cutoff_b: int, cutoff_c: int):
        kron_ab = scipy.sparse.kron(self.get_eye(cutoff_a), self.get_a(cutoff_b))
        return scipy.sparse.kron(kron_ab, self.get_eye(cutoff_c)).tocsc()

    def get_b3(self, cutoff_a: int, cutoff_b: int, cutoff_c: int):
        kron_ab = scipy.sparse.kron(self.get_eye(cutoff_a), self.get_eye(cutoff_b))
        return scipy.sparse.kron(kron_ab, self.get_a(cutoff_c)).tocsc()

    def get_a12(self, cutoff_a: int, cutoff_b: int):
        return self.get_a1(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)

    def get_b123(self, cutoff_a: int, cutoff_b: int, cutoff_c: int):
        return (
            self.get_b1(cutoff_a, cutoff_b, cutoff_c)
            * self.get_b2(cutoff_a, cutoff_b, cutoff_c)
            * self.get_b3(cutoff_a, cutoff_b, cutoff_c)
        )

    def get_a_dag(self, cutoff: int):
        """Creation operator"""
        a = self.get_a(cutoff)
        return a.conjugate().transpose().tocsc()

    def get_N(self, cutoff: int):
        """Number operator"""
        a = self.get_a(cutoff)
        a_dag = self.get_a_dag(cutoff)
        return a_dag * a

    def get_a1_dag(self, cutoff_a: int, cutoff_b: int):
        return self.get_a1(cutoff_a, cutoff_b).conjugate().transpose().tocsc()

    def get_a2_dag(self, cutoff_a: int, cutoff_b: int):
        return self.get_a2(cutoff_a, cutoff_b).conjugate().transpose().tocsc()

    def get_b1_dag(self, cutoff_a: int, cutoff_b: int, cutoff_c: int):
        return self.get_b1(cutoff_a, cutoff_b, cutoff_c).conjugate().transpose().tocsc()

    def get_b2_dag(self, cutoff_a: int, cutoff_b: int, cutoff_c: int):
        return self.get_b2(cutoff_a, cutoff_b, cutoff_c).conjugate().transpose().tocsc()

    def get_b3_dag(self, cutoff_a: int, cutoff_b: int, cutoff_c: int):
        return self.get_b3(cutoff_a, cutoff_b, cutoff_c).conjugate().transpose().tocsc()

    def get_a12_dag(self, cutoff_a: int, cutoff_b: int):
        return self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a2_dag(cutoff_a, cutoff_b)

    def get_b123_dag(self, cutoff_a: int, cutoff_b: int, cutoff_c: int):
        return (
            self.get_b1_dag(cutoff_a, cutoff_b, cutoff_c)
            * self.get_b2_dag(cutoff_a, cutoff_b, cutoff_c)
            * self.get_b3_dag(cutoff_a, cutoff_b, cutoff_c)
        )

    def get_a12dag(self, cutoff_a: int, cutoff_b: int):
        return self.get_a1(cutoff_a, cutoff_b) * self.get_a2_dag(cutoff_a, cutoff_b)

    def get_a1dag2(self, cutoff_a: int, cutoff_b: int):
        return self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)

    def get_eye(self, cutoff: int):
        """Identity matrix"""
        return scipy.sparse.eye(cutoff)

    def r(self, theta, cutoff):
        """Phase space rotation operator

        Args:
            theta (real): rotation

        Returns:
            csc_matrix: operator matrix
        """
        arg = 1j * theta * self.get_N(cutoff)

        return scipy.sparse.linalg.expm(arg)

    def d(self, alpha, cutoff):
        """Displacement operator

        Args:
            alpha (real): displacement

        Returns:
            csc_matrix: operator matrix
        """
        arg = (alpha * self.get_a_dag(cutoff)) - (
            numpy.conjugate(alpha) * self.get_a(cutoff)
        )

        return scipy.sparse.linalg.expm(arg)

    def s(self, theta, cutoff):
        """Single-mode squeezing operator

        Args:
            theta (real): squeeze

        Returns:
            csc_matrix: operator matrix
        """
        a = self.get_a(cutoff)
        a_dag = self.get_a_dag(cutoff)
        a_sqr = a * a
        a_dag_sqr = a_dag * a_dag
        arg = 0.5 * ((numpy.conjugate(theta) * a_sqr) - (theta * a_dag_sqr))

        return scipy.sparse.linalg.expm(arg)

    def s2(self, theta, cutoff_a, cutoff_b):
        """Two-mode squeezing operator

        Args:
            g (real): multiplied by 1j to yield imaginary phase

        Returns:
            csc_matrix: operator matrix
        """

        arg = (numpy.conjugate(theta * 1j) * self.get_a12_dag(cutoff_a, cutoff_b)) - (
            theta * 1j * self.get_a12(cutoff_a, cutoff_b)
        )

        return scipy.sparse.linalg.expm(arg)

    def s3(self, theta, cutoff_a, cutoff_b, cutoff_c):
        """Three-mode squeezing operator

        Args:
            g (real): multiplied by 1j to yield imaginary phase

        Returns:
            csc_matrix: operator matrix
        """

        arg = (
            numpy.conjugate(theta * 1j)
            * self.get_b123_dag(cutoff_a, cutoff_b, cutoff_c)
        ) - (theta * 1j * self.get_b123(cutoff_a, cutoff_b, cutoff_c))

        return scipy.sparse.linalg.expm(arg)

    def bs(self, theta, cutoff_a, cutoff_b):
        """Two-mode beam splitter operator

        Args:
            theta: phase

        Returns:
            csc_matrix: operator matrix
        """

        arg = theta * self.get_a1dag2(cutoff_a, cutoff_b) - numpy.conj(
            theta
        ) * self.get_a12dag(cutoff_a, cutoff_b)

        return scipy.sparse.linalg.expm(arg)

    def cr(self, theta, cutoff):
        """Controlled phase space rotation operator

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(zQB, self.get_N(cutoff)).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def crx(self, theta, cutoff):
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(xQB, self.get_N(cutoff)).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def cry(self, theta, cutoff):
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(yQB, self.get_N(cutoff)).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def cd(self, theta, beta, cutoff):
        """Controlled displacement operator

        Args:
            theta (real): displacement for qubit state 0
            beta (real): displacement for qubit state 1. If None, use -alpha.

        Returns:
            bsr_matrix: operator matrix
        """
        displace0 = (theta * self.get_a_dag(cutoff)) - (
            numpy.conjugate(theta) * self.get_a(cutoff)
        )
        if beta is None:
            beta = -theta
        displace1 = (beta * self.get_a_dag(cutoff)) - (
            numpy.conjugate(beta) * self.get_a(cutoff)
        )

        return scipy.sparse.kron(
            (idQB + zQB) / 2, scipy.sparse.linalg.expm(displace0)
        ) + scipy.sparse.kron((idQB - zQB) / 2, scipy.sparse.linalg.expm(displace1))

    def ecd(self, theta, cutoff):
        """Echoed controlled displacement operator

        Args:
            theta (real): displacement

        Returns:
            csr_matrix: operator matrix
        """
        argm = (theta * self.get_a_dag(cutoff)) - (
            numpy.conjugate(theta) * self.get_a(cutoff)
        )
        arg = scipy.sparse.kron(zQB, argm)

        return scipy.sparse.linalg.expm(arg)

    def cbs(self, theta, cutoff_a, cutoff_b):
        """Controlled phase two-mode beam splitter operator

        Args:
            theta (real): real phase

        Returns:
            csc_matrix: operator matrix
        """

        argm = theta * self.get_a1dag2(cutoff_a, cutoff_b) - numpy.conjugate(
            theta
        ) * self.get_a12dag(cutoff_a, cutoff_b)
        arg = scipy.sparse.kron(zQB, argm).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def cschwinger(self, beta, theta_1, phi_1, theta_2, phi_2, cutoff_a, cutoff_b):
        """General form of a controlled Schwinger gate

        Args:
            params (real): [beta, theta_1, phi_1, theta_2, phi_2]

        Returns:
            csc_matrix: operator matrix
        """

        Sx = (
            self.get_a1(cutoff_a, cutoff_b) * self.get_a2_dag(cutoff_a, cutoff_b)
            + self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)
        ) / 2
        Sy = (
            self.get_a1(cutoff_a, cutoff_b) * self.get_a2_dag(cutoff_a, cutoff_b)
            - self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)
        ) / (2 * 1j)
        Sz = (
            self.get_a2_dag(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)
            - self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a1(cutoff_a, cutoff_b)
        ) / 2

        sigma = (
            numpy.sin(theta_1) * numpy.cos(phi_1) * xQB
            + numpy.sin(theta_1) * numpy.sin(phi_1) * yQB
            + numpy.cos(theta_1) * zQB
        )
        S = (
            numpy.sin(theta_2) * numpy.cos(phi_2) * Sx
            + numpy.sin(theta_2) * numpy.sin(phi_2) * Sy
            + numpy.cos(theta_2) * Sz
        )
        arg = scipy.sparse.kron(sigma, S).tocsc()

        return scipy.sparse.linalg.expm(-1j * beta * arg)

    def snap(self, theta, n, cutoff):
        """SNAP (Selective Number-dependent Arbitrary Phase) operator

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            csc_matrix: operator matrix
        """

        ket_n = numpy.zeros(cutoff)
        ket_n[n] = 1
        projector = numpy.outer(ket_n, ket_n)
        sparse_projector = scipy.sparse.csr_matrix(projector)
        arg = theta * 1j * sparse_projector.tocsc()
        return scipy.sparse.linalg.expm(arg)

    def csnap(self, theta, n, cutoff):
        """SNAP (Selective Number-dependent Arbitrary Phase) operator,
        with explicit sigma_z in exponential. Can be used to generate
        fock-number selective qubit rotations.

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            csc_matrix: operator matrix
        """

        ket_n = numpy.zeros(cutoff)
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
        cutoff = args[-1]
        theta_ns = args[0:-1]
        thetas = theta_ns[: len(theta_ns) // 2]  # arguments
        ns = theta_ns[len(theta_ns) // 2 :]  # Fock states on which they are applied
        if len(thetas) != len(ns):  # one theta per Fock state to apply it to
            raise Exception("len(theta) must be equal to len(n)")

        id = numpy.eye(cutoff)
        gate = scipy.sparse.csr_matrix(id)
        for i in range(len(ns)):
            ket_n = numpy.zeros(cutoff)
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
        cutoff = args[-1]
        theta_ns = args[0:-1]
        thetas = theta_ns[: len(theta_ns) // 2]  # arguments
        ns = theta_ns[len(theta_ns) // 2 :]  # Fock states on which they are applied
        if len(thetas) != len(ns):  # one theta per Fock state to apply it to
            raise Exception("len(theta) must be equal to len(n)")

        id = numpy.eye(cutoff)
        gate = scipy.sparse.csr_matrix(scipy.sparse.kron(idQB, id))
        for i in range(len(ns)):
            ket_n = numpy.zeros(cutoff)
            ket_n[ns[i]] = 1
            projector = numpy.outer(ket_n, ket_n)
            coeff = scipy.sparse.linalg.expm(1j * thetas[i] * zQB) - idQB
            mat = scipy.sparse.kron(coeff, projector).tocsr()
            gate = numpy.add(gate, mat)
        return scipy.sparse.csr_matrix(gate)

    def pnr(self, max, cutoff):
        """Support gate for photon number readout (see Curtis et al., PRA (2021) and Wang et al., PRX (2020))

        Args:
            max (int): the period of the mapping

        Returns:
            csc_matrix: operator matrix
        """
        ket_n = numpy.zeros(cutoff)
        projector = numpy.outer(ket_n, ket_n)
        # binary search
        for j in range(int(max / 2)):
            for i in range(j, cutoff, max):
                ket_n = numpy.zeros(cutoff)
                # fill from right to left
                ket_n[-(i + 1)] = 1
                projector += numpy.outer(ket_n, ket_n)

        # Flip qubit if there is a boson present in any of the modes addressed by the projector
        arg = 1j * (-numpy.pi / 2) * scipy.sparse.kron(xQB, projector).tocsc()
        return scipy.sparse.linalg.expm(arg)

    def eswap(self, theta, cutoff_a, cutoff_b):
        """Exponential SWAP operator

        Args:
            theta (real): rotation

        Returns:
            csc_matrix: operator matrix
        """

        self.mat = numpy.zeros([cutoff_a * cutoff_b, cutoff_a * cutoff_b])
        for j in range(cutoff_a):
            for i in range(cutoff_b):
                self.mat[i + (j * cutoff_a)][i * cutoff_b + j] = 1
        self.sparse_mat = scipy.sparse.csr_matrix(self.mat).tocsc()

        arg = 1j * theta * self.sparse_mat

        return scipy.sparse.linalg.expm(arg)

    def csq(self, theta, cutoff):
        """Single-mode squeezing operator

        Args:
            theta (real): squeeze

        Returns:
            csc_matrix: operator matrix
        """
        a_sqr = self.get_a(cutoff) * self.get_a(cutoff)
        a_dag_sqr = self.get_a_dag(cutoff) * self.get_a_dag(cutoff)
        arg = scipy.sparse.kron(
            zQB, 0.5 * ((numpy.conjugate(theta) * a_sqr) - (theta * a_dag_sqr))
        ).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def testqubitorderf(self, phi):

        arg = 1j * phi * scipy.sparse.kron(xQB, idQB)
        return scipy.sparse.linalg.expm(arg)

    def c_multiboson_sampling(self, max, cutoff):
        """SNAP gate creation for multiboson sampling purposes.

        Args:
            max (int): the period of the mapping

        Returns:
            dia_matrix: operator matrix
        """
        print(max)

        return self.get_eye(cutoff)

    def gate_from_matrix(self, matrix):
        """Converts matrix into gate. Called using ParameterizedUnitaryGate.

        Args:
            matrix (list): the (unitary) matrix that you wish to convert into a gate

        Returns:
            csc_matrix: operator matrix
        """
        return scipy.sparse.csc_matrix(matrix)

    def sum(self, scale, cutoff_a, cutoff_b):
        """Two-qumode sum gate

        Args:
            scale (real): arbitrary scale factor

        Returns:
            csc_matrix: operator matrix
        """
        # TODO verify below implementation
        #     equation 205 from https://arxiv.org/pdf/2407.10381
        #     vs equation 4 from https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.88.097904
        #   Equation 205 with matrix multiplication is not unitary, how to handle different cutoffs
        a_mat = self.get_a(cutoff_a) + self.get_a_dag(cutoff_a)
        b_mat = self.get_a_dag(cutoff_b) - self.get_a(cutoff_b)
        arg = (scale / 2) * (scipy.sparse.kron(a_mat, b_mat))

        return scipy.sparse.linalg.expm(arg)

    def csum(self, scale, cutoff_a, cutoff_b):
        """Conditional two-qumode sum gate

        Args:
            scale (real): arbitrary scale factor

        Returns:
            csc_matrix: operator matrix
        """
        # TODO verify below implementation
        #     equation 205 from https://arxiv.org/pdf/2407.10381
        #     vs equation 4 from https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.88.097904
        #   Equation 205 with matrix multiplication is not unitary, how to handle different cutoffs
        a_mat = self.get_a(cutoff_a) + self.get_a_dag(cutoff_a)
        b_mat = self.get_a_dag(cutoff_b) - self.get_a(cutoff_b)
        arg = (scale / 2) * (scipy.sparse.kron(a_mat, b_mat))
        arg = scipy.sparse.kron(zQB, arg)

        return scipy.sparse.linalg.expm(arg)

    def jc(self, theta, phi, cutoff):
        """Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)

        Returns:
            csc_matrix: operator matrix
        """
        # TODO -- verify use of scipy.sparse.kron vs Table III.3 from https://arxiv.org/pdf/2407.10381
        arg = numpy.exp(1j * phi) * scipy.sparse.kron(
            sigma_minus, self.get_a_dag(cutoff)
        )
        arg += numpy.exp(-1j * phi) * scipy.sparse.kron(sigma_plus, self.get_a(cutoff))
        arg = -1j * theta * arg

        return scipy.sparse.linalg.expm(arg)

    def ajc(self, theta, phi, cutoff):
        """Anti-Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)

        Returns:
            csc_matrix: operator matrix
        """
        # TODO -- verify use of scipy.sparse.kron vs Table III.3 from https://arxiv.org/pdf/2407.10381
        arg = numpy.exp(1j * phi) * scipy.sparse.kron(
            sigma_plus, self.get_a_dag(cutoff)
        )
        arg += numpy.exp(-1j * phi) * scipy.sparse.kron(sigma_minus, self.get_a(cutoff))
        arg = -1j * theta * arg

        return scipy.sparse.linalg.expm(arg)

    def rb(self, theta, cutoff):
        """Rabi interaction gate

        Args:
            theta (real): arbitrary scale factor

        Returns:
            csc_matrix: operator matrix
        """
        # TODO -- verify use of scipy.sparse.kron vs Table III.3 from https://arxiv.org/pdf/2407.10381
        arg = -1j * scipy.sparse.kron(
            xQB,
            (
                theta * self.get_a_dag(cutoff)
                + numpy.conjugate(theta) * self.get_a(cutoff)
            ),
        )

        return scipy.sparse.linalg.expm(arg)
