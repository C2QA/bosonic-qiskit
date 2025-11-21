from collections.abc import Sequence
from typing import Callable, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from qiskit.circuit.parameterexpression import ParameterExpression
from typing_extensions import TypeIs

from .typing import is_int_type

xQB = np.array([[0, 1], [1, 0]])
yQB = np.array([[0, -1j], [1j, 0]])
zQB = np.array([[1, 0], [0, -1]])
idQB = np.array([[1, 0], [0, 1]])
sigma_plus = np.array([[0, 1], [0, 0]])
sigma_minus = np.array([[0, 0], [1, 0]])


UnitaryFunc = Callable[..., np.ndarray | sp.spmatrix | sp.sparray]


class CVOperators:
    """Build operator matrices for continuously variable bosonic gates."""

    @staticmethod
    def call_op(
        op_func: UnitaryFunc,
        params: Sequence[complex | ParameterExpression],
        cutoffs: Sequence[int],
    ) -> NDArray[np.complexfloating]:
        """Call the operator function to build the array using the bound parameter values."""
        # return self.op_func(*map(complex, self.params)).toarray()

        # Add parameters for op_func call
        values: list[complex] = []
        for param in params:
            if isinstance(param, ParameterExpression):
                # if param.is_real():
                #     values.append(float(param))
                # else:
                #     values.append(complex(param))
                values.append(
                    complex(param)
                )  # just cast everything to complex to avoid errors in Ubuntu/MacOS vs Windows
            else:
                values.append(param)

        # Add cutoff for each parameter
        values.extend(cutoffs)

        result = op_func(*values)

        if sp.issparse(result):
            result = result.todense()  # pyright: ignore[reportAttributeAccessIssue]

        # Type checker doesn't know issparse narrows the type
        result = cast(NDArray[np.complexfloating], result)
        return result

    def get_a(self, cutoff: int) -> sp.csc_matrix:
        """Annihilation operator"""
        data = np.sqrt(range(cutoff))
        return sp.spdiags(data=data, diags=[1], m=len(data), n=len(data)).tocsc()

    def get_a1(self, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        return sp.kron(self.get_a(cutoff_a), self.get_eye(cutoff_b)).tocsc()

    def get_a2(self, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        return sp.kron(self.get_eye(cutoff_a), self.get_a(cutoff_b)).tocsc()

    def get_b1(self, cutoff_a: int, cutoff_b: int, cutoff_c: int) -> sp.csc_matrix:
        kron_ab = sp.kron(self.get_a(cutoff_a), self.get_eye(cutoff_b))
        return sp.kron(kron_ab, self.get_eye(cutoff_c)).tocsc()

    def get_b2(self, cutoff_a: int, cutoff_b: int, cutoff_c: int) -> sp.csc_matrix:
        kron_ab = sp.kron(self.get_eye(cutoff_a), self.get_a(cutoff_b))
        return sp.kron(kron_ab, self.get_eye(cutoff_c)).tocsc()

    def get_b3(self, cutoff_a: int, cutoff_b: int, cutoff_c: int) -> sp.csc_matrix:
        kron_ab = sp.kron(self.get_eye(cutoff_a), self.get_eye(cutoff_b))
        return sp.kron(kron_ab, self.get_a(cutoff_c)).tocsc()

    def get_a12(self, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        return self.get_a1(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)

    def get_b123(self, cutoff_a: int, cutoff_b: int, cutoff_c: int) -> sp.csc_matrix:
        return (
            self.get_b1(cutoff_a, cutoff_b, cutoff_c)
            * self.get_b2(cutoff_a, cutoff_b, cutoff_c)
            * self.get_b3(cutoff_a, cutoff_b, cutoff_c)
        )

    def get_a_dag(self, cutoff: int) -> sp.csc_matrix:
        """Creation operator"""
        a = self.get_a(cutoff)
        return a.conjugate().transpose().tocsc()

    def get_N(self, cutoff: int) -> sp.csc_matrix:
        """Number operator"""
        a = self.get_a(cutoff)
        a_dag = self.get_a_dag(cutoff)
        return a_dag * a

    def get_a1_dag(self, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        return self.get_a1(cutoff_a, cutoff_b).conjugate().transpose().tocsc()

    def get_a2_dag(self, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        return self.get_a2(cutoff_a, cutoff_b).conjugate().transpose().tocsc()

    def get_b1_dag(self, cutoff_a: int, cutoff_b: int, cutoff_c: int) -> sp.csc_matrix:
        return self.get_b1(cutoff_a, cutoff_b, cutoff_c).conjugate().transpose().tocsc()

    def get_b2_dag(self, cutoff_a: int, cutoff_b: int, cutoff_c: int) -> sp.csc_matrix:
        return self.get_b2(cutoff_a, cutoff_b, cutoff_c).conjugate().transpose().tocsc()

    def get_b3_dag(self, cutoff_a: int, cutoff_b: int, cutoff_c: int) -> sp.csc_matrix:
        return self.get_b3(cutoff_a, cutoff_b, cutoff_c).conjugate().transpose().tocsc()

    def get_a12_dag(self, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        return self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a2_dag(cutoff_a, cutoff_b)

    def get_b123_dag(
        self, cutoff_a: int, cutoff_b: int, cutoff_c: int
    ) -> sp.csc_matrix:
        return (
            self.get_b1_dag(cutoff_a, cutoff_b, cutoff_c)
            * self.get_b2_dag(cutoff_a, cutoff_b, cutoff_c)
            * self.get_b3_dag(cutoff_a, cutoff_b, cutoff_c)
        )

    def get_a12dag(self, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        return self.get_a1(cutoff_a, cutoff_b) * self.get_a2_dag(cutoff_a, cutoff_b)

    def get_a1dag2(self, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        return self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)

    def get_eye(self, cutoff: int) -> sp.csc_matrix:
        """Identity matrix"""
        return cast(sp.csc_matrix, sp.eye(cutoff, format="csc"))

    def r(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Phase space rotation operator

        Args:
            theta (real): rotation

        Returns:
            csc_matrix: operator matrix
        """
        arg = 1j * theta * self.get_N(cutoff)

        return sp.linalg.expm(arg)

    def d(self, alpha: complex, cutoff: int) -> sp.csc_matrix:
        """Displacement operator

        Args:
            alpha (complex): displacement

        Returns:
            csc_matrix: operator matrix
        """
        arg = (alpha * self.get_a_dag(cutoff)) - (
            np.conjugate(alpha) * self.get_a(cutoff)
        )

        return sp.linalg.expm(arg)

    def s(self, theta: complex, cutoff: int) -> sp.csc_matrix:
        """Single-mode squeezing operator

        Args:
            theta (complex): squeeze

        Returns:
            csc_matrix: operator matrix
        """
        a = self.get_a(cutoff)
        a_dag = self.get_a_dag(cutoff)
        a_sqr = a * a
        a_dag_sqr = a_dag * a_dag
        arg = 0.5 * ((np.conjugate(theta) * a_sqr) - (theta * a_dag_sqr))

        return sp.linalg.expm(arg)

    def s2(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Two-mode squeezing operator

        Args:
            theta (complex): squeezing factor

        Returns:
            csc_matrix: operator matrix
        """
        r, phi = np.abs(theta), np.angle(theta)

        # eq. 183 in arXiv:2407.10381
        arg = (np.exp(1j * phi) * self.get_a12_dag(cutoff_a, cutoff_b)) - (
            np.exp(-1j * phi) * self.get_a12(cutoff_a, cutoff_b)
        )

        return sp.linalg.expm(r * arg)

    def s3(
        self, theta: complex, cutoff_a: int, cutoff_b: int, cutoff_c: int
    ) -> sp.csc_matrix:
        """Three-mode squeezing operator

        Args:
            theta: squeezing amount

        Returns:
            csc_matrix: operator matrix
        """
        r, phi = np.abs(theta), np.angle(theta)

        arg = np.exp(1j * phi) * self.get_b123_dag(
            cutoff_a, cutoff_b, cutoff_c
        ) - np.exp(-1j * phi) * self.get_b123(cutoff_a, cutoff_b, cutoff_c)

        return sp.linalg.expm(r * arg)

    def bs(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Two-mode beam splitter operator

        Args:
            theta: phase

        Returns:
            csc_matrix: operator matrix
        """

        arg = theta * self.get_a1dag2(cutoff_a, cutoff_b) - np.conj(
            theta
        ) * self.get_a12dag(cutoff_a, cutoff_b)

        return sp.linalg.expm(arg)

    def cr(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Controlled phase space rotation operator

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * sp.kron(zQB, self.get_N(cutoff)).tocsc()

        return sp.linalg.expm(arg)

    def crx(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * sp.kron(xQB, self.get_N(cutoff)).tocsc()

        return sp.linalg.expm(arg)

    def cry(self, theta: float, cutoff: int) -> sp.csc_matrix:
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            csc_matrix: operator matrix
        """
        arg = theta * 1j * sp.kron(yQB, self.get_N(cutoff)).tocsc()

        return sp.linalg.expm(arg)

    def cd(self, theta: float, beta: float | None, cutoff: int) -> sp.csc_matrix:
        """Controlled displacement operator

        Args:
            theta (real): displacement for qubit state 0
            beta (real): displacement for qubit state 1. If None, use -alpha.

        Returns:
            bsr_matrix: operator matrix
        """
        displace0 = (theta * self.get_a_dag(cutoff)) - (
            np.conjugate(theta) * self.get_a(cutoff)
        )
        beta = beta or -theta
        displace1 = (beta * self.get_a_dag(cutoff)) - (
            np.conjugate(beta) * self.get_a(cutoff)
        )

        return sp.kron((idQB + zQB) / 2, sp.linalg.expm(displace0)) + sp.kron(
            (idQB - zQB) / 2, sp.linalg.expm(displace1)
        )

    def ecd(self, theta: complex, cutoff: int) -> sp.csc_matrix:
        """Echoed controlled displacement operator

        Args:
            theta (complex): displacement

        Returns:
            csr_matrix: operator matrix
        """
        argm = (theta * self.get_a_dag(cutoff)) - (
            np.conjugate(theta) * self.get_a(cutoff)
        )
        arg = sp.kron(zQB, argm)

        return sp.linalg.expm(arg)

    def cbs(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Controlled phase two-mode beam splitter operator

        Args:
            theta (complex): beamsplitter phase

        Returns:
            csc_matrix: operator matrix
        """

        argm = theta * self.get_a1dag2(cutoff_a, cutoff_b) - np.conjugate(
            theta
        ) * self.get_a12dag(cutoff_a, cutoff_b)
        arg = sp.kron(zQB, argm).tocsc()

        return sp.linalg.expm(arg)

    def cschwinger(
        self,
        beta: float,
        theta_1: float,
        phi_1: float,
        theta_2: float,
        phi_2: float,
        cutoff_a: int,
        cutoff_b: int,
    ) -> sp.csc_matrix:
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
            np.sin(theta_1) * np.cos(phi_1) * xQB
            + np.sin(theta_1) * np.sin(phi_1) * yQB
            + np.cos(theta_1) * zQB
        )
        S = (
            np.sin(theta_2) * np.cos(phi_2) * Sx
            + np.sin(theta_2) * np.sin(phi_2) * Sy
            + np.cos(theta_2) * Sz
        )
        arg = sp.kron(sigma, S).tocsc()

        return sp.linalg.expm(-1j * beta * arg)

    def snap(self, theta: float, n: int, cutoff: int) -> sp.csc_matrix:
        """SNAP (Selective Number-dependent Arbitrary Phase) operator

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            csc_matrix: operator matrix
        """

        ket_n = np.zeros(cutoff)
        ket_n[n] = 1
        projector = np.outer(ket_n, ket_n)
        sparse_projector = sp.csr_matrix(projector)
        arg = theta * 1j * sparse_projector.tocsc()
        return sp.linalg.expm(arg)

    def csnap(self, theta: float, n: int, cutoff: int) -> sp.csc_matrix:
        """SNAP (Selective Number-dependent Arbitrary Phase) operator,
        with explicit sigma_z in exponential. Can be used to generate
        fock-number selective qubit rotations.

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            csc_matrix: operator matrix
        """

        ket_n = np.zeros(cutoff)
        ket_n[n] = 1
        projector = np.outer(ket_n, ket_n)
        # sparse_projector = sp.csc_matrix(projector)
        arg = theta * 1j * sp.kron(zQB, projector).tocsc()
        return sp.linalg.expm(arg)

    def multisnap(self, *args: int | float | np.integer | np.floating) -> sp.csr_matrix:
        """SNAP (Selective Number-dependent Arbitrary Phase) operator for multiple Fock states.
        Generates an arbitrary number of fock-number selective qubit rotations.

        Args:
            args (List[reals, integers]): [List of phases, List of Fock states in which the mode should acquire the associated phase]

        Returns:
            csr_matrix: operator matrix
        """
        # Divide list in two because thetas and ns must be sent in as a single list
        *args, cutoff = args

        if not is_int_type(cutoff):
            raise ValueError(f"Expected integer cutoff, got {cutoff}")

        midpoint = len(args) // 2
        thetas, ns = args[:midpoint], args[:midpoint]

        if len(thetas) != len(ns):  # one theta per Fock state to apply it to
            raise Exception("len(theta) must be equal to len(n)")

        id = np.eye(cutoff)
        gate = sp.csr_matrix(id)
        for theta, n in zip(thetas, ns):
            if not is_int_type(n):
                raise ValueError(f"Got non-integer fock state {n}")

            ket_n = np.zeros(cutoff)
            ket_n[n] = 1
            projector = np.outer(ket_n, ket_n)
            coeff = np.exp(1j * theta) - 1
            mat = sp.csr_matrix(coeff * projector)
            gate = gate + mat

        return sp.csr_matrix(gate)

    def multicsnap(
        self, *args: int | float | np.integer | np.floating
    ) -> sp.csr_matrix:
        """SNAP (Selective Number-dependent Arbitrary Phase) operator for multiple Fock states.
        Generates an arbitrary number of fock-number selective qubit rotations, with the qubit that accrues the geometric phase explicit.

        Args:
            args (List[reals, integers]): [List of phases, List of Fock states in which the mode should acquire the associated phase]

        Returns:
            csr_matrix: operator matrix
        """
        # Divide list in two because thetas and ns must be sent in as a single list
        *args, cutoff = args

        if not is_int_type(cutoff):
            raise ValueError(f"Expected integer cutoff, got {cutoff}")

        midpoint = len(args) // 2
        thetas, ns = args[:midpoint], args[:midpoint]

        if len(thetas) != len(ns):  # one theta per Fock state to apply it to
            raise Exception("len(theta) must be equal to len(n)")

        id = np.eye(cutoff)
        gate = sp.csr_matrix(sp.kron(idQB, id))
        for theta, n in zip(thetas, ns):
            if not is_int_type(n):
                raise ValueError(f"Got non-integer fock state {n}")

            ket_n = np.zeros(cutoff)
            ket_n[n] = 1
            projector = np.outer(ket_n, ket_n)
            coeff = sp.linalg.expm(1j * theta * zQB) - idQB
            mat = sp.kron(coeff, projector).tocsr()
            gate = gate + mat

        return sp.csr_matrix(gate)

    def sqr(self, *args: float) -> sp.csc_matrix:
        """SQR gate (Liu et al, arXiv 2024)

        This function assumes that the parameters (minus the cutoff) are concatenated, so it should
        have length 3*n, where n is the number of distinct fock states to condition on.

        Args:
            params: Gate parameters and cutoff, see `CVCircuit.cv_sqr` for the parameter structure

        Returns
            csc_matrix: The operator matrix
        """
        from qiskit.circuit.library import RGate

        *params, cutoff = args
        cutoff = int(cutoff)

        params = np.atleast_1d(params)
        theta, phi, fock_states = np.array_split(params, 3)
        fock_states = fock_states.astype(int)  # guaranteed by cv_sqr

        blocks = [idQB] * cutoff
        for t, p, n in zip(theta, phi, fock_states):
            blocks[n] = RGate(t, p).to_matrix()

        # Can cast because spmatrix is returned if no block is a sparray. This
        # matrix acts on the space Qumode x Qubit, but we need to put qubit first
        # to match qiskit
        out = cast(sp.csc_matrix, sp.block_diag(blocks, format="csc"))
        perm = np.arange(2 * cutoff).reshape(cutoff, 2).T.flatten()
        return out[perm, :][:, perm]

    def pnr(self, max: int, cutoff: int) -> sp.csc_matrix:
        """Support gate for photon number readout (see Curtis et al., PRA (2021) and Wang et al., PRX (2020))

        Args:
            max (int): the period of the mapping

        Returns:
            csc_matrix: operator matrix
        """
        ket_n = np.zeros(cutoff)
        projector = np.outer(ket_n, ket_n)
        # binary search
        for j in range(int(max / 2)):
            for i in range(j, cutoff, max):
                ket_n = np.zeros(cutoff)
                # fill from right to left
                ket_n[-(i + 1)] = 1
                projector += np.outer(ket_n, ket_n)

        # Flip qubit if there is a boson present in any of the modes addressed by the projector
        arg = 1j * (-np.pi / 2) * sp.kron(xQB, projector).tocsc()
        return sp.linalg.expm(arg)

    def eswap(self, theta, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
        """Exponential SWAP operator

        Args:
            theta (real): rotation

        Returns:
            csc_matrix: operator matrix
        """

        # Todo: why does this store data in self?
        self.mat = np.zeros([cutoff_a * cutoff_b, cutoff_a * cutoff_b])
        for j in range(cutoff_a):
            for i in range(cutoff_b):
                self.mat[i + (j * cutoff_a)][i * cutoff_b + j] = 1
        self.sparse_mat = sp.csr_matrix(self.mat).tocsc()

        arg = 1j * theta * self.sparse_mat

        return sp.linalg.expm(arg)

    def csq(self, theta: complex, cutoff: int) -> sp.csc_matrix:
        """Single-mode squeezing operator

        Args:
            theta (complex): squeeze

        Returns:
            csc_matrix: operator matrix
        """
        a_sqr = self.get_a(cutoff) * self.get_a(cutoff)
        a_dag_sqr = self.get_a_dag(cutoff) * self.get_a_dag(cutoff)
        arg = sp.kron(
            zQB, 0.5 * ((np.conjugate(theta) * a_sqr) - (theta * a_dag_sqr))
        ).tocsc()

        return sp.linalg.expm(arg)

    def c_multiboson_sampling(self, max: int, cutoff: int) -> sp.csc_matrix:
        """SNAP gate creation for multiboson sampling purposes.

        Args:
            max (int): the period of the mapping

        Returns:
            csc_matrix: operator matrix
        """
        return self.get_eye(cutoff)

    def gate_from_matrix(self, matrix: ArrayLike) -> sp.csc_matrix:
        """Converts matrix into gate. Called using ParameterizedUnitaryGate.

        Args:
            matrix (list): the (unitary) matrix that you wish to convert into a gate

        Returns:
            csc_matrix: operator matrix
        """
        return sp.csc_matrix(matrix)

    def sum(self, scale: float, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
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
        arg = (scale / 2) * (sp.kron(a_mat, b_mat))

        return sp.linalg.expm(arg)

    def csum(self, scale: float, cutoff_a: int, cutoff_b: int) -> sp.csc_matrix:
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
        arg = (scale / 2) * (sp.kron(a_mat, b_mat))
        arg = sp.kron(zQB, arg)

        return sp.linalg.expm(arg)

    def jc(self, theta: float, phi: float, cutoff: int) -> sp.csc_matrix:
        """Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)

        Returns:
            csc_matrix: operator matrix
        """
        # TODO -- verify use of sp.kron vs Table III.3 from https://arxiv.org/pdf/2407.10381
        arg = np.exp(1j * phi) * sp.kron(sigma_minus, self.get_a_dag(cutoff))
        arg += np.exp(-1j * phi) * sp.kron(sigma_plus, self.get_a(cutoff))
        arg = -1j * theta * arg

        return sp.linalg.expm(arg)

    def ajc(self, theta: float, phi: float, cutoff: int) -> sp.csc_matrix:
        """Anti-Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)

        Returns:
            csc_matrix: operator matrix
        """
        # TODO -- verify use of sp.kron vs Table III.3 from https://arxiv.org/pdf/2407.10381
        arg = np.exp(1j * phi) * sp.kron(sigma_plus, self.get_a_dag(cutoff))
        arg += np.exp(-1j * phi) * sp.kron(sigma_minus, self.get_a(cutoff))
        arg = -1j * theta * arg

        return sp.linalg.expm(arg)

    def rb(self, theta: complex, cutoff: int):
        """Rabi interaction gate

        Args:
            theta (complex): arbitrary scale factor

        Returns:
            csc_matrix: operator matrix
        """
        # TODO -- verify use of sp.kron vs Table III.3 from https://arxiv.org/pdf/2407.10381
        arg = -1j * sp.kron(
            xQB,
            (theta * self.get_a_dag(cutoff) + np.conjugate(theta) * self.get_a(cutoff)),
        )

        return sp.linalg.expm(arg)
