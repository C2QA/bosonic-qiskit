import functools
import re
from collections.abc import Sequence
from typing import Callable, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.circuit.parameterexpression import ParameterExpression

from .typing import is_int_type

I = np.eye(2)
X = XGate().to_matrix()
Y = YGate().to_matrix()
Z = ZGate().to_matrix()

# Fixme: these imply that |g> = |1> and |e> = |0>. Is that correct?
SPLUS = 0.5 * (X + 1j * Y)  # |1> -> |0>
SMINUS = 0.5 * (X - 1j * Y)  # |0> -> |1>
P0 = (I + Z) / 2  # |0><0|
P1 = (I - Z) / 2  # |1><1|


UnitaryFunc = Callable[..., np.ndarray | sp.sparray]

EXPR_PATTERN = re.compile(r"(\w+)(\d+)")


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

    def get_eye(self, cutoff: int) -> sp.csc_array:
        """Identity matrix"""
        return cast(sp.csc_array, sp.eye_array(cutoff, format="csc"))

    def get_a(self, cutoff: int) -> sp.csc_array:
        """Annihilation operator"""
        data = np.sqrt(np.arange(1, cutoff + 1))
        return cast(
            sp.csc_array,
            sp.diags_array(data, offsets=1, shape=(cutoff, cutoff), format="csc"),
        )

    def get_ad(self, cutoff: int) -> sp.csc_array:
        """Creation operator"""
        a = self.get_a(cutoff)
        return a.conjugate().transpose().tocsc()

    def get_N(self, cutoff: int) -> sp.csc_array:
        """Number operator"""
        a = self.get_a(cutoff)
        a_dag = self.get_ad(cutoff)
        return a_dag @ a

    def get_projector(self, n: int, cutoff: int) -> sp.csc_array:
        out = sp.dok_array((cutoff, cutoff))
        out[n, n] = 1
        return out.tocsc()

    def get_op(self, expr: str, *cutoffs: int) -> sp.csc_array:
        """Helper function to construct creation/annihilation operators symbolically"""
        matrices: dict[int, sp.csc_array] = {}

        for subexpr in expr.split(" "):
            if re_match := EXPR_PATTERN.match(subexpr):
                op = re_match.group(1)
                qumode_idx = int(re_match.group(2))

                if qumode_idx >= len(cutoffs):
                    raise ValueError(
                        f"Expected a qumode index < len(cutoffs), got {qumode_idx}"
                    )

                match op:
                    case "a":
                        matrices[qumode_idx] = self.get_a(cutoffs[qumode_idx])
                    case "ad":
                        matrices[qumode_idx] = self.get_ad(cutoffs[qumode_idx])
                    case "n":
                        matrices[qumode_idx] = self.get_N(cutoffs[qumode_idx])
                    case _:
                        raise ValueError(f"Unrecognized symbol: {op}")

        ordered_mats: list[sp.csc_array] = []
        for i, cutoff in enumerate(cutoffs):
            if (mat := matrices.get(i)) is not None:
                ordered_mats.append(mat)
            else:
                ordered_mats.append(self.get_eye(cutoff))

        return functools.reduce(sp.kron, ordered_mats).tocsc()

    def r(self, theta: float, cutoff: int) -> sp.csc_array:
        """Phase space rotation operator

        Args:
            theta (real): rotation

        Returns:
            csc_array: operator matrix
        """
        arg = 1j * theta * self.get_N(cutoff)
        return sp.linalg.expm(arg)

    def d(self, alpha: complex, cutoff: int) -> sp.csc_array:
        """Displacement operator

        Args:
            alpha (complex): displacement

        Returns:
            csc_array: operator matrix
        """
        arg = alpha * self.get_ad(cutoff)
        hc = arg.conjugate().transpose()
        return sp.linalg.expm(arg - hc)

    def s(self, theta: complex, cutoff: int) -> sp.csc_array:
        """Single-mode squeezing operator

        Args:
            theta (complex): squeeze

        Returns:
            csc_array: operator matrix
        """
        ad = self.get_ad(cutoff)
        ad2 = ad @ ad
        arg = 0.5 * np.conjugate(theta) * ad2
        hc = arg.conjugate().transpose()
        return sp.linalg.expm(arg - hc)

    def s2(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_array:
        """Two-mode squeezing operator

        Args:
            theta (complex): squeezing factor

        Returns:
            csc_array: operator matrix
        """
        r, phi = np.abs(theta), np.angle(theta)

        # eq. 183 in arXiv:2407.10381
        arg = r * np.exp(1j * phi) * self.get_op("ad0 ad1", cutoff_a, cutoff_b)
        hc = arg.conjugate().transpose()
        return sp.linalg.expm(arg - hc)

    def s3(
        self, theta: complex, cutoff_a: int, cutoff_b: int, cutoff_c: int
    ) -> sp.csc_array:
        """Three-mode squeezing operator

        Args:
            theta: squeezing amount

        Returns:
            csc_array: operator matrix
        """
        r, phi = np.abs(theta), np.angle(theta)

        arg = (
            r
            * np.exp(1j * phi)
            * self.get_op("ad0 ad1 ad2", cutoff_a, cutoff_b, cutoff_c)
        )
        hc = arg.conjugate().transpose()
        return sp.linalg.expm(arg - hc)

    def bs(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_array:
        """Two-mode beam splitter operator

        Args:
            theta: phase

        Returns:
            csc_array: operator matrix
        """

        arg = theta * self.get_op("ad0 a1", cutoff_a, cutoff_b)
        hc = arg.conjugate().transpose()
        return sp.linalg.expm(arg - hc)

    def cr(self, theta: float, cutoff: int) -> sp.csc_array:
        """Controlled phase space rotation operator

        Args:
            theta (real): phase

        Returns:
            csc_array: operator matrix
        """
        arg = theta * 1j * sp.kron(Z, self.get_N(cutoff), format="csc")
        return sp.linalg.expm(arg)

    def crx(self, theta: float, cutoff: int) -> sp.csc_array:
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            csc_array: operator matrix
        """
        arg = theta * 1j * sp.kron(X, self.get_N(cutoff), format="csc")
        return sp.linalg.expm(arg)

    def cry(self, theta: float, cutoff: int) -> sp.csc_array:
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            csc_array: operator matrix
        """
        arg = theta * 1j * sp.kron(Y, self.get_N(cutoff), format="csc")
        return sp.linalg.expm(arg)

    def cd(self, alpha: complex, beta: complex | None, cutoff: int) -> sp.csc_array:
        """Controlled displacement operator

        Args:
            alpha (complex): displacement for qubit state 0
            beta (complex): displacement for qubit state 1. If None, use -alpha.

        Returns:
            csc_array: operator matrix
        """
        displace0 = self.d(alpha, cutoff)
        displace1 = self.d(beta or -alpha, cutoff)
        res = sp.kron(P0, displace0) + sp.kron(P1, displace1)
        return res.tocsc()

    def ecd(self, theta: complex, cutoff: int) -> sp.csc_array:
        """Echoed controlled displacement operator

        Args:
            theta (complex): displacement

        Returns:
            csc_array: operator matrix
        """
        return self.cd(theta, -theta, cutoff)

    def cbs(self, theta: complex, cutoff_a: int, cutoff_b: int) -> sp.csc_array:
        """Controlled phase two-mode beam splitter operator

        Args:
            theta (complex): beamsplitter phase

        Returns:
            csc_array: operator matrix
        """

        arg = self.bs(theta, cutoff_a, cutoff_b)
        res = sp.kron(P0, arg) + sp.kron(P1, arg.conjugate().transpose())
        return res.tocsc()

    def cschwinger(
        self,
        beta: float,
        theta_1: float,
        phi_1: float,
        theta_2: float,
        phi_2: float,
        cutoff_a: int,
        cutoff_b: int,
    ) -> sp.csc_array:
        """General form of a controlled Schwinger gate

        Args:
            params (real): [beta, theta_1, phi_1, theta_2, phi_2]

        Returns:
            csc_array: operator matrix
        """

        # Sx = (
        #     self.get_a1(cutoff_a, cutoff_b) * self.get_a2_dag(cutoff_a, cutoff_b)
        #     + self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)
        # ) / 2
        Sx = self.get_op("a0 ad1", cutoff_a, cutoff_b)
        Sx = (Sx + Sx.conjugate().transpose()) / 2

        # Sy = (
        #     self.get_a1(cutoff_a, cutoff_b) * self.get_a2_dag(cutoff_a, cutoff_b)
        #     - self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)
        # ) / (2 * 1j)
        Sy = self.get_op("a0 ad1", cutoff_a, cutoff_b)
        Sy = (Sy - Sy.conjugate().transpose()) / 2j

        # Sz = (
        #     self.get_a2_dag(cutoff_a, cutoff_b) * self.get_a2(cutoff_a, cutoff_b)
        #     - self.get_a1_dag(cutoff_a, cutoff_b) * self.get_a1(cutoff_a, cutoff_b)
        # ) / 2
        Sz = (
            self.get_op("n1", cutoff_a, cutoff_b)
            - self.get_op("n0", cutoff_a, cutoff_b)
        ) / 2

        sigma = (
            np.sin(theta_1) * np.cos(phi_1) * X
            + np.sin(theta_1) * np.sin(phi_1) * Y
            + np.cos(theta_1) * Z
        )
        S = (
            np.sin(theta_2) * np.cos(phi_2) * Sx
            + np.sin(theta_2) * np.sin(phi_2) * Sy
            + np.cos(theta_2) * Sz
        )
        arg = sp.kron(sigma, S).tocsc()

        return sp.linalg.expm(-1j * beta * arg).tocsc()

    def snap(self, theta: float, n: int, cutoff: int) -> sp.csc_array:
        """SNAP (Selective Number-dependent Arbitrary Phase) operator

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            csc_array: operator matrix
        """

        return self.multisnap(theta, n, cutoff)

    def csnap(self, theta: float, n: int, cutoff: int) -> sp.csc_array:
        """SNAP (Selective Number-dependent Arbitrary Phase) operator,
        with explicit sigma_z in exponential. Can be used to generate
        fock-number selective qubit rotations.

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            csc_array: operator matrix
        """

        return self.multicsnap(theta, n, cutoff)

    def multisnap(self, *args: int | float | np.integer | np.floating) -> sp.csc_array:
        """SNAP (Selective Number-dependent Arbitrary Phase) operator for multiple Fock states.
        Generates an arbitrary number of fock-number selective qubit rotations.

        Args:
            args (List[reals, integers]): [List of phases, List of Fock states in which the mode should acquire the associated phase]

        Returns:
            csr_matrix: operator matrix
        """
        *params, cutoff = args
        params = np.atleast_1d(params)

        if not is_int_type(cutoff):
            raise ValueError(f"Expected integer cutoff, got {cutoff}")

        thetas, ns = np.array_split(params, 2)
        ns = ns.astype(int)

        diags = np.ones(cutoff, dtype=complex)
        diags[ns] = np.exp(1j * thetas)
        return cast(
            sp.csc_array,
            sp.diags_array(diags, format="csc"),
        )

    def multicsnap(self, *args: int | float | np.integer | np.floating) -> sp.csc_array:
        """SNAP (Selective Number-dependent Arbitrary Phase) operator for multiple Fock states.
        Generates an arbitrary number of fock-number selective qubit rotations, with the qubit that accrues the geometric phase explicit.

        Args:
            args (List[reals, integers]): [List of phases, List of Fock states in which the mode should acquire the associated phase]

        Returns:
            csr_matrix: operator matrix
        """
        *params, cutoff = args
        params = np.atleast_1d(params)

        if not is_int_type(cutoff):
            raise ValueError(f"Expected integer cutoff, got {cutoff}")

        thetas, ns = np.array_split(params, 2)
        ns = ns.astype(int)

        diags = np.ones(cutoff, dtype=complex)
        diags[ns] = np.exp(1j * thetas)
        diags = np.concatenate([diags, diags.conj()])
        return cast(
            sp.csc_array,
            sp.diags_array(diags, format="csc"),
        )

    def sqr(self, *args: float) -> sp.csc_array:
        """SQR gate (Liu et al, arXiv 2024)

        This function assumes that the parameters (minus the cutoff) are concatenated, so it should
        have length 3*n, where n is the number of distinct fock states to condition on.

        Args:
            params: Gate parameters and cutoff, see `CVCircuit.cv_sqr` for the parameter structure

        Returns
            csc_array: The operator matrix
        """
        from qiskit.circuit.library import RGate

        *params, cutoff = args
        cutoff = int(cutoff)

        params = np.atleast_1d(params)
        theta, phi, fock_states = np.array_split(params, 3)
        fock_states = fock_states.astype(int)  # guaranteed by cv_sqr

        blocks = [sp.eye_array(2)] * cutoff
        for t, p, n in zip(theta, phi, fock_states):
            blocks[n] = sp.csc_array(RGate(t, p).to_matrix())

        # This matrix acts on the space Qumode x Qubit
        out = cast(sp.csc_array, sp.block_diag(blocks, format="csc"))

        # Rearrange to act on Qubit x Qumode for qiskit
        perm = np.arange(2 * cutoff).reshape(cutoff, 2).T.flatten()
        return out[perm, :][:, perm].tocsc()

    def pnr(self, max: int, cutoff: int) -> sp.csc_array:
        """Support gate for photon number readout (see Curtis et al., PRA (2021) and Wang et al., PRX (2020))

        Args:
            max (int): the period of the mapping

        Returns:
            csc_array: operator matrix
        """
        projector = sp.dok_array((cutoff, cutoff))
        # binary search
        for j in range(max // 2):
            for i in range(j, cutoff, max):
                # fill from right to left
                projector += self.get_projector(cutoff - (i + 1), cutoff)

        # Flip qubit if there is a boson present in any of the modes addressed by the projector
        arg = 1j * (-np.pi / 2) * sp.kron(X, projector).tocsc()
        return sp.linalg.expm(arg)

    def eswap(self, theta, cutoff_a: int, cutoff_b: int) -> sp.csc_array:
        """Exponential SWAP operator

        Args:
            theta (real): rotation

        Returns:
            csc_array: operator matrix
        """

        dim = cutoff_a * cutoff_b
        m = np.arange(cutoff_a)[:, None]
        n = np.arange(cutoff_b)[None, :]

        row_indices = n + (m * cutoff_a)
        col_indices = (n * cutoff_b) + m

        data = np.ones(dim)
        swap = sp.coo_array(
            (data, (row_indices.flatten(), col_indices.flatten())), shape=(dim, dim)
        )
        return sp.linalg.expm(1j * theta * swap.tocsc())

    def csq(self, theta: complex, cutoff: int) -> sp.csc_array:
        """Single-mode squeezing operator

        Args:
            theta (complex): squeeze

        Returns:
            csc_array: operator matrix
        """
        a = self.get_a(cutoff)
        a2 = a @ a
        arg = 0.5 * np.conj(theta) * a2
        hc = arg.conjugate().transpose()
        arg = sp.kron(Z, arg - hc)

        return sp.linalg.expm(arg).tocsc()

    def c_multiboson_sampling(self, max: int, cutoff: int) -> sp.csc_array:
        """SNAP gate creation for multiboson sampling purposes.

        Args:
            max (int): the period of the mapping

        Returns:
            csc_array: operator matrix
        """
        # Todo: is this correct?
        return self.get_eye(cutoff)

    def gate_from_matrix(self, matrix: ArrayLike) -> sp.csc_array:
        """Converts matrix into gate. Called using ParameterizedUnitaryGate.

        Args:
            matrix (list): the (unitary) matrix that you wish to convert into a gate

        Returns:
            csc_array: operator matrix
        """
        return sp.csc_array(matrix)

    def sum(self, scale: float, cutoff_a: int, cutoff_b: int) -> sp.csc_array:
        """Two-qumode sum gate

        Args:
            scale (real): arbitrary scale factor

        Returns:
            csc_array: operator matrix
        """
        a_mat = self.get_a(cutoff_a) + self.get_ad(cutoff_a)
        b_mat = self.get_ad(cutoff_b) - self.get_a(cutoff_b)
        arg = (scale / 2) * sp.kron(a_mat, b_mat)
        return sp.linalg.expm(arg)

    def csum(self, scale: float, cutoff_a: int, cutoff_b: int) -> sp.csc_array:
        """Conditional two-qumode sum gate

        Args:
            scale (real): arbitrary scale factor

        Returns:
            csc_array: operator matrix
        """
        a_mat = self.get_a(cutoff_a) + self.get_ad(cutoff_a)
        b_mat = self.get_ad(cutoff_b) - self.get_a(cutoff_b)
        arg = (scale / 2) * sp.kron(a_mat, b_mat)
        arg = sp.kron(Z, arg)
        return sp.linalg.expm(arg)

    def jc(self, theta: float, phi: float, cutoff: int) -> sp.csc_array:
        """Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)

        Returns:
            csc_array: operator matrix
        """
        arg = np.exp(1j * phi) * sp.kron(SMINUS, self.get_ad(cutoff))
        hc = arg.conjugate().transpose()
        arg = -1j * theta * (arg + hc)
        return sp.linalg.expm(arg)

    def ajc(self, theta: float, phi: float, cutoff: int) -> sp.csc_array:
        """Anti-Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)

        Returns:
            csc_array: operator matrix
        """
        arg = np.exp(1j * phi) * sp.kron(SPLUS, self.get_ad(cutoff))
        hc = arg.conjugate().transpose()
        arg = -1j * theta * (arg + hc)
        return sp.linalg.expm(arg)

    def rb(self, theta: complex, cutoff: int):
        """Rabi interaction gate

        Args:
            theta (complex): arbitrary scale factor

        Returns:
            csc_array: operator matrix
        """
        arg = theta * self.get_ad(cutoff)
        hc = arg.conjugate().transpose()
        arg = sp.kron(X, -1j * (arg + hc)).tocsc()
        return sp.linalg.expm(arg)
