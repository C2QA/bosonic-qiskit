import math
from numbers import Complex


import numpy
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.parameter import ParameterExpression
from qiskit.extensions.unitary import UnitaryGate
import scipy.sparse
import scipy.sparse.linalg


xQB = numpy.array([[0, 1], [1, 0]])
yQB = numpy.array([[0, 1j], [-1j, 0]])
zQB = numpy.array([[1, 0], [0, -1]])
idQB = numpy.array([[1, 0], [0, 1]])


class ParameterizedUnitaryGate(Gate):
    """UnitaryGate sublcass that stores the operator matrix for later reference by animation utility."""

    def __init__(self, op_func, params, num_qubits, label=None, duration=100, unit="ns"):
        """Initialize ParameterizedUnitaryGate

        FIXME - Use real duration & units

        Args:
            op_func (function): function to build operator matrix
            params (List): List of parameters to pass to op_func to build operator matrix (supports instances of Qiskit Parameter to be bound later)
            num_qubits (int): Number of qubits in the operator -- this would likely equate to (num_qubits_per_qumode * num_qumodes + num_ancilla).
            label (string, optional): Gate name. Defaults to None.
            duration (int, optional): Duration of gate used for noise modeling. Defaults to 100.
            unit (string, optional): Unit of duration (only supports those allowed by Qiskit).
        """
        super().__init__(name=label, num_qubits=num_qubits, params=params, label=label)

        self.op_func = op_func

        self._parameterized = any(
            isinstance(param, ParameterExpression) and param.parameters for param in params
        )

        self.duration = duration
        self.unit = unit

    def __array__(self, dtype=None):
        """Call the operator function to build the array using the bound parameter values."""
        # return self.op_func(*map(complex, self.params)).toarray()
        values = []
        for param in self.params:
            if isinstance(param, ParameterExpression):
                values.append(float(param))
            else:
                values.append(param)
        values = tuple(values)

        return self.op_func(*values).toarray()

    def _define(self):
        mat = self.to_matrix()
        q = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (UnitaryGate(mat, self.label), [i for i in q], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def validate_parameter(self, parameter):
        """Gate parameters should be int, float, or ParameterExpression"""
        if isinstance(parameter, Complex):
            return parameter
        else:
            return super().validate_parameter(parameter)

    def calculate_matrix(self, current_step: int = 1, total_steps: int = 1, keep_state: bool = False):
        """Calculate the operator matrix by executing the selected function.
        Increment the parameters based upon the current and total steps.
        Args:
            current_step (int, optional): Current step within total_steps. Defaults to 1.
            total_steps (int, optional): Total steps to increment parameters. Defaults to 1.
        Returns:
            ndarray: operator matrix
        """
        if self.is_parameterized():
            raise NotImplementedError("Unable to calculate incremental operator matrices for parameterized gate")

        values = self.calculate_params(current_step, total_steps, keep_state)

        # if self.inverse:
        #     result = scipy.sparse.linalg.inv(self.op_func(*values))
        # else:
        #     result = self.op_func(*values)
        result = self.op_func(*values)

        if hasattr(result, "toarray"):
            result = result.toarray()

        return result

    def calculate_params(self, current_step: int = 1, total_steps: int = 1, keep_state: bool = False):
        if keep_state:
            param_fraction = 1 / total_steps
        else:
            param_fraction = current_step / total_steps

        values = []
        for param in self.params:
            values.append(param * param_fraction)

        return tuple(values)


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
        self.a = scipy.sparse.spdiags(data=data, diags=[1], m=len(data), n=len(data))

        # Creation operator
        self.a_dag = self.a.conjugate().transpose()

        # Number operator
        # self.N = scipy.sparse.matmul(self.a_dag, self.a)
        self.N = self.a_dag * self.a

        self.eye = scipy.sparse.eye(cutoff)

        # 2-qumodes operators
        if num_qumodes > 1:
            self.a1 = scipy.sparse.kron(self.a, self.eye)
            self.a2 = scipy.sparse.kron(self.eye, self.a)
            self.a1_dag = self.a1.conjugate().transpose()
            self.a2_dag = self.a2.conjugate().transpose()

        # For use with eSWAP
        self.mat = numpy.zeros([cutoff * cutoff, cutoff * cutoff])
        for j in range(cutoff):
            for i in range(cutoff):
                self.mat[i + (j * cutoff)][i * cutoff + j] = 1
        self.sparse_mat = scipy.sparse.csr_matrix(self.mat)

        self.cutoff_value = cutoff

    def d(self, alpha):
        """Displacement operator

        Args:
            alpha (real): displacement

        Returns:
            ndarray: operator matrix
        """
        arg = (alpha * self.a_dag) - (numpy.conjugate(alpha) * self.a)

        return scipy.sparse.linalg.expm(arg)

    def cd(self, alpha, beta=None):
        """Displacement operator

        Args:
            alpha (real): displacement for qubit state 0
            beta (real): displacement for qubit state 1. If None, use -alpha.

        Returns:
            ndarray: operator matrix
        """
        displace0 = (alpha * self.a_dag) - (numpy.conjugate(alpha) * self.a)
        if beta is None:
            beta = -alpha
        displace1 = (beta * self.a_dag) - (numpy.conjugate(beta) * self.a)


        return scipy.sparse.kron((idQB+zQB)/2,scipy.sparse.linalg.expm(displace0)) + scipy.sparse.kron((idQB-zQB)/2,scipy.sparse.linalg.expm(displace1))

    def ecd(self, alpha):
        """Displacement operator

        Args:
            alpha (real): displacement

        Returns:
            ndarray: operator matrix
        """
        argm = (alpha * self.a_dag) - (numpy.conjugate(alpha) * self.a)
        arg = scipy.sparse.kron(zQB, argm)/2

        return scipy.sparse.linalg.expm(arg)

    def rh1(self, alpha):
        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        argm = 1j * alpha *(a1dag2 + a12dag)
        arg = scipy.sparse.kron(zQB, argm)

        return scipy.sparse.linalg.expm(arg)

    def rh2(self, alpha):
        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        argm = alpha *(a12dag - a1dag2)
        arg = scipy.sparse.kron(zQB, argm)

        return scipy.sparse.linalg.expm(arg)

    def s(self, zeta):
        """Single-mode squeezing operator

        Args:
            zeta (real): squeeze

        Returns:
            ndarray: operator matrix
        """
        a_sqr = self.a * self.a
        a_dag_sqr = self.a_dag * self.a_dag
        arg = 0.5 * ((numpy.conjugate(zeta) * a_sqr) - (zeta * a_dag_sqr))

        return scipy.sparse.linalg.expm(arg)

    def s2(self, g):
        """Two-mode squeezing operator

        Args:
            g (real): mutliplied by 1j to yield imaginary phase

        Returns:
            ndarray: operator matrix
        """
        a12_dag = self.a1_dag * self.a2_dag
        a12 = self.a1 * self.a2

        arg = (numpy.conjugate(g * 1j) * a12_dag) - (g * 1j * a12)

        return scipy.sparse.linalg.expm(arg)

    def bs(self, theta):
        """Two-mode beam splitter

        Args:
            theta: phase

        Returns:
            ndarray: operator matrix
        """
        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        arg =  (theta * a1dag2 - numpy.conj(theta) * a12dag)

        return scipy.sparse.linalg.expm(arg)

    # def bs(self, g):
    #     """Two-mode beam splitter
    #
    #     Args:
    #         g (real): real phase
    #
    #     Returns:
    #         ndarray: operator matrix
    #     """
    #     a12dag = self.a1 * self.a2_dag
    #     a1dag2 = self.a1_dag * self.a2
    #
    #     arg = (g / 2) * (a1dag2 - a12dag)
    #
    #     return scipy.sparse.linalg.expm(arg)

    # def bs_im(self, weight):
    #     """Two-mode beam splitter
    #
    #     Args:
    #         weight (real): mutliplied by 1j to yield imaginary alpha
    #
    #     Returns:
    #         ndarray: operator matrix
    #     """
    #     a12dag = self.a1 * self.a2_dag
    #     a1dag2 = self.a1_dag * self.a2
    #     alpha = (weight * 1j)
    #
    #     arg = 1j * (alpha * a12dag) - (numpy.conjugate(alpha) * a1dag2)
    #
    #     return scipy.sparse.linalg.expm(arg)

    def cpbs(self, g):
        """Controlled phase two-mode beam splitter

        Args:
            g (real): real phase

        Returns:
            ndarray: operator matrix
        """

        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        argm = (g / 2) * (a1dag2 - a12dag)
        arg = scipy.sparse.kron(zQB, argm)

        return scipy.sparse.linalg.expm(arg)

    def cpbs_z2vqe(self, g):
        """Controlled phase two-mode beam splitter

        Args:
            g (real): real phase

        Returns:
            ndarray: operator matrix
        """

        # NOT CHANGED YET - this is a copy of the cpbs function.

        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        argm = (g / 2) * (a1dag2 - a12dag)
        arg = scipy.sparse.kron(zQB, argm)

        return scipy.sparse.linalg.expm(arg)

    def r(self, theta):
        """Phase space rotation operator

        Args:
            theta (real): rotation

        Returns:
            ndarray: operator matrix
        """
        arg = 1j * theta * self.N

        return scipy.sparse.linalg.expm(arg)

    def qubitDependentCavityRotation(self, theta):
        """Qubit dependent cavity rotation

        Args:
            theta (real): phase

        Returns:
            ndarray: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(zQB, self.N)

        return scipy.sparse.linalg.expm(arg.tocsc())

    def qubitDependentCavityRotationX(self, theta):
        """Qubit dependent cavity rotation

        Args:
            theta (real): phase

        Returns:
            ndarray: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(xQB, self.N)
        return scipy.sparse.linalg.expm(arg.tocsc())

    def qubitDependentCavityRotationY(self, theta):
        """Qubit dependent cavity rotation

        Args:
            theta (real): phase

        Returns:
            ndarray: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(yQB, self.N)
        return scipy.sparse.linalg.expm(arg.tocsc())

    def controlledparity(self, theta):
        """Controlled parity operator
        Rotates the mode if the state of the qubit is such that zQB doesn't give a phase

        Returns:
            ndarray: operator matrix
        """
        arg1 = scipy.sparse.kron(zQB, self.N)
        arg2 = scipy.sparse.kron(idQB, self.N)
        arg = arg1 + arg2
        return scipy.sparse.linalg.expm(1j * theta * arg)

    def snap(self, theta, n):
        """SNAP (Selective Number-dependent Arbitrary Phase) operator

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            ndarray: operator matrix
        """

        ket_n = numpy.zeros(self.cutoff_value)
        ket_n[n] = 1
        projector = numpy.outer(ket_n, ket_n)
        sparse_projector = scipy.sparse.csr_matrix(projector)
        arg = theta * 1j * sparse_projector
        return scipy.sparse.linalg.expm(arg)

    def eswap(self, theta):
        """Exponential SWAP

        Args:
            theta (real): rotation

        Returns:
            ndarray: operator matrix
        """
        arg = 1j * (theta / 2) * self.sparse_mat

        return scipy.sparse.linalg.expm(arg)

    def photonNumberControlledQubitRotation(self, theta, n, qubit_rotation):
        """Photon Number Controlled Qubit Rotation operator
        Rotates the qubit if the mode has a set fock state.

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase
            qubit_rotation (string): Pauli matrix for the qubit rotation

        Returns:
            ndarray: operator matrix
        """

        if qubit_rotation=="X":
            rot=xQB
            print("Somehow the X rotation doesn't work")
        elif qubit_rotation=="Y":
            rot=yQB
        elif qubit_rotation=="Z":
            rot=zQB
            print("Somehow the X rotation doesn't work")
        else:
            print("Please choose pauli X, Y or Z (capitals, ie. 'Y') for the qubit rotation.")

        ket_n = numpy.zeros(self.cutoff_value)
        ket_n[n] = 1
        projector = numpy.outer(ket_n, ket_n)
        sparse_projector = scipy.sparse.csr_matrix(projector)
        argm = theta * 1j * sparse_projector

        arg = scipy.sparse.kron(rot, argm)

        return scipy.sparse.linalg.expm(arg)


