import numpy
import qiskit
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
sigma_plus = numpy.array([[0, 1], [0, 0]])
sigma_minus = numpy.array([[0, 0], [1, 0]])


class ParameterizedUnitaryGate(Gate):
    """UnitaryGate sublcass that stores the operator matrix for later reference by animation utility."""

    def __init__(
        self, op_func, params, num_qubits, label=None, duration=100, unit="ns"
    ):
        """Initialize ParameterizedUnitaryGate

        FIXME - Use real duration & units

        Args:
            op_func (function): function to build operator matrix
            params (List): List of parameters to pass to op_func to build
                operator matrix (supports instances of Qiskit Parameter to be
                bound later)
            num_qubits (int): Number of qubits in the operator -- this would
                likely equate to (num_qubits_per_qumode * num_qumodes + num_ancilla).
            label (string, optional): Gate name. Defaults to None.
            duration (int, optional): Duration of gate used for noise modeling. Defaults to 100.
            unit (string, optional): Unit of duration (only supports those allowed by Qiskit).
        """
        super().__init__(name=label, num_qubits=num_qubits, params=params, label=label)

        self.op_func = op_func

        self._parameterized = any(
            isinstance(param, ParameterExpression) and param.parameters
            for param in params
        )

        self.duration = duration
        self.unit = unit

    def __array__(self, dtype=None):
        """Call the operator function to build the array using the bound parameter values."""
        # return self.op_func(*map(complex, self.params)).toarray()
        values = []
        for param in self.params:
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
        if isinstance(parameter, complex) or (
            isinstance(parameter, ParameterExpression) and not parameter.is_real()
        ):
            return parameter
        elif isinstance(parameter, str):  # accept strings as-is
            return parameter
        else:
            return super().validate_parameter(parameter)

    def calculate_matrix(
        self, current_step: int = 1, total_steps: int = 1, keep_state: bool = False
    ):
        """Calculate the operator matrix by executing the selected function.
        Increment the parameters based upon the current and total steps.
        Args:
            current_step (int, optional): Current step within total_steps. Defaults to 1.
            total_steps (int, optional): Total steps to increment parameters. Defaults to 1.
        Returns:
            ndarray: operator matrix
        """
        if self.is_parameterized():
            raise NotImplementedError(
                "Unable to calculate incremental operator matrices for parameterized gate"
            )

        values = self.calculate_frame_params(current_step, total_steps, keep_state)

        # if self.inverse:
        #     result = scipy.sparse.linalg.inv(self.op_func(*values))
        # else:
        #     result = self.op_func(*values)
        result = self.op_func(*values)

        if hasattr(result, "toarray"):
            result = result.toarray()

        return result


def __calculate_frame_params(
    self, current_step: int = 1, total_steps: int = 1, keep_state: bool = False
):
    """Calculate the parameters at the current step. Return a tuples of the values."""
    if keep_state:
        param_fraction = 1 / total_steps
    else:
        param_fraction = current_step / total_steps

    values = []
    for param in self.params:
        values.append(param * param_fraction)

    return tuple(values)


def __calculate_frame_duration(
    self, current_step: int = 1, total_steps: int = 1, keep_state: bool = False
):
    """Calculate the duration at the current step. Return a tuple of the (duration, unit)."""
    frame_duration = None

    if self.duration:
        if keep_state:
            fraction = 1 / total_steps
        else:
            fraction = current_step / total_steps

        frame_duration = self.duration * fraction
    
    return frame_duration, self.unit


# Monkey patch Qiskit Instruction to support animating base Qiskit Instruction
qiskit.circuit.instruction.Instruction.calculate_frame_params = __calculate_frame_params
qiskit.circuit.instruction.Instruction.calculate_frame_duration = __calculate_frame_duration


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

    def r(self, theta):
        """Phase space rotation operator

        Args:
            theta (real): rotation

        Returns:
            ndarray: operator matrix
        """
        arg = 1j * theta * self.N

        return scipy.sparse.linalg.expm(arg)

    def d(self, theta):
        """Displacement operator

        Args:
            theta (real): displacement

        Returns:
            ndarray: operator matrix
        """
        arg = (theta * self.a_dag) - (numpy.conjugate(theta) * self.a)

        return scipy.sparse.linalg.expm(arg)

    def s(self, theta):
        """Single-mode squeezing operator

        Args:
            theta (real): squeeze

        Returns:
            ndarray: operator matrix
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
            ndarray: operator matrix
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
            ndarray: operator matrix
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
            ndarray: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(zQB, self.N).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def crx(self, theta):
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            ndarray: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(xQB, self.N).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def cry(self, theta):
        """Controlled phase space rotation operator around sigma^x

        Args:
            theta (real): phase

        Returns:
            ndarray: operator matrix
        """
        arg = theta * 1j * scipy.sparse.kron(yQB, self.N).tocsc()

        return scipy.sparse.linalg.expm(arg)

    def cd(self, theta, beta=None):
        """Controlled displacement operator

        Args:
            theta (real): displacement for qubit state 0
            beta (real): displacement for qubit state 1. If None, use -alpha.

        Returns:
            ndarray: operator matrix
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
            ndarray: operator matrix
        """
        argm = (theta * self.a_dag) - (numpy.conjugate(theta) * self.a)
        arg = scipy.sparse.kron(zQB, argm)

        return scipy.sparse.linalg.expm(arg)

    def cbs(self, theta):
        """Controlled phase two-mode beam splitter operator

        Args:
            theta (real): real phase

        Returns:
            ndarray: operator matrix
        """
        self.a1 = scipy.sparse.kron(self.a, self.eye).tocsc()
        self.a2 = scipy.sparse.kron(self.eye, self.a).tocsc()
        self.a1_dag = self.a1.conjugate().transpose().tocsc()
        self.a2_dag = self.a2.conjugate().transpose().tocsc()

        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        argm = theta * (a1dag2 - a12dag)
        arg = scipy.sparse.kron(zQB, argm).tocsc()

        return scipy.sparse.linalg.expm(arg)

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
        arg = theta * 1j * sparse_projector.tocsc()
        return scipy.sparse.linalg.expm(arg)

    def eswap(self, theta):
        """Exponential SWAP operator

        Args:
            theta (real): rotation

        Returns:
            ndarray: operator matrix
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
            ndarray: operator matrix
        """
        a_sqr = self.a * self.a
        a_dag_sqr = self.a_dag * self.a_dag
        arg = scipy.sparse.kron(zQB, 0.5 * ((numpy.conjugate(theta) * a_sqr) - (theta * a_dag_sqr))).tocsc()

        return scipy.sparse.linalg.expm(arg)


    def testqubitorderf(self, phi):

        arg = 1j * phi * scipy.sparse.kron(xQB, idQB)
        return scipy.sparse.linalg.expm(arg)
