import numpy
from qiskit.extensions.unitary import UnitaryGate
from qiskit.quantum_info import Operator
import scipy.sparse
import scipy.sparse.linalg


class ParameterizedOperator(Operator):
    """Support parameterizing operators for circuit animations."""

    def __init__(self, op_func, *params):
        """Initialize ParameterizedOperator.

        Args:
            op_func (function): function to call to generate operator matrix
            params (tuple): function parameters
        """

        super().__init__(op_func(*params).toarray())

        self.op_func = op_func
        self.params = params

    def calculate_matrix(self, current_step: int = 1, total_steps: int = 1):
        """Calculate the operator matrix by executing the selected function. Increment the parameters based upon the current and total steps.

        Args:
            current_step (int, optional): Current step within total_steps. Defaults to 1.
            total_steps (int, optional): Total steps to increment parameters. Defaults to 1.

        Returns:
            ndarray: operator matrix
        """
        param_fraction = current_step / total_steps

        values = []
        for param in self.params:
            values.append(param * param_fraction)

        values = tuple(values)

        return self.op_func(*values).toarray()


class CVGate(UnitaryGate):
    """UnitaryGate sublcass that stores the operator matrix for later reference by animation utility."""

    def __init__(self, data, label=None):
        """Initialize CVGate

        Args:
            data (ndarray): operator matrix
            label (string, optional): Gate name. Defaults to None.
        """
        super().__init__(data, label)

        self.op = data


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

        # 2-qumodes operators
        if num_qumodes > 1:
            eye = scipy.sparse.eye(cutoff)
            self.a1 = scipy.sparse.kron(self.a, eye)
            self.a2 = scipy.sparse.kron(eye, self.a)
            self.a1_dag = self.a1.conjugate().transpose()
            self.a2_dag = self.a2.conjugate().transpose()

    def bs(self, g):
        """Two-mode beam splitter opertor

        Args:
            g (real): real phase

        Returns:
            ndarray: operator matrix
        """
        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        # FIXME -- See Steve 5.4
        #   phi as g(t)
        #   - as +, but QisKit validates that not being unitary
        arg = (g * -1j * a12dag) - (numpy.conjugate(g * -1j) * a1dag2)

        return scipy.sparse.linalg.expm(arg)

    def d(self, alpha):
        """Displacement operator

        Args:
            alpha (real): displacement

        Returns:
            ndarray: operator matrix
        """
        arg = (alpha * self.a_dag) - (numpy.conjugate(alpha) * self.a)

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
            g (real): squeeze

        Returns:
            ndarray: operator matrix
        """
        a12_dag = self.a1_dag * self.a2_dag
        a12 = self.a1 * self.a2

        # FIXME -- See Steve 5.7
        #   zeta as g(t)
        #   use of imaginary, but QisKit validates that is not unitary
        arg = (numpy.conjugate(g) * a12_dag) - (g * a12)

        return scipy.sparse.linalg.expm(arg)
