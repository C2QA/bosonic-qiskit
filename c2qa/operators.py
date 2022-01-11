import numpy
from qiskit.extensions.unitary import UnitaryGate
from qiskit.quantum_info import Operator
import scipy.sparse
import scipy.sparse.linalg


class ParameterizedOperator(Operator):
    """Support parameterizing operators for circuit animations."""

    def __init__(self, op_func, *params, inverse: bool = False):
        """Initialize ParameterizedOperator.

        Args:
            op_func (function): function to call to generate operator matrix
            params (tuple): function parameters
            inverse (bool): True to caclualte the inverse of the operator matrix
        """

        super().__init__(op_func(*params).toarray())

        self.op_func = op_func
        self.params = params
        self.inverse = inverse

    def calculate_matrix(self, current_step: int = 1, total_steps: int = 1):
        """Calculate the operator matrix by executing the selected function.
        Increment the parameters based upon the current and total steps.

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

        if self.inverse:
            result = scipy.sparse.linalg.inv(self.op_func(*values))
        else:
            result = self.op_func(*values)

        return result.toarray()


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

        self.eye = scipy.sparse.eye(cutoff)

        # 2-qumodes operators
        if num_qumodes > 1:
            self.a1 = scipy.sparse.kron(self.a, self.eye)
            self.a2 = scipy.sparse.kron(self.eye, self.a)
            self.a1_dag = self.a1.conjugate().transpose()
            self.a2_dag = self.a2.conjugate().transpose()

    def d(self, alpha):
        """Displacement operator

        Args:
            alpha (real): displacement

        Returns:
            ndarray: operator matrix
        """
        arg = (alpha * self.a_dag) - (numpy.conjugate(alpha) * self.a)

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

    def bs(self, g):
        """Two-mode beam splitter

        Args:
            g (real): real phase

        Returns:
            ndarray: operator matrix
        """
        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        arg = (g/2) * (a12dag - a1dag2)

        return scipy.sparse.linalg.expm(arg)

    def bs_im(self, weight):
        """Two-mode beam splitter

        Args:
            weight (real): mutliplied by 1j to yield imaginary alpha

        Returns:
            ndarray: operator matrix
        """
        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2
        alpha = (weight * 1j)

        arg = 1j * (alpha * a12dag) - (numpy.conjugate(alpha) * a1dag2)

        return scipy.sparse.linalg.expm(arg)

    def cpbs(self, g):
        """Controlled phase two-mode beam splitter

        Args:
            g (real): real phase

        Returns:
            ndarray: operator matrix
        """
        zQB = np.array([[1, 0], [0, -1]])

        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        argm = (g/2) * (a1dag2 - a12dag)
        arg = scipy.sparse.kron(zQB,argm)

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

    def qubitDependentCavityRotation(self):
        """Qubit dependent cavity rotation

        Returns:
            ndarray: operator matrix
        """
        zQB = (1 / 2) * numpy.array([[1, 0], [0, -1]])
        arg=numpy.pi*1j*scipy.sparse.kron(zQB,self.N)
        return scipy.sparse.linalg.expm(arg.tocsc())