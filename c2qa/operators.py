import numpy
from qiskit.extensions.unitary import UnitaryGate
from qiskit.quantum_info import Operator
import scipy.sparse
import scipy.sparse.linalg

xQB = numpy.array([[0, 1], [1, 0]])
yQB = numpy.array([[0, 1j], [-1j, 0]])
zQB = numpy.array([[1, 0], [0, -1]])
idQB = numpy.array([[1, 0], [0, 1]])


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

        # For use with SNAP gate
        self.ket_n = numpy.zeros(cutoff)

        # For use with eSWAP
        self.mat = numpy.zeros([cutoff * cutoff, cutoff * cutoff])
        for j in range(cutoff):
            for i in range(cutoff):
                self.mat[i + (j * cutoff)][i * cutoff + j] = 1
        self.sparse_mat = scipy.sparse.csr_matrix(self.mat)

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

    def bs1(self, g):
        """Two-mode beam splitter

        Args:
            g (real): real phase

        Returns:
            ndarray: operator matrix
        """
        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        arg = (g / 2) * (a12dag - a1dag2)

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

    def controlledparity1(self):
        """Controlled parity operator

        Returns:
            ndarray: operator matrix
        """
        arg1 = scipy.sparse.kron(zQB, self.N)
        arg2 = scipy.sparse.kron(idQB, self.N)
        arg = arg1 + arg2
        return scipy.sparse.linalg.expm(1j * (numpy.pi / 2) * arg)

    def snap1(self, theta, n):
        """SNAP (Selective Number-dependent Arbitrary Phase) operator

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase

        Returns:
            ndarray: operator matrix
        """
        self.ket_n[n] = 1
        projector = numpy.outer(self.ket_n, self.ket_n)
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

    def photonNumberControlledQubitRotation1(self, theta, n, qubit_rotation):
        """Photon Number Controlled Qubit Rotation operator

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase
            qubit_rotation (string): Pauli matrix for the qubit rotation

        Returns:
            ndarray: operator matrix
        """

        if qubit_rotation=="X":
            rot=xQB
        elif qubit_rotation=="Y":
            rot=yQB
        elif qubit_rotation=="Z":
            rot=zQB
        else:
            print("Please choose pauli X, Y or Z (capitals, ie. 'Y') for the qubit rotation.")


        self.ket_n[n] = 1
        projector = numpy.outer(self.ket_n, self.ket_n)
        sparse_projector = scipy.sparse.csr_matrix(projector)
        argm = theta * 1j * sparse_projector

        arg = scipy.sparse.kron(rot, argm)

        return scipy.sparse.linalg.expm(arg)


    def snap(self, theta, n):
        # be careful about adding an extra qubit in here which is in state 1 which will get the negative phase.
        # you can do all the photon number states on one cavity on one ancilla, but each cavity needs an ancilla
        twoOP = scipy.sparse.csr_matrix([[0, 0 ,0 ,0], [0, 0 ,0 ,0], [0 ,0 ,1 ,0], [0, 0 ,0 ,0]])
        arg=numpy.pi*1j*twoOP
        return scipy.sparse.linalg.expm(arg)

    def photonNumberControlledQubitRotation(self, theta, n, qubit_rotation):
        yQB = numpy.array([[0, 1j], [-1j, 0]])
        oneOP = scipy.sparse.csr_matrix([[0, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])
        arg1=-1j*numpy.pi*oneOP/2
        arg = scipy.sparse.kron(yQB, arg1)
        return scipy.sparse.linalg.expm(arg)

    def controlledparity(self):
        zQB = numpy.array([[1, 0], [0, -1]])
        idQB = numpy.array([[1, 0], [0, 1]])
        arg1 = scipy.sparse.kron(zQB,self.N)
        arg2 = scipy.sparse.kron(idQB, self.N)
        arg = arg1 + arg2
        return scipy.sparse.linalg.expm(1j*(numpy.pi/2)*arg)

    def bs(self, g):
        """Two-mode beam splitter opertor"""
        # a12dag = scipy.sparse.matmul(self.a1, self.a2_dag)
        # a1dag2 = scipy.sparse.matmul(self.a1_dag, self.a2)
        a12dag = self.a1 * self.a2_dag
        a1dag2 = self.a1_dag * self.a2

        # FIXME -- See Steve 5.4
        #   phi as g(t)
        #   - as +, but QisKit validates that not being unitary
        # arg = (g * -1j * a12dag) - (np.conjugate(g * -1j) * a1dag2)
        # arg = 1j *((g * a12dag) - (g * a1dag2))
        arg = (g / 2) * (a1dag2 - a12dag)
        return scipy.sparse.linalg.expm(arg)