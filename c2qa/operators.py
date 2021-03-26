import numpy as np
from scipy.linalg import expm


class CVOperators:
    def __init__(self, cutoff: int, num_qumodes: int):
        # FIXME verify sizes of matrices/arrays are correct. 
        #   Large cutoffs make 

        # Annihilation operator
        self.a = np.diag(np.sqrt(range(1, cutoff)), k=1)

        # Creation operator
        self.a_dag = self.a.conj().T

        # Number operator
        self.N = np.matmul(self.a_dag, self.a)

        # 2-qumodes operators
        if num_qumodes > 1:
            eye = np.eye(cutoff)
            self.a1 = np.kron(self.a, eye)  # TODO why is numpy.kron make 3D array? It runs out of memory at larger cutoffs.
            self.a2 = np.kron(eye, self.a)
            self.a1_dag = self.a1.conj().T
            self.a2_dag = self.a2.conj().T

    def bs(self, g):
        """ Two-mode beam splitter opertor """
        a12dag = np.matmul(self.a1, self.a2_dag)
        a1dag2 = np.matmul(self.a1_dag, self.a2)

        # FIXME -- See Steve 5.4
        #   phi as g(t)
        #   - as +, but QisKit validates that not being unitary
        arg = (g * -1j * a12dag) - (np.conjugate(g * -1j) * a1dag2)

        return expm(arg)

    def d(self, alpha):
        """ Displacement operator """
        arg = (alpha * self.a_dag) - (np.conjugate(alpha) * self.a)

        return expm(arg)

    def r(self, theta):
        """ Phase space rotation operator """
        arg = 1j * theta * self.N

        return expm(arg)

    def s(self, zeta):
        """ Single-mode squeezing operator """
        a_sqr = np.matmul(self.a, self.a)
        a_dag_sqr = np.matmul(self.a_dag, self.a_dag)
        arg = 0.5 * ((np.conjugate(zeta) * a_sqr) - (zeta * a_dag_sqr))

        return expm(arg)

    def s2(self, g):
        """ Two-mode squeezing operator """
        a12_dag = np.matmul(self.a1_dag, self.a2_dag)
        a12 = np.matmul(self.a1, self.a2)

        # FIXME -- See Steve 5.7
        #   zeta as g(t)
        #   use of imaginary, but QisKit validates that is not unitary
        arg = (np.conjugate(g) * a12_dag) - (g * a12)

        return expm(arg)
