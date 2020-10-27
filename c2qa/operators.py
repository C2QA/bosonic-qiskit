from c2qa.qumoderegister import QumodeRegister
import numpy as np
from scipy.linalg import expm

class CVOperators:
    def __init__(self, qmr: QumodeRegister):        
        I = np.eye(qmr.cutoff)
        
        # Annihilation operator
        self.a = np.zeros((qmr.cutoff, qmr.cutoff))
        for i in range(qmr.cutoff - 1):
            self.a[i, i + 1]= np.sqrt(i + 1)

        # Creation operator
        self.a_dag = self.a.conj().T

        # Number operator
        self.N = np.matmul(self.a_dag, self.a)

        # 2-qumodes operators
        self.a1 = np.kron(self.a, I)
        self.a2 = np.kron(I, self.a)
        self.a1_dag = self.a1.conj().T
        self.a2_dag = self.a2.conj().T


    def bs(self, phi):
        """ Two-mode beam splitter opertor """
        a12dag = np.matmul(self.a1, self.a2_dag)
        a1dag2 = np.matmul(self.a1_dag, self.a2)

        # FIXME -- See Steve 5.4
        #   phi as g(t)
        #   - as +, but QisKit validates that not being unitary
        arg = (phi * a12dag) - (np.conjugate(phi) * a1dag2)

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


    def s2(self, zeta):
        """ Two-mode squeezing operator """
        a12_dag = np.matmul(self.a1_dag, self.a2_dag)
        a12 = np.matmul(self.a1, self.a2)

        # FIXME -- See Steve 5.7
        #   zeta as g(t)
        #   use of imaginary, but QisKit validates that is not unitary
        arg = (np.conjugate(zeta) * a12_dag) - (zeta * a12)

        return expm(arg)
