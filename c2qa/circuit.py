from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from c2qa.operators import CVOperators
from c2qa.qumoderegister import QumodeRegister
import numpy
#from thewalrus.fock_gradients import displacement


class CVCircuit(QuantumCircuit):
    def __init__(self, qmr: QumodeRegister, qr: QuantumRegister, cr: ClassicalRegister):
        super().__init__(qmr.qreg, qr, cr)

        self.qmr = qmr
        self.qr = qr
        self.cr = cr

        self.ops = CVOperators(self.qmr)


    def initialize(self, fock_states: int):
        for qumode, n in enumerate(fock_states):
            if n >= self.qmr.cutoff:
                raise ValueError("The parameter n should be lower than the cutoff")
            
            vector = numpy.zeros((self.qmr.cutoff,))
            vector[n] = 1

            super().initialize(vector, self.qmr[qumode])

    def cv_bs(self, phi, qumode_a, qumode_b):
        operator = self.ops.BS(phi)
        
        super().unitary(obj = operator, qubits = qumode_a + qumode_b, label = 'BS')

    def cv_d(self, alpha, qumode):       
        operator = self.ops.D(alpha)
        # operator = displacement(r = numpy.abs(alpha), phi = numpy.angle(alpha), cutoff = self.cutoff)

        super().unitary(obj = operator, qubits = qumode, label = 'D')

    def cv_r(self, phi, qumode):
        operator = self.ops.R(phi)
        
        super().unitary(obj = operator, qubits = qumode, label = 'R')

    def cv_s(self, z, qumode):
        operator = self.ops.S(z)
       
        super().unitary(obj = operator, qubits = qumode, label = 'S')

    def cv_s2(self, z, qumode_a, qumode_b):
        operator = self.ops.S2(z)
        
        super().unitary(obj = operator, qubits = qumode_a + qumode_b, label = 'S2')

