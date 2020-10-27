from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from c2qa.operators import CVOperators
from c2qa.qumoderegister import QumodeRegister
import numpy


class CVCircuit(QuantumCircuit):
    def __init__(self, qmr: QumodeRegister, qr: QuantumRegister, cr: ClassicalRegister, name: str = None):
        super().__init__(qmr.qreg, qr, cr, name=name)

        self.qmr = qmr
        self.qr = qr
        self.cr = cr

        self.ops = CVOperators(self.qmr)

    def cv_initialize(self, fock_states):
        """ Initialize qumodes to the given fock_states. """
        for qumode, n in enumerate(fock_states):
            if n >= self.qmr.cutoff:
                raise ValueError("The parameter n should be lower than the cutoff")

            vector = numpy.zeros((self.qmr.cutoff,))
            vector[n] = 1

            super().initialize(vector, self.qmr[qumode])

    def cv_conditional(self, name, op_a, op_b):
        """ Make two operators conditional (i.e., controlled by qubit in either the 0 or 1 state) """
        sub_qmr = QumodeRegister(self.qmr.num_qumodes, self.qmr.num_qubits_per_mode)
        sub_qr = QuantumRegister(1)
        sub_circ = QuantumCircuit(sub_qmr.qreg, sub_qr, name=name)
        sub_circ.append(UnitaryGate(op_a).control(num_ctrl_qubits=1, ctrl_state=0), [sub_qr[0]] + sub_qmr[0])
        sub_circ.append(UnitaryGate(op_b).control(num_ctrl_qubits=1, ctrl_state=1), [sub_qr[0]] + sub_qmr[1])

        return sub_circ.to_instruction()

    def cv_bs(self, phi, qumode_a, qumode_b):
        operator = self.ops.bs(phi)

        self.unitary(obj=operator, qubits=qumode_a + qumode_b, label='BS')

    def cv_d(self, alpha, qumode):
        operator = self.ops.d(alpha)

        self.unitary(obj=operator, qubits=qumode, label='D')

    def cv_cnd_d(self, alpha, beta, ctrl, qumode_a, qumode_b):
        self.append(self.cv_conditional('Dc', self.ops.d(alpha), self.ops.d(beta)), [ctrl] + qumode_a + qumode_b)

    def cv_r(self, phi, qumode):
        operator = self.ops.r(phi)

        self.unitary(obj=operator, qubits=qumode, label='R')

    def cv_s(self, z, qumode):
        operator = self.ops.s(z)

        self.unitary(obj=operator, qubits=qumode, label='S')

    def cv_cnd_s(self, z_a, z_b, ctrl, qumode_a, qumode_b):
        self.append(self.cv_conditional('Sc', self.ops.s(z_a), self.ops.s(z_b)), [ctrl] + qumode_a + qumode_b)

    def cv_s2(self, z, qumode_a, qumode_b):
        operator = self.ops.s2(z)

        self.unitary(obj=operator, qubits=qumode_a + qumode_b, label='S2')
