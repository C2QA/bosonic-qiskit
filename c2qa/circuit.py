import math
import warnings

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions import UnitaryGate

from c2qa.operators import CVOperators
from c2qa.qumoderegister import QumodeRegister


class CVCircuit(QuantumCircuit):
    def __init__(self, *regs, name: str = None, animation_segments: int = math.nan):
        """
        Initialize the registers (at least one must be QumodeRegister), set
        the circuit name, and the number of steps to animate (default is to not animate).
        """
        self.qmregs = []
        registers = []

        num_qumodes = 0

        for reg in regs:
            if isinstance(reg, QumodeRegister):
                if len(self.qmregs) > 0:
                    warnings.warn(
                        "More than one QumodeRegister provided. Using the last one for cutoff.",
                        UserWarning,
                    )
                num_qumodes += reg.num_qumodes
                self.qmregs.append(reg)
                registers.append(reg.qreg)
            else:
                registers.append(reg)

        if len(self.qmregs) == 0:
            raise ValueError("At least one QumodeRegister must be provided.")

        super().__init__(*registers, name=name)

        self.ops = CVOperators(self.cutoff, num_qumodes)

        self.animated = not math.isnan(animation_segments)
        if self.animated and animation_segments < 1:
            self._animation_segments = 1
        else:
            self._animation_segments = animation_segments
        self.animation_steps = 0

    @property
    def cutoff(self):
        return self.qmregs[-1].cutoff

    def get_snapshot_name(self, index: int):
        """Return the string statevector snapshot name for the given frame index."""
        return f"frame_{index}"

    def _snapshot_animation(self):
        """Create a new statevector snapshot."""
        self.snapshot(self.get_snapshot_name(self.animation_steps))
        self.animation_steps += 1

    def cv_initialize(self, fock_state, qumodes):
        """ Initialize the qumode to a Fock state. """

        # Qumodes are already represented as arrays of qubits,
        # but if this is an array of arrays, then we are initializing multiple qumodes.
        modes = qumodes
        if not isinstance(qumodes[0], list):
            modes = [qumodes]

        if fock_state > self.qmregs[-1].cutoff:
            raise ValueError("The given Fock state is greater than the cutoff.")

        for qumode in modes:
            value = np.zeros((self.qmregs[-1].cutoff,))
            value[fock_state] = 1

            super().initialize(value, qumode)

    def cv_conditional(self, name, op_0, op_1, num_qumodes: int = 1):
        """ Make two operators conditional (i.e., controlled by qubit in either the 0 or 1 state) """
        sub_qr = QuantumRegister(1)
        sub_qmr = QumodeRegister(num_qumodes, self.qmregs[-1].num_qubits_per_mode)
        sub_circ = QuantumCircuit(sub_qr, sub_qmr.qreg, name=name)

        # TODO Use size of op_0 and op_1 to calculate the number of qumodes instead of using parameter
        qargs = [sub_qr[0]]
        for i in range(num_qumodes):
            qargs += sub_qmr[i]

        sub_circ.append(
            UnitaryGate(op_0).control(num_ctrl_qubits=1, ctrl_state=0),
            qargs
        )
        sub_circ.append(
            UnitaryGate(op_1).control(num_ctrl_qubits=1, ctrl_state=1),
            qargs
        )

        return sub_circ.to_instruction()

    def cv_bs(self, phi, qumode_a, qumode_b):
        if self.animated:
            segment = phi / self._animation_segments

            for _ in range(self._animation_segments):
                operator = self.ops.bs(segment)
                self.unitary(obj=operator, qubits=qumode_a + qumode_b, label="BS")
                self._snapshot_animation()
        else:
            operator = self.ops.bs(phi)
            self.unitary(obj=operator, qubits=qumode_a + qumode_b, label="BS")

    def cv_cnd_bs(self, phi, chi, ctrl, qumode_a, qumode_b):
        if self.animated:
            segment_phi = phi / self._animation_segments
            segment_chi = chi / self._animation_segments

            for _ in range(self._animation_segments):
                self.append(
                    self.cv_conditional(
                        "BSc", self.ops.bs(segment_phi), self.ops.bs(segment_chi), num_qumodes=2
                    ),
                    [ctrl] + qumode_a + qumode_b
                )
                self._snapshot_animation()
        else:
            self.append(
                self.cv_conditional("BSc", self.ops.bs(phi), self.ops.bs(chi), num_qumodes=2),
                [ctrl] + qumode_a + qumode_b
            )

    def cv_d(self, alpha, qumode):
        if self.animated:
            segment = alpha / self._animation_segments

            for _ in range(self._animation_segments):
                operator = self.ops.d(segment)
                self.unitary(obj=operator, qubits=qumode, label="D")
                self._snapshot_animation()
        else:
            operator = self.ops.d(alpha)
            self.unitary(obj=operator, qubits=qumode, label="D")

    def cv_cnd_d(self, alpha, beta, ctrl, qumode):
        if self.animated:
            segment_alpha = alpha / self._animation_segments
            segment_beta = beta / self._animation_segments

            for _ in range(self._animation_segments):
                self.append(
                    self.cv_conditional(
                        "Dc", self.ops.d(segment_alpha), self.ops.d(segment_beta)
                    ),
                    [ctrl] + qumode
                )
                self._snapshot_animation()
        else:
            self.append(
                self.cv_conditional("Dc", self.ops.d(alpha), self.ops.d(beta)),
                [ctrl] + qumode
            )

    def cv_r(self, phi, qumode):
        if self.animated:
            segment = phi / self._animation_segments

            for _ in range(self._animation_segments):
                operator = self.ops.r(segment)
                self.unitary(obj=operator, qubits=qumode, label="R")
                self._snapshot_animation()
        else:
            operator = self.ops.r(phi)
            self.unitary(obj=operator, qubits=qumode, label="R")

    def cv_s(self, z, qumode):
        if self.animated:
            segment = z / self._animation_segments

            for _ in range(self._animation_segments):
                operator = self.ops.s(segment)
                self.unitary(obj=operator, qubits=qumode, label="S")
                self._snapshot_animation()
        else:
            operator = self.ops.s(z)
            self.unitary(obj=operator, qubits=qumode, label="S")

    def cv_cnd_s(self, z_a, z_b, ctrl, qumode_a):
        if self.animated:
            segment_z_a = z_a / self._animation_segments
            segment_z_b = z_b / self._animation_segments

            for _ in range(self._animation_segments):
                self.append(
                    self.cv_conditional(
                        "Sc", self.ops.s(segment_z_a), self.ops.s(segment_z_b)
                    ),
                    [ctrl] + qumode_a
                )
                self._snapshot_animation()
        else:
            self.append(
                self.cv_conditional("Sc", self.ops.s(z_a), self.ops.s(z_b)),
                [ctrl] + qumode_a
            )

    def cv_s2(self, z, qumode_a, qumode_b):
        if self.animated:
            segment = z / self._animation_segments

            for _ in range(self._animation_segments):
                operator = self.ops.s2(segment)
                self.unitary(obj=operator, qubits=qumode_a + qumode_b, label="S2")
                self._snapshot_animation()
        else:
            operator = self.ops.s2(z)
            self.unitary(obj=operator, qubits=qumode_a + qumode_b, label="S2")
