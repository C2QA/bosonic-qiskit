import copy
import itertools
import warnings
from collections.abc import Sequence
from typing import Any, cast, overload

import numpy as np
import qiskit
import qiskit_aer.library.save_instructions as save
from numpy.typing import ArrayLike
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import InstructionSet, ParameterExpression
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

from bosonic_qiskit.operators import CVOperators
from bosonic_qiskit.parameterized_unitary_gate import ParameterizedUnitaryGate, UnitaryFunc
from bosonic_qiskit.qumoderegister import QumodeRegister
from bosonic_qiskit.typing import Clbit, Qubit, Qumode


class CVCircuit(QuantumCircuit):
    """Extension of QisKit QuantumCircuit to add continuous variable (bosonic) gate support to simulations."""

    def __init__(
        self,
        *regs,
        name: str | None = None,
        probe_measure: bool = False,
        force_parameterized_unitary_gate: bool = True,
    ):
        """Initialize the registers (at least one must be QumodeRegister) and set the circuit name.

        Args:
            name (str, optional): circuit name. Defaults to None.
            probe_measure (bool, optional): automatically support measurement with probe qubits. Defaults to False.
            force_parameterized_unitary_gate (bool, optional): if set to False, improve performance by creating Qiskit UnitaryGate instead of bosonic-qiskit ParamaterizedUnitaryGate and skip transpilation in the util module's simulate() function. Note that bosonic-qiskit ParameterizedUnitaryGate are required for Qiskit parameterized circuits, circuits using photon loss noise passes, and cicruits animated with discretized gates. Defaults to True.

        Raises:
            ValueError: If no QumodeRegister is provided.
        """
        self.qmregs: list[QumodeRegister] = []
        # This needs to be unique from qregs[] in the superclass
        self._qubit_regs: list[QuantumRegister] = []

        registers: list[QuantumRegister] = []

        num_qubits = 0

        for reg in regs:
            if isinstance(reg, QumodeRegister):
                self.qmregs.append(reg)
                registers.append(reg.qreg)
                num_qubits += reg.size
            elif isinstance(reg, QuantumRegister):
                self._qubit_regs.append(reg)
                registers.append(reg)
                num_qubits += reg.size
            else:
                registers.append(reg)

        if len(self.qmregs) == 0:
            raise ValueError("At least one QumodeRegister must be provided.")

        # Support measurement using probe qubits
        self.probe_measure = probe_measure
        self.probe: QuantumRegister | None = None
        if probe_measure:
            self.probe = QuantumRegister(size=num_qubits, name="probe")
            registers.append(self.probe)

        super().__init__(*registers, name=name)

        self.ops = CVOperators()
        self.cv_snapshot_id: int = 0
        self._has_parameterized_gate = False
        self._force_parameterized_unitary_gate = force_parameterized_unitary_gate

    def merge(self, circuit: QuantumCircuit):
        """
        Merge in properties of QisKit QuantumCircuit into this instance.

        Useful if QisKit returned a new instance of QuantumCircuit after
        passing in this instance. Calling merge() can merge the two, keeping
        this instance.

        See https://qiskit.org/documentation/_modules/qiskit/circuit/quantumcircuit.html#QuantumCircuit.copy
        """

        self.qregs = circuit.qregs.copy()
        self.cregs = circuit.cregs.copy()
        self._qubits = circuit._qubits.copy()
        self._ancillas = circuit._ancillas.copy()
        self._clbits = circuit._clbits.copy()
        self._qubit_indices = circuit._qubit_indices.copy()
        self._clbit_indices = circuit._clbit_indices.copy()

        instr_instances = {id(instr): instr for instr, _, __ in circuit._data}

        instr_copies = {id_: instr.copy() for id_, instr in instr_instances.items()}

        self._data = [
            (instr_copies[id(inst)], qargs.copy(), cargs.copy())
            for inst, qargs, cargs in circuit._data
        ]

        self._calibrations = copy.deepcopy(circuit._calibrations)
        self._metadata = copy.deepcopy(circuit._metadata)

    def get_qmr_cutoff(self, qmr_index: int) -> int:
        """Return the qumode cutoff at the given index"""
        return self.qmregs[qmr_index].cutoff

    def get_qmr_num_qubits_per_qumode(self, qmr_index: int) -> int:
        """Return the number of qubits in the qumode register at the given index"""
        return self.qmregs[qmr_index].num_qubits_per_qumode

    @property
    def qumode_qubits(self) -> list[Qubit]:
        """All the qubits representing the qumode registers on the circuit"""
        qubits: list[Qubit] = []
        for reg in self.qmregs:
            qubits.extend(reg.qubits)
        return qubits

    @property
    def qumode_qubit_indices(self) -> list[int]:
        """A qubit index list of the qubits representing the qumode registers on the circuit"""
        qmodes = set(self.qumode_qubits)
        indices: list[int] = []

        for index, qubit in enumerate(self.qubits):
            if qubit in qmodes:
                indices.append(index)

        return indices

    @property
    def qumode_qubits_indices_grouped(self) -> list[list[int]]:
        """Same as qumode_qubit_indices but it groups qubits representing the same qumode together. Returns a nested list."""

        grouped_indices = []
        qubit_indices = {q: i for i, q in enumerate(self.qubits)}
        for qmreg in self.qmregs:
            for qumode in qmreg:
                idx = [qubit_indices[q] for q in qumode]
                grouped_indices.append(idx)

        return grouped_indices

    def get_qubit_index(self, qubit: Qubit) -> int | None:
        """Return the index of the given Qubit"""
        for i, q in enumerate(self.qubits):
            if q == qubit:
                return i
        return None

    def get_qubit_indices(self, qubits: Sequence[Qubit | Sequence[Qubit]]) -> list[int]:
        """Return the indices of the given Qubits"""
        flat_list = []
        for el in qubits:
            if isinstance(el, Sequence):
                flat_list += el
            else:
                flat_list += [el]

        indices = []
        for i, q in enumerate(self.qubits):
            if q in flat_list:
                indices.append(i)
        return indices

    def get_qmr_index(self, qubit: Qubit) -> int:
        """Return the qumode index for the given qubit. If not found, raises ValueError"""
        for index, qmr in enumerate(self.qmregs):
            if qubit in qmr:
                return index
        raise ValueError(f"Bit {qubit} not found in circuit.")

    @property
    def cv_gate_labels(self) -> list[str]:
        """
        All the CV gate names on the current circuit. These will be
        instances of ParameterizedUnitaryGate.
        """
        cv_gates = set()
        for instruction, qargs, cargs in self.data:
            if isinstance(instruction, ParameterizedUnitaryGate):
                cv_gates.add(instruction.label)
        return list(cv_gates)

    def cv_snapshot(self):
        """Wrap the Qiskit QuantumCircuit Snapshot function, giving it a known label for later Wigner function plot generation"""
        self.save_statevector(f"cv_snapshot_{self.cv_snapshot_id}")
        self.cv_snapshot_id += 1

    def add_qubit_register(self, *regs):
        """Add a qubit register to the circuit.

        Args:
            regs (list): List of Registers
        """
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                self._qubit_regs.append(reg)
            else:
                raise ValueError(
                    "Only QuantumRegisters are allowed to be added for now"
                )
        super().add_register(*regs)

    def cv_initialize(
        self, params: int | Sequence[complex], qumodes: Qumode | Sequence[Qumode]
    ):
        """Initialize qumode (or qumodes) to a particular state specified by params

        Args:
            params (list or int): If an int, all specified qumodes will be initialized to the Fock state with n=params.
                                  If a list, all specified qumodes will be initialized to a superposition of Fock states,
                                  with ``params[n]`` the complex amplitude of Fock state ``|n>``. The length of params must be less
                                  than or equal to the cutoff.

            qumodes (list): list of qubits representing a single qumode, or list of multiple qumodes

        Raises:
            ValueError: If the Fock state is greater than the cutoff.
        """
        # Qumodes are already represented as arrays of qubits,
        # but if this is an array of arrays, then we are initializing multiple qumodes.
        modes = qumodes
        if isinstance(modes[0], Qubit):
            modes = [modes]

        modes = cast(Sequence[Qumode], modes)
        params = np.atleast_1d(params)
        if params.size == 1:
            for qumode in modes:
                qumode_index = self.get_qmr_index(qumode[0])
                cutoff = self.get_qmr_cutoff(qumode_index)

                if params[0] >= cutoff:
                    raise ValueError(
                        f"The given Fock state {params} is greater than the cutoff {cutoff}"
                    )

                value = np.zeros(cutoff, dtype=complex)
                value[params] = 1

                super().initialize(value, qumode)
        else:
            for qumode in modes:
                qumode_index = self.get_qmr_index(qumode[0])
                cutoff = self.get_qmr_cutoff(qumode_index)

                if len(params) > cutoff:
                    raise ValueError(
                        f"More parameters provided ({len(params)}) than available Fock states ({cutoff})"
                    )

                amplitudes = np.zeros(cutoff, dtype=complex)
                amplitudes[: len(params)] = params
                amplitudes /= np.linalg.norm(amplitudes)
                super().initialize(amplitudes, qumode)

    def save_circuit(
        self,
        label: str = "statevector",
        conditional: bool = False,
        pershot: bool = False,
    ):
        """Save the simulator statevector using a qiskit class"""
        return save.save_statevector(  # pyright: ignore[reportCallIssue]
            label=label, conditional=conditional, pershot=pershot
        )

    def _new_gate(
        self,
        op_func: UnitaryFunc,
        params: Any,
        num_qubits: int,
        cutoffs: Sequence[int],
        label: str | None = None,
        duration: int = 100,
        unit: str = "ns",
        discretized_param_indices: list[int] | None = None,
    ) -> UnitaryGate | ParameterizedUnitaryGate:
        # If parameters contain compile-time parameters
        is_parameterized = self._force_parameterized_unitary_gate or any(
            isinstance(param, ParameterExpression) and len(param.parameters) > 0
            for param in params
        )

        if is_parameterized:
            self._has_parameterized_gate = True
            gate = ParameterizedUnitaryGate(
                op_func,
                params,
                cutoffs=cutoffs,
                num_qubits=num_qubits,
                label=label,
                duration=duration,
                unit=unit,
                discretized_param_indices=discretized_param_indices or [],
            )
        else:
            data = CVOperators.call_op(op_func, params, cutoffs)
            gate = UnitaryGate(data=data, label=label, num_qubits=num_qubits)

        return gate

    def cv_r(
        self, theta: float, qumode: Qumode, duration: int = 100, unit: str = "ns"
    ) -> InstructionSet:
        """Phase space rotation gate.

        Args:
            theta (real): rotation
            qumode (list): list of qubits representing qumode
            cutoff (int): qumode cutoff

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.r,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode),
                label="R",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode,
        )

    def cv_d(
        self, alpha: complex, qumode: Qumode, duration: int = 100, unit: str = "ns"
    ) -> InstructionSet:
        """Displacement gate.

        Args:
            alpha (real or complex): displacement
            qumode (list): list of qubits representing qumode
            cutoff (int): qumode cutoff

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.d,
                [alpha],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode),
                label="D",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode,
        )

    def cv_sq(
        self, theta: complex, qumode: Qumode, duration: int = 100, unit: str = "ns"
    ) -> InstructionSet:
        """Squeezing gate.

        Args:
            theta (real or complex): squeeze
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.s,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode),
                label="S",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode,
        )

    def cv_sq2(
        self,
        theta: complex,
        qumode_a: Qumode,
        qumode_b: Qumode,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Two-mode squeezing gate

        Args:
            theta (complex): squeeze
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        qumodes = (qumode_a, qumode_b)
        return self.append(
            self._new_gate(
                self.ops.s2,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(m)) for m in qumodes],
                num_qubits=sum(map(len, qumodes)),
                label="S2",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode_a, *qumode_b],
        )

    def cv_sq3(
        self,
        theta: complex,
        qumode_a: Qumode,
        qumode_b: Qumode,
        qumode_c: Qumode,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Three-mode squeezing gate

        Args:
            theta (real or complex): squeeze
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode
            qumode_c (list): list of qubits representing third qumode

        Returns:
            Instruction: QisKit instruction
        """
        qumodes = (qumode_a, qumode_b, qumode_c)
        return self.append(
            self._new_gate(
                self.ops.s3,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(m)) for m in qumodes],
                num_qubits=sum(map(len, qumodes)),
                label="S3",
                duration=duration,
                unit=unit,
            ),
            qargs=list(itertools.chain(*qumodes)),
        )

    def cv_bs(
        self,
        theta: complex,
        qumode_a: Qumode,
        qumode_b: Qumode,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Two-mode beam splitter gate.

        Args:
            theta (real or complex): beamsplitter phase
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.bs,
                [theta],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode_a)),
                    QumodeRegister.calculate_cutoff(len(qumode_b)),
                ],
                num_qubits=len(qumode_a) + len(qumode_b),
                label="BS",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode_a, *qumode_b],
        )

    def cv_c_r(
        self,
        theta: float,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Qubit dependent phase-space rotation gate (i.e., dispersive interaction).

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.cr,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cR",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_c_rx(
        self,
        theta: float,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Qubit dependent phase-space rotation around sigma^x gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.crx,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cRX",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_c_ry(
        self,
        theta: float,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Qubit dependent phase-space rotation around sigma^y gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.cry,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cRY",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_c_d(
        self,
        theta: float,
        qumode: Qumode,
        qubit: Qubit,
        beta: float | None = None,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Conditional displacement gate.

        Args:
            theta (real): displacement
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control qubit
            beta (real): By default is None, and qumode will be displaced by alpha and -alpha for qubit
            state 0 and 1, respectively. If specified, qumode will be displaced by alpha and beta for qubit state 0 and 1.

        Returns:
            Instruction: QisKit instruction
        """
        if beta is None:
            beta = -theta

        return self.append(
            self._new_gate(
                self.ops.cd,
                [theta, beta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cD",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_ecd(
        self,
        theta: float,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Echoed controlled displacement gate.

        Args:
            theta (real): displacement
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.ecd,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="ECD",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_c_bs(
        self,
        theta: complex,
        qumode_a: Qumode,
        qumode_b: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Controlled phase two-mode beam splitter

        Args:
            theta (real or complex): phase
            qubit_ancilla (Qubit): QisKit control Qubit
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.cbs,
                [theta],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode_a)),
                    QumodeRegister.calculate_cutoff(len(qumode_b)),
                ],
                num_qubits=len(qumode_a) + len(qumode_b) + 1,
                label="cBS",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode_a, *qumode_b, qubit],
        )

    def cv_c_schwinger(
        self,
        params: Sequence[float],
        qumode_a: Qumode,
        qumode_b: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """General form of a controlled 'Schwinger' gate, containing both the controlled phase beamsplitter
        and pairs of controlled phase space rotations as special cases.

        It has the form exp(-i*beta*(n1_hat.sigma)(n2_hat.S)),
        where ni_hat = sin(theta_i)*cos(phi_i) + sin(theta_i)*sin(phi_i) + cos(theta_i).
        sigma = [sigmax, sigmay, sigmaz] is the vector of Pauli operators, and
        S = [Sx, Sy, Sz] is a vector of Schwinger boson operators,

        Sx = (a*bdag + adag*b)/2
        Sy = (a*bdag - adag*b)/2i
        Sz = (bdag*b - adag*a)/2,

        obeying the commutation relations [Sj, Sk] = i*epsilon_{ijk}*Sz, where epsilon_{ijk} is the Levi-Civita tensor.

        Args:
            params (real): [beta, theta_1, phi_1, theta_2, phi_2]
            qubit_ancilla (Qubit): QisKit control Qubit
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.cschwinger,
                params,
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode_a)),
                    QumodeRegister.calculate_cutoff(len(qumode_b)),
                ],
                num_qubits=len(qumode_a) + len(qumode_b) + 1,
                label="cSchw",
                duration=duration,
                unit=unit,
                discretized_param_indices=[0],
            ),
            qargs=[*qumode_a, *qumode_b, qubit],
        )

    @overload
    def cv_snap(
        self,
        theta: float,
        n: int,
        qumode: Qumode,
        qubit: Qubit | None = None,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet: ...

    @overload
    def cv_snap(
        self,
        theta: Sequence[float],
        n: Sequence[int],
        qumode: Qumode,
        qubit: Qubit | None = None,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet: ...

    def cv_snap(
        self,
        theta: float | Sequence[float],
        n: int | Sequence[int],
        qumode: Qumode,
        qubit: Qubit | None = None,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """SNAP (Selective Number-dependent Arbitrary Phase) gate. If no qubit is passed,
        then phases are applied to each qumode Fock state specified in theta and n (without
        explicit rotation of the qubit). If a qubit is passed, the phase will be multiplied by
        sigma_z-dependent geometric phase (akin to the implementation of the SNAP gate
        as described in Heeres et al, PRL (2015).

        Args:
            theta (real or list[real]): phase
            n (integer or list[integer]): Fock state in which the mode should acquire the phase
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control qubit. If no qubit is passed, the gate will implement for sigma^z = +1.

        Returns:
            Instruction: QisKit instruction
        """
        cutoff = QumodeRegister.calculate_cutoff(len(qumode))
        label = "cSNAP" if qubit else "SNAP"
        num_qubits = len(qumode) + int(qubit is not None)
        qargs = [*qumode, qubit] if qubit else qumode

        if isinstance(n, Sequence):
            if not isinstance(theta, Sequence) or len(theta) != len(n):
                raise ValueError("Must provide as many parameters as Fock levels")

            if any(x > cutoff for x in n):
                raise ValueError(f"Fock level {max(n)} exceeds the cutoff {cutoff}")

            op_func = self.ops.multicsnap if qubit else self.ops.multisnap

            return self.append(
                self._new_gate(
                    op_func,
                    list(theta) + list(n),
                    cutoffs=[cutoff],
                    num_qubits=num_qubits,
                    label=label,
                    duration=duration,
                    unit=unit,
                )
            )

        else:
            if n > cutoff:
                ValueError(f"Fock state {n} exceeds the cutoff {cutoff}")

            if isinstance(theta, Sequence):
                raise ValueError(
                    "Must provide a scalar parameter for a single fock level"
                )

            op_func = self.ops.csnap if qubit else self.ops.snap

            return self.append(
                self._new_gate(
                    op_func,
                    [theta, n],
                    cutoffs=[cutoff],
                    num_qubits=num_qubits,
                    label=label,
                    duration=duration,
                    unit=unit,
                ),
                qargs=qargs,
            )

    def cv_sqr(
        self,
        theta: ArrayLike,
        phi: ArrayLike,
        n: ArrayLike,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Selective Qubit Rotation (SQR) gate

        This gate applies a qubit rotation conditioned on the Fock state(s) of the oscillator. See eq. 234 of arXiv:2407.10381.
        """
        cutoff = QumodeRegister.calculate_cutoff(len(qumode))
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        n = np.atleast_1d(n)

        if not np.issubdtype(n.dtype, np.integer):
            raise ValueError(f"Must provide integer fock levels, got dtype {n.dtype}")

        out = np.broadcast(theta, phi, n)
        if out.ndim != 1:
            raise ValueError("Theta, phi, and n must be broadcastable to a 1D array")

        if np.unique(n).size != out.size:
            raise ValueError("Must specify different fock levels for each theta, phi")

        if np.any(n >= cutoff):
            raise ValueError("Received Fock level too high for the qumode cutoff")

        # Flatten all the parameters to a single array like the other gates. This will
        # promote the dtype of n -> float
        params = np.concatenate(
            [
                np.broadcast_to(theta, out.shape),
                np.broadcast_to(phi, out.shape),
                np.broadcast_to(n, out.shape),
            ]
        )

        return self.append(
            self._new_gate(
                self.ops.sqr,
                params,
                cutoffs=[cutoff],
                num_qubits=len(qumode) + 1,
                label="SQR",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    # def cv_multisnap(self, thetas, ns, qumode, duration=1, unit="us"):
    #     params = thetas + ns
    #     self.append(
    #         self._new_gate(
    #             self.ops.multisnap, params, num_qubits=len(qumode), label="mSNAP", duration=duration, unit=unit
    #         ),
    #         qargs=qumode,
    #     )

    def cv_c_pnr(
        self,
        max: int,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """PNR (Photon number readout) TODO: Needs comments/explanation/citation!
        Args:
            max (int): the period of the mapping
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control qubit.
        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.pnr,
                [max],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="c_pnr",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_eswap(
        self,
        theta: float,
        qumode_a: Qumode,
        qumode_b: Qumode,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Exponential SWAP gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qumode_b (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.eswap,
                [theta],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode_a)),
                    QumodeRegister.calculate_cutoff(len(qumode_b)),
                ],
                num_qubits=len(qumode_a) + len(qumode_b),
                label="eSWAP",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode_a, *qumode_b],
        )

    def cv_c_sq(
        self,
        theta: complex,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Conditional squeezing gate.

        Args:
            theta (real or complex): squeezing ampltiude
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control Qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.csq,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cS",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_sum(
        self,
        scale: float,
        qumode_a: Qumode,
        qumode_b: Qumode,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Two-mode sum gate.

        Args:
            scale (real): arbitrary real scale factor
            qumode_a (list): list of qubits representing qumode
            qumode_b (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.sum,
                [scale],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode_a)),
                    QumodeRegister.calculate_cutoff(len(qumode_b)),
                ],
                num_qubits=len(qumode_a) + len(qumode_b),
                label="sum",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode_a, *qumode_b],
        )

    def cv_c_sum(
        self,
        scale: float,
        qumode_a: Qumode,
        qumode_b: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Conditional two-mode sum gate.

        Args:
            scale (real): arbitrary real scale factor
            qumode_a (list): list of qubits representing qumode
            qumode_b (list): list of qubits representing qumode
            qubit (Qubit): control Qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.csum,
                [scale],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode_a)),
                    QumodeRegister.calculate_cutoff(len(qumode_b)),
                ],
                num_qubits=len(qumode_a) + len(qumode_b) + 1,
                label="cSum",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode_a, *qumode_b, qubit],
        )

    def cv_jc(
        self,
        theta: float,
        phi: float,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.jc,
                [theta, phi],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode)),
                ],
                num_qubits=len(qumode) + 1,
                label="jc",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_ajc(
        self,
        theta: float,
        phi: float,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Anti-Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.ajc,
                [theta, phi],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode)),
                ],
                num_qubits=len(qumode) + 1,
                label="ajc",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_rb(
        self,
        theta: float,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "ns",
    ) -> InstructionSet:
        """Rabi interaction gate

        Args:
            theta (real): arbitrary scale factor

        Returns:
            InstructionSet: The qiskit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.rb,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="rb",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def measure_z(self, qubit: Qubit, cbit: Clbit) -> InstructionSet:
        """Measure qubit in z using probe qubits

        Args:
            qubit (Qubit): QisKit qubit to measure
            cbit (ClassicalBit): QisKit classical bit to measure into

        Returns:
            Instruction: QisKit measure instruction
        """
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        return self.measure(qubit, cbit)

    def measure_y(self, qubit: Qubit, cbit: Clbit) -> InstructionSet:
        """Measure qubit in y using probe qubits

        Args:
            qubit (Qubit): QisKit qubit to measure
            cbit (ClassicalBit): QisKit classical bit to measure into

        Returns:
            Instruction: QisKit measure instruction
        """
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        self.sdg(qubit)
        self.h(qubit)
        return self.measure(qubit, cbit)

    def measure_x(self, qubit: Qubit, cbit: Clbit) -> InstructionSet:
        """Measure qubit in x using probe qubits

        Args:
            qubit (Qubit): QisKit qubit to measure
            cbit (ClassicalBit): QisKit classical bit to measure into

        Returns:
            Instruction: QisKit measure instruction
        """
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        self.h(qubit)
        return self.measure(qubit, cbit)

    def cv_measure(
        self,
        qargs: Qubit
        | Qumode
        | Sequence[Qubit | Qumode]
        | QuantumRegister
        | QumodeRegister,
        cargs: Clbit | Sequence[Clbit] | ClassicalRegister,
    ) -> InstructionSet:
        """Measure Qumodes and Qubits in qubit_qumode_list and map onto
        classical bits specified in cbit_list.

        Args:
            qubit_qumode_list(List): List of individual Qubits and Qumodes
                (i.e., indexed elements of QubitRegisters and QumodeRegisters)
            cbit_list (List): List of classical bits to map measurements onto.
                Note: Measurement of qumodes requires log(c) classical bits,
                where c is the cutoff.
                If len(cbit_list) is greater than the required number of
                classical bits, excess will be ignored. If len(cbit_list) is
                insufficient, an error will be thrown.

        Returns:
            Instruction: QisKit measure instruction
        """

        if isinstance(qargs, Qubit):
            qargs = [qargs]
        elif isinstance(qargs, QuantumRegister):
            qargs = qargs[:]
        elif isinstance(qargs, QumodeRegister):
            qargs = list(qargs)

        qargs_flat = []
        for arg in qargs:
            if isinstance(arg, Qubit):
                qargs_flat.append(arg)
            else:
                qargs_flat.extend(arg)

        if isinstance(cargs, Clbit):
            cargs = [cargs]

        # Discards unnecessary clbits so the user doesn't need to think about how many clbits are needed
        return self.measure(qargs_flat, cargs[: len(qargs_flat)])

    def cv_delay(
        self, duration: int, qumode: Qumode, unit: str = "ns"
    ) -> InstructionSet:
        """CV_delay. Implements an identity gate of the specified duration.
        This is particularly useful for the implementation of a noise pass.

        Args:
            duration (real): duration of delay gate
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.get_eye,
                [],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode),
                label=f"cv_delay({duration} {unit})",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode,
        )

    def cv_c_multiboson_sampling(
        self,
        max: int,
        qumode: Qumode,
        qubit: Qubit,
        duration: int = 100,
        unit: str = "us",
    ) -> InstructionSet:
        """SNAP (Selective Number-dependent Arbitrary Phase) gates for multiboson sampling.
        Args:
            max (int): the period of the mapping
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control qubit.
        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            self._new_gate(
                self.ops.c_multiboson_sampling,
                [max],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="c_multiboson_sampling",
                duration=duration,
                unit=unit,
            ),
            qargs=[*qumode, qubit],
        )

    def cv_gate_from_matrix(
        self,
        matrix: ArrayLike,
        qumodes: Qumode | Sequence[Qumode] | QumodeRegister | None = None,
        qubits: Qubit | Sequence[Qubit] | QuantumRegister | None = None,
        label: str = "cv_gate_from_matrix",
    ):
        """Converts matrix to gate. Note that if you choose to input some complex gate that would typically be physically
        implemented by multiple successive gate operations, PhotonLossNoisePass, simulate(discretize=True), and animate may
        not be applied in a way that is physical.

        Args:
            matrix (np.array/nested list): Matrix for conversion into gate
            qumodes (QumodeRegister/list): Qumodes initialized by QumodeRegister
            qubits (QuantumRegister/list): Qubits initialized by QuantumRegister

        Returns:
            Instruction: QisKit instruction
        """
        qubits = qubits or []
        if isinstance(qubits, Qubit):
            qubits = [qubits]
        qubits = list(qubits)

        qumodes = qumodes or []
        # Check if the user passed a single qumode
        if len(qumodes) > 0 and isinstance(qumodes[0], Qubit):
            qumodes = [qumodes]
        qumodes = list(qumodes)

        # Flatten everything into a list of qubits
        qargs = list(itertools.chain(*qumodes, qubits))

        matrix = np.atleast_2d(matrix)
        n, m = matrix.shape
        if n != m:
            raise ValueError("Matrix given is not square")

        # Determine if input matrix is same dimension as input qumodes+qubits
        if n != 2 ** len(qargs):
            raise ValueError("Matrix is of different dimension from qumodes + qubits")

        # Checks if input matrix is unitary
        if not is_unitary_matrix(matrix):
            raise ValueError("The mapping provided is not unitary!")

        # Make sure that np.ndarray doesn't get fed into PUG
        matrix = matrix.tolist()

        # TODO Is it safe to ignore cutoff, assuming provided matrix is correct?
        # cutoffs = []
        # for qumode in qumodes:
        #     cutoffs.append(QumodeRegister.calculate_cutoff(len(qumode)))

        # return self.append(
        #     self._new_gate(
        #         self.ops.gate_from_matrix,
        #         [matrix],
        #         cutoffs=cutoffs,
        #         num_qubits=len(qumodes) + len(qubits),
        #         label=label,
        #         duration=duration,
        #         unit=unit
        #     ),
        #     qargs=qumodes + qubits,
        # )
        return self.unitary(matrix, qubits=qargs, label=label)  # pyright: ignore[reportArgumentType]


# Monkey patch Qiskit QuantumCircuit to support parameterizing unitary gates
def __requires_transpile(self):
    return (
        getattr(self, "_force_parameterized_unitary_gate", False)
        or any(
            isinstance(gate, ParameterizedUnitaryGate) or gate.is_parameterized()
            for gate in self.data
        )
        or (hasattr(self, "_has_parameterized_gate") and self._has_parameterized_gate)
    )


CVCircuit.requires_transpile = __requires_transpile

qiskit.circuit.QuantumCircuit.requires_transpile = __requires_transpile
