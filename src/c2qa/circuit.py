import copy
import warnings

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
import qiskit_aer.library.save_instructions as save

from c2qa.operators import CVOperators
from c2qa.parameterized_unitary_gate import ParameterizedUnitaryGate
from c2qa.qumoderegister import QumodeRegister


class CVCircuit(QuantumCircuit):
    """Extension of QisKit QuantumCircuit to add continuous variable (bosonic) gate support to simulations."""

    def __init__(self, *regs, name: str = None, probe_measure: bool = False):
        """Initialize the registers (at least one must be QumodeRegister) and set the circuit name.

        Args:
            name (str, optional): circuit name. Defaults to None.
            probe_measure (bool, optional): automatically support measurement with probe qubits. Defaults to False.

        Raises:
            ValueError: If no QumodeRegister is provided.
        """
        self.qmregs = []
        self._qubit_regs = []  # This needs to be unique from qregs[] in the superclass

        registers = []

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
        if probe_measure:
            self.probe = QuantumRegister(size=num_qubits, name="probe")
            registers.append(self.probe)

        super().__init__(*registers, name=name)

        self.ops = CVOperators()
        self.cv_snapshot_id = 0

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

    def get_qmr_cutoff(self, qmr_index: int):
        """Return the qumode cutoff at the given index"""
        return self.qmregs[qmr_index].cutoff

    def get_qmr_num_qubits_per_qumode(self, qmr_index: int):
        """Return the number of qubits in the qumode register at the given index"""
        return self.qmregs[qmr_index].num_qubits_per_qumode

    @property
    def qumode_qubits(self):
        """All the qubits representing the qumode registers on the circuit"""
        qubits = []
        for reg in self.qmregs:
            qubits += reg[::]
        return qubits

    @property
    def qumode_qubit_indices(self):
        """A qubit index list of the qubits representing the qumode registers on the circuit"""
        qmodes = self.qumode_qubits
        indices = []

        for index, qubit in enumerate(self.qubits):
            if qubit in qmodes:
                indices.append(index)

        return indices

    @property
    def qumode_qubits_indices_grouped(self):
        """Same as qumode_qubit_indices but it groups qubits representing the same qumode together. Returns a nested list."""
        grouped_indices = []

        # Iterate through all qmregs
        for _, qmreg in enumerate(self.qmregs):
            num_qumodes_in_reg = qmreg.num_qumodes
            num_qubits_per_qumode = qmreg.num_qubits_per_qumode

            qmreg_qubit_indices = []

            # For every qubit in circuit, append index of qubit to list if qubit is in qmreg
            for qubit_index, qubit in enumerate(self.qubits):
                if qubit in qmreg[:]:
                    qmreg_qubit_indices.append(qubit_index)

            # Split list according to no. of qumodes in qmreg
            qmreg_qubit_indices = [
                qmreg_qubit_indices[
                    i * num_qubits_per_qumode : (i + 1) * num_qubits_per_qumode
                ]
                for i in range(num_qumodes_in_reg)
            ]

            # Extend final list
            grouped_indices.extend(qmreg_qubit_indices)

        return grouped_indices

    def get_qubit_index(self, qubit):
        """Return the index of the given Qubit"""
        for i, q in enumerate(self.qubits):
            if q == qubit:
                return i
        return None

    def get_qubit_indices(self, qubits: list):
        """Return the indices of the given Qubits"""
        flat_list = []
        for el in qubits:
            if isinstance(el, list):
                flat_list += el
            else:
                flat_list += [el]

        indices = []
        for i, q in enumerate(self.qubits):
            if q in flat_list:
                indices.append(i)
        return indices

    def get_qmr_index(self, qubit):
        """Return the qumode index for the given qubit. If not found, return -1."""
        for index, qmr in enumerate(self.qmregs):
            if qubit in qmr:
                return index
        raise ValueError(f"Bit {qubit} not found in circuit.")

    @property
    def cv_gate_labels(self):
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

    def cv_initialize(self, params, qumodes):
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
        if not isinstance(qumodes[0], list):
            modes = [qumodes]

        if isinstance(params, int):
            for qumode in modes:
                qumode_index = self.get_qmr_index(qumode[0])

                if params >= self.get_qmr_cutoff(qumode_index):
                    raise ValueError("The given Fock state is greater than the cutoff.")

                value = np.zeros(
                    (self.get_qmr_cutoff(qumode_index),), dtype=np.complex128
                )
                value[params] = 1 + 0j

                super().initialize(value, qumode)
        else:
            for qumode in modes:
                qumode_index = self.get_qmr_index(qumode[0])

                if len(params) > self.get_qmr_cutoff(qumode_index):
                    raise ValueError("len(params) exceeds the cutoff.")

                params = np.array(params) / np.linalg.norm(np.array(params))
                amplitudes = np.zeros(
                    (self.get_qmr_cutoff(qumode_index),), dtype=np.complex128
                )
                for ind in range(len(params)):
                    amplitudes[ind] = complex(params[ind])

                super().initialize(amplitudes, qumode)

    def save_circuit(self, conditional, pershot, label="statevector"):
        """Save the simulator statevector using a qiskit class"""
        return save.save_statevector(
            label=label, conditional=conditional, pershot=pershot
        )

    def cv_r(self, theta, qumode, duration=100, unit="ns"):
        """Phase space rotation gate.

        Args:
            theta (real): rotation
            qumode (list): list of qubits representing qumode
            cutoff (int): qumode cutoff

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
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

    def cv_d(self, alpha, qumode, duration=100, unit="ns"):
        """Displacement gate.

        Args:
            alpha (real or complex): displacement
            qumode (list): list of qubits representing qumode
            cutoff (int): qumode cutoff

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
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

    def cv_sq(self, theta, qumode, duration=100, unit="ns"):
        """Squeezing gate.

        Args:
            theta (real or complex): squeeze
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
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

    def cv_sq2(self, theta, qumode_a, qumode_b, duration=100, unit="ns"):
        """Two-mode squeezing gate

        Args:
            theta (real or complex): squeeze
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.s2,
                [theta],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode_a)),
                    QumodeRegister.calculate_cutoff(len(qumode_b)),
                ],
                num_qubits=len(qumode_a) + len(qumode_b),
                label="S2",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode_a + qumode_b,
        )

    def cv_sq3(self, theta, qumode_a, qumode_b, qumode_c, duration=100, unit="ns"):
        """Three-mode squeezing gate

        Args:
            theta (real or complex): squeeze
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode
            qumode_c (list): list of qubits representing third qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.s3,
                [theta],
                cutoffs=[
                    QumodeRegister.calculate_cutoff(len(qumode_a)),
                    QumodeRegister.calculate_cutoff(len(qumode_b)),
                    QumodeRegister.calculate_cutoff(len(qumode_c)),
                ],
                num_qubits=len(qumode_a) + len(qumode_b) + len(qumode_c),
                label="S3",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode_a + qumode_b + qumode_c,
        )

    def cv_bs(self, theta, qumode_a, qumode_b, duration=100, unit="ns"):
        """Two-mode beam splitter gate.

        Args:
            theta (real or complex): beamsplitter phase
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
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
            qargs=qumode_a + qumode_b,
        )

    def cv_c_r(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Qubit dependent phase-space rotation gate (i.e., dispersive interaction).

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        self.append(
            ParameterizedUnitaryGate(
                self.ops.cr,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cR",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def cv_c_rx(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Qubit dependent phase-space rotation around sigma^x gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.crx,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cRX",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def cv_c_ry(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Qubit dependent phase-space rotation around sigma^y gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.cry,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cRY",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def cv_c_d(self, theta, qumode, qubit, beta=None, duration=100, unit="ns"):
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
            ParameterizedUnitaryGate(
                self.ops.cd,
                [theta, beta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cD",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def cv_ecd(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Echoed controlled displacement gate.

        Args:
            theta (real): displacement
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.ecd,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="ECD",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def cv_c_bs(self, theta, qumode_a, qumode_b, qubit, duration=100, unit="ns"):
        """Controlled phase two-mode beam splitter

        Args:
            theta (real or complex): phase
            qubit_ancilla (Qubit): QisKit control Qubit
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        self.append(
            ParameterizedUnitaryGate(
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
            qargs=qumode_a + qumode_b + [qubit],
        )

    def cv_c_schwinger(
        self, params, qumode_a, qumode_b, qubit, duration=100, unit="ns"
    ):
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
        self.append(
            ParameterizedUnitaryGate(
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
            qargs=qumode_a + qumode_b + [qubit],
        )

    def cv_snap(self, theta, n, qumode, qubit=None, duration=100, unit="ns"):
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
        if isinstance(n, int):
            if n > cutoff:
                ValueError("Fock state specified by n exceeds the cutoff.")
            if qubit is None:
                self.append(
                    ParameterizedUnitaryGate(
                        self.ops.snap,
                        [theta, n],
                        cutoffs=[cutoff],
                        num_qubits=len(qumode),
                        label="SNAP",
                        duration=duration,
                        unit=unit,
                    ),
                    qargs=qumode,
                )
            else:
                self.append(
                    ParameterizedUnitaryGate(
                        self.ops.csnap,
                        [theta, n],
                        cutoffs=[cutoff],
                        num_qubits=len(qumode) + 1,
                        label="cSNAP",
                        duration=duration,
                        unit=unit,
                    ),
                    qargs=qumode + [qubit],
                )
        elif isinstance(n, list) and isinstance(theta, list):
            if qubit is None:
                self.append(
                    ParameterizedUnitaryGate(
                        self.ops.multisnap,
                        theta + n,
                        cutoffs=[cutoff],
                        num_qubits=len(qumode),
                        label="SNAP",
                        duration=duration,
                        unit=unit,
                    ),
                    qargs=qumode,
                )
            else:
                self.append(
                    ParameterizedUnitaryGate(
                        self.ops.multicsnap,
                        theta + n,
                        cutoffs=[cutoff],
                        num_qubits=len(qumode) + 1,
                        label="cSNAP",
                        duration=duration,
                        unit=unit,
                    ),
                    qargs=qumode + [qubit],
                )
        else:
            raise ValueError(
                "if theta is passed as a list, then n must also be a list of equal length (and vice versa)."
            )

    # def cv_c_sqr(self, theta, n, qumode, qubit, duration=100, unit="ns"):
    #     """TODO"""

    # def cv_multisnap(self, thetas, ns, qumode, duration=1, unit="us"):
    #     params = thetas + ns
    #     self.append(
    #         ParameterizedUnitaryGate(
    #             self.ops.multisnap, params, num_qubits=len(qumode), label="mSNAP", duration=duration, unit=unit
    #         ),
    #         qargs=qumode,
    #     )

    def cv_c_pnr(self, max, qumode, qubit, duration=100, unit="ns"):
        """PNR (Photon number readout) TODO: Needs comments/explanation/citation!
        Args:
            max (int): the period of the mapping
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control qubit.
        Returns:
            Instruction: QisKit instruction
        """
        self.append(
            ParameterizedUnitaryGate(
                self.ops.pnr,
                [max],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="c_pnr",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def cv_eswap(self, theta, qumode_a, qumode_b, duration=100, unit="ns"):
        """Exponential SWAP gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qumode_b (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        self.append(
            ParameterizedUnitaryGate(
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
            qargs=qumode_a + qumode_b,
        )

    def cv_c_sq(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Conditional squeezing gate.

        Args:
            theta (real or complex): squeezing ampltiude
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control Qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.csq,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="cS",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def cv_testqubitorderf(self, phi, qubit_1, qubit_2, duration=100, unit="ns"):
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.testqubitorderf,
                [phi],
                label="testqubitorderf",
                cutoffs=[],
                num_qubits=2,
                duration=duration,
                unit=unit,
            ),
            qargs=[qubit_1] + [qubit_2],
        )

    def cv_sum(self, scale, qumode_a, qumode_b, duration=100, unit="ns"):
        """Two-mode sum gate.

        Args:
            scale (real): arbitrary real scale factor
            qumode_a (list): list of qubits representing qumode
            qumode_b (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
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
            qargs=qumode_a + qumode_b,
        )

    def cv_c_sum(self, scale, qumode_a, qumode_b, qubit, duration=100, unit="ns"):
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
            ParameterizedUnitaryGate(
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
            qargs=qumode_a + qumode_b + [qubit],
        )

    def cv_jc(self, theta, phi, qumode, qubit, duration=100, unit="ns"):
        """Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
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
            qargs=qumode + [qubit],
        )

    def cv_ajc(self, theta, phi, qumode, qubit, duration=100, unit="ns"):
        """Anti-Jaynes-Cummings gate

        Args:
            theta (real): [0, 2pi)
            phi (real): [0, 2pi)
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
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
            qargs=qumode + [qubit],
        )

    def cv_rb(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Rabi interaction gate

        Args:
            theta (real): arbitrary scale factor

        Returns:
            csc_matrix: operator matrix
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.rb,
                [theta],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="rb",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def measure_z(self, qubit, cbit, duration=100, unit="ns"):
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

        return super.measure(qubit, cbit)

    def measure_y(self, qubit, cbit):
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

    def measure_x(self, qubit, cbit):
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

    def cv_measure(self, qubit_qumode_list, cbit_list):
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

        # Flattens the list (if necessary)
        flat_list = []
        for el in qubit_qumode_list:
            if isinstance(el, list):
                flat_list += el
            else:
                flat_list += [el]

        # Check to see if too many classical registers were passed in.
        # If not, only use those needed (starting with least significant bit).
        # This piece is useful so that the user doesn't need to think about
        # how many bits are needed to read out a list of qumodes, qubits, etc.
        if len(flat_list) < len(cbit_list):
            self.measure(flat_list, cbit_list[0 : len(flat_list)])
        else:
            self.measure(flat_list, cbit_list)

    def cv_delay(self, duration, qumode, unit="ns"):
        """CV_delay. Implements an identity gate of the specified duration.
        This is particularly useful for the implementation of a noise pass.

        Args:
            duration (real): duration of delay gate
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.get_eye,
                [],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode),
                label="cv_delay(" + str(duration) + " " + unit + ")",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode,
        )

    def cv_c_multiboson_sampling(self, max, qumode, qubit, duration=1, unit="us"):
        """SNAP (Selective Number-dependent Arbitrary Phase) gates for multiboson sampling.
        Args:
            max (int): the period of the mapping
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control qubit.
        Returns:
            Instruction: QisKit instruction
        """
        self.append(
            ParameterizedUnitaryGate(
                self.ops.c_multiboson_sampling,
                [max],
                cutoffs=[QumodeRegister.calculate_cutoff(len(qumode))],
                num_qubits=len(qumode) + 1,
                label="c_multiboson_sampling",
                duration=duration,
                unit=unit,
            ),
            qargs=qumode + [qubit],
        )

    def cv_gate_from_matrix(
        self,
        matrix,
        qumodes=[],
        qubits=[],
        duration=100,
        unit="ns",
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
        # If multiple qubits are given, slice to get list of qubits. Otherwise, encase QuantumRegister for single qubit in list.
        try:
            qubits = qubits[:]
        except:
            qubits = [qubits]

        # Slice QumodeRegister to get list of qumodes
        if isinstance(qumodes, QumodeRegister):
            qumodes = qumodes[:]

        # Convert matrix to np.ndarray so that we can compute error flags easier
        matrix = np.array(matrix)

        ## Error flags
        # Matrix needs to be square
        n, m = matrix.shape
        if n != m:
            raise ValueError("Matrix given is not square")

        # Determine if input matrix is same dimension as input qumodes+qubits
        if n != 2 ** (len(qumodes) + len(qubits)):
            raise ValueError("Matrix is of different dimension from qumodes + qubits")

        # Checks if input matrix is unitary
        if not is_unitary_matrix(matrix):
            raise ValueError("The mapping provided is not unitary!")

        # Make sure that np.ndarray doesn't get fed into PUG
        if isinstance(matrix, np.ndarray):
            matrix = matrix.tolist()

        # TODO Is it safe to ignore cutoff, assuming provided matrix is correct?
        cutoffs = []
        # for qumode in qumodes:
        #     cutoffs.append(QumodeRegister.calculate_cutoff(len(qumode)))

        # return self.append(
        #     ParameterizedUnitaryGate(
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
        return self.unitary(matrix, qubits=qumodes + qubits, label=label)
