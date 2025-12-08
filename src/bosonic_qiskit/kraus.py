import math
import warnings
from collections import Counter
from collections.abc import Sequence
from typing import cast

import numpy as np
import scipy
import scipy.sparse as sp
from qiskit.circuit import Instruction, Qubit
from qiskit.utils.units import apply_prefix
from qiskit_aer.noise import QuantumError, kraus_error
from qiskit_aer.noise.noiseerror import NoiseError
from qiskit_aer.noise.passes.local_noise_pass import LocalNoisePass

import bosonic_qiskit

IGNORE_INSTRUCTIONS = ["measure"]


def calculate_kraus(
    photon_loss_rates: Sequence[float],
    time: float,
    circuit: bosonic_qiskit.CVCircuit,
    op_qubits: Sequence[int],
    qumode_qubit_indices: Sequence[int],
) -> list[np.ndarray]:
    """
    Calculate Kraus operator given number of photons and photon loss rate over specified time.

    Apply Kraus operator to provided qubit_indices only, tensor product with identity for remaining qubits.

    Following equation 44 from Bosonic Oprations and Measurements, Girvin

    Args:
        photon_loss_rates (Sequence[float]): kappas, the rate of photon loss per second for each qumode
        time (float): current duration of time in seconds
        circuit (CVCircuit): cq2a.CVCircuit with ops for N and a
        op_qubits (Sequence[int]): qubit int indices in the given CVCircuit used by the current instruction
        qumode_qubit_indices (Sequence[int]): qumode int indices in the given CVCircuit to test if qubits from instruction are a part of a qumode

    Returns:
        List of Kraus operators
    """

    # Identity for individual qubit
    qubit_eye = np.eye(2)

    operators = []
    kraus_tensor = {}
    loss_rate_index = 0
    for op_qubit in op_qubits:
        if qumode_qubit_indices and op_qubit in qumode_qubit_indices:
            qubit = circuit.qubits[op_qubit]
            qmr_index = circuit.get_qmr_index(qubit)
            qumode_index = circuit.qmregs[qmr_index].get_qumode_index(qubit)
            unique_index = f"{qmr_index}-{qumode_index}"

            cutoff = circuit.get_qmr_cutoff(qmr_index)

            # Tensor Kraus operators, if not already done
            if not kraus_tensor.get(
                unique_index, False
            ):  # FIXME need to tensor per qumode, not qumode register
                kraus_ops = __kraus_operators(
                    photon_loss_rates[loss_rate_index],
                    time,
                    cutoff,
                    circuit.ops.get_a(cutoff),
                    circuit.ops.get_N(cutoff),
                )
                operators = __tensor_operators(operators, kraus_ops)
                kraus_tensor[unique_index] = True
                loss_rate_index += 1
        else:
            # Tensor qubit identity
            operators = __tensor_operators(operators, [qubit_eye])

    return operators


def __tensor_operators(
    current: list[np.ndarray], new: list[np.ndarray]
) -> list[np.ndarray]:
    result = []
    if current:
        for current_op in current:
            for new_op in new:
                result.append(np.kron(current_op, new_op))
    else:
        result.extend(new)

    return result


def __kraus_operators(
    photon_loss_rate: float,
    time: float,
    cutoff: int,
    a: sp.csc_array,
    n: sp.csc_array,
) -> list[np.ndarray]:
    """Calculates the Kraus operators for photon loss

    Args:
        photon_loss_rate: The rate of photon loss in the qumode

        time: The duration of the noise channel

        cutoff: The maximum fock level of the qumode

        a: The annihilation operator

        n: The photon number operator

    Returns:
        A list of Kraus operators, where operator i corresponds to losing i photons
    """
    operators: list[np.ndarray] = []
    for photons in range(cutoff + 1):
        kraus = math.sqrt(
            math.pow((1 - math.exp(-1 * photon_loss_rate * time)), photons)
            / math.factorial(photons)
        )
        kraus = kraus * sp.linalg.expm(-1 * (photon_loss_rate / 2) * time * n)
        kraus = cast(sp.csc_array, kraus)

        # a^0 is identity which has no effect and will throw an error
        if photons > 0:
            kraus = kraus @ sp.linalg.matrix_power(a, photons)

        operators.append(kraus.todense())

    return operators


class PhotonLossNoisePass(LocalNoisePass):
    """Add photon loss noise model to a circuit during transpiler transformation pass."""

    def __init__(
        self,
        photon_loss_rates: float | Sequence[float],
        circuit: bosonic_qiskit.CVCircuit,
        instructions: str | Sequence[str] | None = None,
        qumodes: Qubit | Sequence[Qubit] | None = None,
        time_unit: str = "s",
        dt: float | None = None,
    ):
        """
        Initialize the Photon Loss noise pass

        Args:
            photon_loss_rate (float): kappa, the rate of photon loss per second
            circuit (CVCircuit): cq2a.CVCircuit with ops for N and a, and cutoff
            instructions (str or list[str]): the instructions error applies to
            qumode (Sequence[Qubit]): qumode qubits error noise pass applies to
            time_unit (string): string photon loss rate unit of time (default "s" for seconds)
            dt (float): optional conversion factor for photon_loss_rate and operation duration to seconds
        """

        self._circuit = circuit
        self._time_unit = time_unit
        self._dt = dt

        instructions = instructions or []
        self._instructions = set(
            [instructions] if isinstance(instructions, str) else instructions
        )

        # Apply photon loss to all qumodes by default
        qumodes = qumodes or self._circuit.qumode_qubits
        self._qumodes = [qumodes] if isinstance(qumodes, Qubit) else qumodes
        self._qumode_qubit_indices = circuit.get_qubit_indices(self._qumodes)
        self._num_qumodes = self._calculate_num_qumodes(circuit, self._qumodes)

        if isinstance(photon_loss_rates, Sequence):
            self._photon_loss_rates = photon_loss_rates
        else:
            self._photon_loss_rates = [photon_loss_rates]

        # If only one rate was given for multiple qumodes, apply that rate to all qumodes
        if len(self._photon_loss_rates) == 1 and self._num_qumodes > 1:
            self._photon_loss_rates = [
                self._photon_loss_rates[0] for _ in range(self._num_qumodes)
            ]

        # Test that we have the correct number of photon loss rates
        if len(self._photon_loss_rates) != self._num_qumodes:
            raise Exception(
                "List of photon loss rates must have same length as number of qumodes! (i.e., one rate per qumode)"
            )

        # Convert photon loss rate to photons per second
        if self._time_unit == "dt":
            if self._dt is None:
                raise ValueError(
                    "Need to pass a conversion rate for `dt` when using time units `dt`"
                )

            self.photon_loss_rates_sec = [
                rate * self._dt for rate in self._photon_loss_rates
            ]
        else:
            conversion = 1.0 / apply_prefix(1.0, self._time_unit)
            self.photon_loss_rates_sec = [
                rate * conversion for rate in self._photon_loss_rates
            ]

        super().__init__(self._photon_loss_error)

    def _photon_loss_error(
        self, op: Instruction, qubits: Sequence[int]
    ) -> QuantumError | None:
        """Return photon loss error on each operand qubit"""
        if self.applies_to_instruction(op, qubits):
            # FIXME - Qiskit v2.0 removed Instruction duration & unit!
            if getattr(op, "duration", None) is None:
                warnings.warn(
                    "PhotonLossNoisePass ignores instructions without duration,"
                    " you may need to schedule circuit in advance.",
                    UserWarning,
                )
                return None

            # Qiskit `delay` gates are always for one qubit, see https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.delay.html
            if op.name.startswith("delay"):
                warnings.warn(
                    "Qiskit applies delays as single qubit gates. Qumode PhotonLossNoisePass will not be applied"
                )
                return None

            duration = self.duration_to_sec(op)

            kraus_operators = calculate_kraus(
                self.photon_loss_rates_sec,
                duration,
                self._circuit,
                qubits,
                self._qumode_qubit_indices,
            )

            return kraus_error(kraus_operators)

        return None

    def applies_to_instruction(self, op: Instruction, qubits: Sequence[int]) -> bool:
        """Test if this PhotonLossNoisePass applies to the given instruction based on its name and qumodes (qubits)"""
        # FIXME Qiskit v2.0 measure fails in PhotonLossNoisePass, but not in <v1.x?

        inst_whitelisted = (
            op.name not in IGNORE_INSTRUCTIONS
            and not self._instructions
            or op.name in self._instructions
        )
        intersects_qubits = not self._qumode_qubit_indices or any(
            x in qubits for x in self._qumode_qubit_indices
        )

        return inst_whitelisted and intersects_qubits

    def duration_to_sec(self, op: Instruction) -> float:
        """Return the given Instruction's duration in seconds"""

        # FIXME - Qiskit v2.0 removed Instruction duration & unit!
        unit = cast(str | None, getattr(op, "unit", None))
        duration = cast(int | None, getattr(op, "duration", None))

        if unit and duration:
            if unit == "dt":
                if self._dt is None:
                    raise NoiseError(
                        "PhotonLossNoisePass cannot apply noise to a 'dt' unit duration without a dt time set."
                    )
                return duration * self._dt

            return apply_prefix(duration, unit)

        return 0.000_000_1  # 100ns

    def _calculate_num_qumodes(
        self, circuit: bosonic_qiskit.CVCircuit, qumodes: Sequence[Qubit]
    ) -> int:
        # Calculate the number of qumodes based on the number of times the QumodeRegister index changes for the given qumode qubits
        qmr_to_num_qubits: Counter[int] = Counter()
        for qubit in qumodes:
            qmr_index = circuit.get_qmr_index(qubit)
            qmr_to_num_qubits[qmr_index] += 1

        num_qumodes: int = 0
        for qmr_index, num_qubits in qmr_to_num_qubits.items():
            num_qubits_per_qumode = circuit.get_qmr_num_qubits_per_qumode(qmr_index)
            num_qumodes += num_qubits // num_qubits_per_qumode

        return num_qumodes
