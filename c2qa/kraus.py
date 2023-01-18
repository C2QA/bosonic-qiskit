import functools
import math
from typing import Sequence
import warnings


import c2qa
import numpy
from qiskit.circuit import Instruction, Qubit
from qiskit.providers.aer.noise.passes.local_noise_pass import LocalNoisePass
from qiskit.providers.aer.noise import kraus_error
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.utils.units import apply_prefix
import scipy


def calculate_kraus(
    photon_loss_rates: Sequence[float], 
    time: float, 
    circuit: c2qa.CVCircuit,
    op_qubits: Sequence[int],
    qumode_indices: Sequence[int]):
    """
    Calculate Kraus operator given number of photons and photon loss rate over specified time.

    Apply Kraus operator to provided qubit_indices only, tensor product with identity for remaining qubits.

    Following equation 44 from Bosonic Oprations and Measurements, Girvin

    Args:
        photon_loss_rates (Sequence[float]): kappas, the rate of photon loss per second for each qumode
        time (float): current duration of time in seconds
        circuit (CVCircuit): cq2a.CVCircuit with ops for N and a
        op_qubits (Sequence[int]): qubit int indices in the given CVCircuit used by the current instruction
        qumode_indices (Sequence[int]): qumode int indices in the given CVCircuit to test if qubits from instruction are a part of a qumode

    Returns:
        List of Kraus operators
    """

    # Identity for individual qubit
    qubit_eye = numpy.eye(2)

    operators = []
    kraus_tensor = {}

    for op_qubit in op_qubits:
        if qumode_indices and op_qubit in qumode_indices:
            qubit_index = qumode_indices.index(op_qubit)
            qumode_index = math.floor(qubit_index / circuit.num_qubits_per_qumode)

            # Tensor Kraus operators, if not already done
            if not kraus_tensor.get(qumode_index, False):
                kraus_ops = __kraus_operators(photon_loss_rates[qumode_index], time, circuit, circuit.ops.a, circuit.ops.N)
                operators = __tensor_operators(operators, kraus_ops)               
                kraus_tensor[qumode_index] = True
        else:
            # Tensor qubit identity
            operators = __tensor_operators(operators, [qubit_eye])

    return operators


def __tensor_operators(current: list, new: list):
    result = []
    if len(current) > 0:
        for current_op in current:
            for new_op in new:
                result.append(numpy.kron(current_op, new_op))
    else:
        result.extend(new)

    return result



def __kraus_operators(photon_loss_rate: float, time: float, circuit: c2qa.CVCircuit, a, n):
    operators = []
    for photons in range(circuit.cutoff + 1):
        kraus = math.sqrt(
            math.pow((1 - math.exp(-1 * photon_loss_rate * time)), photons)
            / math.factorial(photons)
        )
        kraus = kraus * scipy.sparse.linalg.expm(
            -1 * (photon_loss_rate / 2) * time * n
        )
        kraus = kraus.dot(a**photons)
        operators.append(kraus.toarray())

    return operators


class PhotonLossNoisePass(LocalNoisePass):
    """Add photon loss noise model to a circuit during transpiler transformation pass."""

    def __init__(
        self,
        photon_loss_rates: Sequence[float],
        circuit: c2qa.CVCircuit,
        instructions: Sequence[str] = None,
        qumodes: Sequence[Qubit] = None,
        time_unit: str = "s",
        dt: float = None
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

        if instructions is None:
            self._instructions = None
        elif isinstance(instructions, list):
            self._instructions = instructions
        else:
            self._instructions = [instructions]

        if qumodes is None:
            # Apply photon loss to all qumodes by default
            self._qumodes = self._circuit.qumode_qubits
        elif isinstance(qumodes, list):
            self._qumodes = qumodes
        else:
            self._qumodes = [qumodes]

        self._qumode_indices = circuit.get_qubit_indices(self._qumodes)
        if len(self._qumode_indices) % self._circuit.num_qubits_per_qumode != 0:
            raise Exception("List of qumode indices in PhotonLossNoisePass is not a multiple of the number of qubits per qumode")

        self._num_qumodes = len(self._qumode_indices) // self._circuit.num_qubits_per_qumode

        if isinstance(photon_loss_rates, list):
            self._photon_loss_rates = photon_loss_rates
        else:
            self._photon_loss_rates = [photon_loss_rates]

        # If only one rate was given for multiple qumodes, apply that rate to all qumodes
        if len(self._photon_loss_rates) == 1 and self._num_qumodes > 1:
            self._photon_loss_rates = self._photon_loss_rates * self._num_qumodes

        # Test that we have the correct number of photon loss rates
        if len(self._photon_loss_rates) != self._num_qumodes:
            raise Exception("List of photon loss rates must have same length as number of qumodes! (i.e., one rate per qumode)")

        # Convert photon loss rate to photons per second
        if self._time_unit == "dt":
            self._photon_loss_rates_sec = [rate * self._dt for rate in self._photon_loss_rates] 
        else:
            conversion = 1.0 / apply_prefix(1.0, self._time_unit)
            self._photon_loss_rates_sec = [rate * conversion for rate in self._photon_loss_rates] 

        super().__init__(self._photon_loss_error)

    def _photon_loss_error(self, op: Instruction, qubits: Sequence[int]):
        """Return photon loss error on each operand qubit"""
        error = None
        
        if (self._instructions is None or op.name in self._instructions) and (self._qumode_indices is None or any(x in qubits for x in self._qumode_indices)):
            if not op.duration:
                if op.duration is None:
                    warnings.warn(
                        "PhotonLossNoisePass ignores instructions without duration,"
                        " you may need to schedule circuit in advance.",
                        UserWarning,
                    )
                return None

            # Qiskit `delay` gates are always for one qubit, see https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.delay.html            
            if op.name == "delay":
                warnings.warn("Qiskit applies delays as single qubit gates. Qumode PhotonLossNoisePass will not be applied")
                return None

            # Convert op duration to seconds
            if op.unit == "dt":
                if self._dt is None:
                    raise NoiseError(
                        "PhotonLossNoisePass cannot apply noise to a 'dt' unit duration"
                        " without a dt time set."
                    )
                duration = op.duration * self._dt
            else:
                duration = apply_prefix(op.duration, op.unit)

            kraus_operators = calculate_kraus(
                self._photon_loss_rates_sec, duration, self._circuit, qubits, self._qumode_indices
            )

            error = kraus_error(kraus_operators)

        return error
