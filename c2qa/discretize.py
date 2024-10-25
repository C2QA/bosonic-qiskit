import math
import qiskit

from c2qa.circuit import CVCircuit
from c2qa.kraus import PhotonLossNoisePass
from c2qa.parameterized_unitary_gate import ParameterizedUnitaryGate


def discretize_circuits(
    circuit: CVCircuit,
    segments_per_gate: int = 10,
    keep_state: bool = True,
    qubit: qiskit.circuit.quantumcircuit.QubitSpecifier = None,
    cbit: qiskit.circuit.quantumcircuit.QubitSpecifier = None,
    sequential_subcircuit: bool = False,
):
    """
    Discretize gates into a circuit into segments where each segment ends an indiviudal circuit. Useful for incrementally applying noise or animating the circuit.

    Args:
        circuit (CVCircuit): circuit to simulate and plot
        segments_per_gate (int, optional): Number of segments to split each gate into. Defaults to 10.
        keep_state (bool, optional): True if each gate segments builds on the previous gegment's state vector.
                                     False if each segment starts over from the beginning of the circuit.
                                     If True, it requires sequential simulation of each segment.
        qubit ([QubitSpecifier]): Qubit to measure, if performing Hadamard measure for use with cat states. Defaults to None.
        cbit ([QubitSpecifier]): Classical bit to measure into, if performing Hadamard measure for use with cat states. Defaults to None.
        sequential_subcircuit (bool, optional): boolean flag to animate subcircuits as one gate (False) or as sequential
                                                gates (True). Defautls to False.

    Returns:
        [list]: List of discretized Qiskit circuit
    """

    sim_circuits = []  # Each segment will have its own circuit to simulate

    # base_circuit is copied each gate iteration to build circuit segments to simulate
    base_circuit = circuit.copy()
    base_circuit.data.clear()  # Is this safe -- could we copy without data?

    for inst, qargs, cargs in circuit.data:
        # TODO - get qubit & cbit for measure instead of using parameters
        # qubit = xxx
        # cbit = yyy

        segments = __to_segments(
            inst, segments_per_gate, keep_state, sequential_subcircuit
        )

        for segment in segments:
            sim_circuit = base_circuit.copy()

            sim_circuit.append(instruction=segment, qargs=qargs, cargs=cargs)

            if qubit and cbit:
                # sim_circuit.barrier()
                sim_circuit.h(qubit)
                sim_circuit.measure(qubit, cbit)

            sim_circuits.append(sim_circuit)

            # Start with current circuit for the next segment
            base_circuit = sim_circuit

    return sim_circuits


def discretize_single_circuit(
    circuit: CVCircuit,
    segments_per_gate: int = 10,
    epsilon: float = None,
    sequential_subcircuit: bool = False,
    statevector_per_segment: bool = False,
    statevector_label: str = "segment_",
    noise_passes=None,
):
    """
    Discretize gates into a circuit into segments within a single output circuit. Useful for incrementally applying noise or animating the circuit.

    Args:
        circuit (CVCircuit): circuit to simulate and plot
        segments_per_gate (int, optional): Number of segments to split each gate into. Defaults to 10.
        epsilon (float, optional): float value used to discretize, must specify along with kappa
        kappa (float, optional): float phton loss rate to determine discretization sice, must specify along with epsilon
        sequential_subcircuit (bool, optional): boolean flag to animate subcircuits as one gate (False) or as sequential
                                                gates (True). Defaults to False.
        statevector_per_segment (bool, optional): boolean flag to save a statevector per gate segment. True will call Qiskit
                                                  save_statevector after each segment is simulated, creating statevectors labeled
                                                  "segment_*" that can used after simulation. Defaults to False.
        statevector_label (str, optional): String prefix to use for the statevector saved after each segment
        noise_passes (list of Qiskit noise passes, optional): noise passes to apply

    Returns:
        discretized Qiskit circuit
    """

    # discretized is a copy of the circuit as a whole. Each gate segment be added to simulate
    discretized = circuit.copy()
    discretized.data.clear()  # Is this safe -- could we copy without data?

    if noise_passes:
        if not isinstance(noise_passes, list):
            noise_passes = [noise_passes]

    segment_count = 0
    for inst, qargs, cargs in circuit.data:
        num_segments = segments_per_gate
        qargs_indices = [
            qubit._index for qubit in qargs
        ]  # FIXME -- is there a public API to get the qubit's index in Qiskit v1.0+?

        if noise_passes and not (
            isinstance(inst, qiskit.circuit.instruction.Instruction)
            and inst.name == "initialize"
        ):  # Don't discretize instructions initializing system state:
            noise_pass = None
            for current in noise_passes:
                if isinstance(
                    current, PhotonLossNoisePass
                ) and current.applies_to_instruction(inst, qargs_indices):
                    noise_pass = current
                    break

            if epsilon is not None and noise_pass is not None:
                # FIXME - which of the qumodes' loss rates and QumodeRegister's cutoff should we use?
                photon_loss_rate = noise_pass.photon_loss_rates_sec[0]
                num_segments = math.ceil(
                    (
                        photon_loss_rate
                        * noise_pass.duration_to_sec(inst)
                        * circuit.get_qmr_cutoff(0)
                    )
                    / epsilon
                )

        segments = __to_segments(
            inst=inst,
            segments_per_gate=num_segments,
            keep_state=True,
            sequential_subcircuit=sequential_subcircuit,
        )

        for segment in segments:
            discretized.append(instruction=segment, qargs=qargs, cargs=cargs)

            if statevector_per_segment:
                discretized.save_statevector(
                    label=f"{statevector_label}{segment_count}"
                )
                segment_count += 1

    return discretized, segment_count


def __to_segments(
    inst: qiskit.circuit.instruction.Instruction,
    segments_per_gate: int,
    keep_state: bool,
    sequential_subcircuit: bool,
):
    """Split the instruction into segments_per_gate segments"""

    if isinstance(inst, ParameterizedUnitaryGate):
        # print(f"Discretizing ParameterizedUnitaryGate {inst.name}")
        segments = __discretize_parameterized(inst, segments_per_gate, keep_state)

    # FIXME -- how to identify a gate that was made with QuantumCircuit.to_gate()?
    elif (
        isinstance(inst.definition, qiskit.QuantumCircuit)
        and inst.name != "initialize"
        and inst.label != "cv_gate_from_matrix"
        and len(inst.decompositions) == 0
    ):  # Don't animate subcircuits initializing system state
        # print(f"Discretizing QuantumCircuit {inst.name}")
        segments = __discretize_subcircuit(
            inst.definition, segments_per_gate, keep_state, sequential_subcircuit
        )

    elif (
        isinstance(inst, qiskit.circuit.instruction.Instruction)
        and inst.name != "initialize"
        and inst.label != "cv_gate_from_matrix"
        and len(inst.params) > 0
    ):  # Don't animate instructions initializing system state
        # print(f"Discretizing Instruction {inst.name}")
        segments = __discretize_instruction(inst, segments_per_gate, keep_state)

    else:
        # Else just "discretize" the instruction as a single segment
        # print(f"NOT discretizing {inst.name}")
        segments = [inst]

    return segments


def __discretize_parameterized(
    inst: qiskit.circuit.instruction.Instruction,
    segments_per_gate: int,
    keep_state: bool,
    discretized_param_indices: list = [],
):
    """Split ParameterizedUnitaryGate into multiple segments"""
    segments = []
    for index in range(1, segments_per_gate + 1):
        params = inst.calculate_segment_params(
            current_step=index,
            total_steps=segments_per_gate,
            keep_state=keep_state,
        )
        duration, unit = inst.calculate_segment_duration(
            current_step=index,
            total_steps=segments_per_gate,
            keep_state=keep_state,
        )

        # print(f"Discretized params {params} duration {duration} unit {unit}")

        segments.append(
            ParameterizedUnitaryGate(
                inst.op_func,
                params=params,
                cutoffs=inst.cutoffs,
                num_qubits=inst.num_qubits,
                label=inst.label,
                duration=duration,
                unit=unit,
            )
        )

    return segments


def __discretize_subcircuit(
    subcircuit: qiskit.QuantumCircuit,
    segments_per_gate: int,
    keep_state: bool,
    sequential_subcircuit: bool,
):
    """Create a list of circuits where the entire subcircuit is converted into segments (vs a single instruction)."""

    segments = []
    sub_segments = []

    for inst, qargs, cargs in subcircuit.data:
        sub_segments.append(
            (
                __to_segments(
                    inst, segments_per_gate, keep_state, sequential_subcircuit
                ),
                qargs,
                cargs,
            )
        )

    if sequential_subcircuit:
        # Sequentially animate each gate within the subcircuit
        subcircuit_copy = subcircuit.copy()
        subcircuit_copy.data.clear()  # Is this safe -- could we copy without data?

        for sub_segment in sub_segments:
            gates, gate_qargs, gate_cargs = sub_segment
            for gate in gates:
                subcircuit_copy.append(gate, gate_qargs, gate_cargs)

        segments.append(subcircuit)
    else:
        # Animate the subcircuit as one gate
        for segment in range(segments_per_gate):
            subcircuit_copy = subcircuit.copy()
            subcircuit_copy.data.clear()  # Is this safe -- could we copy without data?

            for sub_segment, qargs, cargs in sub_segments:
                subcircuit_copy.append(sub_segment[segment], qargs, cargs)

            segments.append(subcircuit_copy)

    return segments


def __discretize_instruction(
    inst: qiskit.circuit.instruction.Instruction,
    segments_per_gate: int,
    keep_state: bool,
):
    """Split Qiskit Instruction into multiple segments"""
    segments = []

    for index in range(1, segments_per_gate + 1):
        params = inst.calculate_segment_params(
            current_step=index,
            total_steps=segments_per_gate,
            keep_state=keep_state,
        )
        duration, unit = inst.calculate_segment_duration(
            current_step=index,
            total_steps=segments_per_gate,
            keep_state=keep_state,
        )

        segments.append(
            qiskit.circuit.instruction.Instruction(
                name=inst.name,
                num_qubits=inst.num_qubits,
                num_clbits=inst.num_clbits,
                params=params,
                duration=duration,
                unit=unit,
                label=inst.label,
            )
        )

    return segments
