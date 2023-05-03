import qiskit

from c2qa.circuit import CVCircuit
from c2qa.parameterized_unitary_gate import ParameterizedUnitaryGate


def discretize_circuit(circuit, animation_segments, keep_state, qubit, cbit, sequential_subcircuit):
    sim_circuits = []  # Each frame will have its own circuit to simulate

    # base_circuit is copied each gate iteration to build circuit frames to simulate
    base_circuit = circuit.copy()
    base_circuit.data.clear()  # Is this safe -- could we copy without data?

    for inst, qargs, cargs in circuit.data:
        # TODO - get qubit & cbit for measure instead of using parameters
        # qubit = xxx
        # cbit = yyy

        frames = __to_frames(inst, animation_segments, keep_state, sequential_subcircuit)

        for frame in frames:
            sim_circuit = base_circuit.copy()

            sim_circuit.append(instruction=frame, qargs=qargs, cargs=cargs)

            if qubit and cbit:
                # sim_circuit.barrier()
                sim_circuit.h(qubit)
                sim_circuit.measure(qubit, cbit)

            sim_circuits.append(sim_circuit)

        # Append the full instruction for the next frame
        base_circuit.append(inst, qargs, cargs)
    
    return sim_circuits


def __to_frames(inst, animation_segments, keep_state, sequential_subcircuit):
    """Split the instruction into animation_semgments frames"""

    if isinstance(inst, ParameterizedUnitaryGate):
        frames = __discretize_parameterized(inst, animation_segments, keep_state)

    elif hasattr(inst, "cv_conditional") and inst.cv_conditional:
        frames = __discretize_conditional(inst, animation_segments, keep_state)

    # FIXME -- how to identify a gate that was made with QuantumCircuit.to_gate()?
    elif isinstance(inst.definition, qiskit.QuantumCircuit) and inst.name != "initialize" and len(inst.decompositions) == 0:  # Don't animate subcircuits initializing system state
        frames = __discretize_subcircuit(inst.definition, animation_segments, keep_state, sequential_subcircuit)

    elif isinstance(inst, qiskit.circuit.instruction.Instruction) and inst.name != "initialize":  # Don't animate instructions initializing system state
        frames = __discretize_instruction(inst, animation_segments, keep_state)

    else:
        # Else just "animate" the instruction as a single frame (multiple frames commented out)
        # frames = __discretize_copy(inst, animation_segments)
        frames = [inst]
    
    return frames


def __discretize_parameterized(inst, animation_segments, keep_state):
    """Split ParameterizedUnitaryGate into multiple frames"""
    frames = []
    for index in range(1, animation_segments + 1):
        params = inst.calculate_frame_params(
            current_step=index,
            total_steps=animation_segments,
            keep_state=keep_state,
        )
        duration, unit = inst.calculate_frame_duration(
            current_step=index,
            total_steps=animation_segments,
            keep_state=keep_state,
        )

        frames.append(
            ParameterizedUnitaryGate(
                inst.op_func,
                params=params,
                num_qubits=inst.num_qubits,
                label=inst.label,
                duration=duration,
                unit=unit,
            )
        )

    return frames


def __discretize_conditional(inst, animation_segments, keep_state):
    """Split Qiskit conditional gates into multiple frames"""
    frames = []
    inst_0, qargs_0, cargs_0 = inst.definition.data[0]
    inst_1, qargs_1, cargs_1 = inst.definition.data[1]

    for index in range(1, animation_segments + 1):
        params_0 = inst_0.base_gate.calculate_frame_params(
            current_step=index,
            total_steps=animation_segments,
            keep_state=keep_state,
        )
        params_1 = inst_1.base_gate.calculate_frame_params(
            current_step=index,
            total_steps=animation_segments,
            keep_state=keep_state,
        )

        duration, unit = inst_0.base_gate.calculate_frame_duration(
            current_step=index, total_steps=animation_segments
        )

        frames.append(
            CVCircuit.cv_conditional(
                name=inst.name,
                op=inst_0.base_gate.op_func,
                params_0=params_0,
                params_1=params_1,
                num_qubits_per_qumode=inst.num_qubits_per_qumode,
                num_qumodes=inst.num_qumodes,
                duration=duration,
                unit=unit,
            )
        )

    return frames


def __discretize_subcircuit(subcircuit, animation_segments, keep_state, sequential_subcircuit):
    """Create a list of circuits where the entire subcircuit is converted into frames (vs a single instruction)."""

    frames = []
    sub_frames = []

    for inst, qargs, cargs in subcircuit.data:
        sub_frames.append((__to_frames(inst, animation_segments, keep_state, sequential_subcircuit), qargs, cargs))

    if sequential_subcircuit:
        # Sequentially animate each gate within the subcircuit
        subcircuit_copy = subcircuit.copy()
        subcircuit_copy.data.clear()  # Is this safe -- could we copy without data?

        for sub_frame in sub_frames:
            gates, gate_qargs, gate_cargs = sub_frame
            for gate in gates:
                subcircuit_copy.append(gate, gate_qargs, gate_cargs)
        
        frames.append(subcircuit)
    else:
        # Animate the subcircuit as one gate
        for frame in range(animation_segments):
            subcircuit_copy = subcircuit.copy()
            subcircuit_copy.data.clear()  # Is this safe -- could we copy without data?

            for sub_frame,  qargs, cargs in sub_frames:
                subcircuit_copy.append(sub_frame[frame], qargs, cargs)
            
            frames.append(subcircuit_copy)

    return frames    


def __discretize_instruction(inst, animation_segments, keep_state):
    """Split Qiskit Instruction into multiple frames"""
    frames = []

    for index in range(1, animation_segments + 1):
        params = inst.calculate_frame_params(
            current_step=index,
            total_steps=animation_segments,
            keep_state=keep_state,
        )
        duration, unit = inst.calculate_frame_duration(
            current_step=index,
            total_steps=animation_segments,
            keep_state=keep_state,
        )
        
        frames.append(
            qiskit.circuit.instruction.Instruction(
                name=inst.name,
                num_qubits=inst.num_qubits,
                num_clbits = inst.num_clbits,
                params=params,
                duration=duration,
                unit=unit,
                label=inst.label,
            )
        )

    return frames