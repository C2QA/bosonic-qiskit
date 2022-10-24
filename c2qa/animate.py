import math
import multiprocessing
import os
import pathlib


import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy
import qiskit


from c2qa.circuit import CVCircuit
from c2qa.operators import ParameterizedUnitaryGate
from c2qa.wigner import simulate_wigner


def animate_wigner(
    circuit: CVCircuit,
    qubit: qiskit.circuit.quantumcircuit.QubitSpecifier = None,
    cbit: qiskit.circuit.quantumcircuit.ClbitSpecifier = None,
    animation_segments: int = 10,
    shots: int = 1,
    file: str = None,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    processes: int = None,
    keep_state: bool = True,
    noise_pass=None,
    sequential_subcircuit: bool = False,
):
    """Animate the Wigner function at each step defined in the given CVCirctuit.

    This assumes the CVCircuit was simulated with an animation_segments > 0 to
    act as the frames of the generated movie.

    The ffmpeg binary must be on your system PATH in order to execute this
    function.

    Args:
        circuit (CVCircuit): circuit to simulate and plot
        qubit ([type]): Qubit to measure, if performing Hadamard measure for use with cat states. Defaults to None.
        cbit ([type]): Classical bit to measure into, if performing Hadamard measure for use with cat states. Defaults to None.
        animation_segments (int, optional): Number of segments to split each gate into for animation. Defaults to 10.
        shots (int, optional): Number of simulation shots per frame. Defaults to 1.
        file (str, optional): File path to save (supported formats include MP4 with ffmpeg installed, animated GIF, and APNG). 
                              If None, return plot. Defaults to None.
        axes_min (int, optional): Minimum axes plot value. Defaults to -6.
        axes_max (int, optional): Maximum axes plot value. Defaults to 6.
        axes_steps (int, optional): Steps between axes ticks. Defaults to 200.
        processes (int, optional): Number of parallel Python processes to start.
                                   If None, perform serially in main process. Defaults to None.
        keep_state (bool, optional): True if each frame builds on the previous frame's state vector.
                                     False if each frame starts over from the beginning of the circuit.
                                     If True, it requires sequential simulation of each frame.
        noise_pass (PhotonLossNoisePass, optional): noise pass to apply
        sequential_subcircuit (bool, optional): boolean flag to animate subcircuits as one gate (False) or as sequential 
                                                gates (True). Defautls to False.

    Returns:
        [type]: [description]
    """

    circuits = __animate_circuit(circuit, animation_segments, keep_state, qubit, cbit, sequential_subcircuit)

    # Calculate the Wigner functions for each frame
    if not processes or processes < 1:
        processes = math.floor(multiprocessing.cpu_count() / 2)
        processes = max(processes, 1)  # prevent zero processes with 1 CPU

    # Simulate each frame, storing Wigner function data in w_fock
    xvec = numpy.linspace(axes_min, axes_max, axes_steps)

    if keep_state:
        w_fock = __simulate_wigner_with_state(circuits, qubit, cbit, xvec, shots, noise_pass)
    elif processes == 1:
        w_fock = []
        for circuit in circuits:
            fock, _ = simulate_wigner(
                circuit,
                xvec,
                shots,
                noise_pass=noise_pass,
                conditional=cbit is not None,
            )
            w_fock.append(fock)
    else:
        pool = multiprocessing.Pool(processes)
        results = pool.starmap(
            simulate_wigner,
            (
                (circuit, xvec, shots, noise_pass, cbit is not None)
                for circuit in circuits
            ),
        )
        pool.close()
        w_fock = [i[0] for i in results if i is not None]

    # Remove None values in w_fock if simulation didn't produce results
    w_fock = [i for i in w_fock if i is not None]

    # Animate w_fock Wigner function results
    # Create empty plot to animate
    fig, ax = plt.subplots(constrained_layout=True)

    # Animate
    anim = matplotlib.animation.FuncAnimation(
        fig=fig,
        init_func=_animate_init,
        func=_animate,
        frames=len(w_fock),
        fargs=(fig, ax, xvec, w_fock, file),
        interval=200,
        repeat=True,
    )

    # Save to file using ffmpeg, Pillow (GIF, APNG), or display
    if file:
        save_animation(anim, file)

    return anim


def __animate_circuit(circuit, animation_segments, keep_state, qubit, cbit, sequential_subcircuit):
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
        frames = __animate_parameterized(inst, animation_segments, keep_state)

    elif hasattr(inst, "cv_conditional") and inst.cv_conditional:
        frames = __animate_conditional(inst, animation_segments, keep_state)

    # FIXME -- how to identify a gate that was made with QuantumCircuit.to_gate()?
    elif isinstance(inst.definition, qiskit.QuantumCircuit) and inst.name != "initialize" and len(inst.decompositions) == 0:  # Don't animate subcircuits initializing system state
        frames = __animate_subcircuit(inst.definition, animation_segments, keep_state, sequential_subcircuit)

    elif isinstance(inst, qiskit.circuit.instruction.Instruction) and inst.name != "initialize":  # Don't animate instructions initializing system state
        frames = __animate_instruction(inst, animation_segments, keep_state)

    else:
        frames = __animate_copy(inst, animation_segments)
    
    return frames


def __animate_parameterized(inst, animation_segments, keep_state):
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


def __animate_conditional(inst, animation_segments, keep_state):
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


def __animate_subcircuit(subcircuit, animation_segments, keep_state, sequential_subcircuit):
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


def __animate_instruction(inst, animation_segments, keep_state):
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


def __animate_copy(inst, animation_segments):
    """Return a list of inst of length animation segments"""
    frames = []

    for _ in range(animation_segments):
        frames.append(inst)

    return frames


def save_animation(anim: matplotlib.animation.FuncAnimation, file: str):
    file_path = pathlib.Path(file)

    if file_path.suffix == ".mp4":
        writer = matplotlib.animation.FFMpegWriter(fps=24)
    elif file_path.suffix == ".gif" or file_path.suffix == ".apng":
        writer = matplotlib.animation.PillowWriter(fps=24)
    else:
        print(
            f"Unknown animation file type {file_path.suffix}, defaulting to using PillowWriter"
        )
        writer = matplotlib.animation.PillowWriter(fps=24)

    anim.save(file, writer=writer)


def _animate_init():
    pass  # Prevent rendering frame 0 twice (once for init, once for animate)


def _animate(frame, *fargs):
    """Generate individual matplotlib frame in animation."""
    fig = fargs[0]
    ax = fargs[1]
    xvec = fargs[2]
    w_fock = fargs[3][frame]
    file = fargs[4]

    amax = numpy.amax(w_fock)
    amin = numpy.amin(w_fock)
    abs_max = max(amax, abs(amin))
    color_levels = numpy.linspace(-abs_max, abs_max, 100)

    ax.clear()
    cont = ax.contourf(xvec, xvec, w_fock, color_levels, cmap="RdBu_r")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    if frame == 0:
        fig.colorbar(cont, ax=ax)

    if file:
        os.makedirs(f"{file}_frames", exist_ok=True)
        plt.savefig(f"{file}_frames/frame_{frame}.png")


def __simulate_wigner_with_state(circuits, qubit, cbit, xvec, shots, noise_pass):
    """Simulate Wigner function, preserving state between iterations"""
    w_fock = []
    previous_state = None
    for circuit in circuits:
        if previous_state:
            # Initialize circuit to simulate with the previous frame's state, then append the last instruction
            sim_circuit = circuit.copy()
            sim_circuit.data.clear()  # Is this safe -- could we copy without data?
            sim_circuit.initialize(previous_state)

            if qubit and cbit:
                last_instructions = circuit.data[
                    -3:
                ]  # Get the last instruction, plus the Hadamard/measure
            else:
                last_instructions = circuit.data[-1:]  # Get the last instruction

            for inst in last_instructions:
                sim_circuit.append(*inst)
        else:
            # No previous simulation state, just run the current circuit
            sim_circuit = circuit
        fock, previous_state = simulate_wigner(
            sim_circuit,
            xvec,
            shots,
            noise_pass=noise_pass,
            conditional=cbit is not None,
        )
        w_fock.append(fock)

    return w_fock