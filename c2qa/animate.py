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
from c2qa.discretize import discretize_circuits, discretize_single_circuit
from c2qa.wigner import simulate_wigner, simulate_wigner_multiple_statevectors


def animate_wigner(
    circuit: CVCircuit,
    qubit: qiskit.circuit.quantumcircuit.QubitSpecifier = None,
    cbit: qiskit.circuit.quantumcircuit.ClbitSpecifier = None,
    animation_segments: int = 10,
    discretize_epsilon: float = None,
    shots: int = 1,
    file: str = None,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    processes: int = None,
    keep_state: bool = True,
    noise_passes=None,
    sequential_subcircuit: bool = False,
    draw_grid: bool = False,
    trace: bool = True,
    bitrate: int = -1,
):
    """Animate the Wigner function at each step defined in the given CVCirctuit.

    This assumes the CVCircuit was simulated with an animation_segments > 0 to
    act as the frames of the generated movie.

    The ffmpeg binary must be on your system PATH in order to execute this
    function. See https://ffmpeg.org/download.html to download and install on your system.

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
        noise_passes (list of Qiskit noise passes, optional): noise passes to apply
        sequential_subcircuit (bool, optional): boolean flag to animate subcircuits as one gate (False) or as sequential
                                                gates (True). Defautls to False.
        draw_grid (bool, optional): True if grid lines should be drawn on plot. Defaults to False.
        trace (bool, optional):  True if qubits should be tracedfor each frame prior to calculating Wigner function. Defaults to True.

    Returns:
        [type]: [description]
    """

    if qubit or cbit:
        w_fock, xvec = __discretize_wigner_with_measure(
            circuit,
            qubit,
            cbit,
            animation_segments,
            shots,
            axes_min,
            axes_max,
            axes_steps,
            processes,
            keep_state,
            noise_passes,
            sequential_subcircuit,
            trace,
        )
    else:
        w_fock, xvec = __discretize_wigner_without_measure(
            circuit,
            animation_segments,
            discretize_epsilon,
            shots,
            axes_min,
            axes_max,
            axes_steps,
            noise_passes,
            sequential_subcircuit,
            trace,
        )

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
        fargs=(fig, ax, xvec, w_fock, file, draw_grid),
        interval=200,
        repeat=True,
    )

    # Save to file using ffmpeg, Pillow (GIF, APNG), or display
    if file:
        save_animation(anim, file, bitrate)

    return anim


def __discretize_wigner_with_measure(
    circuit: CVCircuit,
    qubit: qiskit.circuit.quantumcircuit.QubitSpecifier = None,
    cbit: qiskit.circuit.quantumcircuit.ClbitSpecifier = None,
    animation_segments: int = 10,
    shots: int = 1,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    processes: int = None,
    keep_state: bool = True,
    noise_passes=None,
    sequential_subcircuit: bool = False,
    trace: bool = True,
):
    circuits = discretize_circuits(
        circuit, animation_segments, keep_state, qubit, cbit, sequential_subcircuit
    )

    # Calculate the Wigner functions for each frame
    if not processes or processes < 1:
        processes = math.floor(multiprocessing.cpu_count() / 2)
        processes = max(processes, 1)  # prevent zero processes with 1 CPU

    # Simulate each frame, storing Wigner function data in w_fock
    xvec = numpy.linspace(axes_min, axes_max, axes_steps)

    if keep_state:
        w_fock = __simulate_wigner_with_state(
            circuits, qubit, cbit, xvec, shots, noise_passes, trace
        )
    elif processes == 1:
        w_fock = []
        for circuit in circuits:
            fock, _ = simulate_wigner(
                circuit,
                xvec,
                shots,
                noise_passes=noise_passes,
                conditional=cbit is not None,
                trace=trace or cbit is not None,
            )
            w_fock.append(fock)
    else:
        pool = multiprocessing.Pool(processes)
        results = pool.starmap(
            simulate_wigner,
            (
                (
                    circuit,
                    xvec,
                    shots,
                    noise_passes,
                    cbit is not None,
                    trace or cbit is not None,
                )
                for circuit in circuits
            ),
        )
        pool.close()
        w_fock = [i[0] for i in results if i is not None]

    return w_fock, xvec


def __discretize_wigner_without_measure(
    circuit: CVCircuit,
    animation_segments: int = 10,
    discretize_epsilon: float = None,
    shots: int = 1,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    noise_passes=None,
    sequential_subcircuit: bool = False,
    trace: bool = True,
):
    statevector_label = "segment_"

    discretized, num_statevectors = discretize_single_circuit(
        circuit=circuit,
        segments_per_gate=animation_segments,
        epsilon=discretize_epsilon,
        sequential_subcircuit=sequential_subcircuit,
        statevector_per_segment=True,
        statevector_label=statevector_label,
        noise_passes=noise_passes,
    )

    xvec = numpy.linspace(axes_min, axes_max, axes_steps)

    w_fock = simulate_wigner_multiple_statevectors(
        circuit=discretized,
        xvec=xvec,
        shots=shots,
        statevector_label=statevector_label,
        num_statevectors=num_statevectors,
        noise_passes=noise_passes,
        trace=trace,
    )

    return w_fock, xvec


def save_animation(anim: matplotlib.animation.FuncAnimation, file: str, bitrate: int):
    file_path = pathlib.Path(file)

    if file_path.suffix == ".mp4":
        writer = matplotlib.animation.FFMpegWriter(fps=24, bitrate=bitrate)
    elif file_path.suffix == ".gif" or file_path.suffix == ".apng":
        writer = matplotlib.animation.PillowWriter(fps=24, bitrate=bitrate)
    else:
        print(
            f"Unknown animation file type {file_path.suffix}, defaulting to using PillowWriter"
        )
        writer = matplotlib.animation.PillowWriter(fps=24, bitrate=bitrate)

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
    draw_grid = fargs[5]

    amax = numpy.amax(w_fock)
    amin = numpy.amin(w_fock)
    abs_max = max(amax, abs(amin))
    if abs_max == 0:
        abs_max = 5
    color_levels = numpy.linspace(-abs_max, abs_max, 100)

    xvec_int = [int(x) for x in xvec]
    xvec_int = sorted(set(xvec_int))

    ax.clear()
    cont = ax.contourf(xvec, xvec, w_fock, color_levels, cmap="RdBu")

    ax.set_xlabel(r"$x$")
    ax.set_xticks(xvec_int)
    ax.set_ylabel(r"$p$")
    ax.set_yticks(xvec_int)
    if draw_grid:
        ax.grid()

    if frame == 0:
        fig.colorbar(cont, ax=ax)

    time_text = ax.text(
        0.05,
        0.95,
        "",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    time_text.set_text(f"Frame {frame}")

    if file:
        os.makedirs(f"{file}_frames", exist_ok=True)
        plt.savefig(f"{file}_frames/frame_{frame}.png")


def __simulate_wigner_with_state(
    circuits, qubit, cbit, xvec, shots, noise_passes, trace
):
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
            noise_passes=noise_passes,
            conditional=cbit is not None,
            trace=trace or cbit is not None,
        )
        w_fock.append(fock)

    return w_fock
