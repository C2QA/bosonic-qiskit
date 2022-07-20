from copy import copy
from logging import NOTSET
import math
import multiprocessing
import os
import pathlib
from typing import List

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
import scipy.stats

from c2qa import CVCircuit

from c2qa.operators import ParameterizedUnitaryGate

def stateread(stateop, numberofqubits, numberofmodes, cutoff, verbose=True):
    """Print values for states of qubits and qumodes using the result of a simulation of the statevector, e.g. using stateop, _ = c2qa.util.simulate(circuit).

    Returns the states of the qubits and the Fock states of the qumodes with respective amplitudes.
    """

    st = np.array(stateop)  # convert state to np.array
    amp_cv = []
    amp_qb = []

    for i in range(len(st)):
        res = st[
            i]  # go through vector element by element and find the positions of the non-zero elements with next if clause
        if (np.abs(res) > 1e-10):
            pos = i  # position of amplitude (non-zero real)
            # print("position of non-zero real amplitude: ", pos, " res = ", res)
            sln = len(st)  # length of the state vector

            ## Find the qubit states
            qbst = np.empty(numberofqubits, dtype='int')  # stores the qubit state
            iqb = 0  # counts up until the total number of qubits is reached
            # which half of the vector the amplitude is in is the state of the first qubit because of how the kronecker product is made
            while (iqb < numberofqubits):
                if pos < sln / 2:  # if the amplitude is in the first half of the state vector or remaining statevector
                    qbst[iqb] = int(0)  # then the qubit is in 0
                else:
                    qbst[iqb] = int(1)  # if the amplitude is in the second half then it is in 1
                    pos = pos - (
                                sln / 2)  # if the amplitude is in the second half of the statevector, then to find out the state of the other qubits and cavities then we remove the first half of the statevector for simplicity because it corresponds to the qubit being in 0 which isn't the case.
                    # print("pos (sln/2)", pos, "sln ",sln)
                sln = sln / 2  # only consider the part of the statevector corresponding to the qubit state which has just been discovered
                iqb = iqb + 1  # up the qubit counter to start finding out the state of the next qubit
            qbstr = ["".join(item) for item in qbst.astype(str)]
            amp_qb.append((qbst * (np.abs(res) ** 2)).tolist())

            ## Find the qumode states
            qmst = np.empty(numberofmodes, dtype='int')  # will contain the Fock state of each mode
            # print("qmst starting in ", qmst)
            iqm = 0  # counts up the number of modes
            # print("position is now: ",pos)
            while (iqm < numberofmodes):
                # print("mode counter iqm ", iqm)
                # print("cutoff ", cutoff)
                # print("length of vector left to search: sln ", sln)
                lendiv = sln / cutoff  # length of a division is the length of the statevector divided by the cutoff of the hilbert space (which corresponds to the number of fock states which a mode can have)
                # print("lendiv (sln/cutoff)", lendiv)
                val = pos / lendiv
                # print("rough estimate of the position of the non-zero element: val (pos/lendiv) ", val)
                fock = math.floor(val)
                # print("Fock st resulting position in Kronecker product (math.floor(val)) ", fock)
                qmst[iqm] = fock
                pos = pos - (
                            fock * lendiv)  # remove a number of divisions to then search a subsection of the Kronecker product
                # print("new position for next order of depth of Kronecker product/pos: (pos-(fock*lendiv)) ",pos)
                sln = sln - ((cutoff - 1) * lendiv)  # New length of vector left to search
                # print("New length of vector left to search: sln (sln-((cutoff-1)*lendiv))", sln)
                iqm = iqm + 1
            qmstr = ["".join(item) for item in qmst.astype(str)]
            amp_cv.append((qmst*(np.abs(res)**2)).tolist())

            if verbose:
                print("qumodes: ", ''.join(qmstr), " qubits: ", ''.join(qbstr), "    with amplitude: {0:.3f} {1} i{2:.3f}".format(res.real, '+-'[res.imag < 0], abs(res.imag)))


    occupation_cv = [sum(i) for i in zip(*amp_cv)]
    if verbose:
        print("occupation modes ", list(occupation_cv))

    occupation_qb = [sum(i) for i in zip(*amp_qb)]
    if verbose:
        print("occupation qubits ", list(occupation_qb))

    # if (np.abs(np.imag(res)) > 1e-10):
    #     print("\n imaginary amplitude: ", 1j * np.imag(res))

    return [occupation_cv,occupation_qb]

def cv_fockcounts(counts, qubit_qumode_list):
    """Convert counts dictionary from Fock-basis binary representation into base-10 Fock basis (qubit measurements are left unchanged). Accepts a counts dict() as returned by job.result().get_counts()
       along with qubit_qumode_list, a list of Qubits and Qumodes passed into cv_measure(...).

        Returns counts dict()

        Args:
            counts: dict() of counts, as returned by job.result().get_counts() for a circuit which used cv_measure()
            qubit_qumode_list: List of qubits and qumodes measured. This list should be identical to that passed into cv_measure()

        Returns:
            A new counts dict() which lists measurement results for the qubits and qumodes in qubit_qumode_list in little endian order, 
            with Fock-basis qumode measurements reported as a base-10 integer.
        """

    flat_list = []
    for el in qubit_qumode_list:
        if isinstance(el, list):
            flat_list += el
        else:
            flat_list += [el]

    newcounts = {}
    for key in counts:
        counter = len(key) - len(flat_list)
        if counter > 0:
            newkey = ('{0:0' + str(counter) + '}').format(0)
        else:
            newkey = ''
        for registerType in qubit_qumode_list[::-1]:
            if isinstance(registerType, list):
                newkey += str(int(key[counter:counter + len(registerType)], base=2))
                # newkey += str(key[counter:counter+len(registerType)])
                counter += len(registerType)
            else:
                newkey += key[counter]
                counter += 1
        newcounts[newkey] = counts[key]
    return newcounts


def measure_all_xyz(circuit: qiskit.QuantumCircuit):
    """Use QuantumCircuit.measure_all() to measure all qubits in the X, Y, and Z basis.

    Returns state, result tuples each for the X, Y, and Z basis.

    Args:
        circuit (qiskit.QuantumCircuit): circuit to measure qubits one

    Returns:
        x,y,z state & result tuples: (state, result) tuples for each x,y,z measurements
    """

    # QuantumCircuit.measure_all(False) returns a copy of the circuit with measurement gates.
    circuit_z = circuit.measure_all(False)
    state_z, result_z = simulate(circuit_z)

    circuit_x = circuit.copy()
    for qubit in circuit_x.qubits:
        circuit_x.h(qubit)
    circuit_x.measure_all()  # Add measure gates in-place
    state_x, result_x = simulate(circuit_x)

    circuit_y = circuit.copy()
    for qubit in circuit_y.qubits:
        circuit_y.sdg(qubit)
        circuit_y.h(qubit)
    circuit_y.measure_all()  # Add measure gates in-place
    state_y, result_y = simulate(circuit_y)

    return (state_x, result_x), (state_y, result_y), (state_z, result_z)


def get_probabilities(result: qiskit.result.Result):
    """Calculate the probabilities for each of the result's counts.

    Args:
        result (qiskit.result.Result): QisKit result to calculate probabilities from

    Returns:
        dict: probablity dictionary of each state
    """
    shots = 0
    counts = result.get_counts()
    for count in counts:
        shots += counts[count]
    probs = {}
    for count in counts:
        probs[count] = counts[count] / shots

    return probs


def simulate(
    circuit: CVCircuit,
    shots: int = 1024,
    add_save_statevector: bool = True,
    conditional_state_vector: bool = False,
    per_shot_state_vector: bool = False,
    kraus_operators = None,
    error_gates: List[str] = None
):
    """Convenience function to simulate using the given backend.

    Handles calling into QisKit to simulate circuit using defined simulator.

    Args:
        circuit (CVCircuit): circuit to simulate
        shots (int, optional): Number of simulation shots. Defaults to 1024.
        add_save_statevector (bool, optional): Set to True if a state_vector instruction
                                               should be added to the end of the circuit. Defaults to True.
        conditional_state_vector (bool, optional): Set to True if the saved state vector should be contional
                                                   (each state value gets its own state vector). Defaults to False.
        kraus_operator (list)

    Returns:
        tuple: (state, result) tuple from simulation
    """

    # If this is false, the user must have already called save_statevector!
    if add_save_statevector:
        circuit.save_statevector(
            conditional=conditional_state_vector, pershot=per_shot_state_vector
        )

    # Transpile for simulator, with noise error if provided
    if kraus_operators is not None:
        error = qiskit.providers.aer.noise.kraus_error(kraus_operators)
        if not error_gates:
            error_gates = circuit.cv_gate_labels
        noise_model = qiskit.providers.aer.noise.NoiseModel()
        noise_model.add_quantum_error(error, error_gates, circuit.qumode_qubits)
        noise_model.add_basis_gates("unitary")
        # print(noise_model.basis_gates)
        simulator = qiskit.providers.aer.AerSimulator(noise_model=noise_model)
    else:
        simulator = qiskit.providers.aer.AerSimulator()
    circuit_compiled = qiskit.transpile(circuit, simulator)

    # Run and get statevector
    result = simulator.run(circuit_compiled, shots=shots).result()

    # The user may have added their own circuit.save_statevector
    state = None
    if len(result.results):
        try:
            if conditional_state_vector or per_shot_state_vector:
                # Will get a dictionary of state vectors, one for each classical register value
                state = result.data()["statevector"]
            else:
                state = Statevector(result.get_statevector(circuit_compiled))
        except Exception:
            state = None  # result.get_statevector() will fail if add_save_statevector is false

    if add_save_statevector:
        circuit.data.pop()  # Clean up by popping off the SaveStatevector instruction

    return state, result


def plot_wigner_projection(circuit: CVCircuit, qubit, file: str = None):
    """Plot the projection onto 0, 1, +, - for the given circuit.

    This is limited to CVCircuit with only one qubit, also provided as a parameter.

    Args:
        circuit (CVCircuit): circuit to simulate and plot
        qubit (Qubit): qubit to measure
        file (str, optional): File path to save file, if None return plot. Defaults to None.
    """
    # Get unaltered state vector and partial trace
    x, _ = simulate(circuit)
    xT = x.data.conjugate().transpose()

    # Project onto 0 and 1 using Pauli Z
    circuit.z(qubit)
    y, _ = simulate(circuit)
    yT = y.data.conjugate().transpose()

    x_xT = x.data * xT
    x_yT = x.data * yT
    y_xT = y.data * xT
    y_yT = y.data * yT

    trace_x_xT = cv_partial_trace(circuit, x_xT)
    trace_x_yT = cv_partial_trace(circuit, x_yT)
    trace_y_xT = cv_partial_trace(circuit, y_xT)
    trace_y_yT = cv_partial_trace(circuit, y_yT)

    projection_zero = (trace_x_xT + trace_x_yT + trace_y_xT + trace_y_yT) / 4
    projection_one = (trace_x_xT - trace_x_yT - trace_y_xT + trace_y_yT) / 4

    # Clean up by popping off the Pauli Z
    circuit.data.pop()

    # Project onto + and - using Pauli X
    circuit.x(qubit)
    y, _ = simulate(circuit)
    yT = y.data.conjugate().transpose()

    x_xT = x.data * xT
    x_yT = x.data * yT
    y_xT = y.data * xT
    y_yT = y.data * yT

    trace_x_xT = cv_partial_trace(circuit, x_xT)
    trace_x_yT = cv_partial_trace(circuit, x_yT)
    trace_y_xT = cv_partial_trace(circuit, y_xT)
    trace_y_yT = cv_partial_trace(circuit, y_yT)

    projection_plus = (trace_x_xT + trace_x_yT + trace_y_xT + trace_y_yT) / 4
    projection_minus = (trace_x_xT - trace_x_yT - trace_y_xT + trace_y_yT) / 4

    # Clean up by popping of the Pauli X
    circuit.data.pop()

    # Calculate Wigner functions
    xvec = np.linspace(-6, 6, 200)
    wigner_zero = _wigner(projection_zero, xvec, xvec, circuit.cutoff)
    wigner_one = _wigner(projection_one, xvec, xvec, circuit.cutoff)
    wigner_plus = _wigner(projection_plus, xvec, xvec, circuit.cutoff)
    wigner_minus = _wigner(projection_minus, xvec, xvec, circuit.cutoff)

    # Plot using matplotlib on four subplots, at double the default width & height
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12.8, 12.8))

    _add_contourf(ax0, fig, "Projection onto zero", xvec, xvec, wigner_zero)
    _add_contourf(ax1, fig, "Projection onto one", xvec, xvec, wigner_one)
    _add_contourf(ax2, fig, "Projection onto plus", xvec, xvec, wigner_plus)
    _add_contourf(ax3, fig, "Projection onto minus", xvec, xvec, wigner_minus)

    # Save to file or display
    if file:
        plt.savefig(file)
    else:
        plt.show()


def _add_contourf(ax, fig, title, x, y, z):
    """Add a matplotlib contourf plot with color levels based on min/max values in z."""
    amax = np.amax(z)
    amin = abs(np.amin(z))
    max_value = max(amax, amin, 0.0001)  # Force a range if amin/amax are equal
    color_levels = np.linspace(-max_value, max_value, 100)

    cont = ax.contourf(x, y, z, color_levels, cmap="RdBu_r")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(title)
    fig.colorbar(cont, ax=ax)


def _find_cavity_indices(circuit: CVCircuit):
    """
    Return the indices of the cavities from the circuit

    I.e., the indices to the qubits representing the bosonic modes.
    """

    # Find indices of qubits representing qumodes
    qmargs = []
    for reg in circuit.qmregs:
        qmargs.extend(reg.qreg)

    # Trace over the qubits representing qumodes
    index = 0
    indices = []
    for qubit in circuit.qubits:
        if qubit in qmargs:
            indices.append(index)
        index += 1

    return indices


def _find_qubit_indices(circuit: CVCircuit):
    """
    Return the indices of the qubits from the circuit that are not in a QumodeRegister

    I.e., the indices to the qubits themselves, not the qubits representing the bosonic modes.
    """

    # Find indices of qubits representing qumodes
    qmargs = []
    for reg in circuit.qmregs:
        qmargs.extend(reg.qreg)

    # Trace over the qubits not representing qumodes
    index = 0
    indices = []
    for qubit in circuit.qubits:
        if qubit not in qmargs:
            indices.append(index)
        index += 1

    return indices


def cv_qubits_reduced_density_matrix(circuit: CVCircuit, state_vector):
    """Return reduced density matrix of the qubits by tracing out the cavities from the given Fock state vector.

    Args:
        circuit (CVCircuit): circuit yielding the results to trace over
        state_vector (Statevector): simulation results to trace over

    Returns:
        DensityMatrix: density matrix of the qubits from a partial trace over the cavities
    """

    indices = _find_cavity_indices(circuit)

    return partial_trace(state_vector, indices)


def cv_partial_trace(circuit: CVCircuit, state_vector):
    """Return reduced density matrix of the cavities by tracing out the qubits from the given Fock state vector.

    Args:
        circuit (CVCircuit): circuit with results to trace (to find Qubit index)
        state_vector (Statevector): simulation results to trace over

    Returns:
        DensityMatrix: partial trace
    """

    indices = _find_qubit_indices(circuit)

    return partial_trace(state_vector, indices)


def plot_wigner(
        circuit: CVCircuit,
        state_vector: Statevector,
        trace: bool = True,
        file: str = None,
        axes_min: int = -6,
        axes_max: int = 6,
        axes_steps: int = 200,
        num_colors: int = 100,
):
    """Produce a Matplotlib figure for the Wigner function on the given state vector.

    Optionally perform partial trace.

    Args:
        circuit (CVCircuit): circuit with results to trace (to find Qubit index)
        state_vector (Statevector): simulation results to trace over and plot
        trace (bool, optional): True if qubits should be traced. Defaults to True.
        file (str, optional): File path to save plot. If none, return plot. Defaults to None.
        axes_min (int, optional): Minimum axes plot value. Defaults to -6.
        axes_max (int, optional): Maximum axes plot value. Defaults to 6.
        axes_steps (int, optional): Steps between axes ticks. Defaults to 200.
        num_colors (int, optional): Number of color gradients in legend. Defaults to 100.
    """
    if trace:
        state = cv_partial_trace(circuit, state_vector)
    else:
        state = state_vector

    w_fock = wigner(state, circuit.cutoff, axes_min, axes_max, axes_steps)

    plot(
        data=w_fock,
        axes_min=axes_min,
        axes_max=axes_max,
        axes_steps=axes_steps,
        file=file,
        num_colors=num_colors,
    )


def plot(
        data,
        axes_min: int = -6,
        axes_max: int = 6,
        axes_steps: int = 200,
        file: str = None,
        num_colors: int = 100,
):
    """Contour plot the given data array"""
    xvec = np.linspace(axes_min, axes_max, axes_steps)

    amax = np.amax(data)
    amin = np.amin(data)
    abs_max = max(amax, abs(amin))
    color_levels = np.linspace(-abs_max, abs_max, num_colors)

    fig, ax = plt.subplots(constrained_layout=True)
    cont = ax.contourf(xvec, xvec, data, color_levels, cmap="RdBu_r")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    cb=fig.colorbar(cont, ax=ax)
    cb.set_label(r"$W(x,p)$")

    if file:
        plt.savefig(file)
    else:
        plt.show()


def animate_wigner(
    circuit: CVCircuit,
    qubit,
    cbit,
    animation_segments: int = 10,
    shots: int = 1024,
    file: str = None,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    processes: int = None,
    kraus_operators = None,
    error_gates: List[str] = None,
    keep_state: bool = False
):
    """Animate the Wigner function at each step defined in the given CVCirctuit.

    This assumes the CVCircuit was simulated with an animation_segments > 0 to
    act as the frames of the generated movie.

    The ffmpeg binary must be on your system PATH in order to execute this
    function.

    Args:
        circuit (CVCircuit): circuit to simulate and plot
        qubit ([type]): qubit to measure
        cbit ([type]): classical bit to measure into
        animation_segments (int, optional): Number of segments to split each gate into for animation. Defaults to 10.
        shots (int, optional): Number of simulation shots per frame. Defaults to 1024.
        file (str, optional): File path to save. If None, return plot. Defaults to None.
        axes_min (int, optional): Minimum axes plot value. Defaults to -6.
        axes_max (int, optional): Maximum axes plot value. Defaults to 6.
        axes_steps (int, optional): Steps between axes ticks. Defaults to 200.
        processes (int, optional): Number of parallel Python processes to start.
                                   If None, perform serially in main process. Defaults to None.
        keep_state (bool, optional): True if each frame builds on the previous frame's state vector. 
                                     False if each frame starts over from the beginning of the circuit.
                                     If True, it requires sequential simulation of each frame.

    Returns:
        [type]: [description]
    """

    # Simulate each frame, storing Wigner function data in w_fock
    xvec = np.linspace(axes_min, axes_max, axes_steps)

    circuits = []  # Each frame will have its own circuit to simulate

    # base_circuit is copied each gate iteration to build circuit frames to simulate
    base_circuit = circuit.copy()
    base_circuit.data.clear()  # Is this safe -- could we copy without data?
    for inst, qargs, cargs in circuit.data:
        # TODO - get qubit & cbit for measure instead of using parameters
        # qubit = xxx
        # cbit = yyy

        if isinstance(inst, ParameterizedUnitaryGate):
            for index in range(1, animation_segments + 1):
                sim_circuit = base_circuit.copy()

                sim_circuit.unitary(
                    inst.calculate_matrix(current_step=index, total_steps=animation_segments, keep_state=keep_state),
                    qargs,
                    label=inst.name,
                )

                # sim_circuit.barrier()
                sim_circuit.h(qubit)
                sim_circuit.measure(qubit, cbit)

                circuits.append(sim_circuit)
        elif hasattr(inst, "cv_conditional") and inst.cv_conditional:
            inst_0, qargs_0, cargs_0 = inst.definition.data[0]
            inst_1, qargs_1, cargs_1 = inst.definition.data[1]

            for index in range(1, animation_segments + 1):
                sim_circuit = base_circuit.copy()

                params_0 = inst_0.base_gate.calculate_params(current_step=index, total_steps=animation_segments, keep_state=keep_state)
                params_1 = inst_1.base_gate.calculate_params(current_step=index, total_steps=animation_segments, keep_state=keep_state)

                sim_circuit.append(
                    CVCircuit.cv_conditional(
                        inst.name,
                        inst_0.base_gate.op_func,
                        params_0,
                        params_1,
                        inst.num_qubits_per_qumode,
                        inst.num_qumodes,
                    ),
                    qargs,
                    cargs,
                )

                # sim_circuit.barrier()
                sim_circuit.h(qubit)
                sim_circuit.measure(qubit, cbit)

                circuits.append(sim_circuit)
        else:
            sim_circuit = base_circuit.copy()
            sim_circuit.append(inst, qargs, cargs)

            # sim_circuit.barrier()
            sim_circuit.h(qubit)
            sim_circuit.measure(qubit, cbit)

            circuits.append(sim_circuit)

        # Append the full instruction for the next frame
        base_circuit.append(inst, qargs, cargs)

    # Calculate the Wigner functions for each frame
    if not processes or processes < 1:
        processes = math.floor(multiprocessing.cpu_count() / 2)
        processes = max(processes, 1)  # prevent zero processes with 1 CPU

    if keep_state:
        w_fock = []
        previous_state = None
        for circuit in circuits:
            if previous_state:
                # Initialize circuit to simulate with the previous frame's state, then append the last instruction
                sim_circuit = circuit.copy()
                sim_circuit.data.clear()  # Is this safe -- could we copy without data?
                sim_circuit.initialize(previous_state)
                last_instructions = circuit.data[-3:]  # Get the last instruction, plus the Hadamard/measure
                for inst in last_instructions:
                    sim_circuit.append(*inst)
            else:
                # No previous simulation state, just run the current circuit
                sim_circuit = circuit
            fock, previous_state = simulate_wigner(sim_circuit, xvec, shots, kraus_operators=kraus_operators, error_gates=error_gates)
            w_fock.append(fock)
    elif processes == 1:
        w_fock = []
        for circuit in circuits:
            fock, _ = simulate_wigner(circuit, xvec, shots, kraus_operators=kraus_operators, error_gates=error_gates)
            w_fock.append(fock)
    else:
        pool = multiprocessing.Pool(processes)
        results = pool.starmap(
            simulate_wigner, ((circuit, xvec, shots, kraus_operators, error_gates) for circuit in circuits)
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
        func=_animate,
        frames=len(w_fock),
        fargs=(fig, ax, xvec, w_fock, file),
        interval=200,
        repeat=True,
    )

    # Save to file using ffmpeg, Pillow (GIF), or display
    if file:
        save_animation(anim, file)

    return anim


def save_animation(anim: matplotlib.animation.FuncAnimation, file: str):
    file_path = pathlib.Path(file)

    if file_path.suffix == ".mp4":
        writer = matplotlib.animation.FFMpegWriter(fps=24)
    elif file_path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=24)
    else:
        print(
            f"Unknown animation file type {file_path.suffix}, defaulting to animated GIF"
        )
        writer = matplotlib.animation.PillowWriter(fps=24)

    anim.save(file, writer=writer)


def _animate(frame, *fargs):
    """Generate individual matplotlib frame in animation."""
    fig = fargs[0]
    ax = fargs[1]
    xvec = fargs[2]
    w_fock = fargs[3][frame]
    file = fargs[4]

    amax = np.amax(w_fock)
    amin = np.amin(w_fock)
    abs_max = max(amax, abs(amin))
    color_levels = np.linspace(-abs_max, abs_max, 100)

    ax.clear()
    cont = ax.contourf(xvec, xvec, w_fock, color_levels, cmap="RdBu_r")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    if frame == 0:
        fig.colorbar(cont, ax=ax)

    if file:
        os.makedirs(f"{file}_frames", exist_ok=True)
        plt.savefig(f"{file}_frames/frame_{frame}.png")


def simulate_wigner(
    circuit: CVCircuit, 
    xvec: np.ndarray, 
    shots: int,
    kraus_operators = None,
    error_gates: List[str] = None
):
    """Simulate the circuit, partial trace the results, and calculate the Wigner function."""
    state, _ = simulate(circuit, shots=shots, conditional_state_vector=True, kraus_operators=kraus_operators, error_gates=error_gates)
    
    if state:
        even_state = state["0x0"]
        # odd_state = state["0x1"]

        density_matrix = cv_partial_trace(circuit, even_state)
        wigner_result = _wigner(density_matrix, xvec, xvec, circuit.cutoff)
    else:
        print("WARN: No state vector returned by simulation -- unable to calculate Wigner function!")
        wigner_result = None
        even_state = None
    
    return wigner_result, even_state


def wigner(
        state,
        cutoff: int,
        axes_min: int = -6,
        axes_max: int = 6,
        axes_steps: int = 200,
        hbar: int = 2,
):
    """
    Calculate the Wigner function on the given state vector.

    Args:
        state (array-like): state vector to calculate Wigner function
        cutoff (int): cutoff used during simulation
        axes_min (int, optional): Minimum axes plot value. Defaults to -6.
        axes_max (int, optional): Maximum axes plot value. Defaults to 6.
        axes_steps (int, optional): Steps between axes ticks. Defaults to 200.
        hbar (int, optional): hbar value to use in Wigner function calculation. Defaults to 2.

    Returns:
        array-like: Results of Wigner function calculation
    """
    xvec = np.linspace(axes_min, axes_max, axes_steps)
    return _wigner(state, xvec, xvec, cutoff, hbar)


def wigner_mle(
        states,
        cutoff: int,
        axes_min: int = -6,
        axes_max: int = 6,
        axes_steps: int = 200,
        hbar: int = 2,
):
    """
    Find the maximum likelihood estimation for the given state vectors and calculate the Wigner function on the result.

    Args:
        states (array-like of array-like): state vectors to calculate MLE and Wigner function
        cutoff (int): cutoff used during simulation
        axes_min (int, optional): Minimum axes plot value. Defaults to -6.
        axes_max (int, optional): Maximum axes plot value. Defaults to 6.
        axes_steps (int, optional): Steps between axes ticks. Defaults to 200.
        hbar (int, optional): hbar value to use in Wigner function calculation. Defaults to 2.

    Returns:
        array-like: Results of Wigner function calculation
    """
    mle_state = []
    for qubit_states in zip(*states):
        # TODO what distribution are the qubit states? (using normal)
        # scipy.stats normal distribution defaults to MLE fit, returns tuple[0] mean, tuple[1] std dev
        mle = scipy.stats.norm.fit(qubit_states)
        mle_state.append(mle[0])

    mle_normalized = mle_state / np.linalg.norm(mle_state)

    return wigner(mle_normalized, cutoff, axes_min, axes_max, axes_steps, hbar)


def _wigner(state, xvec, pvec, cutoff: int, hbar: int = 2):
    r"""
    Copy of Xanadu Strawberry Fields Wigner function, placed here to reduce dependencies.
    Starwberry Fields used the QuTiP "iterative" implementation.

    Strawberry Fields is released under the Apache License: https://github.com/XanaduAI/strawberryfields/blob/master/LICENSE

    QuTiP is released under the BSD 3-clause license: https://github.com/qutip/qutip/blob/master/LICENSE.txt

    See:
        <https://github.com/XanaduAI/strawberryfields/blob/e46bd122faff39976cc9052cc1a6472762c415b4/strawberryfields/backends/states.py#L725-L780>


    Calculates the discretized Wigner function of the specified mode.
    .. note::
        This code is a modified version of the "iterative" method of the
        `wigner function provided in QuTiP
        <http://qutip.org/docs/4.0.2/apidoc/functions.html?highlight=wigner#qutip.wigner.wigner>`_,
        which is released under the BSD license, with the following
        copyright notice:
        Copyright (C) 2011 and later, P.D. Nation, J.R. Johansson,
        A.J.G. Pitchford, C. Granade, and A.L. Grimsmo. All rights reserved.
    Args:
        mode (int): the mode to calculate the Wigner function for
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
    Returns:
        array: 2D array of size [len(xvec), len(pvec)], containing reduced Wigner function
        values for specified x and p values.
    """
    if isinstance(state, DensityMatrix):
        rho = state.data
    else:
        rho = DensityMatrix(state).data
    Q, P = np.meshgrid(xvec, pvec)
    A = (Q + P * 1.0j) / (2 * np.sqrt(hbar / 2))

    Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(cutoff)])

    # Wigner function for |0><0|
    Wlist[0] = np.exp(-2.0 * np.abs(A) ** 2) / np.pi

    # W = rho(0,0)W(|0><0|)
    W = np.real(rho[0, 0]) * np.real(Wlist[0])

    for n in range(1, cutoff):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
        W += 2 * np.real(rho[0, n] * Wlist[n])

    for m in range(1, cutoff):
        temp = copy(Wlist[m])
        # Wlist[m] = Wigner function for |m><m|
        Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wlist[m - 1]) / np.sqrt(m)

        # W += rho(m,m)W(|m><m|)
        W += np.real(rho[m, m] * Wlist[m])

        for n in range(m + 1, cutoff):
            temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = copy(Wlist[n])
            # Wlist[n] = Wigner function for |m><n|
            Wlist[n] = temp2

            # W += rho(m,n)W(|m><n|) + rho(n,m)W(|n><m|)
            W += 2 * np.real(rho[m, n] * Wlist[n])

    return W / (hbar)
