from copy import copy
import math

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit.providers.aer.library.save_instructions.save_statevector import save_statevector
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit.result import Result

from c2qa import CVCircuit


def simulate(circuit: CVCircuit, backend_name: str = "aer_simulator"):
    """
    Convenience function to simulate using the given backend.

    Returns Statevector
    """
    save_statevector(circuit)

    # Transpile for simulator
    simulator = qiskit.Aer.get_backend(backend_name)
    circuit_compiled = qiskit.transpile(circuit, simulator)

    # Run and get statevector
    result = simulator.run(circuit_compiled).result()
    state = Statevector(result.get_statevector(circuit_compiled))

    # Clean up by popping off the SaveStatevector instruction
    circuit.data.pop()

    return state, result


def plot_wigner_interference(circuit: CVCircuit, qubit, file: str = None):
    """
    Plot the projection onto 0, 1, +, - for the given circuit.

    This is limited to CVCircuit with only one qubit, also provided as a parameter.
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
    xvec = np.linspace(-5, 5, 200)
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
    max_value = max(amax, amin)
    color_levels = np.linspace(-max_value, max_value, 100)

    cont = ax.contourf(x, y, z, color_levels, cmap="RdBu_r")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(title)
    fig.colorbar(cont, ax=ax)


def plot_wigner_interference_old(circuit: CVCircuit, state_vector: Statevector, file: str = None):
    """Produce a Matplotlib figure for the Wigner function on the given state vector."""

    # Create identity
    #   TODO What size should it be?
    state_len = len(state_vector.data)
    # TODO shouldn"t have to math.floorm should check that length is even
    eye = np.identity(math.floor(state_len / 2), dtype=int)
    # eye = np.identity(2, dtype=int)

    # Calculate projectors for zero and one
    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    zero_projector = np.kron(zero, eye)
    one_projector = np.kron(one, eye)

    # TODO Should we tensor the state vector or the density matrix array?
    #   QisKit partial_trace() fails with "Input not a quantum state" error when using state vector.
    # state = DensityMatrix(state_vector).data
    state = state_vector.data

    # Project state onto zero and one
    zero_projection = zero_projector * state
    one_projection = one_projector * state

    # Trace over qubit
    #   TODO Does QisKit partial_trace() work correctly after projection?
    #     The projection isn't the same size/shape as original state vector.
    #     The qubit indices from the circuit and original state vector won"t match the indices in the new matrices.
    zero_trace = cv_partial_trace(circuit, zero_projection)
    one_trace = cv_partial_trace(circuit, one_projection)

    state_trace = cv_partial_trace(circuit, state)

    projection_zero = (state_trace + zero_trace) / 2
    projection_one = (state_trace - one_trace) / 2

    # Calculate Wigner functions
    xvec = np.linspace(-5, 5, 200)
    zero_wigner = _wigner(projection_zero, xvec, xvec, circuit.cutoff)
    one_wigner = _wigner(projection_one, xvec, xvec, circuit.cutoff)

    # Plot using matplot lib on two horizontal subplots, at double the default width
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
    cont = axs[0].contourf(xvec, xvec, zero_wigner, 100)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("p")
    axs[0].set_title("Projection onto zero")
    fig.colorbar(cont, ax=axs[0])

    cont = axs[1].contourf(xvec, xvec, one_wigner, 100)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("p")
    axs[1].set_title("Projection onto one")
    fig.colorbar(cont, ax=axs[1])

    # Save to file or display
    if file:
        plt.savefig(file)
    else:
        plt.show()


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


def cv_partial_trace(circuit: CVCircuit, state_vector):
    """ Return reduced density matrix by tracing out the qubits from the given Fock state vector. """

    indices = _find_qubit_indices(circuit)

    return partial_trace(state_vector, indices)


def plot_wigner_fock_state(
    circuit: CVCircuit, state_vector: Statevector, trace: bool = True, file: str = None
):
    """
    Produce a Matplotlib figure for the Wigner function on the given state vector.

    Optionally perform partial trace.
    """
    if trace:
        density_matrix = cv_partial_trace(circuit, state_vector)
    else:
        density_matrix = state_vector

    plot_wigner(density_matrix, circuit.cutoff, file)

def plot_wigner(state, cutoff: int, file: str = None):
    """Produce a Matplotlib figure for the Wigner function on the given state vector."""
    xvec = np.linspace(-5, 5, 200)
    w_fock = _wigner(state, xvec, xvec, cutoff)

    amax = np.amax(w_fock)
    amin = abs(np.amin(w_fock))
    max_value = max(amax, amin)
    color_levels = np.linspace(-max_value, max_value, 100)

    fig, ax = plt.subplots(constrained_layout=True)
    cont = ax.contourf(xvec, xvec, w_fock, color_levels, cmap="RdBu_r")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    fig.colorbar(cont, ax=ax)

    if file:
        plt.savefig(file)
    else:
        plt.show()


def animate_wigner_fock_state(circuit: CVCircuit, result: Result, file: str = None):
    """
    Animate the Wigner function at each step defined in the given CVCirctuit.

    This assumes the CVCircuit was simulated with an animation_segments > 0 to
    act as the frames of the generated movie.

    The ffmpeg binary must be on your system PATH in order to execute this
    function.
    """
    # Calculate the Wigner functions for each frame
    xvec = np.linspace(-5, 5, 200)
    w_fock = []
    for frame in range(circuit.animation_steps):
        state_vector = result.data(circuit)["snapshots"]["statevector"][
            circuit.get_snapshot_name(frame)
        ][0]
        density_matrix = cv_partial_trace(circuit, state_vector)
        w_fock.append(_wigner(density_matrix, xvec, xvec, circuit.cutoff))

    # Create empty plot to animate
    fig, ax = plt.subplots(constrained_layout=True)

    # Animate
    anim = matplotlib.animation.FuncAnimation(
        fig=fig,
        func=_animate,
        frames=circuit.animation_steps,
        fargs=(fig, ax, xvec, w_fock),
        interval=200,
        repeat=True,
    )

    # Save to file using ffmpeg or display
    if file:
        writervideo = matplotlib.animation.FFMpegWriter(fps=60)
        anim.save(file, writer=writervideo)

    return anim


def _animate(frame, *fargs):
    ax = fargs[1]
    xvec = fargs[2]
    w_fock = fargs[3]

    ax.clear()
    ax.contourf(xvec, xvec, w_fock[frame], levels=100)
    ax.set_xlabel("x")
    ax.set_ylabel("p")


def _wigner(state, xvec, pvec, cutoff: int, hbar: int = 2):
    r"""
    Copy of Xanadu Strawberry Fields Wigner function, placed here to reduce dependencies.

    Strawberry Fields is released under the Apache License: https://github.com/XanaduAI/strawberryfields/blob/master/LICENSE

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
    if isinstance(state, Statevector):
        rho = DensityMatrix(state).data
    elif isinstance(state, DensityMatrix):
        rho = state.data
    else:
        rho = state
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
