from copy import copy
from pathlib import Path


from c2qa.circuit import CVCircuit
from c2qa.util import cv_partial_trace, simulate


import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.result import Result
import scipy.stats


def simulate_wigner(
    circuit: CVCircuit,
    xvec: np.ndarray,
    shots: int,
    noise_pass=None,
    conditional: bool = True,
):
    """Simulate the circuit, partial trace the results, and calculate the Wigner function."""
    states, _ = simulate(
        circuit,
        shots=shots,
        noise_pass=noise_pass,
        conditional_state_vector=conditional,
    )

    if states:
        if conditional:
            state = states["0x0"]  # even state
            # state = states["0x1"]  # odd state
            density_matrix = cv_partial_trace(circuit, state)
        else:
            state = states
            density_matrix = state

        wigner_result = _wigner(density_matrix, xvec, xvec, circuit.cutoff)
    else:
        print(
            "WARN: No state vector returned by simulation -- unable to calculate Wigner function!"
        )
        wigner_result = None
        state = None

    return wigner_result, state


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

    # Prevent DeprecationWarning from Qiskit returning Statevector instead of array
    states_data = [state.data for state in states]

    for qubit_states in zip(*states_data):
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
    cb = fig.colorbar(cont, ax=ax)
    cb.set_label(r"$W(x,p)$")

    if file:
        plt.savefig(file)
    else:
        plt.show()


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


def plot_wigner_snapshot(
    circuit: CVCircuit,
    result: Result, 
    folder: Path = None,
    trace: bool = True,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    num_colors: int = 100,
):
    snapshots = result.data()['snapshots']['statevector']

    for cv_snapshot_id in range(circuit.cv_snapshot_id):
        label = f"cv_snapshot_{cv_snapshot_id}"

        if folder:
            file = Path(folder, f"{label}.png")
        else:
            file = f"{label}.png"
        
        snapshot = snapshots[label]
        index = 0
        if len(snapshot) > 1:
            print(f"Simulation had {len(snapshot)} shots, plotting last one")
            index = len(snapshot) - 1

        plot_wigner(circuit, snapshot[index], trace, file, axes_min, axes_max, axes_steps, num_colors)


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
