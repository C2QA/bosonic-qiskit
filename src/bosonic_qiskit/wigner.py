from pathlib import Path
from typing import Literal, Sequence, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import qutip
import scipy.stats
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.result import Result
from qiskit_aer.noise import LocalNoisePass

from bosonic_qiskit.circuit import CVCircuit
from bosonic_qiskit.util import simulate, trace_out_qubits

from .typing import NoisePassLike, Qubit

# Available method names in Qutip
WignerMethods = Literal["clenshaw", "iterative", "laguerre", "fft"]
WignerResult = tuple[np.ndarray, ...]


def simulate_wigner(
    circuit: CVCircuit,
    xvec: np.ndarray,
    shots: int,
    noise_passes: NoisePassLike | None = None,
    conditional_state: str | None = None,
    trace: bool = False,
    method: WignerMethods = "clenshaw",
    g: float = np.sqrt(2),
) -> tuple[WignerResult, Statevector]:
    """Simulate the circuit, optionally partial trace the results, and calculate the Wigner function

    Args:
        circuit: The circuit to simulate

        xvec: The grid points to evaluate on. This will also be used to define the grid in the y direction

        shots: The number of shots to execute

        noise_passes: Optional noise passes to apply to the simulation

        conditional_state: An optional state to select

        trace: Whether or not to trace out all qubits, leaving the qumodes

        method: The method used to calculate the wigner function.

    Returns:
        A tuple containing the wigner function and the statevector
    """
    states, _, _ = simulate(
        circuit,
        shots=shots,
        noise_passes=noise_passes,
        conditional_state_vector=True if conditional_state else False,
        return_fockcounts=False,
    )

    if not states:
        raise RuntimeError("No state vector returned from the simulation")

    state = states[conditional_state] if conditional_state else states
    state = cast(Statevector, state)
    dm = trace_out_qubits(circuit, state) if trace else DensityMatrix(state)
    wigner_result = _wigner(dm, xvec, g=g, method=method)
    return wigner_result, state


def simulate_wigner_multiple_statevectors(
    circuit: CVCircuit,
    xvec: ArrayLike,
    shots: int,
    statevector_label: str,
    num_statevectors: int,
    noise_passes: NoisePassLike | None = None,
    trace: bool = False,
    g: float = np.sqrt(2),
    method: WignerMethods = "clenshaw",
) -> list[WignerResult]:
    """Simulate the circuit, optionally partial trace the results, and calculate the Wigner function on each statevector starting with the given label."""
    state, result, _ = simulate(circuit, shots=shots, noise_passes=noise_passes)

    if not result.results:
        raise RuntimeError("No statevector was returned from simulation.")

    wigner_results = []
    for num in range(num_statevectors):
        state = result.data()[f"{statevector_label}{num}"]
        state = cast(Statevector, state)
        dm = trace_out_qubits(circuit, state) if trace else DensityMatrix(state)
        wigner_results.append(_wigner(dm, xvec, g=g, method=method))

    return wigner_results


def wigner(
    state: ArrayLike | Statevector | DensityMatrix,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    g: float = np.sqrt(2),
    method: WignerMethods = "clenshaw",
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
    if isinstance(state, Statevector):
        state = DensityMatrix(state)

    if not isinstance(state, DensityMatrix):
        state = np.atleast_1d(state)

        # If we have a statevector, turn it into density matrix
        if state.ndim == 1:
            state = np.outer(state, state)

        state = DensityMatrix(state)

    xvec = np.linspace(axes_min, axes_max, axes_steps)
    return _wigner(state, xvec, g=g, method=method)


def wigner_mle(
    states: Sequence[Statevector],
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    g: float = np.sqrt(2),
    method: WignerMethods = "clenshaw",
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

    mle_state = np.array(mle_state)
    mle_normalized = mle_state / np.linalg.norm(mle_state)

    return wigner(mle_normalized, axes_min, axes_max, axes_steps, g=g, method=method)


def _wigner(
    state: DensityMatrix,
    xvec: ArrayLike,
    yvec: ArrayLike | None = None,
    g: float = np.sqrt(2),
    method: WignerMethods = "clenshaw",
) -> WignerResult:
    rho = state.data

    if not yvec:
        yvec = xvec

    return qutip.wigner(psi=qutip.Qobj(rho), xvec=xvec, yvec=yvec, g=g, method=method)  # pyright: ignore[reportReturnType]


def plot_wigner(
    circuit: CVCircuit,
    state_vector: Statevector,
    trace: bool = True,
    file: str | Path | None = None,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    num_colors: int = 100,
    draw_grid: bool = False,
    dpi: int = 100,
    **wigner_kwargs,
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
        draw_grid (bool, optional): True if grid lines should be drawn on plot. Defaults to False.
    """
    state = (
        trace_out_qubits(circuit, state_vector)
        if trace
        else DensityMatrix(state_vector)
    )

    w_fock = wigner(
        state=state,
        axes_min=axes_min,
        axes_max=axes_max,
        axes_steps=axes_steps,
        **wigner_kwargs,
    )

    plot(
        data=w_fock,
        axes_min=axes_min,
        axes_max=axes_max,
        axes_steps=axes_steps,
        file=file,
        num_colors=num_colors,
        draw_grid=draw_grid,
        dpi=dpi,
    )


def plot(
    data,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    file: str | Path | None = None,
    num_colors: int = 100,
    draw_grid: bool = False,
    dpi: int = 100,
):
    """Contour plot the given data array"""
    xvec = np.linspace(axes_min, axes_max, axes_steps)

    amax = np.amax(data)
    amin = np.amin(data)
    if amax == 0 and amin == 0:
        amax = 1
        amin = -1
    abs_max = max(amax, abs(amin))
    color_levels = np.linspace(-abs_max, abs_max, num_colors)

    fig, ax = plt.subplots(constrained_layout=True)
    cont = ax.contourf(xvec, xvec, data, color_levels, cmap="RdBu")

    xvec_int = [int(x) for x in xvec]
    xvec_int = sorted(set(xvec_int))
    ax.set_xlabel(r"$x$")
    # ax.set_xticks(xvec_int)
    ax.set_ylabel(r"$p$")
    # ax.set_yticks(xvec_int)
    ax.set_aspect("equal", "box")
    if draw_grid:
        ax.grid()

    cb = fig.colorbar(cont, ax=ax, format=tick.FormatStrFormatter("%.2f"))
    cb.set_label(r"$W(x,p)$", rotation=270, labelpad=25)

    if file:
        plt.savefig(file, dpi=dpi)
    else:
        plt.show()


def plot_wigner_projection(
    circuit: CVCircuit,
    qubit: Qubit,
    file: str | Path | None = None,
    draw_grid: bool = False,
    **wigner_kwargs,
):
    """Plot the projection onto 0, 1, +, - for the given circuit.

    This is limited to CVCircuit with only one qubit, also provided as a parameter.

    Args:
        circuit (CVCircuit): circuit to simulate and plot
        qubit (Qubit): qubit to measure
        file (str, optional): File path to save file, if None return plot. Defaults to None.
        draw_grid (bool, optional): True if gridlines should be drawn on plots. Defaults to False.
    """
    # Get unaltered state vector and partial trace
    x, _, _ = simulate(circuit)
    x = cast(Statevector, x)
    xT = x.data.conjugate().transpose()

    # Project onto 0 and 1 using Pauli Z
    circuit.z(qubit)
    y, _, _ = simulate(circuit)
    yT = y.data.conjugate().transpose()

    x_xT = x.data * xT
    x_yT = x.data * yT
    y_xT = y.data * xT
    y_yT = y.data * yT

    trace_x_xT = trace_out_qubits(circuit, x_xT)
    trace_x_yT = trace_out_qubits(circuit, x_yT)
    trace_y_xT = trace_out_qubits(circuit, y_xT)
    trace_y_yT = trace_out_qubits(circuit, y_yT)

    projection_zero = (trace_x_xT + trace_x_yT + trace_y_xT + trace_y_yT) / 4
    projection_one = (trace_x_xT - trace_x_yT - trace_y_xT + trace_y_yT) / 4

    # Clean up by popping off the Pauli Z
    circuit.data.pop()

    # Project onto + and - using Pauli X
    circuit.x(qubit)
    y, _, _ = simulate(circuit)
    yT = y.data.conjugate().transpose()

    x_xT = x.data * xT
    x_yT = x.data * yT
    y_xT = y.data * xT
    y_yT = y.data * yT

    trace_x_xT = trace_out_qubits(circuit, x_xT)
    trace_x_yT = trace_out_qubits(circuit, x_yT)
    trace_y_xT = trace_out_qubits(circuit, y_xT)
    trace_y_yT = trace_out_qubits(circuit, y_yT)

    projection_plus = (trace_x_xT + trace_x_yT + trace_y_xT + trace_y_yT) / 4
    projection_minus = (trace_x_xT - trace_x_yT - trace_y_xT + trace_y_yT) / 4

    # Clean up by popping of the Pauli X
    circuit.data.pop()

    # Calculate Wigner functions
    xvec = np.linspace(-6, 6, 200)
    wigner_zero = _wigner(projection_zero, xvec, **wigner_kwargs)
    wigner_one = _wigner(projection_one, xvec, **wigner_kwargs)
    wigner_plus = _wigner(projection_plus, xvec, **wigner_kwargs)
    wigner_minus = _wigner(projection_minus, xvec, **wigner_kwargs)

    # Plot using matplotlib on four subplots, at double the default width & height
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12.8, 12.8))

    _add_contourf(ax0, fig, "Projection onto zero", xvec, xvec, wigner_zero, draw_grid)
    _add_contourf(ax1, fig, "Projection onto one", xvec, xvec, wigner_one, draw_grid)
    _add_contourf(ax2, fig, "Projection onto plus", xvec, xvec, wigner_plus, draw_grid)
    _add_contourf(
        ax3, fig, "Projection onto minus", xvec, xvec, wigner_minus, draw_grid
    )

    # Save to file or display
    if file:
        plt.savefig(file)
    else:
        plt.show()


def plot_wigner_snapshot(
    circuit: CVCircuit,
    result: Result,
    folder: str | Path | None = None,
    trace: bool = True,
    axes_min: int = -6,
    axes_max: int = 6,
    axes_steps: int = 200,
    num_colors: int = 100,
    **wigner_kwargs,
):
    for cv_snapshot_id in range(circuit.cv_snapshot_id):
        label = f"cv_snapshot_{cv_snapshot_id}"
        file = Path(folder) / f"{label}.png" if folder else f"{label}.png"

        snapshot = result.data()[label]
        # index = 0
        # if len(snapshot) > 1:
        #     print(f"Simulation had {len(snapshot)} shots, plotting last one")
        #     index = len(snapshot) - 1

        # plot_wigner(circuit, snapshot[index], trace, file, axes_min, axes_max, axes_steps, num_colors, g=g, method=method)
        plot_wigner(
            circuit,
            snapshot,
            trace,
            file,
            axes_min,
            axes_max,
            axes_steps,
            num_colors,
            **wigner_kwargs,
        )


def _add_contourf(
    ax: Axes,
    fig: Figure,
    title: str,
    x: Sequence[int | float],
    y: Sequence[int | float],
    z: Sequence[int | float],
    draw_grid: bool = False,
):
    """Add a matplotlib contourf plot with color levels based on min/max values in z."""
    amax = np.amax(z)
    amin = abs(np.amin(z))
    max_value = max(amax, amin, 0.0001)  # Force a range if amin/amax are equal
    color_levels = np.linspace(-max_value, max_value, 100)

    cont = ax.contourf(x, y, z, color_levels, cmap="RdBu")

    xvec_int = [int(value) for value in x]
    xvec_int = sorted(set(xvec_int))
    ax.set_xlabel("x")
    ax.set_xticks(xvec_int)

    yvec_int = [int(value) for value in y]
    yvec_int = sorted(set(yvec_int))
    ax.set_ylabel("p")
    ax.set_yticks(yvec_int)

    if draw_grid:
        ax.grid()

    ax.set_title(title)
    fig.colorbar(cont, ax=ax)
