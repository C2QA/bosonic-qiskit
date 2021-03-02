from copy import copy

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit.result import Result

from c2qa import CVCircuit


def _project(a: np.ndarray, b: np.ndarray):
    """ Project vector a on vector b """

    # Find norm of the vector v
    b_norm = np.sqrt(sum(b**2))

    # Project a onto b using np.dot() 
    return (np.dot(a, b) / b_norm**2) * b


def plot_wigner_interference(circuit: CVCircuit, state_vector: Statevector, file: str = None):
    """Produce a Matplotlib figure for the Wigner function on the given state vector."""


    # FIXME -- Build appropriately sized matrix as projection operator

    # zero = np.array([[1, 0], [0, 0]])
    # one = np.array([[0, 0], [0, 1]])

    zero = np.zeros(len(state_vector.data))
    zero[0] = 1

    one = np.zeros(len(state_vector.data))
    one[1] = 1


    xvec = np.linspace(-5, 5, 200)
    state = np.array(state_vector.data)

    # Two horizontal subplots, at double the default width
    fig, axs = plt.subplots(1, 2, figsize=(12.8,4.8))

    projection = _project(state, zero)
    # projection = zero.dot(state)
    w_fock = _wigner(projection, xvec, xvec, circuit.cutoff)
    cont = axs[0].contourf(xvec, xvec, w_fock, 100)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("p")
    fig.colorbar(cont, ax=axs[0])

    projection = _project(state, one)
    # projection = one.dot(state)
    w_fock = _wigner(projection, xvec, xvec, circuit.cutoff)
    cont = axs[1].contourf(xvec, xvec, w_fock, 100)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("p")
    fig.colorbar(cont, ax=axs[1])

    if file:
        plt.savefig(file)
    else:
        plt.show()


def cv_partial_trace(circuit: CVCircuit, state_vector: Statevector):
    """ Return reduced density matrix by tracing out the qubits from the given Fock state vector. """

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

    return partial_trace(state_vector, indices)


def plot_wigner_fock_state(
    circuit: CVCircuit, state_vector: Statevector, file: str = None
):
    """Produce a Matplotlib figure for the Wigner function on the given state vector."""
    xvec = np.linspace(-5, 5, 200)
    density_matrix = cv_partial_trace(circuit, state_vector)
    w_fock = _wigner(density_matrix, xvec, xvec, circuit.cutoff)

    fig, ax = plt.subplots(constrained_layout=True)
    cont = ax.contourf(xvec, xvec, w_fock, 100)
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
    fig = fargs[0]
    ax = fargs[1]
    xvec = fargs[2]
    w_fock = fargs[3]

    ax.clear()
    cont = ax.contourf(xvec, xvec, w_fock[frame], levels=100)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    # fig.colorbar(cont, ax=ax)  # FIXME Colorbar shifts position in animation?

def _wigner(state, xvec, pvec, cutoff: int, hbar: int = 2):
    r"""
    Copy of Xanadu Strawberry Fields Wigner function, placed here to reduce dependencies.

    Strawberry Fields is released under the Apache License: https://github.com/XanaduAI/strawberryfields/blob/master/LICENSE

    See:
        <https://github.com/XanaduAI/strawberryfields/blob/e46bd122faff39976cc9052cc1a6472762c415b4/strawberryfields/backends/states.py#L725-L780>


    Calculates the discretized Wigner function of the specified mode.
    .. note::
        This code is a modified version of the 'iterative' method of the
        `wigner function provided in QuTiP <http://qutip.org/docs/4.0.2/apidoc/functions.html?highlight=wigner#qutip.wigner.wigner>`_,
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
