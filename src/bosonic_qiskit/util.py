import math
import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import qiskit
import qiskit.quantum_info
import qiskit_aer
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.result import Result
from qiskit_aer.noise import LocalNoisePass, NoiseModel

from bosonic_qiskit import CVCircuit
from bosonic_qiskit.discretize import discretize_circuits

from .typing import Clbit, NoisePassLike, Qubit, Qumode


def flatten(l: Sequence[Sequence[Any]]) -> Sequence[Any]:
    return [item for sublist in l for item in sublist]


def cv_ancilla_fock_measure(
    circuit: CVCircuit, list_qumodes_to_sample: Sequence[Qumode], qmr_number: int = 0
) -> NDArray[np.integer]:
    """Simulate a circuit with an appended binary search for boson number, and determine the Fock state of a set of qumodes using
    phase kickback on the qubit. For more information, see Curtis et al., PRA (2021) and Wang et al., PRX (2020).

    Returns the Fock state of the qumodes in list_qumodes_to_sample, in qumode register qmr_number.
    """
    # Count number of qubits in circuit so far
    num_qubits = len(flatten(circuit._qubit_regs))
    # Collect qumode register from circuit
    qmr = circuit.qmregs[qmr_number]
    # Add one ancilla qubits to the circuit per qumode to measure
    qbr_extra = qiskit.QuantumRegister(
        size=len(list_qumodes_to_sample), name="qbr_sampling"
    )
    # Add classical bits to readout measurement results
    cbr_extra = qiskit.ClassicalRegister(
        len(list_qumodes_to_sample) * circuit.num_qubits_per_qumode, name="cbr_sampling"
    )
    circuit.add_register(qbr_extra, cbr_extra)
    # Iterate over the qumodes
    qumode_counter = 0
    for j in list_qumodes_to_sample:
        # Set the initial maximum Fock state
        max = circuit.cutoff
        # Iterate a number of time corresponding to the number of bits required to represent the maximum Fock state in binary (remove useless characters at the front)
        for iteration in range(len(bin(circuit.cutoff)) - 3):
            # Make sure the ancilla qubit is always reset to 0
            circuit.initialize("0", qbr_extra[qumode_counter])
            # Apply a circuit which flips the ancilla if the qumode occupation is odd etc. see (Curtis et al., PRA, 2021 and Wang et al.,  PRX, 2020)
            circuit.cv_c_pnr(
                max, qmr[list_qumodes_to_sample[j]], qbr_extra[qumode_counter]
            )
            # Measure the qubit onto the classical bits (from left to right)
            classical_bit = (
                circuit.num_qubits_per_qumode
                - 1
                - iteration
                + (qumode_counter * circuit.num_qubits_per_qumode)
            )
            circuit.measure(qbr_extra[qumode_counter], classical_bit)
            # Update the maximum value for the SNAP gate creation
            max = int(max / 2)
        circuit.barrier()
        qumode_counter += 1
    # Simulate circuit with a single shot
    _, result, _ = simulate(circuit, shots=1)
    # Return integer value of boson number occupation, converted from the bits which make up a binary number
    print(result.get_counts())
    full_set_of_binary = list(result.get_counts().keys())[0].encode("ascii")
    results_integers = np.zeros([len(list_qumodes_to_sample)], dtype=int)
    for j in range(len(list_qumodes_to_sample)):
        binary_number = full_set_of_binary[
            j * circuit.num_qubits_per_qumode : (
                (j + 1) * circuit.num_qubits_per_qumode
            )
        ]
        print(binary_number)
        results_integers[-(j + 1)] = int(binary_number, 2)
    return results_integers


def stateread(
    stateop: ArrayLike,
    numberofqubits: int,
    numberofmodes: int,
    cutoff: int,
    verbose: bool = True,
    little_endian: bool = False,
):
    """Print values for states of qubits and qumodes using the result of a
    simulation of the statevector, e.g. using stateop, _, _, _ = bosonic_qiskit.util.simulate(circuit).

    Returns the states of the qubits and the Fock states of the qumodes with respective amplitudes.
    """
    st = np.array(stateop)  # convert state to np.array
    amp_cv = []
    amp_qb = []
    state = []

    cutoff = 2 ** int(
        np.ceil(np.log2(cutoff))
    )  # The cutoff needs to be a power of 2 for this code to work

    for i in range(len(st)):
        res = st[
            i
        ]  # go through vector element by element and find the positions of the non-zero elements with next if clause
        if np.abs(res) > 1e-10:
            pos = i  # position of amplitude (non-zero real)
            # print("position of non-zero real amplitude: ", pos, " res = ", res)
            sln = len(st)  # length of the state vector

            # Find the qubit states
            qbst = np.empty(numberofqubits, dtype="int")  # stores the qubit state
            iqb = 0  # counts up until the total number of qubits is reached

            # which half of the vector the amplitude is in is the state of the
            # first qubit because of how the kronecker product is made
            while iqb < numberofqubits:
                # if the amplitude is in the first half of the state vector or remaining statevector
                if pos < sln / 2:
                    qbst[iqb] = int(0)  # then the qubit is in 0
                else:
                    # if the amplitude is in the second half then it is in 1
                    qbst[iqb] = int(1)

                    # if the amplitude is in the second half of the statevector,
                    # then to find out the state of the other qubits and cavities
                    # then we remove the first half of the statevector for simplicity
                    # because it corresponds to the qubit being in 0 which isn't the case.
                    pos = pos - (sln / 2)
                    # print("pos (sln/2)", pos, "sln ",sln)

                # only consider the part of the statevector corresponding to the qubit state which has just been discovered
                sln = sln / 2

                # up the qubit counter to start finding out the state of the next qubit
                iqb = iqb + 1
            amp_qb.append((qbst * (np.abs(res) ** 2)).tolist())

            # Find the qumode states
            qmst = np.empty(
                numberofmodes, dtype="int"
            )  # will contain the Fock state of each mode
            # print("qmst starting in ", qmst)
            iqm = 0  # counts up the number of modes
            # print("position is now: ",pos)
            while iqm < numberofmodes:
                # print("mode counter iqm ", iqm)
                # print("cutoff ", cutoff)
                # print("length of vector left to search: sln ", sln)

                # length of a division is the length of the statevector divided
                # by the cutoff of the hilbert space (which corresponds to the
                # number of fock states which a mode can have)
                lendiv = sln / cutoff
                # print("lendiv (sln/cutoff)", lendiv)
                val = pos / lendiv
                # print("rough estimate of the position of the non-zero element: val (pos/lendiv) ", val)
                fock = math.floor(val)
                # print("Fock st resulting position in Kronecker product (math.floor(val)) ", fock)
                qmst[iqm] = fock

                # remove a number of divisions to then search a subsection of the Kronecker product
                pos = pos - (fock * lendiv)
                # print("new position for next order of depth of Kronecker product/pos: (pos-(fock*lendiv)) ",pos)

                # New length of vector left to search
                sln = sln - ((cutoff - 1) * lendiv)
                # print("New length of vector left to search: sln (sln-((cutoff-1)*lendiv))", sln)
                iqm = iqm + 1
            amp_cv.append((qmst * (np.abs(res) ** 2)).tolist())

            state.append([qmst.tolist(), qbst.tolist(), res])

            if verbose:
                if little_endian:
                    qmstr = ["".join(str(item)) for item in qmst]
                    qbstr = ["".join(str(item)) for item in qbst]
                    print(
                        "qumodes: ",
                        "".join(qmstr),
                        " qubits: ",
                        "".join(qbstr),
                        "    with amplitude: {0:.3f} {1} i{2:.3f}".format(
                            res.real, "-" if res.imag < 0 else "+", abs(res.imag)
                        ),
                        "(little endian)",
                    )
                else:
                    qmstr = ["".join(str(item)) for item in qmst[::-1]]
                    qbstr = ["".join(str(item)) for item in qbst[::-1]]
                    print(
                        "qumodes: ",
                        "".join(qmstr),
                        " qubits: ",
                        "".join(qbstr),
                        "    with amplitude: {0:.3f} {1} i{2:.3f}".format(
                            res.real, "-" if res.imag < 0 else "+", abs(res.imag)
                        ),
                        "(big endian)",
                    )

    occupation_cv = [sum(i) for i in zip(*amp_cv)]
    occupation_qb = [sum(i) for i in zip(*amp_qb)]

    if not little_endian:
        for i in range(len(state)):
            state[i][0] = state[i][0][::-1]
            state[i][1] = state[i][1][::-1]
        occupation_cv = occupation_cv[::-1]
        occupation_qb = occupation_qb[::-1]

    return [occupation_cv, occupation_qb], state


def counts_to_fockcounts(circuit: CVCircuit, result: Result) -> dict[str, int]:
    """Convert Qiskit simulation counts dictionary it to a Fock-basis counts dictionary.

    The Qiskit counts dictionary key is a string representing the Little Endian ordering classical bit values
    for each qubit and the value is the total count of simulated shots (runs) that had those values.

    See https://docs.quantum.ibm.com/api/qiskit/qiskit.result.Result#get_counts for Qiskit documentation on its
    counts histogram data structure.

    The returned value is the Fock-basis state key to the total count of simulated shots (runs) that had that value.

    Args:
        circuit (CVCircuit): simulated circuit
        result (Result): Qiskit simulation results with simulation counts

    Returns:
        New dict with Fock-basis key and total simulation counts value
    """

    qubit_counts = result.get_counts()
    qumode_bit_mapping = _final_qumode_mapping(circuit)

    fock_counts = {}
    for qubit_key in qubit_counts:
        max_iter_index = len(qubit_key) - 1
        fock_basis_key = qubit_key

        # Using the nested list of qumode bit mappings, convert the relevant bits to base-10 integer and
        # form new key by concatenating the unchanged bits of key around the decimal value.
        for index in range(len(qubit_key)):
            for qumode in qumode_bit_mapping:
                if index == min(qumode):
                    fock_decimal = str(
                        int(
                            qubit_key[
                                max_iter_index - max(qumode) : max_iter_index
                                - min(qumode)
                                + 1
                            ],
                            base=2,
                        )
                    )
                    fock_basis_key = (
                        fock_basis_key[: max_iter_index - max(qumode)]
                        + fock_decimal
                        + fock_basis_key[max_iter_index - min(qumode) + 1 :]
                    )

        fock_counts[fock_basis_key] = qubit_counts[qubit_key]

    return fock_counts


def _final_qumode_mapping(circuit: CVCircuit) -> list[list[Clbit]]:
    """
    Return the classical bits that active qumode qubits are mapped onto. Bits corresponding to distinct qumodes are grouped together
    """
    active_qumode_bit_indices_grouped = []

    final_measurement_mapping = _final_measurement_mapping(circuit)

    # If no explicit measurements are in the circuit, just assume all qumode qbits were "measured"
    if len(final_measurement_mapping) == 0:
        for qubit_index in circuit.qumode_qubit_indices:
            final_measurement_mapping[qubit_index] = qubit_index

    # For each qumode qubit group, extract list of bits that map to qubits in group. Append list only if list is not empty
    for qumode_qubit_group in circuit.qumode_qubits_indices_grouped:
        qumode_bit_group = [
            key
            for key, val in final_measurement_mapping.items()
            for qubit in qumode_qubit_group
            if val == qubit
        ]

        if qumode_bit_group != []:
            active_qumode_bit_indices_grouped.append(qumode_bit_group)

    # Sort nested list by first item in each sublist
    active_qumode_bit_indices_grouped = sorted(
        active_qumode_bit_indices_grouped, key=lambda x: x[0]
    )

    return active_qumode_bit_indices_grouped


# This code is part of Mthree.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=no-name-in-module
def _final_measurement_mapping(circuit: CVCircuit) -> dict[Qubit, Clbit]:
    """Return the measurement mapping for the circuit.

    Dict keys label classical bits, whereas the values indicate the
    physical qubits that are measured to produce those bit values.

    Parameters:
        circuit (QuantumCircuit): Input Qiskit QuantumCircuit.

    Returns:
        dict: Mapping of classical bits to qubits for final measurements.
    """
    active_qubits = list(range(circuit.num_qubits))
    active_cbits = list(range(circuit.num_clbits))

    # Map registers to ints
    qint_map = {}
    for idx, qq in enumerate(circuit.qubits):
        qint_map[qq] = idx

    cint_map = {}
    for idx, qq in enumerate(circuit.clbits):
        cint_map[qq] = idx

    # Find final measurements starting in back
    qmap = []
    cmap = []
    for item in circuit._data[::-1]:
        if item.name == "measure":
            cbit = cint_map[item[2][0]]
            qbit = qint_map[item[1][0]]
            if cbit in active_cbits and qbit in active_qubits:
                qmap.append(qbit)
                cmap.append(cbit)
                active_cbits.remove(cbit)

        if not active_cbits or not active_qubits:
            break
    mapping = {}
    if cmap and qmap:
        for idx, qubit in enumerate(qmap):
            mapping[cmap[idx]] = qubit

    # Sort so that classical bits are in numeric order low->high.
    mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))
    return mapping


def measure_all_xyz(circuit: QuantumCircuit) -> tuple["SimulateResult", ...]:
    """Use QuantumCircuit.measure_all() to measure all qubits in the X, Y, and Z basis.

    Returns state, result, fockcounts tuples each for the X, Y, and Z basis.

    Args:
        circuit (QuantumCircuit): circuit to measure qubits one

    Returns:
        x,y,z state & result tuples: (state, result, fockcounts) tuples for each x,y,z measurements
    """

    # QuantumCircuit.measure_all(False) returns a copy of the circuit with measurement gates.
    circuit_z = cast(QuantumCircuit, circuit.measure_all(False))
    state_z, result_z, fockcounts_z = simulate(circuit_z)

    circuit_x = circuit.copy()
    for qubit in circuit_x.qubits:
        circuit_x.h(qubit)
    circuit_x.measure_all()  # Add measure gates in-place
    state_x, result_x, fockcounts_x = simulate(circuit_x)

    circuit_y = circuit.copy()
    for qubit in circuit_y.qubits:
        circuit_y.sdg(qubit)
        circuit_y.h(qubit)
    circuit_y.measure_all()  # Add measure gates in-place
    state_y, result_y, fockcounts_y = simulate(circuit_y)

    return (
        (state_x, result_x, fockcounts_x),
        (state_y, result_y, fockcounts_y),
        (state_z, result_z, fockcounts_z),
    )


def get_probabilities(result: Result) -> dict[str, float] | Sequence[dict[str, float]]:
    """Calculate the probabilities for each of the result's counts.

    Args:
        result (qiskit.result.Result): QisKit result to calculate probabilities from

    Returns:
        dict[str, float] or list[dict[str, float]]: normalized histogram for each experiment
    """

    counts: dict[str, int] | list[dict[str, int]] = result.get_counts()

    single_experiment = False
    if isinstance(counts, dict):
        counts = [counts]
        single_experiment = True

    output = []
    for c in counts:
        shots = sum(c.values())
        probs = {k: n / shots for k, n in c.items()}
        output.append(probs)

    if single_experiment:
        return output[0]

    return output


SimulateResult = tuple[
    Statevector | dict[str, Statevector] | None, Result, dict[str, int] | None
]


def simulate(
    cvcircuit: QuantumCircuit | CVCircuit,
    shots: int = 1024,
    return_fockcounts: bool = True,
    add_save_statevector: bool = True,
    conditional_state_vector: bool = False,
    per_shot_state_vector: bool = False,
    noise_model: NoiseModel | None = None,
    noise_passes: NoisePassLike | None = None,
    max_parallel_threads: int = 0,
    discretize: bool = False,
) -> SimulateResult:
    """Convenience function to simulate using the given backend.

    Handles calling into QisKit to simulate circuit using defined simulator.

    Args:
        circuit (CVCircuit): circuit to simulate
        shots (int, optional): Number of simulation shots. Defaults to 1024.
        return_fockcounts (bool, optional): Set to True if measurement results should be returned. Defaults to False
        add_save_statevector (bool, optional): Set to True if a state_vector instruction
                                               should be added to the end of the circuit. Defaults to True.
        conditional_state_vector (bool, optional): Set to True if the saved state vector should be contional
                                                   (each state value gets its own state vector). Defaults to False.
        per_shot_state_vector (bool, optional): Set to Ture if the simulation should return a separate state vector for
                                                every simulation shot.
        noise_model (NoiseModel, optional): Custom noise model to pass into AerSimulator to use on each simulation shot
        noise_passes (list of LocalNoisePass, optional): Custom noise pass to apply on each gate.
        max_parallel_threads (int, opational): Sets the maximum number of CPU cores used by OpenMP for parallelization.
                                               If set to 0 the maximum will be set to the number of CPU cores (Default: 0).
        discretize (bool, optional): Set to True if circuit should be discretized to apply noise passes. Defaults to False.

    Returns:
        tuple: (state, result, fock_counts) tuple from [optionally discretized] simulations
    """

    if discretize and not noise_passes:
        warnings.warn(
            "Discretization of circuit intended for use with noise passes, but none provided"
        )

    sim_circuit = discretize_circuits(cvcircuit)[-1] if discretize else cvcircuit

    # If this is false, the user must have already called save_statevector!
    if add_save_statevector:
        sim_circuit.save_statevector(
            conditional=conditional_state_vector, pershot=per_shot_state_vector
        )

    circuit_compiled = sim_circuit

    # Run noise pass, if provided
    noise_pass_lst = None
    if noise_passes:
        noise_pass_lst = noise_passes
        if isinstance(noise_pass_lst, LocalNoisePass):
            noise_pass_lst = [noise_pass_lst]

        for noise_pass in noise_pass_lst:
            circuit_compiled = noise_pass(circuit_compiled)

    # Transpile for simulator
    simulator = qiskit_aer.AerSimulator()

    if circuit_compiled.requires_transpile() or noise_pass_lst:
        # TODO do we need more than the translation pass manager?
        # circuit_compiled = qiskit.transpile(circuit_compiled, simulator)

        pm = qiskit.transpiler.preset_passmanagers.common.generate_translation_passmanager(
            target=simulator.target
        )
        circuit_compiled = pm.run(circuit_compiled)

    # Run and get statevector
    result = simulator.run(
        circuit_compiled,
        shots=shots,
        max_parallel_threads=max_parallel_threads,
        noise_model=noise_model,
    ).result()

    # The user may have added their own circuit.save_statevector
    state = None
    if len(result.results):
        try:
            if conditional_state_vector or per_shot_state_vector:
                # Will get a dictionary of state vectors, one for each classical register value
                state = result.data()["statevector"]
                state = cast(dict[str, Statevector], state)
            else:
                state = Statevector(result.get_statevector(circuit_compiled))
        except Exception:
            state = None  # result.get_statevector() will fail if add_save_statevector is false

    if add_save_statevector:
        sim_circuit.data.pop()  # Clean up by popping off the SaveStatevector instruction

    if return_fockcounts and add_save_statevector:
        # Fixme: this error message is uninformative -- why did it fail?
        try:
            fockcounts = counts_to_fockcounts(sim_circuit, result)
            return (state, result, fockcounts)
        except Exception as e:
            raise Exception("counts_to_fockcounts() was not able to execute") from e
    else:
        return (state, result, None)


def _find_qubit_indices(circuit: CVCircuit) -> list[int]:
    """
    Return the indices of the qubits from the circuit that are not in a QumodeRegister

    I.e., the indices to the qubits themselves, not the qubits representing the bosonic modes.
    """

    # Find indices of qubits representing qumodes
    to_exclude = set(circuit.qumode_qubits)

    # Trace over the qubits not representing qumodes
    indices = []
    for index, qubit in enumerate(circuit.qubits):
        if qubit not in to_exclude:
            indices.append(index)

    return indices


def trace_out_qumodes(circuit: CVCircuit, state_vector: Statevector) -> DensityMatrix:
    """Return reduced density matrix of the qubits by tracing out the cavities of the CVCircuit from the given Fock state vector.

    Args:
        circuit (CVCircuit): circuit yielding the results to trace over
        state_vector (Statevector): simulation results to trace over

    Returns:
        DensityMatrix: density matrix of the qubits from a partial trace over the cavities
    """

    indices = circuit.qumode_qubit_indices

    return qiskit.quantum_info.partial_trace(state_vector, indices)


def trace_out_qubits(circuit: CVCircuit, state_vector: Statevector) -> DensityMatrix:
    """Return reduced density matrix of the cavities by tracing out the all qubits of the CVCircuit from the given Fock state vector.

    Args:
        circuit (CVCircuit): circuit with results to trace (to find Qubit index)
        state_vector (Statevector or DensityMatrix): simulation results to trace over

    Returns:
        DensityMatrix: partial trace
    """

    indices = _find_qubit_indices(circuit)

    return qiskit.quantum_info.partial_trace(state_vector, indices)


def cv_partial_trace(
    circuit: CVCircuit, state_vector: Statevector, qubits: Sequence[Qubit]
) -> DensityMatrix:
    """Return reduced density matrix over the given Qiskit Qubits.

    First find the indices of the given Qubits, then call qiskit.quantum_info.partial_trace

    Args:
        circuit (CVCircuit): circuit with results to trace (to find Qubit index)
        state_vector (Statevector or DensityMatrix): simulation results to trace over
        qubits (list): list of Qiskit Qubit to trace over

    Returns:
        DensityMatrix: partial trace
    """

    if isinstance(qubits, Qubit):
        qubits = [qubits]

    indices = circuit.get_qubit_indices(qubits)
    return qiskit.quantum_info.partial_trace(state_vector, indices)


def fockmap(
    matrix: ArrayLike,
    fock_input: ArrayLike,
    fock_output: ArrayLike,
    amplitude: ArrayLike = 1 + 0j,
) -> NDArray[np.complexfloating]:
    """Generates matrix corresponding to some specified mapping of Fock states for a single qumode.
    First feed function empty matrix, then call fmap_matrix however many times needed to fully define intended mapping.
    Maps ith element in fock_input to ith element in fock_output with amplitude specified by ith element in amplitude.
    If amplitude is left blank, function assumes amp = 1 for all mappings.

    Two use cases
    1) int + list datatype combination (length of amp list must match length of either fock_input or fock_output, whichever is longer):
    >fockmap(matrix, 1, [0, 1])
    ->> ``|0><1| + |1><1|``

    >fockmap(matrix, [3, 2], 0, [0.5j, 1])
    ->> ``0.5j|0><3| + |0><2|``

    2) list datatype
    >fockmap(matrix, [3, 2], [2, 1], [0.1j, 0.8])
    ->> ``0.1j|2><3| + 0.8|1><2|``

    >fockmap(matrix, [1, 1], [2, 4])
    ->> ``|2><1| + |4><1|``


    Args:
        matrix (ArrayLike): Matrix that you want to change
        fock_input (ArrayLike): Input state(s) for mapping, corresponds to bra
        fock_output (ArrayLike): Output states(s) for mapping, corresponds to ket
        amplitude (ArrayLike): Amplitudes corresponding to final mapped states. Defaults to 1

    Returns:
        np.array: Edited matrix"""

    fock_input = np.atleast_1d(fock_input)
    fock_output = np.atleast_1d(fock_output)
    amplitude = np.atleast_1d(amplitude)

    matrix = np.atleast_2d(matrix).astype(complex)
    n, m = matrix.shape
    if n != m:
        raise ValueError("Matrix given is not square")

    # Errors if these can't be broadcast together, or if any fock state is not an integer
    matrix[fock_output, fock_input] = amplitude
    return matrix


def avg_photon_num(
    circuit: CVCircuit, state: Statevector | DensityMatrix, decimals: int = 2
) -> list[float]:
    """Returns average photon number of state for each qumode within the circuit using the number operator.

    Args:
        circuit (CVCircuit): Circuit definine qumodes present in given state
        state (Statevector or DensityMatrix): full state to operate on
        decimals: Determines precision of calculation

    Returns:
        float: Average photon number to specified precision
    """

    qumode_qubits = circuit.qumode_qubits_indices_grouped
    averages = []
    for qumode in range(len(qumode_qubits)):
        traced_qubits = []
        for traced_qumode in range(len(qumode_qubits)):
            if traced_qumode != qumode:
                traced_qubits.extend(qumode_qubits[traced_qumode])
        traced_state = qiskit.quantum_info.partial_trace(state, traced_qubits)
        averages.append(qumode_avg_photon_num(traced_state, decimals))

    return averages


def qumode_avg_photon_num(
    state: Statevector | DensityMatrix, decimals: int = 2
) -> float:
    """Returns average photon number of an individual qumode's state using the number operator.

    Args:
        state (Statevector or DensityMatrix): State to operate on for an individual qumode
        decimals: Determines precision of calculation

    Returns:
        float: Average photon number to specified precision
    """

    # Generate number operator based on dimension of state
    dim = state.dim
    N = np.diag(range(dim))

    # Normalise state
    if isinstance(state, Statevector):
        for_norm = state.inner(state)
    elif isinstance(state, DensityMatrix):
        for_norm = state.trace()
    else:
        raise TypeError(
            "Only Statevector or DensityMatrix are accepted as valid types."
        )

    # Calculate average photon number
    avg_photon = state.expectation_value(N) / for_norm

    if round(avg_photon.imag, 6) != 0:
        raise Exception(
            "Magnitude of average photon is complex, check inputs. Imaginary portion = {}".format(
                avg_photon.imag
            )
        )

    return np.round(avg_photon.real, decimals)
