import copy
import os
import sys
from typing import List, Tuple

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import qiskit
import scipy

import c2qa


def h1h2h3(
        circuit: c2qa.CVCircuit,
        qma: List[qiskit.circuit.Qubit],
        qmb: List[qiskit.circuit.Qubit],
        qb: qiskit.circuit.Qubit,
        theta_1: float,
        theta_2: float,
        theta_3: float
        ) -> c2qa.CVCircuit:
    circuit.cv_rh1(theta_1, qmb, qma, qb)
    circuit.cv_rh2(theta_2, qmb, qma, qb)
    circuit.rx(-2*theta_3, qb) # the factors in front of the theta_3 enable us to change the qiskit Rx gate to exp^{i theta}
    return circuit

def apply_h1h2h3(circuit: c2qa.CVCircuit, gate_angles: List[float]) -> None:
    if len(circuit.qmregs) != 1:
        raise ValueError('Only support a single qumode register right now!')
    if len(circuit._qubit_regs) != 1:
        raise ValueError('Only support a single qubit register right now!')

    theta_1, theta_2, theta_3 = gate_angles

    qumode_reg = circuit.qmregs[0]
    qubit_reg  = circuit._qubit_regs[0]

    # brickwork format
    for j in range(0,qumode_reg.num_qumodes - 1,2):
        h1h2h3(circuit, qumode_reg[j+1], qumode_reg[j], qubit_reg[j], theta_1, theta_2, theta_3)
    for j in range(1,qumode_reg.num_qumodes - 1,2):
        h1h2h3(circuit, qumode_reg[j+1], qumode_reg[j], qubit_reg[j], theta_1, theta_2, theta_3)

def construct_circuit(params: List[float], circuit: c2qa.CVCircuit) -> c2qa.CVCircuit:
    for j in range(len(params) // 3):
        apply_h1h2h3(circuit, params[3*j:3*j+3])
    return circuit

def z2_vqe(num_qubits: int, num_qumodes: int, qubits_per_mode: int,
           initial_qumode_state: List[int], num_layers: int = 1,
           gauge_fluctuations=1, optimizer='COBYLA') -> Tuple:
    """Implements a VQE loop on a Z2 LGT

    The input parameters define the specific Z2 LGT for which the VQE will
    attempt to prepare the ground state. After iterating through the
    variational loop, this function will return the optimal parameters and
    energy that were found.
    """
    # cutoff = 2 ** qubits_per_mode

    qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode)
    qbr = qiskit.QuantumRegister(size=num_qubits)
    cbr_measure_field = qiskit.ClassicalRegister(size=num_qubits, name='measure_field')
    cbr_measure_hopping = qiskit.ClassicalRegister(size=num_qubits, name='measure_hopping')

    init_circuit = c2qa.CVCircuit(qmr, qbr, cbr_measure_field, cbr_measure_hopping)

    # initialize the qumodes
    for i in range(qmr.num_qumodes):
        init_circuit.cv_initialize(initial_qumode_state[i], qmr[i])
    # initialize the qubits
    #for qubit in qbr:
    #    init_circuit.h(qubit)

    trace = []
    occupation_history = []
    def f(params):
        """The objective function that scipy will minimize"""
        # First, construct the parameterized ansatz
        z2_ansatz = construct_circuit(params, copy.deepcopy(init_circuit)) # deepcopy allows us to reuse the circuit creation and initialization

        # Take the measurement outcomes and compute the expected energy
        energy = compute_z2_expected_energy(z2_ansatz, gauge_fluctuations=gauge_fluctuations)

        # check the occupation
        z2_ansatz.measure_all()
        stateop, result = c2qa.util.simulate(z2_ansatz)
        # Fock counts for each qumode, occupation = [fock_count_qumode_0 , ..., fock_count_qumode_n]
        occupation = c2qa.util.stateread(stateop, 1, 2, 4, verbose=False)[0][::-1]
        occupation_history.append(occupation)

        trace.append(energy)

        return energy

    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=3 * num_layers)
    out = scipy.optimize.minimize(f, x0=init_params, method=optimizer)

    return out, trace, occupation_history

def compute_z2_expected_energy(circuit: c2qa.CVCircuit, gauge_fluctuations: float) -> float:
    if len(circuit.qmregs) != 1:
        raise ValueError('Only support a single qumode register right now!')
    if len(circuit._qubit_regs) != 1:
        raise ValueError('Only support a single qubit register right now!')

    qumode_reg = circuit.qmregs[0]
    qubit_reg  = circuit._qubit_regs[0]
    # TODO: write a check to grab the correct creg by name
    cbit_field_reg   = circuit.cregs[0]
    cbit_hopping_reg = circuit.cregs[1]

    circuit.barrier()

    hopping_contribution = append_and_measure_hopping_term(copy.deepcopy(circuit), qumode_reg, qubit_reg, cbit_hopping_reg)
    field_contribution = append_and_measure_field_term(copy.deepcopy(circuit), qubit_reg, cbit_field_reg)

    return hopping_contribution + gauge_fluctuations * field_contribution

def append_and_measure_hopping_term(
        circuit: c2qa.CVCircuit,
        qumode_reg: c2qa.QumodeRegister,
        qubit_reg: qiskit.QuantumRegister,
        cbit_hopping_reg: qiskit.ClassicalRegister
        ) -> float:
    for n in range(qumode_reg.num_qumodes - 1):
        measureE_hoppingterm(circuit, qumode_reg[n], qumode_reg[n+1],
                             qubit_reg[n], cbit_hopping_reg[n])

    # Simulate and compute expected energy

    # 1. Measure <Z>
    stateop, result = c2qa.util.simulate(circuit)

    # TODO: return to this and come up with a robust way to get the correct creg results
    counts = result.get_counts()
    hopping_counts = {bitstr.split()[0]: val for bitstr, val in counts.items()}
    shots = sum(hopping_counts.values())

    avg_Z = []
    for n in range(qubit_reg.size):
        avg_z_n = 0
        for bitstr, count in hopping_counts.items():
            bit = bitstr[::-1][n]
            if bit == '0':
                avg_z_n += count / shots
            elif bit == '1':
                avg_z_n += -1 * count / shots
        avg_Z.append(avg_z_n)

    # 2. Measure a, a_dag
    #circuit.measure_all()
    stateop, result = c2qa.util.simulate(circuit)
    # Fock counts for each qumode, occupation = [fock_count_qumode_0 , ..., fock_count_qumode_n]
    occupation = c2qa.util.stateread(stateop, qubit_reg.size, qumode_reg.num_qumodes, qumode_reg.cutoff, verbose=False)[0][::-1]
    diffs = []
    for n in range(qumode_reg.num_qumodes - 1):
        diffs.append(occupation[n+1] - occupation[n])

    assert(len(diffs) == len(avg_Z))
    return sum([diff_val * z_val for diff_val, z_val in zip(diffs, avg_Z)])


def append_and_measure_field_term(
        circuit: c2qa.CVCircuit,
        qubit_reg: qiskit.QuantumRegister,
        cbit_field_reg: qiskit.ClassicalRegister
        ) -> float:
    for qubit, cbit in zip(qubit_reg, cbit_field_reg):
        measureE_fieldterm(circuit, qubit, cbit)

    # Simulate and compute expected energy
    stateop, result = c2qa.util.simulate(circuit)

    # TODO: return to this and come up with a robust way to get the correct creg results
    counts = result.get_counts()
    field_counts = {bitstr.split()[-1]: val for bitstr, val in counts.items()}
    shots = sum(field_counts.values())

    avg_X = []
    for n in range(qubit_reg.size):
        avg_x_n = 0
        for bitstr, count in field_counts.items():
            bit = bitstr[::-1][n]
            if bit == '0':
                avg_x_n += count / shots
            elif bit == '1':
                avg_x_n += -1 * count / shots
        avg_X.append(avg_x_n)

    return sum(avg_X)

def measureE_fieldterm(
        circuit: c2qa.CVCircuit,
        qubit: qiskit.circuit.Qubit,
        cbit: qiskit.circuit.Clbit
        ) -> None:
    circuit.h(qubit)
    circuit.measure(qubit, cbit)

def measureE_hoppingterm(
        circuit: c2qa.CVCircuit,
        qumode1: List[qiskit.circuit.Qubit],
        qumode2: List[qiskit.circuit.Qubit],
        qubit: qiskit.circuit.Qubit,
        cbit: qiskit.circuit.Clbit
        ) -> None:
    circuit.cv_bs(np.pi/4, qumode1, qumode2)
    circuit.measure(qubit, cbit)


#def measure_gauge_invariant_propagator(circuit: c2qa.CVCircuit, qumode_idx: int,
#                                       qubit_idx: int, alpha: float = 1.0,
#                                       verbose: int = 0, shots: int = 1000) -> float:
    # TODO implement this function after debugging it in Z2LGT_BQ_VQE.ipynb
    #qubit_reg = None
    #qumode_reg = None
    #circuit.h(qubit_idx)
#    return None

def schwinger_vqe(num_qubits: int, num_qumodes: int, qubits_per_mode: int,
           initial_qumode_state: List[int], num_layers: int = 1,
           g: float = 1, theta: float = 1, J: float = 1.0, Lambda: float = 1.0,
           optimizer='COBYLA') -> Tuple:
    """Implements a VQE loop on a Schwinger Model LGT

    The input parameters define the specific Schwinger LGT for which the VQE will
    attempt to prepare the ground state. After iterating through the
    variational loop, this function will return the optimal parameters and
    energy that were found.
    """
    # cutoff = 2 ** qubits_per_mode

    qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode)
    qbr = qiskit.QuantumRegister(size=num_qubits)
    cbr_measure_field = qiskit.ClassicalRegister(size=num_qubits, name='measure_field')
    cbr_measure_hopping = qiskit.ClassicalRegister(size=num_qubits, name='measure_hopping')

    init_circuit = c2qa.CVCircuit(qmr, qbr, cbr_measure_field, cbr_measure_hopping)

    # initialize the qumodes
    for i in range(qmr.num_qumodes):
        init_circuit.cv_initialize(initial_qumode_state[i], qmr[i])
    # initialize the qubits
    #for qubit in qbr:
    #    init_circuit.h(qubit)

    trace = []
    occupation_history = []
    def f(params):
        """The objective function that scipy will minimize"""
        # First, construct the parameterized ansatz
        ansatz = construct_circuit(params, copy.deepcopy(init_circuit)) # deepcopy allows us to reuse the circuit creation and initialization

        # Take the measurement outcomes and compute the expected energy
        energy = compute_schwinger_expected_energy(ansatz, g, theta, J, Lambda)

        # check the occupation
        #ansatz.measure_all()
        #stateop, result = c2qa.util.simulate(ansatz)
        # Fock counts for each qumode, occupation = [fock_count_qumode_0 , ..., fock_count_qumode_n]
        # TODO: the line below is currently HARDCODED to noq=1, nom=2, cutoff=4
        #occupation = c2qa.util.stateread(stateop, 1, 2, 4, verbose=False)[0][::-1]
        #occupation_history.append(occupation)

        trace.append(energy)

        return energy

    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=3 * num_layers)
    out = scipy.optimize.minimize(f, x0=init_params, method=optimizer)

    return out, trace, occupation_history

def compute_schwinger_expected_energy(circuit: c2qa.CVCircuit, g: float, theta: float, J: float, Lambda: float) -> float:
    if len(circuit.qmregs) != 1:
        raise ValueError('Only support a single qumode register right now!')
    if len(circuit._qubit_regs) != 1:
        raise ValueError('Only support a single qubit register right now!')

    qumode_reg = circuit.qmregs[0]
    qubit_reg  = circuit._qubit_regs[0]
    # TODO: write a check to grab the correct creg by name
    cbit_magnetic_reg = circuit.cregs[0]
    cbit_e_and_m_reg  = circuit.cregs[1]

    circuit.barrier()

    electric_contribution = measure_electric_contribution(copy.deepcopy(circuit), qumode_reg, qubit_reg, theta)
    magnetic_contribution = measure_magnetic_contribution(copy.deepcopy(circuit), qubit_reg, cbit_magnetic_reg)
    e_and_m_contribution  = measure_e_and_m_contribution(copy.deepcopy(circuit), qumode_reg, qubit_reg, cbit_e_and_m_reg, Lambda)

    return (1/g) * electric_contribution + (1/g) * magnetic_contribution - (J) * e_and_m_contribution

def make_number_operator(matrix, index, num_modes, cutoff, num_qubits):
    identity = scipy.sparse.eye(cutoff)
    if index == 0:
        op = matrix
    else:
        op = identity

    for i in range(1, num_modes):
        if i == index:
            op = scipy.sparse.kron(op, matrix)
        else:
            op = scipy.sparse.kron(op, identity)

    qubit_identity = scipy.sparse.eye(2)
    for _ in range(num_qubits):
        op = scipy.sparse.kron(op, qubit_identity)

    return op

def measure_electric_contribution(
        circuit: c2qa.CVCircuit,
        qumode_reg: c2qa.QumodeRegister,
        qubit_reg: qiskit.QuantumRegister,
        theta: float,
        ) -> float:
    """Measure the energy of H_E = 1/g * SUM_n{(Na_n - Nb_n - theta / 2pi) ^ 2}"""
    stateop, result = c2qa.util.simulate(circuit)

    # Fock counts for each qumode, occupation = [fock_count_qumode_0 , ..., fock_count_qumode_n]
    occupation = c2qa.util.stateread(stateop, qubit_reg.size, qumode_reg.num_qumodes, qumode_reg.cutoff, verbose=False)[0][::-1]
    summation_terms = []
    for n in range(qumode_reg.num_qumodes - 1):
        summation_terms.append(-2 * occupation[n] * occupation[n+1] - theta * (occupation[n] + occupation[n+1]))

    #qumode_statevector = c2qa.util.cv_partial_trace(circuit, stateop).to_statevector()
    qumode_statevector = stateop

    N_matrix = c2qa.operators.CVOperators(qumode_reg.cutoff, 1).N
    N_squared = N_matrix @ N_matrix
    for index in range(qumode_reg.num_qumodes):
        number_operator = make_number_operator(N_squared, index,
                                               qumode_reg.num_qumodes, qumode_reg.cutoff,
                                               qubit_reg.size).toarray()

        statevec_dag = qumode_statevector.data.conj().T
        statevec     = qumode_statevector.data

        expectation_val = np.dot(statevec_dag, np.dot(number_operator, statevec))
        real_part = np.real(expectation_val)
        imag_part = np.imag(expectation_val)
        if abs(imag_part) > 1e-10:
            raise Exception(f'Imaginary part of expectation value should be 0 not: {imag_part}')
        summation_terms.append(real_part)

    return sum(summation_terms)

def measure_magnetic_contribution(
        circuit: c2qa.CVCircuit,
        qubit_reg: qiskit.QuantumRegister,
        cbit_reg: qiskit.ClassicalRegister
        ) -> float:
    """Measure the energy of H_M = 1/g * SUM_n{-1^n * Z_n}"""

    for qubit, cbit in zip(qubit_reg, cbit_reg):
        circuit.measure(qubit, cbit)

    # Simulate and compute expected energy
    stateop, result = c2qa.util.simulate(circuit)

    # TODO: return to this and come up with a robust way to get the correct creg results
    counts = result.get_counts()
    magnetic_counts = {bitstr.split()[-1]: val for bitstr, val in counts.items()}
    shots = sum(magnetic_counts.values())

    avg_Z = []
    for n in range(qubit_reg.size):
        avg_z_n = 0
        for bitstr, count in magnetic_counts.items():
            bit = bitstr[::-1][n]
            if bit == '0':
                avg_z_n += count / shots
            elif bit == '1':
                avg_z_n += -1 * count / shots
        avg_z_n = (-1 ** n) * avg_z_n
        avg_Z.append(avg_z_n)

    return sum(avg_Z)

def measure_e_and_m_contribution(
        circuit: c2qa.CVCircuit,
        qumode_reg: c2qa.QumodeRegister,
        qubit_reg: qiskit.QuantumRegister,
        cbit_reg: qiskit.ClassicalRegister,
        Lambda: float,
        ) -> float:
    """Measure the energy of H_EM = """
    # First apply our beamsplitter trick to XX(ab + ba) and YY(ab + ba)
    # 1. Simulate the circuit twice to get the X and Y expectations on the qubits
    X_expectations, Y_expectations = [], []
    for basis in ['X', 'Y']:
        measure_circuit = copy.deepcopy(circuit)
        for qubit, cbit in zip(qubit_reg, cbit_reg):
            if basis == 'Y':
                measure_circuit.s(qubit)
            measure_circuit.h(qubit)
            measure_circuit.measure(qubit, cbit)

        # Simulate and compute expected energy
        stateop, result = c2qa.util.simulate(measure_circuit)

        # TODO: return to this and come up with a robust way to get the correct creg results
        raw_counts = result.get_counts()
        counts = {bitstr.split()[-2]: val for bitstr, val in raw_counts.items()}
        shots = sum(counts.values())

        for n in range(qubit_reg.size):
            avg_n = 0
            for bitstr, count in counts.items():
                bit = bitstr[::-1][n]
                if bit == '0':
                    avg_n += count / shots
                elif bit == '1':
                    avg_n += -1 * count / shots
            if basis == 'X':
                X_expectations.append(avg_n)
            else:
                Y_expectations.append(avg_n)

    # 2. "Cheat" and use the statevector to get the occupation of the different qumodes
    # Fock counts for each qumode, occupation = [fock_count_qumode_0 , ..., fock_count_qumode_n]
    occupation = c2qa.util.stateread(stateop, qubit_reg.size, qumode_reg.num_qumodes, qumode_reg.cutoff, verbose=False)[0][::-1]
    xx_beamsplitter_terms = []
    yy_beamsplitter_terms = []
    xy_beamsplitter_terms = []
    yx_beamsplitter_terms = []
    for n in range(qumode_reg.num_qumodes - 1):
        diff_plus_term = occupation[n+1] - occupation[n]
        # TODO: update how minus terms are computed
        diff_minus_term = occupation[n+1] - occupation[n]
        # TODO: the line below will throw ValueOutOfBounds if #qubits != #qumodes
        xx_beamsplitter_terms.append(diff_plus_term * X_expectations[n+1] * X_expectations[n])
        yy_beamsplitter_terms.append(diff_plus_term * Y_expectations[n+1] * Y_expectations[n])
        xy_beamsplitter_terms.append(-1 * diff_minus_term * X_expectations[n+1] * Y_expectations[n])
        yx_beamsplitter_terms.append(diff_minus_term * Y_expectations[n+1] * X_expectations[n])

    return sum(xx_beamsplitter_terms) + sum(yy_beamsplitter_terms) + sum(xy_beamsplitter_terms) + sum(yx_beamsplitter_terms)
