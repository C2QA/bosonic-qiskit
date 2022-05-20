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


def h1h2h3(circuit, qma, qmb, qb, theta_1, theta_2, theta_3):
    circuit.cv_rh1(theta_1, qmb, qma, qb)
    circuit.cv_rh2(theta_1, qmb, qma, qb)
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
           initial_qumode_state: List[int], num_layers: int = 1) -> Tuple:
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
    for qubit in qbr:
        init_circuit.h(qubit)

    def f(params):
        """The objective function that scipy will minimize"""
        # First, construct the parameterized ansatz
        z2_ansatz = construct_circuit(params, copy.deepcopy(init_circuit)) # deepcopy allows us to reuse the circuit creation and initialization

        # Take the measurement outcomes and compute the expected energy
        energy = compute_z2_expected_energy(z2_ansatz)

        return energy

    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=3 * num_layers)
    out = scipy.optimize.minimize(f, x0=init_params, method="COBYLA")

    return out['fun'], out['x']

def compute_z2_expected_energy(circuit):
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

    field_contribution = append_and_measure_field_term(copy.deepcopy(circuit), qubit_reg, cbit_field_reg)

    hopping_contribution = append_and_measure_hopping_term(copy.deepcopy(circuit), qumode_reg, qubit_reg, cbit_hopping_reg)

    return field_contribution + hopping_contribution

def append_and_measure_hopping_term(circuit, qumode_reg, qubit_reg, cbit_hopping_reg):
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
    circuit.measure_all()
    stateop, result = c2qa.util.simulate(circuit)
    # Fock counts for each qumode, occupation = [fock_count_qumode_0 , ..., fock_count_qumode_n]
    occupation = c2qa.util.stateread(stateop, qubit_reg.size, qumode_reg.num_qumodes, qumode_reg.cutoff, verbose=False)[0][::-1]
    diffs = []
    for n in range(qumode_reg.num_qumodes - 1):
        diffs.append(occupation[n+1] - occupation[n])

    assert(len(diffs) == len(avg_Z))
    return sum([diff_val * z_val for diff_val, z_val in zip(diffs, avg_Z)])


def append_and_measure_field_term(circuit, qubit_reg, cbit_field_reg):
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

def measureE_fieldterm(circuit, qubit, cbit):
    circuit.h(qubit)
    circuit.measure(qubit, cbit)

def measureE_hoppingterm(circuit, qumode1, qumode2, qubit, cbit):
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
