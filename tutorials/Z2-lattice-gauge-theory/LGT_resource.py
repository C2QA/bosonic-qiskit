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

def vary_Z2LGT(circuit, numberofmodes, qmr, qbr, theta_1, theta_2, theta_3):
    # print("initial state ")
    # stateop, _ = c2qa.util.simulate(circuit)
    # util.stateread(stateop, qbr.size, numberofmodes, cutoff)

    # brickwork format
    for j in range(0,numberofmodes-1,2):
        h1h2h3(circuit, qmr[j+1], qmr[j], qbr[j], theta_1, theta_2, theta_3)
    for j in range(1,numberofmodes-1,2):
        h1h2h3(circuit, qmr[j+1], qmr[j], qbr[j], theta_1, theta_2, theta_3)

def apply_h1h2h3(circuit: c2qa.CVCircuit, gate_angles: List[float]) -> None:
    if len(circuit.qmregs) != 1:
        raise ValueError('Only support a single qumode register right now!')
    if len(circuit._qubit_regs) != 1:
        raise ValueError('Only support a single qubit register right now!')

    theta_1, theta_2, theta_3 = gate_angles

    qumode_reg = circuit.qmregs[0]
    qubit_reg  = circuit._qubit_regs[0]

    # Apply the H1 unitary
    for n in range(qumode_reg.num_qumodes - 1, 2):
        #circuit.cv_rh1(-1 * np.sqrt(theta_1), qumode_reg[n], qumode_reg[n+1], qubit_reg[n])
        circuit.cv_rh1(theta_1, qumode_reg[n], qumode_reg[n+1], qubit_reg[n])

    # Apply the H2 unitary
    for n in range(qumode_reg.num_qumodes - 1, 2):
        # circuit.cv_rh2(np.sqrt(theta_2), qumode_reg[n], qumode_reg[n+1], qubit_reg[n])
        circuit.cv_rh2(theta_2, qumode_reg[n], qumode_reg[n+1], qubit_reg[n])

    # Apply the H3 unitary
    for qubit in qubit_reg:
        # circuit.rx(2 * theta_3, qubit)
        circuit.rx(2 * theta_3, qubit)

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
    init_circuit = c2qa.CVCircuit(qmr, qbr)

    # initialize the qumodes
    for i in range(qmr.num_qumodes):
        init_circuit.cv_initialize(initial_qumode_state[i], qmr[i])
    # initialize the qubits
    for qubit in qbr:
        init_circuit.h(qubit)

    def f(params):
        """The objective function that scipy will minimize"""
        # First, construct the parameterized ansatz
        z2_ansatz = construct_circuit(params, copy.deepcopy(init_circuit))

        # Take the measurement outcomes and compute the expected energy
        energy = compute_z2_expected_energy(copy.deepcopy(z2_ansatz))

        # Finally, simulate its execution
        stateop, result = c2qa.util.simulate(z2_ansatz)

        # stateop, result = c2qa.util.simulate(circuit)
        # counts = result.get_counts()
        # c2qa.util.cv_fockcounts(counts, (qmr[0], qbr[0]))
        # stateop, result = c2qa.util.simulate(circuit)
        # occupation = util.stateread(stateop, qbr.size, numberofmodes, 4)
        # occs[0][i]=np.array(list(occupation[0]))
        # occs[1][i]=np.array(list(occupation[1]))

        return 0

    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=3 * num_layers)
    out = scipy.optimize.minimize(f, x0=init_params, method="COBYLA")

    return out['fun'], out['x']

def compute_z2_expected_energy(result):
    for i in range(numberofqubits):
        measureE_fieldterm(circuit, qmr, qbr, i)
    circuit.barrier()
    for i in range(numberofqubits):
        measureE_fieldterm(circuit, qmr, qbr, i)
    circuit

def measureE_fieldterm(circuit, qmr, qbr, i):
    circuit.x(qbr[i])
    # figure out which qubit corresponds to i in the small endian format etc. Or just make the measure function.
    circuit.measure(-i, 0)

def measureE_hoppingterm(circuit, numberofmodes, numberofqubits, qmr, qbr, i):
    occs=[np.zeros((numberofmodes,numberofqubits))]
    circuit.cv_bs(np.pi/4, qmr[i], qmr[i+1])
    # figure out which qubit corresponds to i in the small endian format etc. Or just make the measure function.
    circuit.measure(-i, 0)
    # return occs


#def measure_gauge_invariant_propagator(circuit: c2qa.CVCircuit, qumode_idx: int,
#                                       qubit_idx: int, alpha: float = 1.0,
#                                       verbose: int = 0, shots: int = 1000) -> float:
    # TODO implement this function after debugging it in Z2LGT_BQ_VQE.ipynb
    #qubit_reg = None
    #qumode_reg = None
    #circuit.h(qubit_idx)
#    return None
