import os
import sys

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import c2qa
import qiskit
import numpy as np
import c2qa.util as util


def h1h2h3(circuit, qma, qmb, qb, theta_1, theta_2, theta_3):
    circuit.cv_cpbs(theta_1, qmb, qma, qb)
    circuit.cv_cpbs_z2vqe(theta_2, qmb, qma, qb)
    circuit.rx(theta_3, qb)
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


def measureE_Z2LGT(circuit, numberofmodes, numberofqubits, qmr, qbr, cutoff, N, m, g, dt):
    occs=[np.zeros((N,numberofmodes)),np.zeros((N,numberofqubits))]

    # print("initial state ")
    # stateop, _ = c2qa.util.simulate(circuit)
    # util.stateread(stateop, qbr.size, numberofmodes, cutoff)

    # brickwork format
    for j in range(0,numberofmodes-1,2):
        h1h2h3(circuit, qmr[j+1], qmr[j], qbr[j], m, g, dt)
    for j in range(1,numberofmodes-1,2):
        h1h2h3(circuit, qmr[j+1], qmr[j], qbr[j], m, g, dt)
    # stateop, result = c2qa.util.simulate(circuit)
    # occupation = util.stateread(stateop, qbr.size, numberofmodes, 4)
    # occs[0][i]=np.array(list(occupation[0]))
    # occs[1][i]=np.array(list(occupation[1]))

    # return occs