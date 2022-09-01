import c2qa
import numpy as np


def eiht(circuit, qma, qmb, J, U, dt):
    circuit.cv_bs(-J * dt, qmb, qma)
    circuit.cv_r(-U * dt, qma)
    circuit.cv_r(-U * dt, qmb)
    return circuit


def trotterise_Z2LGT(
    circuit, numberofmodes, numberofqubits, qmr, qbr, cutoff, N, m, g, dt
):
    occs = [np.zeros((N, numberofmodes)), np.zeros((N, numberofqubits))]

    for i in range(numberofqubits):
        circuit.h(
            qbr[i]
        )  # Inititialises the qubit to a plus state (so that pauli Z flips it)
    # print("initial state ")
    # stateop, _ = c2qa.util.simulate(circuit)
    # util.stateread(stateop, qbr.size, numberofmodes, cutoff)

    # Trotterise. i*dt corresponds to the timestep i of length from the previous timestep dt.
    for i in range(N):
        print("dt+1", i * dt)
        # Trotterise according to the brickwork format to make depth of circuit
        # 2 and not number of timesteps (because each site needs to be part of
        # a gate with the site to the left and a gate with the site to the right.
        for j in range(0, numberofmodes - 1, 2):
            eiht(circuit, qmr[j + 1], qmr[j], qbr[j], m, g, dt)
        for j in range(1, numberofmodes - 1, 2):
            eiht(circuit, qmr[j + 1], qmr[j], qbr[j], m, g, dt)
        stateop, result = c2qa.util.simulate(circuit)
        occupation = c2qa.util.stateread(stateop, qbr.size, numberofmodes, 4)
        occs[0][i] = np.array(list(occupation[0]))
        occs[1][i] = np.array(list(occupation[1]))

    return occs
