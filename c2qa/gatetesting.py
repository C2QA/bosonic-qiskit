import circuit, projectors
import numpy as np
import c2qa

def initi(qmr, circuit, diffstallmodes):
    circuit.cv_initialize(diffstallmodes[0], qmr[0])
    circuit.cv_initialize(diffstallmodes[1], qmr[1])

def differentModeInitialisation(qmr, circuit, numberofmodes, qbinist, samestallmodes):
    diffstallmodes = [0, 0]
    initi(qmr, circuit, diffstallmodes)
    circuit.cv_bs(np.pi, qmr[0], qmr[1])
    state, _ = c2qa.util.simulate(circuit)
    projectors.overlap(state, numberofmodes, qbinist, samestallmodes, diffstallmodes, "diffstallmodes" ,"all")

    diffstallmodes=[1,1]
    initi(qmr, circuit, diffstallmodes)
    circuit.cv_bs(np.pi, qmr[0], qmr[1])
    state, _ = c2qa.util.simulate(circuit)
    projectors.overlap(state, numberofmodes, qbinist, samestallmodes, diffstallmodes, "diffstallmodes" ,"all")

    diffstallmodes=[1,0]
    initi(qmr, circuit, diffstallmodes)
    circuit.cv_bs(np.pi, qmr[0], qmr[1])
    state, _ = c2qa.util.simulate(circuit)
    projectors.overlap(state, numberofmodes, qbinist, samestallmodes, diffstallmodes, "diffstallmodes" ,"all")

def differentThetaInitialisation(qmr, circuit, numberofmodes, qbinist, samestallmodes, diffstallmodes):
    print("pi")
    initi(qmr, circuit, diffstallmodes)
    circuit.cv_bs(np.pi, qmr[1], qmr[0])
    state, _ = c2qa.util.simulate(circuit)
    projectors.overlap(state, numberofmodes, qbinist, samestallmodes, diffstallmodes, "diffstallmodes" ,"all")
    print("pi/2")
    initi(qmr, circuit, diffstallmodes)
    circuit.cv_bs(np.pi/2, qmr[0], qmr[1])
    state, _ = c2qa.util.simulate(circuit)
    projectors.overlap(state, numberofmodes, qbinist, samestallmodes, diffstallmodes, "diffstallmodes" ,"all")
    print("pi/4")
    initi(qmr, circuit, diffstallmodes)
    circuit.cv_bs(np.pi/4, qmr[1], qmr[0])
    state, _ = c2qa.util.simulate(circuit)
    projectors.overlap(state, numberofmodes, qbinist, samestallmodes, diffstallmodes, "diffstallmodes" ,"all")
    print("pi/8")
    initi(qmr, circuit, diffstallmodes)
    circuit.cv_bs(np.pi/8, qmr[0], qmr[1])
    state, _ = c2qa.util.simulate(circuit)
    projectors.overlap(state, numberofmodes, qbinist, samestallmodes, diffstallmodes, "diffstallmodes" ,"all")
