import math
import random


import c2qa
import numpy as np
import qiskit


def test_noise_model(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 2
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        qr = qiskit.QuantumRegister(2)
        circuit = c2qa.CVCircuit(qmr, qr)

        for qumode in range(num_qumodes):
            circuit.cv_initialize(0, qmr[qumode])

        circuit.initialize([0, 1], qr[1])  # qr[0] will init to zero

        alpha = random.random()
        beta = random.random()
        circuit.cv_cnd_d(alpha, -beta, qr[0], qmr[0])
        circuit.cv_cnd_d(-alpha, beta, qr[0], qmr[0])

        circuit.cv_cnd_d(alpha, -beta, qr[1], qmr[0])
        circuit.cv_cnd_d(-alpha, beta, qr[1], qmr[0])

        # prob = 0.01
        # kraus = circuit.ops.a * circuit.ops.a_dag
        # kraus_operator = [circuit.ops.a.toarray().tolist(), circuit.ops.a_dag.toarray().tolist()]

        # theta = np.pi / 2
        # kraus_operator = [[[1, 0],[0, math.sin(theta)]], [[0, math.cos(theta)],[0, 0]]]

        num_photons = circuit.cutoff - 1
        photon_loss_rate = 0.01
        time = 10.0
        kraus_operator = c2qa.kraus.calculate_kraus(num_photons, photon_loss_rate, time, circuit.ops.a, circuit.ops.a_dag)

        print("a")
        print(circuit.ops.a.toarray())
        print("a_dag")
        print(circuit.ops.a_dag.toarray())
        print("a * a_dag")
        print((circuit.ops.a * circuit.ops.a_dag).toarray())
        print("kraus")
        print(kraus_operator)

        state, result = c2qa.util.simulate(circuit, kraus_operator=kraus_operator)