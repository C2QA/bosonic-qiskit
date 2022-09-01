import pytest
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
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        for qumode in range(num_qumodes):
            circuit.cv_initialize(0, qmr[qumode])

        circuit.initialize([0, 1], qr[1])  # qr[0] will init to zero

        alpha = random.random()
        circuit.cv_c_d(alpha, qmr[0], qr[0])
        circuit.cv_c_d(-alpha, qmr[0], qr[0])

        circuit.cv_c_d(-alpha, qmr[0], qr[1])
        circuit.cv_c_d(alpha, qmr[0], qr[1])

        photon_loss_rate = 0.01
        time = 5.0
        kraus_operators = c2qa.kraus.calculate_kraus(photon_loss_rate, time, circuit)

        print("kraus")
        print(kraus_operators)

        state, result = c2qa.util.simulate(circuit)


def test_kraus_operators(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 2
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        qr = qiskit.QuantumRegister(2)
        circuit = c2qa.CVCircuit(qmr, qr)

        photon_loss_rate = 0.1
        time = 1.0
        kraus_operators = c2qa.kraus.calculate_kraus(photon_loss_rate, time, circuit)

        kraus = qiskit.quantum_info.operators.channel.Kraus(kraus_operators)
        assert kraus.is_cp(), "Is not completely positive"

        print()
        print("Kraus Operators")
        accum = 0j
        for index, op in enumerate(kraus_operators):
            print(f"op {index}")
            print(op)

            op_dag = np.transpose(np.conj(op))
            print(f"op_dag {index}")
            print(op_dag)

            op_dot = np.dot(op_dag, op)
            print(f"op_dot {index}")
            print(op_dot)

            accum += op_dot
            print()

        print("Sum")
        print(accum)

        is_identity = (accum.shape[0] == accum.shape[1]) and np.allclose(
            accum, np.eye(accum.shape[0])
        )
        print(f"Sum is identity {is_identity}")
        assert is_identity, "Sum is not identity"

        assert kraus.is_tp(), "Is not trace preserving"
        assert kraus.is_cptp(), "Is not CPTP"


@pytest.mark.skip(reason="GitHub actions build environments do not have ffmpeg")
def test_photon_loss_pass_no_displacement(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        circuit = c2qa.CVCircuit(qmr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_d(0, qmr[0])

        photon_loss_rate = 1000000  # At duration of 100ns, this makes kt = 0.1
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate, circuit)

        # state, result = c2qa.util.simulate(circuit, noise_pass=noise_pass)

        wigner_filename = "tests/noise_model_pass_no_displacement.mp4"
        c2qa.util.animate_wigner(
            circuit,
            animation_segments=200,
            file=wigner_filename,
            noise_pass=noise_pass,
        )


@pytest.mark.skip(reason="GitHub actions build environments do not have ffmpeg")
def test_photon_loss_pass_slow_displacement(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        circuit = c2qa.CVCircuit(qmr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_d(1, qmr[0])

        photon_loss_rate = 1000000  # At duration of 100ns, this makes kt = 0.1
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate, circuit)

        # state, result = c2qa.util.simulate(circuit, noise_pass=noise_pass)

        wigner_filename = "tests/noise_model_pass_slow_displacement.mp4"
        c2qa.util.animate_wigner(
            circuit,
            animation_segments=200,
            file=wigner_filename,
            noise_pass=noise_pass,
        )
