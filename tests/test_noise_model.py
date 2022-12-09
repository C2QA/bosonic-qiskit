from pathlib import Path
import pytest
import random


import c2qa
import numpy as np


import qiskit
import qiskit.providers.aer.noise as noise


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

        photon_loss_rate = 1000000  # per second
        time = 5.0  # seconds
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

        photon_loss_rate = 1000000  # per second
        time = 1.0  # seconds
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


def test_noise_with_beamsplitter(capsys):
    with capsys.disabled():
        num_qumodes = 2
        qubits_per_mode = 2
        num_qubits = 1

        qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode)
        qbr = qiskit.QuantumRegister(size=num_qubits)
        init_circuit = c2qa.CVCircuit(qmr, qbr)
        init_circuit.cv_initialize(2, qmr[0])
        init_circuit.cv_c_bs(1, qmr[1], qmr[0], qbr[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate=photon_loss_rate, circuit=init_circuit, time_unit=time_unit)
        state, result = c2qa.util.simulate(init_circuit, noise_pass=noise_pass)

def test_photon_loss_pass_with_conditional(capsys):
    with capsys.disabled():
        num_qumodes = 1
        qubits_per_mode = 2
        num_qubits = 1

        qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode)
        qbr = qiskit.QuantumRegister(size=num_qubits)
        init_circuit = c2qa.CVCircuit(qmr, qbr)
        init_circuit.cv_initialize(2, qmr[0])
        init_circuit.cv_c_d(1, qmr[0], qbr[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate=photon_loss_rate, circuit=init_circuit, time_unit=time_unit)
        state, result = c2qa.util.simulate(init_circuit, noise_pass=noise_pass)

def test_animate_photon_loss_pass(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        circuit = c2qa.CVCircuit(qmr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_d(0, qmr[0], duration=100, unit="ns")

        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate=photon_loss_rate, circuit=circuit, time_unit=time_unit)

        wigner_filename = "tests/test_animate_photon_loss_pass.gif"
        c2qa.animate.animate_wigner(
            circuit,
            animation_segments=10,
            file=wigner_filename,
            noise_pass=noise_pass,
        )
        assert Path(wigner_filename).is_file()

@pytest.mark.skip(reason="GitHub actions build environments do not have ffmpeg")
def test_photon_loss_pass_no_displacement(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        circuit = c2qa.CVCircuit(qmr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_d(0, qmr[0], duration=100, unit="ns")

        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate=photon_loss_rate, circuit=circuit, time_unit=time_unit)

        # state, result = c2qa.util.simulate(circuit, noise_pass=noise_pass)

        wigner_filename = "tests/noise_model_pass_no_displacement.mp4"
        c2qa.animate.animate_wigner(
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

        circuit.cv_d(1.5, qmr[0], duration=100, unit="ns")

        photon_loss_rate = 0.02
        time_unit = "ns"
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate=photon_loss_rate, circuit=circuit, time_unit=time_unit)

        # state, result = c2qa.util.simulate(circuit, noise_pass=noise_pass)

        wigner_filename = "tests/noise_model_pass_slow_displacement.mp4"
        c2qa.animate.animate_wigner(
            circuit,
            animation_segments=200,
            file=wigner_filename,
            noise_pass=noise_pass,
            # draw_grid=True
        )


@pytest.mark.skip(reason="GitHub actions build environments do not have ffmpeg")
def test_photon_loss_pass_slow_conditional_displacement(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        num_qubits = 1

        qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode)
        qbr = qiskit.QuantumRegister(size=num_qubits)
        circuit = c2qa.CVCircuit(qmr, qbr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_c_d(1, qmr[0], qbr[0], duration=100, unit="ns")

        photon_loss_rate = 0.02
        time_unit = "ns"
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate=photon_loss_rate, circuit=circuit, time_unit=time_unit)

        # state, result = c2qa.util.simulate(circuit, noise_pass=noise_pass)

        wigner_filename = "tests/noise_model_pass_slow_condiational_displacement.mp4"
        c2qa.animate.animate_wigner(
            circuit,
            animation_segments=200,
            file=wigner_filename,
            noise_pass=noise_pass,
        )

def test_photon_loss_and_phase_damping(capsys):
    with capsys.disabled():
        state_a, result_a = _build_photon_loss_and_amp_damping_circuit(0.0)
        print(state_a)
        assert result_a.success

        state_b, result_b = _build_photon_loss_and_amp_damping_circuit(1.0)
        print(state_b)
        assert result_b.success

        assert not allclose(state_a, state_b)
    
def _build_photon_loss_and_amp_damping_circuit(amp_damp = 0.3, photon_loss_rate = 0.01):
    num_qumodes = 1
    qubits_per_mode = 2
    num_qubits = 1

    qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode)
    qbr = qiskit.QuantumRegister(size=num_qubits)
    circuit = c2qa.CVCircuit(qmr, qbr)
    circuit.cv_initialize(2, qmr[0])
    circuit.x(qbr[0])
    circuit.cv_d(1, qmr[0], duration=100, unit="ns")

    # Initialize phase damping NoiseModel
    noise_model = noise.NoiseModel()
    phase_error = noise.amplitude_damping_error(amp_damp)
    noise_model.add_quantum_error(phase_error, ["x"], [circuit.get_qubit_index(qbr[0])])

    # Initialize PhotonLossNoisePass
    noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rate=photon_loss_rate, circuit=circuit, time_unit="ns")

    return c2qa.util.simulate(circuit, noise_model=noise_model, noise_pass=noise_pass)

def allclose(a, b) -> bool:
    """Convert SciPy sparse matrices to ndarray and test with Numpy"""

    # If a and b are SciPy sparse matrices, they'll have a "toarray()" function
    if hasattr(a, "toarray"):
        a = a.toarray()

    if hasattr(b, "toarray"):
        b = b.toarray()

    return np.allclose(a, b)