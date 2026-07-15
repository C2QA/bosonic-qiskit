from typing import cast
from pathlib import Path
import pytest
import random


import bosonic_qiskit
import numpy as np
from scipy import stats


import qiskit
import qiskit_aer.noise as noise
from qiskit_aer.noise.passes.relaxation_noise_pass import RelaxationNoisePass


def test_noise_model(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 2
        qmr = bosonic_qiskit.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        qr = qiskit.QuantumRegister(2)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = bosonic_qiskit.CVCircuit(qmr, qr, cr)

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
        kraus_operators = bosonic_qiskit.kraus.calculate_kraus(
            photon_loss_rates=[photon_loss_rate, photon_loss_rate],
            time=time,
            circuit=circuit,
            op_qubits=[0, 1, 2],
            qumode_qubit_indices=[0, 1],
        )

        print("kraus")
        print(kraus_operators)

        state, result, fock_counts = bosonic_qiskit.util.simulate(circuit)


def test_kraus_operators(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 2
        qmr = bosonic_qiskit.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        qr = qiskit.QuantumRegister(2)
        circuit = bosonic_qiskit.CVCircuit(qmr, qr)

        photon_loss_rate = 1000000  # per second
        time = 1.0  # seconds
        kraus_operators = bosonic_qiskit.kraus.calculate_kraus(
            photon_loss_rates=[photon_loss_rate, photon_loss_rate],
            time=time,
            circuit=circuit,
            op_qubits=[0, 1, 2],
            qumode_qubit_indices=[0, 1],
        )

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


def test_beamsplitter_kraus_operators(capsys):
    with capsys.disabled():
        num_qumodes = 2
        qubits_per_mode = 2

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
        )
        circuit = bosonic_qiskit.CVCircuit(qmr)
        circuit.cv_initialize(2, qmr[0])
        circuit.cv_bs(1, qmr[1], qmr[0], duration=100, unit="ns")
        photon_loss_rate = 1000000  # per second
        time = 1.0  # seconds
        kraus_operators = bosonic_qiskit.kraus.calculate_kraus(
            photon_loss_rates=[photon_loss_rate, photon_loss_rate],
            time=time,
            circuit=circuit,
            op_qubits=[2, 3, 0, 1],
            qumode_qubit_indices=[0, 1, 2, 3],
        )

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


def test_invalid_photon_loss_rate_length(capsys):
    with pytest.raises(Exception), capsys.disabled():
        num_qumodes = 2
        qubits_per_mode = 2

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
        )
        init_circuit = bosonic_qiskit.CVCircuit(qmr)
        init_circuit.cv_initialize(2, qmr[0])
        init_circuit.cv_bs(1, qmr[1], qmr[0], duration=100, unit="ns")
        photon_loss_rates = [1, 2, 3]  # Should only have two loss rates
        time_unit = "ns"

        # Should raise Exception for not having proper number of loss rates
        bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rates,
            circuit=init_circuit,
            time_unit=time_unit,
        )


def test_valid_photon_loss_rate_length(capsys):
    with capsys.disabled():
        num_qumodes = 2
        qubits_per_mode = 2

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
        )
        init_circuit = bosonic_qiskit.CVCircuit(qmr)
        init_circuit.cv_initialize(2, qmr[0])
        init_circuit.cv_bs(1, qmr[1], qmr[0], duration=100, unit="ns")
        photon_loss_rates = [1, 2]  # Should only have two loss rates
        time_unit = "ns"

        # Should not raise exception as has proper number of loss rates
        bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rates,
            circuit=init_circuit,
            time_unit=time_unit,
        )


def test_noise_with_beamsplitter(capsys):
    with capsys.disabled():
        num_qumodes = 2
        qubits_per_mode = 2

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
        )
        init_circuit = bosonic_qiskit.CVCircuit(qmr)
        init_circuit.cv_initialize(2, qmr[0])
        init_circuit.cv_bs(1, qmr[1], qmr[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=init_circuit,
            time_unit=time_unit,
        )
        state, result, fock_counts = bosonic_qiskit.util.simulate(
            init_circuit, noise_passes=noise_pass
        )


def test_noise_with_beamsplitter_diff_cutoff(capsys):
    with capsys.disabled():
        qmr1 = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qmr2 = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=3)
        init_circuit = bosonic_qiskit.CVCircuit(qmr1, qmr2)
        init_circuit.cv_initialize(2, qmr1[0])
        init_circuit.cv_initialize(2, qmr2[0])
        init_circuit.cv_bs(1, qmr1[0], qmr2[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=init_circuit,
            time_unit=time_unit,
        )
        state, result, fock_counts = bosonic_qiskit.util.simulate(
            init_circuit, noise_passes=noise_pass
        )


@pytest.mark.skip(reason="This test takes nearly 30 minutes to pass on Github...")
def test_noise_with_cbs_diff_cutoff(capsys):
    with capsys.disabled():
        qmr1 = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qmr2 = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=3)
        qbr = qiskit.QuantumRegister(1)
        init_circuit = bosonic_qiskit.CVCircuit(qmr1, qmr2, qbr)
        init_circuit.cv_initialize(2, qmr1[0])
        init_circuit.cv_initialize(2, qmr2[0])
        init_circuit.cv_c_bs(1, qmr1[0], qmr2[0], qbr[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=init_circuit,
            time_unit=time_unit,
        )
        state, result, fock_counts = bosonic_qiskit.util.simulate(
            init_circuit, noise_passes=noise_pass
        )


def test_noise_with_sq2_diff_cutoff(capsys):
    with capsys.disabled():
        qmr1 = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qmr2 = bosonic_qiskit.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=3)
        init_circuit = bosonic_qiskit.CVCircuit(qmr1, qmr2)
        init_circuit.cv_initialize(2, qmr1[0])
        init_circuit.cv_initialize(2, qmr2[0])
        init_circuit.cv_sq2(1, qmr1[0], qmr2[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=init_circuit,
            time_unit=time_unit,
        )
        state, result, fock_counts = bosonic_qiskit.util.simulate(
            init_circuit, noise_passes=noise_pass
        )


def test_noise_with_cnd_beamsplitter(capsys):
    with capsys.disabled():
        num_qumodes = 2
        qubits_per_mode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)
        init_circuit = bosonic_qiskit.CVCircuit(qmr, qbr)
        init_circuit.cv_initialize(2, qmr[0])
        init_circuit.cv_c_bs(1, qmr[1], qmr[0], qbr[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=init_circuit,
            time_unit=time_unit,
        )
        state, result, fock_counts = bosonic_qiskit.util.simulate(
            init_circuit, noise_passes=noise_pass
        )


def test_photon_loss_pass_with_conditional(capsys):
    with capsys.disabled():
        num_qumodes = 1
        qubits_per_mode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)
        init_circuit = bosonic_qiskit.CVCircuit(qmr, qbr)
        init_circuit.cv_initialize(2, qmr[0])
        init_circuit.cv_c_d(1, qmr[0], qbr[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=init_circuit,
            time_unit=time_unit,
        )
        state, result, fock_counts = bosonic_qiskit.util.simulate(
            init_circuit, noise_passes=noise_pass
        )


def test_photon_loss_pass_delay_without_unit(capsys):
    with capsys.disabled():
        num_qumodes = 1
        qubits_per_mode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)

        fail_circuit = bosonic_qiskit.CVCircuit(qmr, qbr)
        fail_circuit.cv_initialize(2, qmr[0])
        fail_circuit.delay(duration=100)  # , unit="ns")
        fail_circuit.cv_c_d(1, qmr[0], qbr[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=fail_circuit,
            time_unit=time_unit,
        )
        state, result, fock_counts = bosonic_qiskit.util.simulate(
            fail_circuit, noise_passes=noise_pass
        )
        assert result.success


def test_photon_loss_pass_delay_with_unit(capsys):
    with capsys.disabled():
        num_qumodes = 1
        qubits_per_mode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)

        pass_circuit = bosonic_qiskit.CVCircuit(qmr, qbr)
        pass_circuit.cv_initialize(2, qmr[0])
        pass_circuit.delay(duration=100, unit="ns")
        pass_circuit.cv_c_d(1, qmr[0], qbr[0], duration=100, unit="ns")
        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=pass_circuit,
            time_unit=time_unit,
        )
        state, result, fock_counts = bosonic_qiskit.util.simulate(
            pass_circuit, noise_passes=noise_pass
        )
        assert result.success


def test_animate_photon_loss_pass(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        qmr = bosonic_qiskit.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        circuit = bosonic_qiskit.CVCircuit(qmr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_d(0, qmr[0], duration=100, unit="ns")

        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate, circuit=circuit, time_unit=time_unit
        )

        wigner_filename = "tests/test_animate_photon_loss_pass.gif"
        bosonic_qiskit.animate.animate_wigner(
            circuit,
            animation_segments=10,
            file=wigner_filename,
            noise_passes=noise_pass,
        )
        assert Path(wigner_filename).is_file()


def test_animate_photon_loss_pass_with_epsilon(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        qmr = bosonic_qiskit.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        circuit = bosonic_qiskit.CVCircuit(qmr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_d(0, qmr[0], duration=100, unit="ns")

        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate, circuit=circuit, time_unit=time_unit
        )

        wigner_filename = "tests/test_animate_photon_loss_pas_with_epsilon.gif"
        bosonic_qiskit.animate.animate_wigner(
            circuit,
            discretize_epsilon=0.1,
            file=wigner_filename,
            noise_passes=noise_pass,
        )
        assert Path(wigner_filename).is_file()


def test_photon_loss_pass_no_displacement(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        qmr = bosonic_qiskit.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        circuit = bosonic_qiskit.CVCircuit(qmr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_d(0, qmr[0], duration=100, unit="ns")

        photon_loss_rate = 0.01
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate, circuit=circuit, time_unit=time_unit
        )

        # state, result, fock_counts = bosonic_qiskit.util.simulate(circuit, noise_passes=noise_pass)

        wigner_filename = "tests/test_photon_loss_pass_no_displacement.gif"
        bosonic_qiskit.animate.animate_wigner(
            circuit,
            animation_segments=200,
            file=wigner_filename,
            noise_passes=noise_pass,
        )


def test_photon_loss_pass_slow_displacement(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        qmr = bosonic_qiskit.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        circuit = bosonic_qiskit.CVCircuit(qmr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_d(1.5, qmr[0], duration=100, unit="ns")

        photon_loss_rate = 0.02
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate, circuit=circuit, time_unit=time_unit
        )

        # state, result, fock_counts = bosonic_qiskit.util.simulate(circuit, noise_passes=noise_pass)

        wigner_filename = "tests/test_photon_loss_pass_slow_displacement.gif"
        bosonic_qiskit.animate.animate_wigner(
            circuit,
            animation_segments=200,
            file=wigner_filename,
            noise_passes=noise_pass,
            # draw_grid=True
        )


def test_photon_loss_pass_slow_conditional_displacement(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 4
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)
        circuit = bosonic_qiskit.CVCircuit(qmr, qbr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_c_d(1, qmr[0], qbr[0], duration=100, unit="ns")

        photon_loss_rate = 0.02
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate, circuit=circuit, time_unit=time_unit
        )

        # state, result, fock_counts = bosonic_qiskit.util.simulate(circuit, noise_passes=noise_pass)

        wigner_filename = (
            "tests/test_photon_loss_pass_slow_conditional_displacement.gif"
        )
        bosonic_qiskit.animate.animate_wigner(
            circuit,
            animation_segments=200,
            file=wigner_filename,
            noise_passes=noise_pass,
        )


def test_photon_loss_instruction(capsys):
    with capsys.disabled():
        num_qumodes = 2
        num_qubits_per_qumode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)
        circuit = bosonic_qiskit.CVCircuit(qmr, qbr)

        circuit.cv_initialize(1, qmr[0])
        circuit.cv_initialize(1, qmr[1])

        circuit.cv_d(1, qmr[0], duration=100, unit="ns")
        circuit.cv_c_d(1, qmr[1], qbr[0], duration=100, unit="ns")

        photon_loss_rate = 0.02
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=circuit,
            time_unit=time_unit,
            instructions=["cD"],
        )

        state, result, fock_counts = bosonic_qiskit.util.simulate(
            circuit, noise_passes=noise_pass
        )
        assert result.success


def test_photon_loss_qumode(capsys):
    with capsys.disabled():
        num_qumodes = 2
        num_qubits_per_qumode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)
        circuit = bosonic_qiskit.CVCircuit(qmr, qbr)

        circuit.cv_initialize(1, qmr[0])

        circuit.cv_d(1, qmr[0], duration=100, unit="ns")
        circuit.cv_c_d(1, qmr[1], qbr[0], duration=100, unit="ns")

        photon_loss_rate = 0.02
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=circuit,
            time_unit=time_unit,
            qumodes=qmr[1],
        )

        state, result, fock_counts = bosonic_qiskit.util.simulate(
            circuit, noise_passes=noise_pass
        )
        assert result.success


def test_photon_loss_instruction_qumode(capsys):
    with capsys.disabled():
        num_qumodes = 2
        num_qubits_per_qumode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)
        circuit = bosonic_qiskit.CVCircuit(qmr, qbr)

        circuit.cv_initialize(1, qmr[0])

        circuit.cv_d(1, qmr[0], duration=100, unit="ns")
        circuit.cv_c_d(1, qmr[1], qbr[0], duration=100, unit="ns")

        photon_loss_rate = 0.02
        time_unit = "ns"
        noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
            photon_loss_rates=photon_loss_rate,
            circuit=circuit,
            time_unit=time_unit,
            instructions=["cD"],
            qumodes=qmr[0],
        )

        state, result, fock_counts = bosonic_qiskit.util.simulate(
            circuit, noise_passes=noise_pass
        )
        assert result.success


def test_photon_loss_and_phase_damping(capsys):
    with capsys.disabled():
        state_a, result_a, fock_counts = _build_photon_loss_and_amp_damping_circuit(0.0)
        print(state_a)
        assert result_a.success

        state_b, result_b, fock_counts = _build_photon_loss_and_amp_damping_circuit(1.0)
        print(state_b)
        assert result_b.success

        assert not allclose(state_a, state_b)


def _build_photon_loss_and_amp_damping_circuit(amp_damp=0.3, photon_loss_rate=0.01):
    num_qumodes = 1
    qubits_per_mode = 2
    num_qubits = 1

    qmr = bosonic_qiskit.QumodeRegister(
        num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode
    )
    qbr = qiskit.QuantumRegister(size=num_qubits)
    circuit = bosonic_qiskit.CVCircuit(qmr, qbr)
    circuit.cv_initialize(2, qmr[0])
    circuit.x(qbr[0])
    circuit.cv_d(1, qmr[0], duration=100, unit="ns")

    # Initialize phase damping NoiseModel
    noise_model = noise.NoiseModel()
    phase_error = noise.amplitude_damping_error(amp_damp)
    noise_model.add_quantum_error(phase_error, ["x"], [circuit.get_qubit_index(qbr[0])])

    # Initialize PhotonLossNoisePass
    noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass(
        photon_loss_rates=photon_loss_rate, circuit=circuit, time_unit="ns"
    )

    return bosonic_qiskit.util.simulate(
        circuit, noise_model=noise_model, noise_passes=noise_pass
    )


def allclose(a, b) -> bool:
    """Convert SciPy sparse matrices to ndarray and test with Numpy"""

    # If a and b are SciPy sparse matrices, they'll have a "toarray()" function
    if hasattr(a, "toarray"):
        a = a.toarray()

    if hasattr(b, "toarray"):
        b = b.toarray()

    return np.allclose(a, b)


def test_relaxation_noise_pass(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)
        circuit = bosonic_qiskit.CVCircuit(qmr, qbr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_c_d(1, qmr[0], qbr[0], duration=100, unit="ns")

        t1s = np.ones(circuit.num_qubits).tolist()
        t2s = np.ones(circuit.num_qubits).tolist()
        noise_pass = RelaxationNoisePass(t1s, t2s)

        filename = "tests/test_relaxation_noise_pass.gif"
        bosonic_qiskit.animate.animate_wigner(
            circuit,
            animation_segments=200,
            file=filename,
            noise_passes=noise_pass,
        )
        assert Path(filename).is_file()


def test_relaxation_and_photon_loss_noise_passes(capsys):
    with capsys.disabled():
        num_qumodes = 1
        num_qubits_per_qumode = 2
        num_qubits = 1

        qmr = bosonic_qiskit.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode
        )
        qbr = qiskit.QuantumRegister(size=num_qubits)
        circuit = bosonic_qiskit.CVCircuit(qmr, qbr)

        circuit.cv_initialize(3, qmr[0])

        circuit.cv_c_d(1, qmr[0], qbr[0], duration=100, unit="ns")

        noise_passes = []

        t1s = np.ones(circuit.num_qubits).tolist()
        t2s = np.ones(circuit.num_qubits).tolist()
        noise_passes.append(RelaxationNoisePass(t1s, t2s))

        photon_loss_rate = 0.02
        noise_passes.append(
            bosonic_qiskit.kraus.PhotonLossNoisePass(
                photon_loss_rates=photon_loss_rate, circuit=circuit, time_unit="ns"
            )
        )

        filename = "tests/test_relaxation_and_photon_loss_noise_passes.gif"
        bosonic_qiskit.animate.animate_wigner(
            circuit,
            animation_segments=200,
            file=filename,
            noise_passes=noise_passes,
        )
        assert Path(filename).is_file()


def test_multi_qumode_loss_probability():
    t = 100  # gate duration in ns
    kappa = 10_000_000 / 3  # photon loss rate per second

    num_qumodes = 2
    num_qubits_per_qumode = 2
    qmr = bosonic_qiskit.QumodeRegister(num_qumodes, num_qubits_per_qumode)
    circuit = bosonic_qiskit.CVCircuit(qmr)

    # This circuit applies a 50/50 beamsplitter followed by its inverse to realize a no-op,
    # starting from the |1,1> state, the first BS gate puts us in state |2,0> + |0,2> / sqrt(2),
    # and the second BS gate returns us to |1,1> if no photons have been lost.
    circuit.cv_initialize(1, qmr[0])
    circuit.cv_initialize(1, qmr[1])
    circuit.cv_bs(np.pi / 4, qmr[0], qmr[1], duration=t, unit="ns")
    circuit.cv_bs(-np.pi / 4, qmr[0], qmr[1], duration=t, unit="ns")

    # Note that we return the state vector for each shot, because at the end of the circuit
    # otherwise, qiskit will randomly return us one of the possible outcomes and we would prefer
    # to average over all of them
    noise_pass = bosonic_qiskit.kraus.PhotonLossNoisePass([kappa], circuit)
    shots = 4000
    state_vectors, *_ = bosonic_qiskit.util.simulate(
        circuit,
        noise_passes=noise_pass,
        per_shot_state_vector=True,
        shots=shots,
        return_fockcounts=False,
    )

    # Read off the resulting state probabilities per shot and aggregate
    probs_dict = {}
    for state_vector in state_vectors:  # ty:ignore[not-iterable]
        _, fock_states = bosonic_qiskit.util.stateread(
            state_vector, 0, num_qumodes, 2**num_qubits_per_qumode, verbose=False
        )
        for qumode_state, _, amplitude in fock_states:
            n_a = cast(int, qumode_state[0])
            n_b = cast(int, qumode_state[1])
            if (n_a, n_b) not in probs_dict:
                probs_dict[(n_a, n_b)] = 0.0

            prob = cast(float, amplitude.real**2)
            probs_dict[(n_a, n_b)] += prob / shots

            # BS preserves total excitation number (2), and photon loss can only decrease it
            if n_a + n_b > 2:
                assert prob == pytest.approx(0), (
                    "This process shouldn't generate photons"
                )

    # Now we'll model what we _expect_ to happen using a markov chain:
    # Let eta be the transmission efficiency, the probability a single photon survives a gate.
    eta = np.exp(-kappa / 1e9 * t)
    rv2 = stats.binom(n=2, p=eta)
    rv1 = stats.binom(n=1, p=eta)

    # Represent the population in the basis {|2,0> or |0,2>, |1,1>, |1,0> or |0,1>, |0,0>}, with
    # our initial state being |1,1>
    initial_probs = np.array([0, 1, 0, 0], dtype=float)

    # The action of the first bs gate will rotate in the subspaces preserving particle number
    u_bs = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    new_probs = u_bs @ initial_probs

    # The dissipation step is applied after each gate discretely.
    dissipation = np.array(
        [
            [rv2.pmf(2), 0, 0, 0],
            [0, rv2.pmf(2), 0, 0],
            [rv2.pmf(1), rv2.pmf(1), rv1.pmf(1), 0],
            [rv2.pmf(0), rv2.pmf(0), rv1.pmf(0), 1],
        ]
    )
    new_probs = dissipation @ new_probs

    # Second beamsplitter gate is the inverse of the first, and then we apply dissipation again
    new_probs = u_bs.T @ new_probs
    expected_probs = dissipation @ new_probs

    # Put the dict above from the simulation in the same basis:
    actual_probs = np.array(
        [
            probs_dict.get((2, 0), 0) + probs_dict.get((0, 2), 0),
            probs_dict.get((1, 1), 0),
            probs_dict.get((1, 0), 0) + probs_dict.get((0, 1), 0),
            probs_dict.get((0, 0), 0),
        ]
    )

    assert actual_probs[0] == pytest.approx(0)  # |2,0> or |0,2> should be impossible

    # Test the remaining outcomes using a chi-square test
    f_obs = (actual_probs[1:] * shots).astype(int)
    f_exp = (expected_probs[1:] * shots).astype(int)
    res = stats.chisquare(f_obs, f_exp)
    assert res.pvalue > 0.05, (
        f"Chi-squre test failed with actual_probs={actual_probs}, expected_probs={expected_probs}, pvalue={res.pvalue}"
    )
