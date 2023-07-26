import c2qa
import pytest
import qiskit
import random


def test_cv_c_d(capsys):
    """The cv_c_d gate should discretize all params (i.e., default behavior)"""
    with capsys.disabled():
        num_qumodes = 2
        num_qubits_per_qumode = 2
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        qr = qiskit.QuantumRegister(2)
        circuit = c2qa.CVCircuit(qmr, qr)

        theta = random.random()
        beta = -theta
        circuit.cv_c_d(theta=theta, beta=beta, qumode=qmr[0], qubit=qr[0])

        gate = circuit.data[0].operation
        total_steps = 2
        discretized_params = gate.calculate_segment_params(current_step=1, total_steps=total_steps, keep_state=True)

        print(f"Original theta={theta}")
        print(f"Discretized params {discretized_params}")

        assert discretized_params[0] == (theta / total_steps)
        assert discretized_params[1] == (beta / total_steps)


def test_cv_c_schwinger(capsys):
    """The cv_c_schwinger gate should discretize the first param, but the others not"""
    with capsys.disabled():
        num_qumodes = 2
        num_qubits_per_qumode = 2
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        qr = qiskit.QuantumRegister(2)
        circuit = c2qa.CVCircuit(qmr, qr)

        beta = random.random()
        theta_1 = random.random()
        phi_1 = random.random()
        theta_2 = random.random()
        phi_2 = random.random()
        circuit.cv_c_schwinger([beta, theta_1, phi_1, theta_2, phi_2], qmr[0], qmr[1], qr[0])

        gate = circuit.data[0].operation
        total_steps = 2
        discretized_params = gate.calculate_segment_params(current_step=1, total_steps=total_steps, keep_state=True)

        print(f"Original params {(beta, theta_1, phi_1, theta_2, phi_2)}")
        print(f"Discretized params {discretized_params}")

        assert discretized_params[0] == (beta / total_steps)
        assert discretized_params[1] == theta_1
        assert discretized_params[2] == phi_1
        assert discretized_params[3] == theta_2
        assert discretized_params[4] == phi_2


@pytest.mark.skip(reason="Enable and test manually with debug breakpoint in__calculate_segment_params to ensure only first param is discretized")
def test_cv_c_schwinger_animate(capsys):
    """The cv_c_schwinger gate should discretize the first param, but the others not"""
    with capsys.disabled():
        num_qumodes = 2
        num_qubits_per_qumode = 2
        qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
        qr = qiskit.QuantumRegister(2)
        circuit = c2qa.CVCircuit(qmr, qr)

        beta = random.random()
        theta_1 = random.random()
        phi_1 = random.random()
        theta_2 = random.random()
        phi_2 = random.random()
        circuit.cv_c_schwinger([beta, theta_1, phi_1, theta_2, phi_2], qmr[0], qmr[1], qr[0])

        c2qa.animate.animate_wigner(
            circuit,
            file="tests/test_cv_c_schwinger_animate.gif",
            axes_min=-8,
            axes_max=8,
            animation_segments=2,
            shots=1,
        )


def test_discretize_with_pershot_statevector(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(1, 3)
        creg = qiskit.ClassicalRegister(3)
        circ = c2qa.CVCircuit(qmr, creg)
        circ.cv_initialize(7, qmr[0])

        circ.cv_delay(duration=100, qumode=qmr[0], unit="ns")
        circ.cv_measure(qmr[0], creg)

        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rates=0.02, circuit=circ, time_unit="ns")
        results = c2qa.util.simulate(circ, noise_passes=noise_pass, discretize=True, shots=2, per_shot_state_vector=True)

        for state, result, accumulated_counts, fock_counts in results:
            print(accumulated_counts)
            assert result.success


def test_accumulated_counts(capsys):
    def simulate_test(discretize: bool):
        qmr = c2qa.QumodeRegister(1, 3)
        anc = qiskit.circuit.AncillaRegister(1)
        cr = qiskit.circuit.ClassicalRegister(1)
        circ = c2qa.CVCircuit(qmr, anc, cr)

        circ.initialize([1, 0], anc[0]) # Ancilla in |g>
        circ.cv_initialize(3, qmr[0]) # Qumode in |3>

        # Photon number parity circuit
        circ.h(anc[0])
        circ.cv_c_r(3.14 / 2, qmr[0], anc[0], duration=1, unit="µs")
        circ.h(anc[0])
        circ.measure(anc[0], cr[0])

        # Simulate
        noise_pass = c2qa.kraus.PhotonLossNoisePass(photon_loss_rates=0.1, circuit=circ, time_unit="µs")
        if discretize:
            results = c2qa.util.simulate(circ, noise_passes=noise_pass, discretize=discretize, shots=3000, add_save_statevector=False)

            for state, result, accumulated_counts, fock_counts in results:
                print("##############")
                print(f"Result counts: {result.get_counts()}")
                print(f"Accumulated counts: {accumulated_counts}")
                print(f"Fock counts: {fock_counts}")
                assert result.success
        else:
            state, result, accumulated_counts, fock_counts = c2qa.util.simulate(circ, noise_passes=noise_pass, discretize=discretize, shots=3000)
            print("##############")
            print(f"Result counts: {result.get_counts()}")
            print(f"Accumulated counts: {accumulated_counts}")
            print(f"Fock counts: {fock_counts}")
            assert result.success

    with capsys.disabled():
        print()
        print("NOT DISCRETIZED")
        simulate_test(discretize=False)
        print()
        print("DISCRETIZED")
        simulate_test(discretize=True)
