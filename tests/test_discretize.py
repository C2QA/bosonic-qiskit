import c2qa
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

        assert discretized_params[0] == (theta / 2)
        assert discretized_params[1] == (beta / 2)


def test_cv_c_schwinger(capsys):
    """The cv_c_schwinger gate should discretize the first param, but the others not"""
    with capsys.disabled():
        pass
