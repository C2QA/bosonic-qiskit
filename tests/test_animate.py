from pathlib import Path


import numpy
import pytest
import qiskit


import c2qa


def __build_subcircuit():
    # Define Hamiltonian parameters
    omega_R = 2
    omega_Q = 5
    chi = 0.1

    # Set number of qubits per qumode
    num_qubits_per_qumode = 3

    # Choose alpha for coherent state
    alpha = 1

    # Choose total animation time
    total_time = 1 * 2 * numpy.pi / omega_R

    # Create new circuit
    qmr = c2qa.QumodeRegister(
        num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode
    )
    qbr = qiskit.QuantumRegister(1)
    U_JC = c2qa.CVCircuit(qmr, qbr)

    # Append U_R
    U_JC.cv_r(-omega_R * total_time, qmr[0])
    # Append U_Q
    U_JC.rz(omega_Q * total_time, qbr[0])
    # Append U_\chi -- KS: this needs to be updated to reflect naming conventions in manuscript
    U_JC.cv_c_r(-chi * total_time / 2, qmr[0], qbr[0])
    # Compile this circuit into a single parameterized gate
    U_JC = U_JC.to_gate(label="U_JC")

    # Instantiate the circuit and initialize the qubit to the '0' state.
    circuit_0 = c2qa.CVCircuit(qmr, qbr)
    circuit_0.initialize([1, 0], qbr)

    # Squeeze so we can visually see rotation
    circuit_0.cv_sq(0.5, qmr[0])

    # Now initialize the qumode in a coherent state
    # cutoff = 2**num_qubits_per_qumode
    # coeffs = [numpy.exp(-numpy.abs(alpha)**2/2)*alpha**n/(numpy.sqrt(numpy.math.factorial(n))) for n in range(0,cutoff)]
    # circuit_0.cv_initialize(coeffs,qmr[0])

    # Append time evolution unitary
    circuit_0.append(U_JC, qmr[0] + [qbr[0]])
    # circuit_0.assign_parameters({dt : total_time})

    # dt = total_time
    # # Append U_R
    # circuit_0.cv_r(-omega_R*dt,qmr[0])
    # # Append U_Q
    # circuit_0.rz(omega_Q*dt,qbr[0])
    # # Append U_\chi -- KS: this needs to be updated to reflect naming conventions in manuscript
    # circuit_0.cv_c_r(-chi*dt/2,qmr[0],qbr[0])

    # Compile this circuit into a single parameterized gate
    # U_JC = U_JC.to_gate(label='U_JC')

    # # Now repeat the above steps for a qubit initialized to the '1' state:
    # circuit_1 = c2qa.CVCircuit(qmr,qbr)
    # circuit_1.initialize([0,1], qbr)
    # circuit_1.cv_d(alpha,qmr[0])
    # circuit_1.append(U_JC,qmr[0] + [qbr[0]])
    # circuit_1 = circuit_1.assign_parameters({dt : total_time})

    return circuit_0


def test_animate_subcircuit_one_gate(capsys):
    """Test animating a circuit with a composite gate built from another circuit.
    Composite gate borrowed from Jaynes-Cummings model tutorial"""

    with capsys.disabled():
        circuit = __build_subcircuit()

        # Animate wigner function of each circuit
        c2qa.animate.animate_wigner(
            circuit, file="tests/composite_gate.gif", animation_segments=20
        )


def test_animate_subcircuit_sequential(capsys):
    """Test animating a circuit with a composite gate built from another circuit.
    Composite gate borrowed from Jaynes-Cummings model tutorial"""

    with capsys.disabled():
        circuit = __build_subcircuit()

        # Animate wigner function of each circuit
        c2qa.animate.animate_wigner(
            circuit,
            file="tests/sequential_subcircuit.gif",
            animation_segments=20,
            sequential_subcircuit=True,
        )


def test_animate_parameterized(capsys):
    with capsys.disabled():
        a = qiskit.circuit.Parameter("𝛼")

        qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=4)
        qbr = qiskit.QuantumRegister(1)
        cbr = qiskit.ClassicalRegister(1)

        minimal_circuit = c2qa.CVCircuit(qmr, qbr, cbr)

        minimal_circuit.h(qbr[0])

        minimal_circuit.cv_c_d(1j * a, qmr[0], qbr[0])

        bound_circuit = minimal_circuit.assign_parameters({a: 2})

        wigner_filename = "tests/animate_parameterized.apng"
        c2qa.animate.animate_wigner(
            bound_circuit,
            qubit=qbr[0],
            cbit=cbr[0],
            file=wigner_filename,
            axes_min=-8,
            axes_max=8,
            animation_segments=5,
            processes=1,
            shots=25,
        )
        assert Path(wigner_filename).is_file()


def test_animate_gif(capsys):
    with capsys.disabled():
        __animate_with_cbit("tests/displacement.gif")


def test_animate_apng(capsys):
    with capsys.disabled():
        __animate_with_cbit("tests/displacement.apng")


def __animate_with_cbit(filename: str):
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=4)
    qr = qiskit.QuantumRegister(size=1)
    cr = qiskit.ClassicalRegister(size=1)
    circuit = c2qa.CVCircuit(qmr, qr, cr)

    dist = 3

    circuit.initialize([1, 0], qr[0])
    circuit.cv_initialize(0, qmr[0])

    circuit.h(qr[0])
    circuit.cv_c_d(dist, qmr[0], qr[0])

    c2qa.animate.animate_wigner(
        circuit,
        qubit=qr[0],
        cbit=cr[0],
        file=filename,
        axes_min=-8,
        axes_max=8,
        animation_segments=5,
        processes=1,
        shots=25,
    )
    assert Path(filename).is_file()


def __animate_without_cbit(filename: str, trace: bool = False):
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=4)
    qr = qiskit.QuantumRegister(size=1)
    circuit = c2qa.CVCircuit(qmr, qr)

    dist = 3

    circuit.initialize([1, 0], qr[0])
    circuit.cv_initialize(0, qmr[0])

    circuit.h(qr[0])
    circuit.cv_c_d(dist, qmr[0], qr[0])

    c2qa.animate.animate_wigner(
        circuit,
        qubit=qr[0],
        file=filename,
        axes_min=-8,
        axes_max=8,
        animation_segments=5,
        processes=1,
        shots=25,
        trace=trace,
    )
    assert Path(filename).is_file()


def test_animate_with_trace(capsys):
    with capsys.disabled():
        __animate_without_cbit("tests/animate_with_trace.gif", True)


def test_animate_without_trace(capsys):
    with capsys.disabled():
        __animate_without_cbit("tests/animate_without_trace.gif", False)


def test_calibration_animate_mp4(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=6)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        dist = 3

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])
        circuit.cv_d(1j * dist, qmr[0])
        circuit.cv_c_d(-dist, qmr[0], qr[0])
        circuit.cv_d(-1j * dist, qmr[0])

        c2qa.animate.animate_wigner(
            circuit,
            qubit=qr[0],
            cbit=cr[0],
            file="tests/displacement.gif",
            axes_min=-8,
            axes_max=8,
            animation_segments=48,
            shots=128,
        )
