import c2qa
import numpy
from pathlib import Path
import pytest
import qiskit
from qiskit.visualization import plot_histogram


def test_partial_trace_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        circuit.initialize([0, 1], qr[0])
        circuit.cv_initialize(0, qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)

        assert state.dims() == (2, 2, 2)
        assert trace.dims() == (2, 2)
        prob = trace.probabilities_dict()
        numpy.testing.assert_almost_equal(prob["00"], 1.0)

        # print("Partial trace Fock state zero")
        # print(DensityMatrix(state).probabilities_dict())
        # print(trace.probabilities_dict())


def test_partial_trace_one(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(1, qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)

        assert state.dims() == (2, 2, 2)
        assert trace.dims() == (2, 2)
        prob = trace.probabilities_dict()
        numpy.testing.assert_almost_equal(prob["01"], 1.0)

        # print("Partial trace Fock state one")
        # print(DensityMatrix(state).probabilities_dict())
        # print(trace.probabilities_dict())


def test_plot_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 1.0

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])

        state, _ = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)
        c2qa.util.plot_wigner(circuit, trace, file="tests/zero.png", trace=False)


def test_plot_one(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(1, qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        # print("Qumode initialized to one:")
        # print(state)
        c2qa.util.plot_wigner(circuit, state, file="tests/one.png")


def test_animate_composite_gate(capsys):
    """ Test animating a circuit with a composite gate built from another circuit.
    Composite gate borrowed from Jaynes-Cummings model tutorial """

    with capsys.disabled():
        # Define Hamiltonian parameters
        omega_R = 2
        omega_Q = 5
        chi = 0.1

        # Set number of qubits per qumode
        num_qubits_per_qumode = 3

        # Choose alpha for coherent state
        alpha = 1

        # Choose total animation time
        total_time = 1*2*numpy.pi/omega_R

        # First construct circuit_qubit0
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode)
        qbr = qiskit.QuantumRegister(1)

        # Create new circuit
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode)
        qbr = qiskit.QuantumRegister(1)
        U_JC = c2qa.CVCircuit(qmr,qbr)
        # Append U_R
        U_JC.cv_r(-omega_R*total_time,qmr[0])
        # Append U_Q
        U_JC.rz(omega_Q*total_time,qbr[0])
        # Append U_\chi -- KS: this needs to be updated to reflect naming conventions in manuscript
        U_JC.cv_c_r(-chi*total_time/2,qmr[0],qbr[0])
        # Compile this circuit into a single parameterized gate
        U_JC = U_JC.to_gate(label='U_JC')

        # Instantiate the circuit and initialize the qubit to the '0' state.
        circuit_0 = c2qa.CVCircuit(qmr,qbr)
        circuit_0.initialize([1,0], qbr)

        # Now initialize the qumode in a coherent state
        cutoff = 2**num_qubits_per_qumode
        coeffs = [numpy.exp(-numpy.abs(alpha)**2/2)*alpha**n/(numpy.sqrt(numpy.math.factorial(n))) for n in range(0,cutoff)]
        circuit_0.cv_initialize(coeffs,qmr[0])


        # Append time evolution unitary
        circuit_0.append(U_JC,qmr[0] + [qbr[0]]);
        # circuit_0.bind_parameters({dt : total_time})


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
        # circuit_1.append(U_JC,qmr[0] + [qbr[0]]);
        # circuit_1 = circuit_1.bind_parameters({dt : total_time})

        # Animate wigner function of each circuit
        c2qa.util.animate_wigner(circuit_0,file="tests/composite_gate.gif", animation_segments = 1000)


def test_animate_gif(capsys):
    with capsys.disabled():
        __animate("tests/displacement.gif")


def test_animate_apng(capsys):
    with capsys.disabled():
        __animate("tests/displacement.apng")


def __animate(filename: str):
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=4)
    qr = qiskit.QuantumRegister(size=1)
    cr = qiskit.ClassicalRegister(size=1)
    circuit = c2qa.CVCircuit(qmr, qr, cr)

    dist = 3

    circuit.initialize([1, 0], qr[0])
    circuit.cv_initialize(0, qmr[0])

    circuit.h(qr[0])
    circuit.cv_c_d(dist, qmr[0], qr[0])

    c2qa.util.animate_wigner(
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


@pytest.mark.skip(reason="GitHub actions build environments do not have ffmpeg")
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

        c2qa.util.animate_wigner(
            circuit,
            qubit=qr[0],
            cbit=cr[0],
            file="tests/displacement.mp4",
            axes_min=-8,
            axes_max=8,
            animation_segments=48,
            shots=128,
        )


def test_plot_wigner_projection(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 1.0

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])
        # circuit.cv_d(dist, qmr[0])

        c2qa.util.plot_wigner_projection(circuit, qr[0], file="tests/interference.png")


def test_measure_all_xyz(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = numpy.sqrt(2)

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])

        (
            (state_x, result_x),
            (state_y, result_y),
            (state_z, result_z),
        ) = c2qa.util.measure_all_xyz(circuit)

        print("state_x.probabilities_dict()")
        print(state_x.probabilities_dict())

        print("result_x.get_counts() calculated probabilities")
        print(c2qa.util.get_probabilities(result_x))

        print("result_x.to_dict()")
        print(result_x.to_dict())

        plot_histogram(result_x.get_counts(), title="X", figsize=(9, 7)).savefig(
            "tests/plot_histogram_x.png"
        )
        plot_histogram(result_y.get_counts(), title="Y", figsize=(9, 7)).savefig(
            "tests/plot_histogram_y.png"
        )
        plot_histogram(result_z.get_counts(), title="Z", figsize=(9, 7)).savefig(
            "tests/plot_histogram_z.png"
        )


def test_cat_state_wigner_plot(capsys):
    with capsys.disabled():
        num_qubits_per_qumode = 4
        dist = 2

        qmr = c2qa.QumodeRegister(
            num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode
        )
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])

        # conditional_state_vector=True will return two state vectors, one for 0 and 1 classical register value
        state, _ = c2qa.util.simulate(circuit, conditional_state_vector=True)
        even_state = state["0x0"]
        odd_state = state["0x1"]

        wigner_filename = "tests/cat_wigner_even.png"
        c2qa.util.plot_wigner(
            circuit,
            even_state,
            file=wigner_filename,
            trace=True,
            axes_min=-6,
            axes_max=6,
        )
        assert Path(wigner_filename).is_file()

        wigner_filename = "tests/cat_wigner_odd.png"
        c2qa.util.plot_wigner(
            circuit,
            odd_state,
            file=wigner_filename,
            trace=True,
            axes_min=-6,
            axes_max=6,
        )
        assert Path(wigner_filename).is_file()

        # # Need to recreate circuit state prior to measure collapsing qubit state for projections
        # qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode)
        # qr = qiskit.QuantumRegister(size=1)
        # cr = qiskit.ClassicalRegister(size=1)
        # circuit = c2qa.CVCircuit(qmr, qr, cr)
        # circuit.initialize(state)

        # wigner_filename = "tests/repeat_projection_wigner.png"
        # c2qa.util.plot_wigner_projection(circuit, qr[0], file=wigner_filename)
        # assert Path(wigner_filename).is_file()


def test_wigner_mle(capsys):
    with capsys.disabled():
        num_qubits_per_qumode = 4
        dist = 2

        qmr = c2qa.QumodeRegister(
            num_qumodes=1, num_qubits_per_qumode=num_qubits_per_qumode
        )
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])
        circuit.h(qr[0])

        states, result = c2qa.util.simulate(circuit, per_shot_state_vector=True)
        wigner = c2qa.util.wigner_mle(states, circuit.cutoff)
        assert wigner is not None
        print(wigner)


def test_stateread(capsys):
    with capsys.disabled():
        num_qumodes = 2
        qubits_per_mode = 3

        qmr = c2qa.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode, name="qmr"
        )

        circuit = c2qa.CVCircuit(qmr)

        circuit.cv_initialize(2, qmr[0])
        circuit.cv_initialize(0, qmr[1])
        state, result = c2qa.util.simulate(circuit)
        c2qa.util.stateread(
            state,
            numberofqubits=0,
            numberofmodes=num_qumodes,
            cutoff=circuit.cutoff,
            verbose=True,
        )
