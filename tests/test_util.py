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
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

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


def test_animate(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=4)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        dist = 3

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        wigner_filename = "tests/displacement.gif"
        c2qa.util.animate_wigner(
            circuit,
            qubit=qr[0],
            cbit=cr[0],
            file=wigner_filename,
            axes_min=-8,
            axes_max=8,
            animation_segments=5,
            processes=1,
        )
        assert Path(wigner_filename).is_file()


@pytest.mark.skip(reason="GitHub actions build environments do not have ffmpeg")
def test_calibration_animate(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=6)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        dist = 3

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        circuit.cv_d(1j * dist, qmr[0])
        circuit.cv_cnd_d(-dist, dist, qr[0], qmr[0])
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
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
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
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

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
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
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
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        circuit.h(qr[0])

        states, result = c2qa.util.simulate(circuit, per_shot_state_vector=True)
        wigner = c2qa.util.wigner_mle(states, circuit.cutoff)
        assert wigner is not None
        print(wigner)


def test_stateread(capsys):
    with capsys.disabled():
        num_qumodes=2
        qubits_per_mode=3        

        qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode, name="qmr")

        circuit = c2qa.CVCircuit(qmr)

        circuit.cv_initialize(2, qmr[0])
        circuit.cv_initialize(0, qmr[1])
        state, result = c2qa.util.simulate(circuit)
        occs = c2qa.util.stateread(state, numberofqubits=0, numberofmodes=num_qumodes, cutoff=circuit.cutoff, verbose=True)
