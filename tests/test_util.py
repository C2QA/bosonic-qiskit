from qiskit.quantum_info.states.densitymatrix import DensityMatrix
import c2qa
import matplotlib.pyplot as plt
import numpy
from pathlib import Path
import pytest
import scipy.special as ssp
import qiskit
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit.providers.aer.library.save_instructions.save_statevector import save_statevector

def test_partial_trace_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
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
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
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
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
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
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(1, qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        # print("Qumode initialized to one:")
        # print(state)
        c2qa.util.plot_wigner(circuit, state, file="tests/one.png")


@pytest.mark.skip(reason="GitHub actions build environments do not have ffmpeg")
def test_animate(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr, animation_segments=10)

        dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        circuit.cv_d(1j * dist, qmr[0])
        circuit.cv_cnd_d(-dist, dist, qr[0], qmr[0])
        circuit.cv_d(-1j * dist, qmr[0])
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])

        backend = qiskit.Aer.get_backend("statevector_simulator")
        job = qiskit.execute(circuit, backend)
        result = job.result()

        c2qa.util.animate_wigner(
            circuit, result, file="tests/displacement.mp4"
        )


def test_plot_wigner_projection(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
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


def test_simulate_plot(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=3)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 1.0

        # qr[0] will init to zero
        # circuit.cv_initialize(0, qmr[0])
        # circuit.initialize([0,1], qr[0])
        circuit.initialize([0,1], qmr[0][0])

        # circuit.h(qr[0])
        # circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        state, result = c2qa.util.simulate(circuit)

        print(state)
        plot_state_city(state).savefig("tests/plot_state_city.png")
        plot_histogram(result.get_counts(), figsize=(9, 7)).savefig("tests/plot_histogram.png")


def test_measure_all_xyz(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = numpy.sqrt(2)

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        (state_x, result_x), (state_y, result_y), (state_z, result_z) = c2qa.util.measure_all_xyz(circuit)

        print("state_x.probabilities_dict()")
        print(state_x.probabilities_dict())

        print("result_x.get_counts() calculated probabilities")
        print(c2qa.util.get_probabilities(result_x))

        print("result_x.to_dict()")
        print(result_x.to_dict())

        plot_histogram(result_x.get_counts(), title="X", figsize=(9, 7)).savefig("tests/plot_histogram_x.png")
        plot_histogram(result_y.get_counts(), title="Y", figsize=(9, 7)).savefig("tests/plot_histogram_y.png")
        plot_histogram(result_z.get_counts(), title="Z", figsize=(9, 7)).savefig("tests/plot_histogram_z.png")

def test_repeat_until_success(capsys):
    with capsys.disabled():
        success = False
        num_qubits_per_qumode = 4

        while not success:
            qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=num_qubits_per_qumode)
            qr = qiskit.QuantumRegister(size=1)
            cr = qiskit.ClassicalRegister(size=1)
            circuit = c2qa.CVCircuit(qmr, qr, cr)

            dist = 2

            circuit.initialize([1, 0], qr[0])
            circuit.cv_initialize(0, qmr[0])

            circuit.h(qr[0])
            circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
            circuit.h(qr[0])
            # save_statevector(circuit)
            circuit.measure(qr[0], cr[0])

            state, result = c2qa.util.simulate(circuit, shots=1, add_save_statevector=True)
            counts = result.get_counts()
            print(counts)
            success = "0" in counts and counts["0"] == 1

            if success:
                wigner_filename = "tests/repeat_wigner.png"
                c2qa.util.plot_wigner(circuit, state, file=wigner_filename, trace=True, axes_min=-6, axes_max=6)
                assert Path(wigner_filename).is_file()

                # # Need to recreate circuit state prior to measure collapsing qubit state for projections
                # qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=num_qubits_per_qumode)
                # qr = qiskit.QuantumRegister(size=1)
                # cr = qiskit.ClassicalRegister(size=1)
                # circuit = c2qa.CVCircuit(qmr, qr, cr)
                # circuit.initialize(state)

                # wigner_filename = "tests/repeat_projection_wigner.png"
                # c2qa.util.plot_wigner_projection(circuit, qr[0], file=wigner_filename)
                # assert Path(wigner_filename).is_file()