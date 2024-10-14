from pathlib import Path


import c2qa
import numpy
import qiskit


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

        state, result, fock_counts = c2qa.util.simulate(circuit)
        trace = c2qa.util.trace_out_qubits(circuit, state)
        c2qa.wigner.plot_wigner(circuit, trace, file="tests/zero.png", trace=False)


def test_plot_one(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(1, qmr[0])

        state, result, fock_counts = c2qa.util.simulate(circuit)
        # print("Qumode initialized to one:")
        # print(state)
        c2qa.wigner.plot_wigner(circuit, state, file="tests/one.png")


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

        c2qa.wigner.plot_wigner_projection(
            circuit, qr[0], file="tests/interference.png"
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
        state, result, fock_counts = c2qa.util.simulate(
            circuit, conditional_state_vector=True
        )
        even_state = state["0x0"]
        odd_state = state["0x1"]

        wigner_filename = "tests/cat_wigner_even.png"
        c2qa.wigner.plot_wigner(
            circuit,
            even_state,
            file=wigner_filename,
            trace=True,
            axes_min=-6,
            axes_max=6,
        )
        assert Path(wigner_filename).is_file()

        wigner_filename = "tests/cat_wigner_odd.png"
        c2qa.wigner.plot_wigner(
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
        # c2qa.wigner.plot_wigner_projection(circuit, qr[0], file=wigner_filename)
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

        state, result, fock_counts = c2qa.util.simulate(
            circuit, per_shot_state_vector=True
        )
        wigner = c2qa.wigner.wigner_mle(state)
        assert wigner is not None
        print(wigner)


def test_plot_wigner_snapshot(capsys):
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

        circuit.cv_snapshot()

        circuit.h(qr[0])

        circuit.cv_snapshot()
        circuit.cv_c_d(dist, qmr[0], qr[0])

        circuit.cv_snapshot()

        circuit.h(qr[0])

        circuit.cv_snapshot()

        state, result, fock_counts = c2qa.util.simulate(circuit)

        c2qa.wigner.plot_wigner_snapshot(circuit, result, "tests")


def test_plot_zero_contour(capsys):
    with capsys.disabled():
        data = numpy.zeros((200, 200)).tolist()
        filename = "tests/test_plot_zero_contour.png"
        c2qa.wigner.plot(data, file=filename)
        assert Path(filename).is_file()
