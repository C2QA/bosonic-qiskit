import c2qa
import numpy
from pathlib import Path
import qiskit
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, DensityMatrix


def test_trace_out_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        circuit.initialize([0, 1], qr[0])
        circuit.cv_initialize(0, qmr[0])

        state, result, fock_counts = c2qa.util.simulate(circuit)
        trace = c2qa.util.trace_out_qubits(circuit, state)

        assert state.dims() == (2, 2, 2)
        assert trace.dims() == (2, 2)
        prob = trace.probabilities_dict()
        numpy.testing.assert_almost_equal(prob["00"], 1.0)

        # print("Partial trace Fock state zero")
        # print(DensityMatrix(state).probabilities_dict())
        # print(trace.probabilities_dict())


def test_trace_out_one(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(1, qmr[0])

        state, result, fock_counts = c2qa.util.simulate(circuit)
        trace = c2qa.util.trace_out_qubits(circuit, state)

        assert state.dims() == (2, 2, 2)
        assert trace.dims() == (2, 2)
        prob = trace.probabilities_dict()
        numpy.testing.assert_almost_equal(prob["01"], 1.0)

        # print("Partial trace Fock state one")
        # print(DensityMatrix(state).probabilities_dict())
        # print(trace.probabilities_dict())


def test_trace_out_qubit(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qbr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qbr)

        circuit.initialize([1, 0], qbr[0])
        circuit.cv_initialize(1, qmr[0])

        state, result, fock_counts = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state, qbr[0])

        assert state.dims() == (2, 2, 2)
        assert trace.dims() == (2, 2)
        prob = trace.probabilities_dict()
        numpy.testing.assert_almost_equal(prob["01"], 1.0)

        # print("Partial trace Fock state one")
        # print(DensityMatrix(state).probabilities_dict())
        # print(trace.probabilities_dict())


def test_trace_out_qumode(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qbr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qbr)

        circuit.initialize([1, 0], qbr[0])
        circuit.cv_initialize(1, qmr[0])

        state, result, fock_counts = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state, qmr[0])


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
            (state_x, result_x, _),
            (state_y, result_y, _),
            (state_z, result_z, _),
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
        state, result, fock_counts = c2qa.util.simulate(circuit)
        c2qa.util.stateread(
            state,
            numberofqubits=0,
            numberofmodes=num_qumodes,
            cutoff=qmr.cutoff,
            verbose=True,
        )

def test_fockmap(capsys):
    with capsys.disabled():

        # Build rand array of rand dim between 1 and 100, use fockmap to populate initally empty array, and assert that final array is equal to rand array
        for _ in range(10): # Repeat test 10 times
            dim = numpy.random.randint(low=1, high=101)
            randarray = numpy.random.uniform(low=0, high=1, size=(dim, dim))

            testmatrix = numpy.zeros((dim, dim))
            for i in range(dim):
                for j in range (dim):
                    testmatrix = c2qa.util.fockmap(testmatrix, j, i, randarray[i, j])
                     
            assert((testmatrix == randarray).all())

        # Check fockmap using numpy.outer
        matrix = numpy.zeros((4, 4))
        assert((c2qa.util.fockmap(matrix, 0, 0) == numpy.outer([1, 0, 0 ,0], [1, 0, 0, 0])).all()) # |0><0|
        assert((c2qa.util.fockmap(matrix, 1, [3, 2], [1, 0.5]) == (numpy.outer([0, 0, 0 ,1], [0, 1, 0, 0]) + 0.5 * numpy.outer([0, 0, 1 ,0], [0, 1, 0, 0]))).all()) # |3><1| + 0.5|2><1|
        assert((c2qa.util.fockmap(matrix, 1, [3, 2, 1]) == c2qa.util.fockmap(matrix, [1, 1, 1], [3, 2, 1])).all()) # |3><1| + |2><1| + |1><1|

        # Check the types which are accepted for each arg.
        for i in range(10):
            # Nested list, numpy.ndarray
            m_types = [[[0, 0],[0 ,0]], numpy.zeros((2, 2))] 

            # int, list
            fi_types = [0, [1, 0]]

            # int, list
            fo_types = [1, [1, 0]]

            #int, float, complex, empty list, list, numpy.ndarray
            amp_types = [1, 1.0, 1j, [], [1, 1], numpy.array([1, 1])]

            # Generate random indices to test for
            m_index = numpy.random.randint(0, 2)
            fi_index = numpy.random.randint(0, 2)
            fo_index = numpy.random.randint(0, 2)

            if (fi_index == 0) & (fo_index == 0):
                amp_index = numpy.random.randint(0, 4)
            else:
                amp_index = numpy.random.randint(3, 5)
            
            # Assert that output is a numpy.ndarray
            assert(type(c2qa.util.fockmap(m_types[m_index], fi_types[fi_index], fo_types[fo_index], amp_types[amp_index])) == numpy.ndarray)


def test_counts_to_fockcounts(capsys):
    with capsys.disabled():
        for i in range(5):
            # Generate 3 qmreg with random no. of qumodes each
            num_qubits_per_qmr = 2
            num_qm = [numpy.random.randint(low=1, high=3) for _ in range(3)]

            qmr_0 = c2qa.QumodeRegister(num_qm[0], num_qubits_per_qmr)
            qmr_1 = c2qa.QumodeRegister(num_qm[1], num_qubits_per_qmr)
            qmr_2 = c2qa.QumodeRegister(num_qm[2], num_qubits_per_qmr)

#            q_0 = qiskit.QuantumRegister(2)
#            q_1 = qiskit.QuantumRegister(2)

            cr = qiskit.ClassicalRegister(3 * 2) # We will measure 3 qumodes

            circuit = c2qa.CVCircuit(qmr_0, qmr_1, qmr_2, cr)

            # Initialize random state on random qumode within each qmreg
            randstate = []
            for _ in range(3):
                _randstate = [0, 0, 0, 0]
                _randstate[numpy.random.randint(low=0, high=4)] = 1
                randstate.append(_randstate)

            rand_qm = [numpy.random.randint(low=0, high=num_qm[i]) for i in range(3)]

            circuit.cv_initialize(randstate[0], qmr_0[rand_qm[0]])
            circuit.cv_initialize(randstate[1], qmr_1[rand_qm[1]])
            circuit.cv_initialize(randstate[2], qmr_2[rand_qm[2]])

            # # Initalize random state on random qubit within each q
            # randstate = []
            # for _ in range(2):
            #     _randstate = [0, 0]
            #     _randstate[numpy.random.randint(low=0, high=2)] = 1
            #     randstate.append(_randstate)

            # rand_q = [numpy.random.randint(low=0, high=2) for i in range(2)]

            # circuit.initialize(randstate[0], q_0[rand_q[0]])
            # circuit.initialize(randstate[1], q_1[rand_q[1]])

            # Measure the circuit with random bits on classical register
            rand_bit = numpy.random.permutation([0, 1, 2]) #[3])

            circuit.cv_measure(qmr_0[rand_qm[0]], cr[rand_bit[0] * 2:(rand_bit[0] + 1) * 2])
            circuit.cv_measure(qmr_1[rand_qm[1]], cr[rand_bit[1] * 2:(rand_bit[1] + 1) * 2])
            circuit.cv_measure(qmr_2[rand_qm[2]], cr[rand_bit[2] * 2:(rand_bit[2] + 1) * 2 ])

            # circuit.measure(q_0[rand_q[0]], cr[rand_bit[3] * 2])
            # circuit.measure(q_1[rand_q[1]], cr[rand_bit[3] * 2 + 1])

            # Generate regs arg for cv_fockcounts
            _regs = [qmr_0[rand_qm[0]], qmr_1[rand_qm[1]], qmr_2[rand_qm[2]]] #[q_0[rand_q[0]], q_1[rand_q[1]]]
            regs = []
            for i in rand_bit:
                # if i == 3: # Since q regs are always measured on consecutive bits, append both at same time
                #     regs.append(_regs[3])
                #     regs.append(_regs[4])
                # else:
                    regs.append(_regs[i])

            # Compare output of cv_fockcounts vs counts_to_fockcounts
            _, result, fock_counts = c2qa.util.simulate(circuit, return_fockcounts=True)
#            print(result.get_counts(), c2qa.util._final_qumode_mapping(circuit))
            assert(fock_counts == c2qa.util.cv_fockcounts(result.get_counts(), regs))
            
            
def test_avg_photon_num(capsys):
    with capsys.disabled():
        for _ in range(5): # Repeat test 5 times
            # Decimals
            decimals = numpy.random.randint(1, 6)
            
            # Generate random vector
            dim = numpy.random.randint(1, 11)
            vector = numpy.random.uniform(-1, 1, dim) + 1.j * numpy.random.uniform(-1, 1, dim)
            
            # Compute magnitude of each element within vector, and norm of vector
            element_norm = numpy.multiply(vector, numpy.conjugate(vector))
            norm = numpy.sum(element_norm)

            # Dot product between number operator and magnitude
            avg_num = numpy.mean(numpy.dot(element_norm, numpy.array(range(dim))))/norm
            
            # Average photon number of statevector, density matrix, and random vector must all match
            assert(c2qa.util.avg_photon_num(Statevector(vector), decimals) == c2qa.util.avg_photon_num(DensityMatrix(vector), decimals) == round(avg_num.real, decimals))
            
            