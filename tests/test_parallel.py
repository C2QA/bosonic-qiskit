import os

def test_parallel_qasmsimulator(capsys):
    with capsys.disabled():
        print()
        print("-------------")
        print("QasmSimulator")
        from qiskit.providers.aer import QasmSimulator
        __simulate(QasmSimulator())

def test_parallel_aersimulator(capsys):
    with capsys.disabled():
        print()
        print("------------")
        print("AerSimulator")
        from qiskit.providers.aer import AerSimulator
        __simulate(AerSimulator())



def __simulate(simulator):        
    # mpt = 8
    # ncores = str(mpt)
    # os.environ["OMP_NUM_THREADS"] = ncores
    # os.environ["OPENBLAS_NUM_THREADS"] = ncores
    # os.environ["MKL_NUM_THREADS"] = ncores
    # os.environ["VECLIB_MAXIMUM_THREADS"] = ncores
    # os.environ["NUMEXPR_NUM_THREADS"] = ncores

    from qiskit import execute
    from qiskit.circuit.library import QFT
    # from qiskit.providers.aer import QasmSimulator
    from qiskit.providers.aer.noise import NoiseModel, amplitude_damping_error

    # simulator = QasmSimulator()
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(amplitude_damping_error(1e-3), ['h'])

    qubits = 11
    qft = QFT(qubits).inverse()
    for q in range(qubits):
        qft.h(q)
    qft.measure_all()
    # print(qft)
    backend_options = {}
    backend_options['method'] = 'statevector'

    faster_count = 0
    test_runs = 100
    for _ in range(test_runs):
        result = execute(qft, simulator, shots=10000, max_parallel_threads=8, backend_options=backend_options).result()
        parallel_time = result.to_dict()['time_taken']
        # print(f"parallel statevector: {parallel_time}")
        result = execute(qft, simulator, shots=10000, max_parallel_threads=1, backend_options=backend_options).result()
        serial_time = result.to_dict()['time_taken']
        # print(f"serial statevector: {serial_time}")
        if parallel_time < serial_time:
            faster_count += 1
    print(f"Faster {faster_count} times out of {test_runs}")
    # assert serial_time > parallel_time