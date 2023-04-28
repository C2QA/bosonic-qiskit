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


def test_parallel_bosonic_qiskit(capsys):
    with capsys.disabled():
        print()

        import c2qa
        import matplotlib.pyplot as plt
        import time
        import qiskit
        from qiskit_aer.noise.passes.relaxation_noise_pass import RelaxationNoisePass


        qbr=qiskit.QuantumRegister(2)
        qmr = c2qa.QumodeRegister(num_qumodes = 3, num_qubits_per_qumode = 3)
        init_circuit = c2qa.CVCircuit(qmr,qbr)
        init_circuit.cv_initialize(4, qmr[0])
        init_circuit.h(qbr[0])
        init_circuit.cv_initialize(1, qmr[0])
        for i in range(20):
            init_circuit.cv_c_bs(i, qmr[0], qmr[1], qbr[0], duration=2.25, unit="us")
            init_circuit.cv_c_bs(i, qmr[0], qmr[1], qbr[0], duration=2.25, unit="us")
            init_circuit.rx(-2 * i,qbr[0])
            init_circuit.cv_c_bs(i*2, qmr[1], qmr[2], qbr[1], duration=2.25, unit="us")
            init_circuit.cv_c_bs(i*3, qmr[1], qmr[2], qbr[1], duration=2.25, unit="us")
            init_circuit.rx(-2 * i*4,qbr[1])
        init_circuit.measure_all()
        qb_T1 = 2e-4  # 200 us - https://arxiv.org/pdf/2211.09116.pdf
        qm_T1 = 100000

        list = []
        for i in range(len(qmr)):
            for j in range(qmr.num_qubits_per_qumode):
                list.append(qm_T1)

        qb_noise_pass = RelaxationNoisePass(list, list) # T1, T2

        # Transpile for simulator
        # circuit_compiled = qb_noise_pass(init_circuit)
        simulator = qiskit.providers.aer.AerSimulator()
        circuit_compiled_n = qiskit.transpile(init_circuit, simulator)

        def run(shots,max_parallel_threads):
            # Run
            time1=time.time()
            # _, results = c2qa.util.simulate(init_circuit, max_parallel_threads=max_parallel_threads,
            #                                 shots=shots, add_save_statevector=False)#, noise_pass=qb_noise_pass)
            # print(results.get_counts())
            results=simulator.run(circuit_compiled_n, method="statevector",max_parallel_shots=0,shots=shots, max_parallel_threads=max_parallel_threads).result() #max_parallel_experiments=0,method="statevector",
            # print(results)
            return time.time()-time1
        # run(10,1)
        shotss=[1,10,100]

        one_thread_times = [run(shots,1) for shots in shotss]
        print(f"1 thread {one_thread_times}")
        # plt.plot(shotss, one_thread_times, 'x',label="1")

        two_thread_times = [run(shots,2) for shots in shotss]
        print(f"2 thread {two_thread_times}")
        # plt.plot(shotss, two_thread_times, 'x',label="2")

        four_thread_times = [run(shots,4) for shots in shotss]
        print(f"4 thread {four_thread_times}")
        # plt.plot(shotss, four_thread_times, 'x',label="4")

        eight_thread_times = [run(shots,8) for shots in shotss]
        print(f"8 thread  {eight_thread_times}")
        # plt.plot(shotss, eight_thread_times, 'x',label="8")

        # plt.legend()
        # plt.xlabel("shots")
        # plt.ylabel("seconds")
        # plt.plot(shotss,[run(shots,0) for shots in shotss])
        # plt.xscale('log')
        # plt.yscale('log')
        # %timeit run(4)
        # %timeit run(1)
        # %timeit run(0)

        # Github Windows, Linux runners have 2 cores, macOS has 3 cores.
        # It doens't make much sense to test more than 2 threads...
        all_less = True
        for i in range(1, len(shotss)): # Can't parallelize only one shot
            if two_thread_times[i] > one_thread_times[i]:
                all_less = False
            # if four_thread_times[i] > two_thread_times[i]:
            #     all_less = False
            # if eight_thread_times[i] > four_thread_times[i]:
            #     all_less = False
        assert all_less