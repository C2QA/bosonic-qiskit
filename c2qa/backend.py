import qiskit
import qiskit_aer

class CVBackend(qiskit_aer.AerSimulator):

    def __init__(self):
        
        target = qiskit.transpiler.Target()
        target.add_instruction(qiskit.circuit.library.IGate(), name="cR")
        target.add_instruction(qiskit.circuit.library.IGate(), name="cD")
        target.add_instruction(qiskit.circuit.library.IGate(), name="D")

        # super().__init__(target=target, basis_gates = ["cR", "cD", "D"])

        configuration = qiskit_aer.backends.backendconfiguration.AerBackendConfiguration.from_dict(qiskit_aer.AerSimulator._DEFAULT_CONFIGURATION)
        
        # configuration.custom_instructions = ["cR", "cD", "D"]
        # super().__init__(configuration = configuration)

        basis_gates = configuration.basis_gates
        basis_gates.extend(["cR", "cD", "D"])
        super().__init__(target=target, basis_gates=basis_gates)

        
        
        # self._target = target
        # self._from_backend = True