import c2qa
from c2qa.operators import CVOperators
from qiskit.quantum_info.operators.predicates import is_unitary_matrix


class TestUnitary:

    def setup_method(self, method):
        qmr = c2qa.QumodeRegister(1, 1)
        self.ops = CVOperators(qmr)

    def test_bs(self):
        assert is_unitary_matrix(self.ops.bs(1))

    def test_d(self):
        assert is_unitary_matrix(self.ops.d(1))
    
    def test_r(self):
        assert is_unitary_matrix(self.ops.r(1))

    def test_s(self):
        assert is_unitary_matrix(self.ops.s(1))

    def test_s2(self):
        assert is_unitary_matrix(self.ops.s2(1))
