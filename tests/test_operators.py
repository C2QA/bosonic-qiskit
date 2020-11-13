import c2qa
import numpy
from c2qa.operators import CVOperators
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
import random

class TestUnitary:
    """Verify operators are unitary"""

    def setup_method(self, method):
        qmr = c2qa.QumodeRegister(2, 2)
        self.ops = CVOperators(qmr)

    def test_bs(self):
        assert is_unitary_matrix(self.ops.bs(random.random()))

    def test_d(self):
        assert is_unitary_matrix(self.ops.d(random.random()))

    def test_r(self):
        assert is_unitary_matrix(self.ops.r(random.random()))

    def test_s(self):
        assert is_unitary_matrix(self.ops.s(random.random()))

    def test_s2(self):
        assert is_unitary_matrix(self.ops.s2(random.random()))


class TestMatrices:
    """Test that the operators produce the values we expect.
    TODO - would be better to test against known input & output values vs simply non-zero
    """

    def setup_method(self, method):
        qmr = c2qa.QumodeRegister(2, 2)
        self.ops = CVOperators(qmr)
        
    def test_bs(self):
        op = self.ops.bs(random.random())
        assert numpy.count_nonzero(op)

    def test_d(self):
        op = self.ops.d(random.random())
        assert numpy.count_nonzero(op)

    def test_r(self):
        op = self.ops.r(random.random())
        assert numpy.count_nonzero(op)

    def test_s(self):
        op = self.ops.s(random.random())
        assert numpy.count_nonzero(op)

    def test_s2(self):
        op = self.ops.s2(random.random())
        assert numpy.count_nonzero(op)
