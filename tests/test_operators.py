import random

from c2qa.operators import CVOperators
import numpy


def allclose(a, b) -> bool:
    """Convert SciPy sparse matrices to ndarray and test with Numpy"""

    # If a and b are SciPy sparse matrices, they'll have a "toarray()" function
    if hasattr(a, "toarray"):
        a = a.toarray()

    if hasattr(b, "toarray"):
        b = b.toarray()

    return numpy.allclose(a, b)


def is_unitary_matrix(mat) -> bool:
    """Convert SciPy sparse matrix to ndarray and test with QisKit"""
    from qiskit.quantum_info.operators.predicates import is_unitary_matrix

    return is_unitary_matrix(mat.toarray())


class TestUnitary:
    """Verify operators are unitary"""

    def setup_method(self, method):
        self.cutoff = 4
        self.ops = CVOperators()

    def test_bs(self):
        assert is_unitary_matrix(self.ops.bs(random.random(), self.cutoff, self.cutoff))

    def test_d(self):
        assert is_unitary_matrix(self.ops.d(random.random(), self.cutoff))

    def test_r(self):
        assert is_unitary_matrix(self.ops.r(random.random(), self.cutoff))

    def test_s(self):
        assert is_unitary_matrix(self.ops.s(random.random(), self.cutoff))

    def test_s2(self):
        assert is_unitary_matrix(self.ops.s2(random.random(), self.cutoff, self.cutoff))


class TestMatrices:
    """Test that the operators produce the values we expect.
    TODO - would be better to test against known input & output values vs simply non-zero
    """

    def setup_method(self, method):
        self.cutoff = 4
        self.num_qumodes = 2
        self.ops = CVOperators()

    def test_a(self, capsys):
        # From https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/backends/fockbackend/ops.py#L208-L215
        trunc = self.cutoff  # equal to CVOperators cutoff
        ret = numpy.zeros((trunc, trunc), dtype=numpy.complex128)
        for i in range(1, trunc):
            ret[i - 1][i] = numpy.sqrt(i)

        assert allclose(self.ops.get_a(self.cutoff), ret)

    def test_bs(self):
        one = self.ops.bs(1, self.cutoff, self.cutoff)
        rand = self.ops.bs(random.random(), self.cutoff, self.cutoff)

        assert not allclose(one, rand)

    def test_bs_across_os(self, capsys):
        """Doesn't actually test anything, but as it is run across platforms by GitHub
        Actions a manual comparison can be made between Linux, MacOS, and Windows"""
        with capsys.disabled():
            op = self.ops.bs(numpy.pi / 4, self.cutoff, self.cutoff)
            # print()
            # print(op)

            assert op.getnnz()

    def test_d(self, capsys):
        with capsys.disabled():
            one = self.ops.d(1, self.cutoff)
            # rand = self.ops.d(random.random())

            print()
            # print("a")
            # print(self.ops.a)
            # print("a_dag")
            # print(self.ops.a_dag)
            print("1")
            print(one)

            neg_one = self.ops.d(-1, self.cutoff)
            print("-1")
            print(neg_one)

            # assert not allclose(one, rand)

    def test_d_across_os(self, capsys):
        """Doesn't actually test anything, but as it is run across platforms by GitHub
        Actions a manual comparison can be made between Linux, MacOS, and Windows"""
        with capsys.disabled():
            op = self.ops.d(numpy.pi / 2, self.cutoff)
            # print()
            # print(op)

            assert op.nnz

    def test_r(self):
        one = self.ops.r(1, self.cutoff)
        rand = self.ops.r(random.random(), self.cutoff)

        assert not allclose(one, rand)

    def test_s(self):
        one = self.ops.s(1, self.cutoff)
        rand = self.ops.s(random.random(), self.cutoff)

        assert not allclose(one, rand)

    def test_s2(self):
        one = self.ops.s2(1, self.cutoff, self.cutoff)
        rand = self.ops.s2(random.random(), self.cutoff, self.cutoff)

        assert not allclose(one, rand)
