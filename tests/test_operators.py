import random

from c2qa.operators import CVOperators
import numpy


def allclose(a, b) -> bool:
    """Convert SciPy sparse matrices to ndarray and test with Numpy"""
    from numpy import allclose
    from scipy.sparse.csr import csr_matrix

    if isinstance(a, csr_matrix):
        a = a.toarray()

    if isinstance(b, csr_matrix):
        b = b.toarray()

    return allclose(a, b)


def is_unitary_matrix(mat) -> bool:
    """Convert SciPy sparse matrix to ndarray and test with QisKit"""
    from qiskit.quantum_info.operators.predicates import is_unitary_matrix
    return is_unitary_matrix(mat.toarray())


class TestUnitary:
    """Verify operators are unitary"""

    def setup_method(self, method):
        self.ops = CVOperators(cutoff=4, num_qumodes=2)

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
        self.ops = CVOperators(cutoff=4, num_qumodes=2)

    def test_a(self, capsys):
        # From https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/backends/fockbackend/ops.py#L208-L215
        trunc = 4  # equal to CVOperators cutoff
        ret = numpy.zeros((trunc, trunc), dtype=numpy.complex128)
        for i in range(1, trunc):
            ret[i - 1][i] = numpy.sqrt(i)

        assert allclose(self.ops.a, ret)

    def test_bs(self):
        one = self.ops.bs(1)
        rand = self.ops.bs(random.random())

        assert not allclose(one, rand)

    def test_bs_across_os(self, capsys):
        """Doesn't actually test anything, but as it is run across platforms by GitHub
        Actions a manual comparison can be made between Linux, MacOS, and Windows"""
        with capsys.disabled():
            op = self.ops.bs(numpy.pi / 4)
            # print()
            # print(op)

            assert op.getnnz()

    def test_d(self, capsys):
        with capsys.disabled():
            one = self.ops.d(1)
            rand = self.ops.d(random.random())

            print()
            # print("a")
            # print(self.ops.a)
            # print("a_dag")
            # print(self.ops.a_dag)
            print("1")
            print(one)

            neg_one = self.ops.d(-1)
            print("-1")
            print(neg_one)

            assert not allclose(one, rand)

    def test_compare_d(self, capsys):
        with capsys.disabled():
            ops = CVOperators(cutoff=40, num_qumodes=1)
            # d = ops.d(0.3+0.5j)
            # sf = d[0:5, 0:5]
            d = ops.d(1)
            sf = d[0:4, 0:4]
            print(d)
            print()
            print(sf)

    def test_d_across_os(self, capsys):
        """Doesn't actually test anything, but as it is run across platforms by GitHub
        Actions a manual comparison can be made between Linux, MacOS, and Windows"""
        with capsys.disabled():
            op = self.ops.d(numpy.pi / 2)
            # print()
            # print(op)

            assert op.getnnnz()

    def test_r(self):
        one = self.ops.r(1)
        rand = self.ops.r(random.random())

        assert not allclose(one, rand)

    def test_s(self):
        one = self.ops.s(1)
        rand = self.ops.s(random.random())

        assert not allclose(one, rand)

    def test_s2(self):
        one = self.ops.s2(1)
        rand = self.ops.s2(random.random())

        assert not allclose(one, rand)
