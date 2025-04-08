import itertools

import numpy as np
import pytest
import qiskit as qk
from qiskit.primitives.containers import BitArray
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy import stats

import c2qa as bq
from c2qa.primitives import FockArray
from c2qa.primitives import FockSampler as Sampler


@pytest.fixture()
def reset_classical_reg():
    # Reset so each classical register will start from c0. If this is not used,
    # sometimes the `test_large_qumode_register` test has a c1 register instead of c0
    # because the `test_bellstate` test ran first. This is why global variables can be bad, kids!
    qk.ClassicalRegister.instances_counter = itertools.count()
    yield


class TestFockSampler:
    def test_bellstate(self, reset_classical_reg):
        # This test checks the bell state |0>|0> + |2>|3>,
        # which allows us to test endianness of the resulting samples
        # (since the |2>|3> component has differing energy levels)
        qmr = bq.QumodeRegister(2, 2)
        qc = bq.CVCircuit(qmr)

        statevec = np.zeros(2**qmr.size, dtype=np.complex128)
        statevec[0] = 1 / np.sqrt(2)
        statevec[14] = 1 / np.sqrt(2)  # for little endian, |2>|3> -> |3>|2> (14)
        qc.cv_initialize(statevec, qmr, mode="full")
        qc.cv_measure_all()

        sampler = Sampler.from_sampler(AerSampler())
        job = sampler.run([qc], shots=1024)
        results = job.result()

        # Test there's only one circuit
        assert len(results) == 1
        pub_result = results[0].data

        # Check that there's only one register in the data and that its name matches
        # 'c0', which is auto-assigned by cv_measure_all
        assert set(pub_result) == {"c0"}

        # Check properties of the fock array and its underlying data
        fockarray = pub_result.c0
        assert isinstance(fockarray, FockArray)
        assert fockarray.array.shape == (1024, 1)
        assert len(np.unique(fockarray.array)) == 2

        # Perform a statistical test that we haven't deviated too much
        # from the 1/2 expected proportions of the bell state (the noise
        # coming from finite shots). In principle this part can fail, but it
        # should be very rare, and we can always suppress that by increasing
        # the shot count.
        fock_counts = fockarray.get_fock_counts()
        assert set(fock_counts) == {0, 14}
        result = stats.binomtest(fock_counts[0], 1024, p=0.5)
        assert result.pvalue >= 0.05

    def test_large_qumode_register(self, reset_classical_reg):
        # We want to test that we can properly interpret multiple
        # bytes in the original BitArray as a single integer in the
        # FockArray. This means we'll need a statevector with more than 2^8 elements
        qmr = bq.QumodeRegister(3, 4)
        qc = bq.CVCircuit(qmr)

        # Put them in the |1>|1>|1> state. This should correspond to
        # 1(16^2) + 1(16) + 1 = 273
        qc.cv_initialize([0, 1], qmr)
        qc.cv_measure_all()

        sampler = Sampler.from_sampler(AerSampler())
        job = sampler.run([qc], shots=1024)
        results = job.result()

        assert len(results) == 1
        pub_result = results[0].data
        assert set(pub_result) == {"c0"}

        # Check properties of the fock array and its underlying data. The internal bit array
        # will have shape (..., 1024, 2) with dtype np.uint8, but the FockArray object should
        # view it as an array of shape (..., 1024, 1) with a dtype holding 2 bytes
        fockarray = pub_result.c0
        assert isinstance(fockarray, FockArray)
        assert fockarray.array.shape == (1024, 1)
        assert fockarray.array.dtype.itemsize == 2
        assert fockarray.num_bits == 12

        # Check the outcome matches what we prepared above
        fock_counts = fockarray.get_fock_counts()
        assert set(fock_counts) == {273}

    def test_controlled_displacement(self, reset_classical_reg):
        # This test checks behavior of the FockSampler when there's hybrid
        # CV/DV registers, and it tests slicing.
        qmr = bq.QumodeRegister(1, 6)
        q = qk.QuantumRegister(1)
        c0 = qk.ClassicalRegister(qmr.size, "c0")
        c1 = qk.ClassicalRegister(1, "c1")
        qc = bq.CVCircuit(qmr, q, c0, c1)

        # Make a superposition of coherent states with c-D gate.
        # Should have |0>|0> + |1>|a>
        alpha = 0.7
        qc.h(q[0])
        qc.cv_c_d(alpha / 2, qmr[0], q[0])
        qc.cv_d(alpha / 2, qmr[0])
        qc.x(q[0])
        qc.cv_measure([qmr, q], [c0, c1])

        shots = 1024
        sampler = Sampler.from_sampler(AerSampler())
        qc = qk.transpile(
            qc, backend=AerSimulator()
        )  # have to transpile this time because of cv gates
        job = sampler.run([qc], shots=shots)
        result = job.result()[0].data

        # The FockSampler should have left the qubit register alone, not promoting it
        # to a FockArray
        assert set(result) == {"c0", "c1"}
        assert isinstance(result.c0, FockArray)
        assert isinstance(result.c1, BitArray) and not isinstance(result.c1, FockArray)

        # Test that we get 50/50 |0> or |1> for the qubit register
        qubit_counts = result.c1.get_counts()
        assert set(qubit_counts) == {"0", "1"}
        stat_result = stats.binomtest(qubit_counts["0"], shots, p=0.5)
        assert stat_result.pvalue >= 0.05

        # For the |0>|0> state, check that we always get back |0> for the qumode
        qubit0_idx = np.nonzero(result.c1.array.flat == 0)[0]
        fock0 = result.c0.slice_shots(qubit0_idx)
        fock_counts = fock0.get_fock_counts()
        assert set(fock_counts) == {0}

        # Check that for the |1>|a> state, we have a coherent state for the qumode.
        # We perform this test with a chi-squared check that p(n) ~= poisson(|alpha|^2)
        expected_dist = stats.poisson(np.abs(alpha) ** 2)
        qubit1_idx = np.nonzero(result.c1.array.flat)[0]
        focka = result.c0.slice_shots(qubit1_idx)
        fock_counts = focka.get_fock_counts()

        f_exp = [expected_dist.pmf(x) for x in range(qmr.cutoff)]
        f_obs = [fock_counts.get(x, 0) / shots for x in range(qmr.cutoff)]
        stat_result = stats.chisquare(f_obs, f_exp, sum_check=False)
        assert stat_result.pvalue >= 0.05
