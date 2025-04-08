from typing import Iterable, Set

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import (
    BaseSamplerV2,
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
    PrimitiveJob,
)
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.primitives.containers.sampler_pub import SamplerPub

from .circuit import CVCircuit
from . import util


class FockArray(BitArray):
    """Wrapper around `BitArray` to make it easier to handle fock data"""

    def __init__(self, *args):
        super().__init__(*args)

        _bytes = self._array.shape[-1]
        self.dtype = np.dtype(f">u{_bytes}")

    @property
    def array(self):
        return self._array.view(self.dtype)

    def get_fock_counts(self, loc=None):
        return self.get_int_counts(loc=loc)

    def slice_shots(self, indices):
        res = super().slice_shots(indices)
        return FockArray(res._array, res._num_bits)

    def __repr__(self):
        desc = f"<shape={self.shape}, num_shots={self.num_shots}>"
        return f"FockArray({desc})"


class FockJob(PrimitiveJob):
    def __init__(self, cv_pubs, function, future, *args, **kwargs):
        super().__init__(function, *args, **kwargs)

        self._future = future
        self.cv_pubs = cv_pubs

    @classmethod
    def wrap_primitive_job(cls, cv_pubs, job: PrimitiveJob):
        new_job = cls(cv_pubs, job._function, job._future, *job._args, **job._kwargs)
        new_job._job_id = job._job_id
        return new_job

    def result(self):
        result: PrimitiveResult = super().result()
        return PrimitiveResult(
            [
                FockSamplerPubResult.wrap_sampler_result(pub, r)
                for pub, r in zip(self.cv_pubs, result)
            ],
            metadata=result.metadata,
        )


class FockSamplerPubResult(SamplerPubResult):
    def __init__(self, cv_pub: SamplerPub, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data = self._to_fock(cv_pub)

    @classmethod
    def wrap_sampler_result(cls, cv_pub: SamplerPub, result: SamplerPubResult):
        return cls(cv_pub, data=result.data, metadata=result.metadata)

    def _to_fock(self, cv_pub: SamplerPub):
        databin = self._data
        new_result = dict(self._data.items())
        if self._is_likely_cvcircuit(cv_pub.circuit):
            fock_regs = {creg.name for creg in self._get_fock_registers(cv_pub.circuit)}
            for creg in fock_regs:
                data = databin[creg]

                if not isinstance(data, BitArray):
                    raise ValueError(
                        f"Cannot convert data for classical register {creg}, expected a BitArray"
                    )

                new_result[creg] = FockArray(data.array, data.num_bits)

            result = DataBin(**new_result, shape=databin.shape)

        return result

    def _is_likely_cvcircuit(self, circuit: QuantumCircuit):
        return (
            isinstance(circuit, CVCircuit) or CVCircuit.metadata_key in circuit.metadata
        )

    def _get_fock_registers(self, circuit: QuantumCircuit) -> Set[ClassicalRegister]:
        if isinstance(circuit, CVCircuit):
            qmregs = {qmreg.qreg for qmreg in circuit.qmregs}
        else:
            reg_names = set(
                circuit.metadata[CVCircuit.metadata_key]["qumode_registers"]
            )
            qmregs = set(filter(lambda x: x.name in reg_names, circuit.qregs))

        # Get mapping from classical bit -> qubit
        mapping = util._final_measurement_mapping(circuit)
        mapping = {circuit.clbits[i]: circuit.qubits[j] for i, j in mapping.items()}

        # Now just collect unique classical registers that represent qumodes
        return {
            clbit._register
            for clbit, qubit in mapping.items()
            if qubit._register in qmregs
        }


class FockSampler(BaseSamplerV2):
    """Sampler primitive that wraps qiskit sampler to handle qumode data

    This handles converting results of qumode registers back to
    decimal Fock values instead of bit strings. For any qubit registers
    in the original `CVCircuit`, those bitstrings are left unchanged.
    """

    def __init__(self, base_sampler: BaseSamplerV2):
        self.base_sampler = base_sampler

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> FockJob:
        cv_pubs = [SamplerPub.coerce(pub, shots=shots) for pub in pubs]

        job = self.base_sampler.run(cv_pubs, shots=shots)
        return FockJob.wrap_primitive_job(cv_pubs, job)

    @classmethod
    def from_sampler(cls, sampler: BaseSamplerV2):
        return cls(sampler)
