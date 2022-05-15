# VQE of the $Z_2$ lattice gauge theory
## (Tentative layout)

### Ground state preparation
Implement a variational quantum eigensolver (VQE) to prepare the ground state of
the Z2 LGT Hamiltonian.

(**add equations and links to where the code lives**)
    1. Ansatz generation
        - `resource.vary_Z2LGT`
    2. Energy measurement
        - `resource.measureE_fieldterm` and `resource.measureE_hoppingterm`
    3. Classical optimization (gradient-based or gradient-free?)

### Observing a phase transition
Once the ground state has been prepared, measure the long-distance correlation operator O_n,n'

    1. Phase kickback measurement of $\hat{x}$ and $\hat{p}$
        - `resource.measure_gauge_invariant_propagator`
