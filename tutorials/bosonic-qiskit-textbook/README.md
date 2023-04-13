# Bosonic Qiskit Textbook

## Forward
This textbook presents a zero-to-sixty introduction to continuous-variable quantum computing. The intended audience has experience with quantum computing using qubits at the level of the [Qiskit Textbook](https://qiskit.org/textbook/preface.html) but is not assumed to have any previous experience with physics. The textbook is intended to be self-contained, in that mathematical formulas and quantum mechanics concepts are always introduced alongside proofs or guided exercises. The guide mixes together the core concepts of quantum computing using bosonic modes with code examples from Bosonic Qiskit wherever appropriate. 

This work was produced by the Bosonic Qiskit project group under the Yale undergraduate Quantum Computing club, consisting of Ben McDonough, Jeb Cui, and Gabriel Marous. Writing this was an incredible learning experience. We found the richest and most helpful source for quantum optics to be **Introductory Quantum Optics** by Gerry and Knight, which we recommend if you are interested in learning more. This project was supervised by Prof. Steven M. Girvin and Dr. Kevin Smith.

The original project repository can be found here: https://github.com/benmcdonough20/bosonic-qiskit-tutorials/tree/main/bosonic_qiskit_textbook

## Structure
This textbook in its current form is divided into five parts:

### Introduction
An introduction to the state space of bosonic modes. Comparison between qubits and continuous-variable systems. Physical examples of bosonic modes. Importing Bosonic Qiskit and creating a simple circuit.

### Operators
Ladder operators, the number operator, quadrature operators, and phase-space operators. Outer products and exponentials of operators. Working with continuous spectra.

### Coherent States
Definition and representation in the number basis. Photon number distribution. Overlap of two coherent states. Derivation of the uncertainty principle and coherent states as minimum-uncertainty states. Time-evolution of a coherent state and correspondence with the Harmonic oscillator.

### Wigner Distributions
Introduction and motivation for considering Wigner distributions. Weyl quantization and recovering the trace from the Weyl transformation. The Wigner function as a Weyl transformation of the density matrix. Properties of the Wigner function including normalization, negativity, and real-valuedness. More generalized discussion of minimum-uncertainty states as Gaussians and a derivation of the Wigner function of a minimum-uncertainty state.

### Gates
The Baker-Campbell-Hausedorff Lemma and Heisenburg's equations of motion. Discussion of the properties of Gaussian gates including the following:

* Phase space rotation
* Displacement
* Single-mode squeezing
* Two-mode squeezing
* Beamsplitter

Definition of conditional gates as derived from Gaussian gates. Illustration by measuring the geometrical phase of sequential displacement gates forming a loop.

## Dependencies
* Bosonic Qiskit 8.0
