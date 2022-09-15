from setuptools import setup, find_packages


setup(
    name="bosonic-qiskit",
    version_config=True,
    setup_requires=["setuptools-git-versioning"],
    url="https://github.com/C2QA/bosonic-qiskit",
    author="Tim Stavenger",
    author_email="timothy.stavenger@pnnl.gov",
    description="National Quantum Initiative Co-design Center for Quantum Advantage bosonic Qiskit simulator",
    long_description="Qiskit extension supporting simulation of bosonic qumode Fock states reprsented as qubits within Qiskit",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=["qiskit==0.34.2", "matplotlib==3.5.2", "pylatexenc==2.10"],
)
