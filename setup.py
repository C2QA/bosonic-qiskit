from setuptools import setup, find_packages


setup(
    name="c2qa-qiskit",
    version_config=True,
    setup_requires=["setuptools-git-versioning"],
    url="https://github.com/C2QA/c2qa-qiskit.git",
    author="Tim Stavenger",
    author_email="timothy.stavenger@pnnl.gov",
    description="National Quantum Initiative Co-design Center for Quantum Advantage bosonic Qiskit simulator",
    long_description="IBM Qiskit extension supporting simulation of bosonic qumodes reprsented as 2^n qubits within Qiskit",
    packages=find_packages(),
    install_requires=[
        "qiskit==0.34.2",
        "matplotlib==3.2.2",
        "pylatexenc==2.8",
        "Pillow==8.2.0",
        "cryptography==3.2.1",
    ],
)
