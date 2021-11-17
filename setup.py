from setuptools import setup, find_packages


setup(
    name="c2qa-qiskit",
    version_config=True,
    setup_requires=["setuptools-git-versioning"],
    url="https://github.com/C2QA/c2qa-qiskit.git",
    author="Tim Stavenger",
    author_email="timothy.stavenger@pnnl.gov",
    description="National Quantum Initiative Co-design Center for Quantum Advantage simulator for hybrid qubits in QisKit",
    packages=find_packages(),
    install_requires=[
        "qiskit==0.25.0",
        # For drawing circuits, state vectors, Wigner function plots (matplotlib 3.3.0+ is incompatible)
        "matplotlib==3.2.2",
        "pylatexenc==2.8",
        "Pillow==8.2.0",
        # Installing current cryptography 3.3.1 failed in Windows 10, force previous minor build version.
        "cryptography==3.2.1",
    ],
)
