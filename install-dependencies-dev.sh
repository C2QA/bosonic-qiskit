#!/bin/bash

python3 -m venv bosonic-qiskit
source bosonic-qiskit/bin/activate
pip3 install -r requirements_dev.txt
pre-commit install
