#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements_dev.txt
pre-commit install
pip3 install -e .
