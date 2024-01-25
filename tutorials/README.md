The sub-folders within `tutorials` offer sample usage of the `bosonic-qiskit` Python package.

Please see each tutorial's README.md and text markdown cells within the notebook for details on what they demonstrate.

Please note that unless you are using the `bosonic-qiskit` published to PyPI installed on your system or in a virtual environment, the root of the repository must be added to the path instead of adding the `bosonic-qiskit` package to your Python environment. For example:
```
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
```
