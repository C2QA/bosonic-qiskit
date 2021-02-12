The sub-folders within `tutorials` offer sample usage of the `c2qa-qiskit` Python package.

Please note that until `c2qa-qiskit` is published to PyPI, the root of the repository must be added to the path instead of adding the `c2qa-qiskit` package to your Python environment. For example:
```
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
```

Note that currently all Wigner function plotting tutorials are non-functional.