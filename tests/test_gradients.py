"""Finite-difference check of the physics gradient.

This one runs in a subprocess because it needs float64, and JAX only honours
`jax_enable_x64` when it is set before the library is first used. Toggling it
mid-session silently leaves the computation in float32, where a central
difference of two nearly-equal values is quantised and the check passes or
fails for reasons that have nothing to do with the gradient.

Everything else in the suite deliberately runs in float32, because that is
what the published results were produced in.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CHECK = Path(__file__).parent / "_gradcheck.py"


def test_physics_gradients_match_central_differences() -> None:
    env = {**os.environ, "JAX_ENABLE_X64": "1"}
    result = subprocess.run(
        [sys.executable, str(CHECK)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
