"""Inverse PINN recovery of a PMSM's permanent-magnet flux linkage.

The package is layered so each piece can be tested and ported on its own:

``params``      frozen configuration records
``transforms``  Clarke and Park
``machine``     the dq state equations
``inverter``    magnitude limit, conduction drop, dead time
``control``     the two field-oriented current loops
``simulate``    fixed-step RK4 over the closed loop
``scenarios``   the excitation, including the speed step that governs
                identifiability
``windows``     cutting a record into estimation windows
``pinn``        the network, the physics loss and the optimiser
``fixtures``    golden vectors for the TypeScript twin
"""

from .params import (
    InverterParams,
    MotorParams,
    PinnConfig,
    SimParams,
    WindowConfig,
)
from .simulate import Trace, simulate

__all__ = [
    "InverterParams",
    "MotorParams",
    "PinnConfig",
    "SimParams",
    "Trace",
    "WindowConfig",
    "simulate",
]
