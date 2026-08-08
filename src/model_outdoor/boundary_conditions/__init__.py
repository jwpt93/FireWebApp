"""Boundary-condition registry (Phase 23 Refactor 2A).

See :mod:`.base` for the ``BoundaryCondition`` abstract class contract.

Registered kinds
----------------
"outdoor_wind"
    x-face wind inlet + outflow + z-bed + y-periodic pattern used by
    every pre-Phase-23 validation case (Cheney 1993, Marsden-Smedley
    1995).  Behaviour-preserving wrap of the inline code that used
    to live directly in :func:`model_outdoor.spread_3d.run_3d_spread`.

Coming in Refactor 2B
---------------------
"cup_burner"
    z-min fuel-jet + coflow inlet, z-max outflow, solid side walls.
    For the ISO 14520 / NIST SP 890 cup-burner MEC test.

To register a new BC:

1. Add ``boundary_conditions/<name>.py`` with a ``BoundaryCondition``
   subclass.  Its ``.kind`` attribute must match the registry key.
2. Add ``from . import <module>; _REGISTRY[<module>.<Cls>.kind] =
   <module>.<Cls>`` to this file.
3. Add unit tests under ``tests/outdoor/test_boundary_condition_<name>.py``
   per Rule #18.
"""
from .base import BoundaryCondition
from .outdoor_wind import OutdoorWindBC
from .cup_burner import CupBurnerBC


_REGISTRY: dict[str, type[BoundaryCondition]] = {
    OutdoorWindBC.kind: OutdoorWindBC,
    CupBurnerBC.kind:   CupBurnerBC,
}


def available() -> tuple[str, ...]:
    """Sorted tuple of registered BC-kind names."""
    return tuple(sorted(_REGISTRY.keys()))


def get_bc_class(kind: str) -> type[BoundaryCondition]:
    """Look up a BC class by kind name. Raises ValueError on unknown."""
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown boundary_condition_kind={kind!r}.  "
            f"Registered: {available()}"
        )
    return _REGISTRY[kind]
