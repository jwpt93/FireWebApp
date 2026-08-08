"""Pluggable chemistry-closure registry (Phase 15-0).

See :mod:`._interface` for the closure-module contract.

Public API
----------
``run(closure_name, **kwargs)``
    Dispatch chemistry source computation to the named closure.

``available()``
    Sorted tuple of registered closure names.

To add a new closure:

1. Add ``chemistry_closures/<name>.py`` exposing a ``run(**kwargs) -> None``
   function (see :mod:`._interface`).
2. Add ``from . import <name>`` and ``_REGISTRY[<name>] = <name>.run``
   to this file.
3. Add unit tests in ``tests/outdoor/test_<name>_closure.py`` (Rule #18).

Phase 15 work-in-progress closures (to be added):
    - 15A: ``edc_chi_st_hybrid`` — EDC + χ_st gradient-based source band
    - 15C: ``level_set_fsd``     — Boger-Veynante FSD on smoothed phi_flame
    - 15B: ``mixture_fraction``  — Mell 2007 / FIRETEC mixture-fraction Z transport
"""
from . import edc
from . import ebu_bootstrap
from . import pasr
from . import level_set_fsd
from . import edc_2step_methane    # Phase 23 Westbrook-Dryer 2-step

_REGISTRY = {
    'edc':                 edc.run,
    'ebu_bootstrap':       ebu_bootstrap.run,
    'pasr':                pasr.run,
    'level_set_fsd':       level_set_fsd.run,       # Phase 15C
    'edc_2step_methane':   edc_2step_methane.run,   # Phase 23
}


def available() -> tuple[str, ...]:
    """Return sorted tuple of registered closure names."""
    return tuple(sorted(_REGISTRY))


def run(closure_name: str, **kwargs) -> None:
    """Dispatch chemistry source computation to the named closure.

    Each closure pulls what it needs from kwargs and ignores the rest.
    See :mod:`._interface` for the keyword-argument contract.

    Raises
    ------
    ValueError
        If ``closure_name`` is not in :func:`available`.
    """
    fn = _REGISTRY.get(closure_name)
    if fn is None:
        raise ValueError(
            f"combustion_closure={closure_name!r} not in {available()!r}"
        )
    fn(**kwargs)
