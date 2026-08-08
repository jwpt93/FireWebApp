"""BoundaryCondition abstract base class (Phase 23 Refactor 2A).

A ``BoundaryCondition`` object encapsulates the inflow/outflow pattern
for a specific simulation type.  The outdoor wildfire spread cases
(Cheney 1993, Marsden-Smedley 1995) use the wind-inlet pattern
(``OutdoorWindBC``): x-min face is an inflow (log-law or canopy BL
wind profile), x-max is outflow, y-faces are periodic, z-min is a
solid wall (bed base), z-max is outflow.

Phase 23 introduces a second pattern for the cup burner
(``CupBurnerBC``, to be added in Refactor 2B): z-min has an axi-
symmetric fuel-jet + coflow inlet, z-max is outflow, side walls are
solid, and there is no bed.

The registry pattern is deliberately loose: each BC subclass owns
which faces it configures.  Kernels consume ghost values from
``proj_solver`` and other state that the BC populates via
``.configure()`` at solver setup.

To add a new BC:

1. Add ``boundary_conditions/<name>.py`` with a subclass of
   ``BoundaryCondition``.
2. Register it in ``boundary_conditions/__init__.py``.
3. Add unit tests in ``tests/outdoor/test_boundary_condition_<name>.py``
   per Rule #18.

Rule #17 (determinism): any BC that computes stochastic quantities
(e.g. SEM eddies) must use a caller-provided seed for bit-exact
reproducibility across runs.
"""
from __future__ import annotations


class BoundaryCondition:
    """Abstract BC. Subclasses implement configure() and optionally
    apply_per_step().

    Attributes
    ----------
    kind : str
        Registry key (e.g. "outdoor_wind", "cup_burner").
    """
    kind: str = ""

    def configure(self, proj_solver, grid, state) -> None:
        """One-time BC setup at solver init.

        Called AFTER grid + state are built but BEFORE the timestepping
        loop starts.  A BC that populates a static inflow profile
        (like the outdoor wind BC) does its full setup here.  A BC
        with a time-varying inlet (e.g. gust ramp, agent injection
        pulse) may store its params here and update per step in
        ``apply_per_step()``.

        Parameters
        ----------
        proj_solver : ProjectionSolver3D
            The pressure-projection solver.  Access the ``.set_inlet_BC``
            or (future) ``.set_bottom_inlet_BC`` hooks to install
            face-value arrays that the solver consumes for divergence
            and gradient boundary corrections.
        grid : Grid3D
            The mesh geometry (dx, dy, dz_arr, x_mid, y_mid, z_mid,
            Nx, Ny, Nz).
        state : CellState3D
            The full field state.  A BC may initialize interior fields
            (e.g. u = u_inlet profile as spin-up IC) here.
        """
        raise NotImplementedError

    def apply_per_step(self, state, grid, t: float) -> None:
        """Per-step BC update.  Default no-op.

        Override when the BC changes with time (composite wind hodograph,
        agent injection pulse, etc.).  Called BEFORE momentum and
        species updates each timestep.
        """
        pass
