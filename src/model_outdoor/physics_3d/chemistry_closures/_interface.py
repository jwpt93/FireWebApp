"""Chemistry-closure plug-in interface (Phase 15-0).

A chemistry closure is a module under chemistry_closures/ that implements:

    def run(**kwargs) -> None:
        '''Compute and apply the chemistry source term in place.

        Updates T_g, Y_fuel, Y_O2 in place; writes time-averaged ω to
        omega_out.  Returns None.

        Required keyword arguments for every closure:
            rho:        (Nz,Ny,Nx) float64 [kg/m³]
            T_g:        (Nz,Ny,Nx) float64 [K]   — updated in place
            Y_fuel:     (Nz,Ny,Nx) float64 [-]   — updated in place
            Y_O2:       (Nz,Ny,Nx) float64 [-]   — updated in place
            chi_rad:    float
            cp_g:       float [J/kg/K]
            dt:         float [s]
            n_substeps: int
            omega_out:  (Nz,Ny,Nx) float64 [kg/m³/s] — overwritten

        Closure-specific optional kwargs (current registry):
            edc:           k_turb, eps_turb
            ebu_bootstrap: tau_mix, omega_O2, omega_max_T, T_pin
            pasr:          tau_mix

        Closures MUST silently accept and ignore kwargs they do not use
        (use ``**_unused`` in the signature).  This lets the caller
        construct one kwarg bag in the main loop and dispatch without
        per-closure conditional argument construction.
        '''

To register a new closure:
  1. Add ``chemistry_closures/<name>.py`` with the ``run`` function above.
  2. Add ``from . import <name>`` and ``_REGISTRY[<name>] = <name>.run``
     to ``__init__.py``.
  3. Add unit tests in ``tests/outdoor/test_<name>_closure.py`` (Rule #18).

Rule #17 (determinism): the underlying @njit kernel must use the
double-buffer pattern when @njit(parallel=True) is combined with
reductions, to guarantee bit-exact reproducibility across thread counts.

Rule #18 (unit tests required): every new closure must ship
determinism + sanity tests in the same commit that adds the module.
"""
