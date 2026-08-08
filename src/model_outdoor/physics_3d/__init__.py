"""Physics modules for the bottom-up 3D PDE spread model (Phase 13).

Each module under this package implements one physical effect (pyrolysis,
drag, momentum, species transport, combustion, radiation, gas-solid
coupling, boundary conditions) with a Numba-JIT step function and a
literature-verified isolation test in tests/outdoor/test_3d_components.py.

Modules are designed to be composable: spread_3d.py imports their step
functions and orchestrates the time loop.  No module imports from
another physics_3d module.
"""
