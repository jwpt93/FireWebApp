"""CupBurnerBC — z-min fuel-jet + coflow inlet for ISO 14520 / NIST SP 890
cup-burner MEC validation (Phase 23 Refactor 2B).

Geometry (per ISO 14520-1 Annex B, verified against Takahashi 2007):

    z=Lz  ┌─────────────────────────┐  free outflow (atmospheric)
          │                         │
          │      diffusion          │
          │       flame             │
          │      ─────              │
          │     ─  ─   ─            │
          │     ─ (F) ─             │  fuel jet at cup rim, ~4 mm ASL
          │      ─   ─              │
          │                         │
          │                         │
          │                         │  chimney interior (side walls)
          │                         │
    z=0   ├──┬──┬──────┬──┬──┬──┬───┤
             │  │      │  │  │  │       ← bottom face BC:
          wall  │      │  │  │  wall
              coflow F coflow            F = fuel jet (Y_F=1, w=U_fuel)
                                          coflow = air ± N2/CO2, w=U_coflow

Default dimensions (Takahashi 2007 Table 1):
    fuel cup ID     = 28 mm  → R_fuel   = 14 mm
    chimney ID      = 85 mm  → R_chim   = 42.5 mm
    U_fuel (CH4)    = 0.06 m/s (~40 cm³/s at cup exit)
    U_coflow (air)  = 0.10 m/s (~100 cm³/s)

This class holds the geometry math + parameters.  Full activation
requires additional plumbing in projection_3d.py (z-min inlet BC in
the divergence + gradient computations) and every species/energy
advection kernel (per-face inlet ghost values for Y_F, Y_O2, Y_H2O,
T_g).  ``configure()`` currently raises NotImplementedError with a
message pointing to the plumbing plan.

Refactor 2B (this file + skeleton) — landed 2026-07-25
Refactor 2C (projection + kernel plumbing) — planned
Refactor 2D (first CH4/N2 MEC CAL run) — planned

References
----------
- Takahashi, F., Katta, V.R., Grosshandler, W.L. (2007) "Extinguishment
  of cup-burner flames," Proc. Combust. Inst. 31:2721-2729
- Grosshandler et al. (1995) NIST SP 890 vol.2 — cup burner MEC tables
- ISO 14520-1 Annex B — reference cup-burner geometry
"""
from __future__ import annotations

import numpy as np

from .base import BoundaryCondition


# Air composition at NTP.  Same value as spread_3d.py hard-codes for
# the x-min outdoor inlet (`state.Y_O2[:, :, 0] = 0.232`).
Y_O2_AIR_DEFAULT = 0.232


class CupBurnerBC(BoundaryCondition):
    """z-min axisymmetric-analog fuel-jet + coflow inlet.

    Slab-symmetric approximation: the domain is (x, y, z) but the
    fuel-jet + coflow pattern is applied along the x-axis with
    y-periodic thin-slab convention (Ly ≪ Lx = chimney diameter).
    A true 3D axisymmetric cup burner would require r-θ-z geometry;
    the slab analog matches our validated Cheney/Marsden-Smedley
    convention and is sufficient for MEC-scalar validation.
    """

    kind = "cup_burner"

    def __init__(
        self,
        *,
        fuel_jet_radius_m: float = 0.014,
        chimney_radius_m: float = 0.0425,
        fuel_jet_velocity_m_s: float = 0.06,
        coflow_velocity_m_s: float = 0.10,
        Y_F_fuel: float = 1.0,
        Y_O2_coflow: float = Y_O2_AIR_DEFAULT,
        # Suppressant dilution: fraction of N2 (or CO2) mixed into coflow
        # AT THE EXPENSE OF O2.  Y_O2_coflow_effective = Y_O2_coflow ×
        # (1 - Y_agent_coflow / Y_N2_air_baseline).  Sweeping this from
        # 0 up to MEC identifies the extinguishing concentration.
        Y_agent_coflow: float = 0.0,
        T_inlet_K: float = 298.0,
        # ── Phase 23 wick option B ─────────────────────────────────────
        # When wick_enable=True, apply_per_step() dirichlet-forces
        # Y_fuel in the fuel-jet-column region at z ∈ [wick_z_lo,
        # wick_z_hi].  Bypasses the "fuel needs to travel up from
        # z=0 inlet in ~3 s" transient that killed slab v6/v7.
        # wick_Y_F: pure fuel (1.0) or stoich (0.055).  1.0 makes
        # the wick region behave like a fuel-vapor source (like a
        # real liquid-fuel evaporating wick above a cup rim);
        # combustion happens at the wick-coflow interface.
        wick_enable: bool = False,
        wick_Y_F: float = 1.0,
        wick_z_lo: float = 0.003,   # 3 mm above cup rim
        wick_z_hi: float = 0.009,   # 9 mm
        **_ignored_outdoor_kwargs,   # absorb unused outdoor kwargs
    ):
        """
        Parameters
        ----------
        fuel_jet_radius_m : float
            Half-width of the central fuel-jet band (default 14 mm =
            ISO 14520 28-mm cup ID / 2).
        chimney_radius_m : float
            Half-width of the coflow annulus outer boundary (default
            42.5 mm = 85 mm chimney ID / 2).
        fuel_jet_velocity_m_s : float
            Bulk velocity at fuel-jet outlet (default 0.06 m/s ≈ Takahashi
            2007 methane rate).
        coflow_velocity_m_s : float
            Bulk velocity of oxidizer coflow (default 0.10 m/s).
        Y_F_fuel : float
            Fuel-side mass fraction of Y_F at the jet outlet (default
            1.0 = pure gaseous fuel).
        Y_O2_coflow : float
            Baseline mass fraction of O2 in coflow when Y_agent_coflow=0
            (default 0.232 = air).
        Y_agent_coflow : float
            Extra suppressant added to the coflow AT THE EXPENSE OF O2.
            This is the sweep parameter for MEC identification.
        T_inlet_K : float
            Inlet gas temperature (default 298 K = ambient).
        """
        self.fuel_jet_radius_m = float(fuel_jet_radius_m)
        self.chimney_radius_m = float(chimney_radius_m)
        self.fuel_jet_velocity_m_s = float(fuel_jet_velocity_m_s)
        self.coflow_velocity_m_s = float(coflow_velocity_m_s)
        self.Y_F_fuel = float(Y_F_fuel)
        self.Y_O2_coflow = float(Y_O2_coflow)
        self.Y_agent_coflow = float(Y_agent_coflow)
        self.T_inlet_K = float(T_inlet_K)
        # Wick config (option B)
        self.wick_enable = bool(wick_enable)
        self.wick_Y_F    = float(wick_Y_F)
        self.wick_z_lo   = float(wick_z_lo)
        self.wick_z_hi   = float(wick_z_hi)
        self._wick_k_lo  = None    # populated lazily in apply_per_step
        self._wick_k_hi  = None
        # Populated by _build_masks at configure time.
        self.fuel_mask: np.ndarray | None = None
        self.coflow_mask: np.ndarray | None = None
        self.wall_mask: np.ndarray | None = None
        self.w_inlet_zmin: np.ndarray | None = None

    def _build_masks(self, grid) -> None:
        """Compute the (Ny, Nx) fuel-jet / coflow / wall masks at z=0.

        Two geometry modes:
        - Slab (Ny == 1): "radial" coordinate is |x - x_c| only; masks
          are y-invariant lines.  Matches thin-slab convention used by
          Cheney/Marsden-Smedley outdoor cases.
        - Axisymmetric analog (Ny > 1): true 2D radial distance
          r = sqrt((x - x_c)^2 + (y - y_c)^2).  Fuel jet is a disc,
          coflow is an annulus, wall is outside the chimney.  This
          gives the physical axisymmetric focusing (1/r geometric
          concentration of fuel-coflow mixing near the fuel-jet edge)
          that the slab approximation misses.  Refactor 2D-3D
          extension per lift-off failure of v6/v7 slab sims.
        """
        Nx, Ny = grid.Nx, grid.Ny
        x_center = 0.5 * (grid.x_mid[0] + grid.x_mid[-1])
        if Ny == 1:
            r_x = np.abs(grid.x_mid - x_center)   # (Nx,)
            fuel_1d   = r_x <= self.fuel_jet_radius_m
            coflow_1d = (r_x > self.fuel_jet_radius_m) & \
                        (r_x <= self.chimney_radius_m)
            wall_1d   = r_x > self.chimney_radius_m
            self.fuel_mask   = np.broadcast_to(fuel_1d,   (Ny, Nx)).copy()
            self.coflow_mask = np.broadcast_to(coflow_1d, (Ny, Nx)).copy()
            self.wall_mask   = np.broadcast_to(wall_1d,   (Ny, Nx)).copy()
        else:
            y_center = 0.5 * (grid.y_mid[0] + grid.y_mid[-1])
            xx, yy = np.meshgrid(grid.x_mid, grid.y_mid, indexing="xy")
            r = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)
            self.fuel_mask   = r <= self.fuel_jet_radius_m
            self.coflow_mask = (r > self.fuel_jet_radius_m) & \
                               (r <= self.chimney_radius_m)
            self.wall_mask   = r > self.chimney_radius_m

    def _build_w_inlet_zmin(self, grid) -> np.ndarray:
        """(Ny, Nx) vertical velocity at the z=0 face.

        Fuel jet cells get U_fuel; coflow cells get U_coflow; wall
        cells get 0 (no penetration).
        """
        if self.fuel_mask is None:
            self._build_masks(grid)
        w = np.zeros((grid.Ny, grid.Nx))
        w[self.fuel_mask]   = self.fuel_jet_velocity_m_s
        w[self.coflow_mask] = self.coflow_velocity_m_s
        # wall cells already 0
        self.w_inlet_zmin = w
        return w

    def effective_Y_O2_coflow(self) -> float:
        """Y_O2 in coflow after N2/CO2 dilution.

        Model: agent is added AT THE EXPENSE OF O2 (constant total
        moles / mass in the coflow stream).  So Y_O2 drops linearly:
            Y_O2_eff = Y_O2_baseline × (1 - Y_agent / Y_N2_baseline)
        where Y_N2_baseline = 1 - Y_O2_baseline ≈ 0.768 for air.
        """
        Y_N2_baseline = 1.0 - self.Y_O2_coflow
        if Y_N2_baseline <= 0.0:
            return self.Y_O2_coflow
        return self.Y_O2_coflow * (1.0 - self.Y_agent_coflow / Y_N2_baseline)

    def configure(self, proj_solver, grid, state) -> None:
        """Install z-min inlet BC + initialize interior for a cup burner run.

        1. Build fuel/coflow/wall geometry masks.
        2. Build w_inlet_zmin (U_fuel in fuel cells, U_coflow in coflow
           cells, 0 in wall cells) and install into the projection solver.
        3. Compute per-cell species inlet ghosts (Y_F, Y_O2, Y_H2O)
           and expose them on the BC as ``.Y_F_inlet_zmin`` etc. so the
           main loop can pass them into species_3d.step_species_transport
           and the T_g advection.
        4. Initialize state.T_g to T_inlet (298 K) everywhere.
        5. Seed a small hot spot at the fuel-coflow interface to
           bootstrap the diffusion flame.
        """
        self._build_masks(grid)
        w = self._build_w_inlet_zmin(grid)

        # (1) Velocity BC into projection solver
        proj_solver.set_bottom_inlet_BC(w)

        # (2) Species inlet ghost arrays — (Ny, Nx), one per species.
        # Fuel cells: Y_F=1, Y_O2=0.  Coflow cells: Y_F=0, Y_O2=effective.
        # Wall cells: irrelevant (velocity ghost is 0 so no advective flux).
        Y_O2_eff = self.effective_Y_O2_coflow()

        self.Y_F_inlet_zmin = np.zeros((grid.Ny, grid.Nx))
        self.Y_F_inlet_zmin[self.fuel_mask] = self.Y_F_fuel

        self.Y_O2_inlet_zmin = np.zeros((grid.Ny, grid.Nx))
        self.Y_O2_inlet_zmin[self.coflow_mask] = Y_O2_eff

        # Y_H2O is zero in both streams (dry air + dry fuel)
        self.Y_H2O_inlet_zmin = np.zeros((grid.Ny, grid.Nx))

        # (3) Temperature inlet ghost — uniform T_inlet
        self.T_inlet_zmin = np.full((grid.Ny, grid.Nx), self.T_inlet_K)

        # (4) Initialize interior — no bed, no flame yet.  Just set
        # atmosphere to T_inlet (ambient).  Y_O2 = 0.232 everywhere
        # (coflow composition, ignoring dilution — the inlet ghost
        # will supply the diluted mix at z=0 anyway).
        state.T_g[:, :, :] = self.T_inlet_K

        # (5) Ignition kernel — a pre-mixed stoichiometric CH4-air pocket
        # at 1500 K spanning the full chimney at z ∈ [3, 30] mm above
        # the cup rim (Refactor 2D-ignition-B: extended from the 3-9 mm
        # v6 attempt which burned out in ~4 s before the diffusion flame
        # could establish).  Rationale:
        #  - Bigger pocket → more fuel + O2 inventory → longer burn
        #    (~22 s at v6's decay rate) → diffusion flame has plenty
        #    of time to establish at the natural stabilization point.
        #  - Composition: Y_F = 0.055, Y_O2 = 0.220 (stoichiometric
        #    CH4-air by mass, Drysdale 2011 §3.3).  Chemistry can fire
        #    immediately without waiting for fuel + O2 to diffuse.
        #  - Temperature: 1500 K — above methane's Arrhenius knee
        #    (~800 K) so ignition is instant.
        #  - Density: ρ = P₀/(R·T) so the EoS pass doesn't see a
        #    discontinuity that the projection interprets as a
        #    supersonic transient (learned from the v4→v5 smoke test).
        #  - Extent x: full chimney; y: all y-slabs.
        z_ign_lo = 0.003    # 3 mm above cup rim
        z_ign_hi = 0.030    # 30 mm (10× thicker than v6)
        _z_cum = np.cumsum(grid.dz_arr)
        k_lo = int(np.searchsorted(_z_cum, z_ign_lo))
        k_hi = int(np.searchsorted(_z_cum, z_ign_hi))
        k_lo = max(k_lo, 1)
        k_hi = min(max(k_hi, k_lo + 1), grid.Nz - 1)
        # x extent: full chimney (excluding walls outside chimney_radius)
        x_center = 0.5 * (grid.x_mid[0] + grid.x_mid[-1])
        i_lo = int(np.searchsorted(grid.x_mid, x_center - self.chimney_radius_m))
        i_hi = int(np.searchsorted(grid.x_mid, x_center + self.chimney_radius_m))
        T_ign = 1500.0
        Y_F_ign = 0.055    # stoichiometric CH4 in air by mass
        Y_O2_ign = 0.220
        state.T_g[k_lo:k_hi, :, i_lo:i_hi]    = T_ign
        state.Y_fuel[k_lo:k_hi, :, i_lo:i_hi] = Y_F_ign
        state.Y_O2[k_lo:k_hi, :, i_lo:i_hi]   = Y_O2_ign
        # Consistent ρ = P₀/(R·T)
        rho_ign = 101325.0 / (287.0 * T_ign)
        state.rho[k_lo:k_hi, :, i_lo:i_hi] = rho_ign

    def species_inlet_kwargs(self, species_name: str) -> dict:
        """Return the kwargs for step_species_transport for a given
        species name so the main loop can plug them in via **kwargs."""
        key = {
            "Y_fuel": "Y_F_inlet_zmin",
            "Y_O2":   "Y_O2_inlet_zmin",
            "Y_H2O":  "Y_H2O_inlet_zmin",
        }.get(species_name)
        if key is None or getattr(self, key, None) is None:
            return dict(Y_inlet_zmin=np.zeros((1, 1)),
                        z_min_inlet_active=False)
        return dict(Y_inlet_zmin=getattr(self, key),
                    z_min_inlet_active=True)

    def apply_per_step(self, state, grid, t: float) -> None:
        """Wick fuel source — force Y_fuel to stoichiometric value in a
        rectangular region above the cup rim at every timestep.

        Motivation (Phase 23 wick option B): the slab (Ny=1) diffusion-
        flame smoke sims lift off + blow out because heat from the
        ignition pocket buoys upward faster than fuel can diffuse across
        the shear layer to sustain a flame at the natural stabilization
        height.  A "wick" injects fresh fuel volumetrically right where
        the flame stabilizes — the FDS/FireFOAM standard for cup burner
        sims.  Extinction physics is preserved: as Y_agent rises in the
        coflow, the local mixture at the wick gets diluted → adiabatic
        flame T drops → chemistry rate collapses → flame extinguishes.
        We DELIBERATELY do NOT pin T here — that would eliminate the
        MEC-relevant extinction mode.

        Wick geometry: same fuel-jet disc (or stripe) in x-y as the
        z-min inlet, but at z ∈ [z_wick_lo, z_wick_hi] (default 3-9 mm)
        instead of z = 0.  Only fuel_mask cells are wick cells.
        """
        # Only active if wick is configured (init-time flag)
        if not getattr(self, "wick_enable", False):
            return
        # Compute wick z-range indices lazily (first call caches them)
        if getattr(self, "_wick_k_lo", None) is None:
            z_cum = np.cumsum(grid.dz_arr)
            self._wick_k_lo = max(int(np.searchsorted(z_cum, self.wick_z_lo)),
                                  1)
            self._wick_k_hi = min(max(int(np.searchsorted(z_cum, self.wick_z_hi)),
                                       self._wick_k_lo + 1),
                                  grid.Nz - 1)
        k_lo, k_hi = self._wick_k_lo, self._wick_k_hi
        # Broadcast fuel_mask (Ny, Nx) to (k_hi - k_lo, Ny, Nx)
        wick_mask = np.broadcast_to(self.fuel_mask,
                                     (k_hi - k_lo, grid.Ny, grid.Nx))
        # Dirichlet on Y_F: force to stoich (0.055) in wick region.
        # Also zero out Y_O2 there (fuel jet displaces O2) so the
        # combustion happens at the wick-coflow interface, not inside
        # the wick.  Density updated from EoS below.
        state.Y_fuel[k_lo:k_hi][wick_mask] = self.wick_Y_F
        state.Y_O2  [k_lo:k_hi][wick_mask] = 0.0
