/**
 * The one configuration the live-solver page ships.
 *
 * Kept in its own module so it can be imported and asserted by tests without
 * pulling in the page's DOM code. Every value here was measured; the comments
 * say what by.
 */
import { A_CH } from './cheney.js';

// Phase 20 "Option B" blend. Below 2.5 m/s the Cheney fit drives the front
// entirely; 2.5-3.5 ramps; at and above 3.5 the resolved solver drives it.
export const U_THRESH = 3.5, U_BLEND_W = 1.0;

// Hard floor at 1 m/s. Cheney 1993 did not measure below it, so there is no
// basis for the fit there and no sub-1 m/s relation to fall back on.
export const U_MIN = 1.0;

// ONE configuration, and every value in it is measured rather than guessed.
// See SOLVER_PORT.md section 8 for the numbers behind each choice.
export const CFG_NOTE =
  '12 m domain, dx 0.10 m, atmosphere growing to a 1 m cap. Every setting here '
  + 'was measured, not guessed: N_SUB 1 (upstream study, 6 cases, worst 1.9%), '
  + 'projection tolerance 1e-4 (ROS unchanged to 4 dp), n_z_bed 4 (8 costs 4.5x '
  + 'wall time for +13% at one wind and -5% at another), and a passive level set '
  + 'with the spread rate read off the solid-fuel front, which is how the parent '
  + "project's validation workers measure it.";

export const CFG = {
  // Mesh: the growing atmosphere the deck asks for (26 z-cells, not 320).
  Lx: 12.0, dx: 0.10, Lz: 8.0, bedXStart: 1.0, bedXEnd: 9.0,
  wallBlN: 1, wallBlFirstDz: 0.025, wallBlGrowth: 1.0,
  Ly: 0.10, dy: 0.10, nZBed: 4, hBed: 0.10, rhoB: 1.07,
  sigmaSav: 2000.0, canopyCd: 0.30, initialMoistureFrac: 0.04,
  atmGrowth: 1.20, atmMaxDz: 1.0,
  cflFactor: 0.40, minDtS: 1.0e-4,
  ignitionDurationS: 3.0, ignitionQMult: 3.0, ignitionWidthMult: 3.0,
  ignitionTPinEnable: false,
  solidPhaseIgnitionEnable: true, solidPhaseIgnitionTsK: 1000.0,
  lagrangianBedNPerCell: 4, lagrangianBedDryingMode: 'combined',
  lagrangianBedHConv: 250.0, lagrangianBedViewFactorGeometric: true,
  domSubcycleEvery: 5, wallFunction: false,
  // Passive level set is the CANONICAL high-wind setting, not a disabled one.
  // With it the resolved bed physics carries the fire and the solver reports
  // ROS_Ts, the solid-fuel front -- the metric the validation workers use.
  levelSetPassive: true,
  // Phase 19/20 hybrid ON -- the production low-wind configuration. Without
  // it the resolved closure cannot propagate below ~3.5 m/s and the applet
  // would quietly under-predict instead of failing.
  // a_ch = CUT, not natural. The bed here is rho_b = 1.07, h_bed = 0.10,
  // sav = 2000 -- which is Cheney 1993 *Cut* grass verbatim (see
  // Outdoor_Cheney_Cut4 deck). Feeding the natural-sward coefficient 0.406 to
  // a cut-grass bed overstated the reference ROS by 1/0.845 = 1.18x, so both
  // the empirical branch and every ratio measured against it were wrong.
  empiricalRosEnable: true, empiricalRosACh: A_CH.cut,
  empiricalRosUThresholdMs: U_THRESH, empiricalRosBlendWidthMs: U_BLEND_W,
  // N_SUB = 1, not the upstream default of 10.
  //
  // Upstream notes that N_SUB "has never had a convergence study -- it is a
  // hardcoded constant justified by splitting theory, not by measurement".
  // The study was run (scripts/run_2d_nsub_validation.py, 2D production mesh):
  // six Cheney cases at N_SUB 10 vs 1, worst ROS deviation 1.9% against a 5%
  // band, all pass. Reproduced independently in this port at -0.01% on a 6 s
  // run. It is worth 1.78x here -- the chemistry sub-loop drops from 48% of
  // step time to 8%.
  //
  // The SOLVER's own default stays at 10, faithful to upstream. This is the
  // applet making an explicit, measured choice.
  nSub: 1,
  // Projection inner tolerance 1e-4, not the upstream 1e-6.
  //
  // The Krylov solve feeds an OUTER loop that iterates on the actual
  // divergence residual to projDivTol = 1e-3. An inner tolerance three orders
  // tighter than the thing consuming it is resolving detail that gets thrown
  // away. Measured, 12 m / dx 0.10, ROS identical to 4 decimals throughout:
  //
  //   rtol    ms/step   projection   proj iters   div residual
  //   1e-6     16.4       10.06         1.00        6.1e-6
  //   1e-5     13.9        7.58         1.00        5.9e-5
  //   1e-4     11.7        5.23         1.00        5.7e-4
  //   1e-3     34.2       21.68         1.71        1.0e-3   <- cliff
  //
  // There is a cliff, not a gradient. Past ~3e-4 the divergence residual
  // reaches projDivTol and the outer loop needs a SECOND projection, which
  // costs far more than the loosened inner tolerance saved. 1e-4 keeps about
  // 2x margin to it. If a stiffer case ever crosses anyway the failure is
  // graceful -- an extra outer iteration, so slower, not wrong.
  //
  // As with nSub, the SOLVER's own default stays at the upstream 1e-6.
  projectionCgRtol: 1.0e-4,
};

