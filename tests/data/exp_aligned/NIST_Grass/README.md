# NIST_Grass — Retired Validation Target

**Status: NO DATA AVAILABLE — this directory is retired as a validation target.**

## Finding (2026-03-26)

A systematic search identified no piloted cone calorimeter HRRPUA time-series
data for any fescue species (*Festuca* spp.) or cheat grass (*Bromus tectorum*)
in any freely accessible database or published literature:

| Source | Grass data present? | Test type | Usable? |
|--------|---------------------|-----------|---------|
| NIST Fire Calorimetry Database (FCD) | Little bluestem only | Large-scale open crib burning (NFRL, ≥50 kW scale) — NOT ISO 5660 cone | No |
| NIST TN 1481 (Pitts 2007) | Tall fescue at 35.2 kW/m², cheat grass at 45 kW/m² | Non-piloted, smoldering focus — not ISO 5660 | No |
| BGS/NERC wildfire dataset (doi:10.5285/45af9c3d) | UK grass species at 50 kW/m² | Scalar peak HRR only — no time series | No |
| White & Zipperer (2010) Int. J. Wildland Fire | Ornamental plants (no fescue) | Scalar peak HRR only | No |
| Weise & White USDA FS papers | Ornamental / western shrubs | Scalar peak HRR only | No |

**Bottom line**: No piloted ISO 5660 cone calorimeter HRRPUA time-series for
fescue or cheat grass exists in any identified source. The original L2-A
validation case (tall fescue 25 kW/m² CAL / 50 kW/m² VAL from NIST FCD) was
based on an incorrect assumption that the NIST FCD contained ISO 5660 cone data
for these species. This has been corrected.

The NIST FCD bluestem grass data is large-scale pile/crib burning from the
National Fire Research Laboratory — a different test geometry entirely.

## Current Grass Validation Status

Grass-class kinetics are validated through the Chen et al. (2021) crop straw
cone calorimeter dataset:
- Wheat straw (*Triticum aestivum*) 50 kW/m² — CALIBRATION CASE ✓ PASS
- Rice straw (*Oryza sativa*) 50 kW/m² — VALIDATION ✓ PASS
- Corn straw (*Zea mays*) 50 kW/m²  — VALIDATION (see `Chen_Straw_Cone/` directory)

GR1 outdoor fuel bed decks (`Outdoor_Grass_GR1__CONE_25.txt`, `CONE_50.txt`)
are forward predictions using the transferred wheat straw kinetics. No EXP
comparison is planned until cone calorimeter data for an appropriate grass
species becomes available.

## If Grass Cone Data Becomes Available in Future

If piloted ISO 5660 HRRPUA time-series data for any native grass species
(fescue, bluestem, brome) at 25 kW/m² and 50 kW/m² is obtained:
1. Declare the cal/val split in this README before examining the data (Rule #2)
2. Declare acceptance bands before running (Rule #3)
3. Update `inputs/validation_cases/Outdoor_Grass_GR1__CONE_25.txt` designation
   to CALIBRATION and tune A_py/E_py to the 25 kW/m² case only
4. Report results and update `docs/user_guide/ch_outdoor_wip.tex`
