# Chen et al. (2021) Crop Straw Cone Calorimeter Data

## Source

Chen, X., Xu, Y., Jiang, Y., Li, Z., Zhou, Y., & Li, J. (2021).
Fire behavior characteristics of crop straw in cone calorimeter tests.
*Materials*, 14(12), 3407.
https://doi.org/10.3390/ma14123407
Open access (MDPI + PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8234294/)

## Test Conditions (ISO 5660)

- Apparatus: cone calorimeter, spark ignition
- Incident flux: 50 kW/m²
- Sample holder: 100 × 100 × 35 mm stainless steel
- Sample mass: 60 g per specimen
- Fuel load (dry basis): ~5.66 kg/m²  (60 g / 0.01 m² at 6% dry-basis MC)
- Bulk density (packed): ~171 kg/m³  (5.66 kg/m² / 0.035 m)
- Exhaust flow: 24 L/s, test duration: 900 s

## Cal/Val Split (Rule #2 — declared before data examined)

- CALIBRATION: Wheat straw (*Triticum aestivum*) at 50 kW/m²
  → Used to tune A_py/E_py for dry grass kinetics
- VALIDATION:  Rice straw (*Oryza sativa*) at 50 kW/m²
  → Independent species check; same flux, same geometry

Corn straw excluded (higher lignin content, likely different kinetics class).

## Acceptance Bands (Rule #3 — declared before running)

| Case | Peak R/E | Avg R/E (30–300 s) |
|------|----------|---------------------|
| Wheat straw 50 kW/m² (CAL) | 0.70–1.30 | 0.70–1.30 |
| Rice straw 50 kW/m² (VAL) | 0.70–1.30 | 0.70–1.30 |

Wide initial bands: kinetics are starting values only (A_py=5e10, E_py=1.25e5).
Tighten to 0.85–1.15 after calibration is complete and Rule #3 is re-declared.

## Scalar Metrics (Table 3, Chen et al. 2021)

| Parameter | Rice | Wheat | Corn |
|-----------|------|-------|------|
| TTI [s] | 8 ± 1 | 7 ± 1 | 10 ± 1 |
| Peak HRRPUA [kW/m²] | 104 ± 3 | 114 ± 1 | 167 ± 2 |
| Time to peak [s] | 21 ± 2 | 17 ± 3 | 13 ± 2 |
| THR at 900 s [MJ/m²] | 46 ± 2 | 55 ± 4 | 57 ± 3 |
| Moisture [%, dry basis] | 7.5 | 6.1 | 8.2 |

## Moisture Content Note

Chen reports moisture content on dry basis. Deck parameters use the packed
bulk density (171 kg/m³) computed from the known sample geometry; moisture
is not separately tracked (fuel_state mode, M1=1.0 = fully loaded fuel bed).

## HRR Curve Shape

"Thermally thick charring" — sharp early peak (at TTI + 10–14 s), followed
by slow monotone decline to a plateau (~50–65 kW/m²), with slight upturn
near 900 s from char oxidation.

## Physical Justification as Grass Surrogate

Crop straws (wheat, rice) are cellulosic annual grass stems. Their chemical
composition (cellulose ~40%, hemicellulose ~25%, lignin ~15%) is representative
of the grass fuel class. A_py/E_py calibrated from these data represent the
thermal decomposition kinetics of dry cellulosic fine fuels, and are physically
transferable to the GR1 outdoor fuel bed geometry (Anderson 1982) because the
Arrhenius parameters characterise the specific decomposition rate [1/s] at a
given temperature, independent of fuel bed packing or geometry.
(Per Rule #1: physical justification required for surrogate use.)

## CSV Format

Time [s], HRRPUA [kW/m²]
Data digitized from Figure 3(a) of Chen et al. (2021).
Digitization uncertainty: ±5 kW/m² (visual read from published figure).
Scalar anchor points (peak, TTI) taken from Table 3 of the paper.
