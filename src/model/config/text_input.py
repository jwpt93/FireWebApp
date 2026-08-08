# backward compat shim — text_input.py moved to model.io.text_input
from model.io.text_input import *  # noqa: F401, F403
from model.io.text_input import (  # noqa: F401
    load_text_input,
    RomInputs,
    q_in_callable,
    ramped_q_in_callable,
    q_inc_ramp_factor,
    resolve_geometry,
    apply_material_geometry,
    hoc_eff_to_j_per_kg,
    hoc_eff_to_kj_per_kg,
    normalize_hoc_units,
    convert_q_in,
    convert_m_py,
)
