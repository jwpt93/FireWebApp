from __future__ import annotations

# Prepend cheney-web/src to sys.path (model/, model_outdoor/ live there)
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import sys
import os
import shutil
from typing import Callable

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config.defaults import (
    default_env_config,
    default_fuel_config,
    default_sim_config,
    default_thresholds as _default_thresholds,
)
from model.fuel.pyrolysis import pyrolysis_flux
from tests.utils.plot_paths import get_test_plots_dir


@pytest.fixture(scope="session", autouse=True)
def _prepare_test_plots_dir() -> Path:
    """Prepare plot output dir once per pytest invocation."""
    base = get_test_plots_dir()
    do_clean = os.environ.get("TEST_PLOTS_CLEAN", "1").strip().lower() not in {"0", "false", "no"}
    if do_clean:
        for p in base.iterdir():
            if p.name == ".gitignore":
                continue
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    return base


@dataclass
class DeterministicForcing:
    """Deterministic forcing with callable heat and moisture terms."""

    q_in_func: Callable[[float], float]
    rewet_rate_func: Callable[[float], float] = lambda t: 0.0
    M1_eq_func: Callable[[float], float] = lambda t: 0.0

    def q_in(self, t: float) -> float:
        return float(self.q_in_func(t))

    def rewet_rate(self, t: float) -> float:
        return float(self.rewet_rate_func(t))

    def M1_eq(self, t: float) -> float:
        return float(self.M1_eq_func(t))


def pyrolysis_flux_vectorized(
    T1_array: np.ndarray, M1_array: np.ndarray, fuel_cfg
) -> np.ndarray:
    """Vectorized helper for pyrolysis flux [kg/m^2/s]."""

    return np.array(
        [pyrolysis_flux(float(t1), float(m1), fuel_cfg) for t1, m1 in zip(T1_array, M1_array)]
    )


@pytest.fixture
def dump_debug_output():
    def _dump(test_name: str, result, extra_info: str) -> None:
        """Dump debug outputs for failed tests."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = ROOT / "test_debug" / f"{test_name}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        t = np.asarray(result.t, dtype=float)
        y = np.asarray(result.y, dtype=float)
        if y.ndim != 2:
            raise ValueError("result.y must be a 2D array")
        if y.shape[0] == 3 and y.shape[1] == t.size:
            T1, T2, M1 = y[0], y[1], y[2]
        elif y.shape[1] == 3 and y.shape[0] == t.size:
            T1, T2, M1 = y[:, 0], y[:, 1], y[:, 2]
        else:
            raise ValueError("result.y shape does not match time dimension")

        fuel_cfg = getattr(result, "fuel_cfg", default_fuel_config())
        m_py = pyrolysis_flux_vectorized(T1, M1, fuel_cfg)

        csv_path = out_dir / "result.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "T1", "T2", "M1", "pyrolysis_flux"])
            for row in zip(t, T1, T2, M1, m_py):
                writer.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6e}"])

        summary = f"test: {test_name}\n{extra_info}\n"
        (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

        print(f"\n[DEBUG] {test_name} -> {out_dir}")  # noqa: T201
        print(extra_info)  # noqa: T201

    return _dump


@pytest.fixture
def default_fuel_cfg():
    return default_fuel_config()


@pytest.fixture
def default_env_cfg():
    return default_env_config()


@pytest.fixture
def default_sim_cfg():
    return default_sim_config()


@pytest.fixture
def default_thresholds():
    return _default_thresholds()


@pytest.fixture
def forcing_heat_on() -> DeterministicForcing:
    def q_in(t: float) -> float:
        if t < 10.0:
            return 2.0e4 * (t / 10.0)
        return 2.0e4

    return DeterministicForcing(q_in_func=q_in)


@pytest.fixture
def forcing_heat_off() -> DeterministicForcing:
    return DeterministicForcing(q_in_func=lambda t: 0.0)


@pytest.fixture
def forcing_pulse() -> DeterministicForcing:
    def q_in(t: float) -> float:
        return 3.0e4 if t < 20.0 else 0.0

    return DeterministicForcing(q_in_func=q_in)
