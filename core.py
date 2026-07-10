"""Core logic of the EEG storage sizing tool.

Simulates a battery storage operating on community load and
generation time series and determines the storage capacity required
to reach target values for self-sufficiency (Autarkiegrad) and
self-consumption (Eigenverbrauchsquote).

All time series are energy amounts per timestep in kWh sharing a
common ``DatetimeIndex`` (e.g. 15-min metered data or hourly
synthetic profiles).

KPI definitions:
    autarky          = 1 - grid_import / load
    self_consumption = 1 - grid_export / generation
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class StorageParams:
    """Technical parameters of the simulated battery storage.

    Args:
        c_rate (float): Power-to-capacity ratio [1/h]. 0.5 is a
            typical value for residential/commercial batteries.
        roundtrip_eff (float): Round-trip efficiency (0-1), split
            symmetrically between charging and discharging.
    """

    c_rate: float = 0.5
    roundtrip_eff: float = 0.90


def timestep_hours(index: pd.Index) -> float:
    """Return the timestep length of a datetime index in hours.

    Args:
        index (pd.Index): Datetime index with at least two entries
            and (mostly) regular spacing.

    Returns:
        float: Timestep length in hours (e.g. 0.25 for 15-min data).
    """
    # Median statt erster Differenz, damit einzelne Lücken in den
    # Messdaten das Ergebnis nicht verfälschen.
    diffs = np.diff(index.values).astype('timedelta64[s]')
    return float(np.median(diffs.astype(float)) / 3600.0)


def simulate_storage(generation: pd.Series, load: pd.Series,
                     capacity_kwh: float,
                     params: Optional[StorageParams] = None
                     ) -> pd.DataFrame:
    """Simulate the charge/discharge logic of a battery storage.

    The battery charges whenever generation exceeds load and
    discharges whenever load exceeds generation
    (self-consumption-oriented, not grid-supportive).

    Args:
        generation (pd.Series): Generation per timestep [kWh].
        load (pd.Series): Load per timestep [kWh]. Must share the
            index of *generation*.
        capacity_kwh (float): Usable storage capacity [kWh].
        params (StorageParams, optional): Technical storage
            parameters. Defaults to ``StorageParams()``.

    Returns:
        pd.DataFrame: Timestep results with columns ``generation``,
            ``load``, ``grid_import``, ``grid_export``, ``charge``,
            ``discharge``, ``soc`` (all kWh). ``df.attrs`` carries
            ``capacity_kwh`` and ``dt_h``.

    Example:
        result = simulate_storage(gen, load, capacity_kwh=100)
        kpis = compute_kpis(result)
    """
    params = params or StorageParams()

    if not isinstance(generation, pd.Series) \
            or not isinstance(load, pd.Series):
        raise TypeError("generation and load must be pandas Series")
    if len(load) < 2:
        raise ValueError("time series must contain at least two "
                         "timesteps")
    if not generation.index.equals(load.index):
        raise ValueError("generation and load must share the same "
                         "index")
    if capacity_kwh < 0:
        raise ValueError("capacity_kwh must be >= 0")

    dt_h = timestep_hours(load.index)
    gen = generation.to_numpy(dtype=float)
    dem = load.to_numpy(dtype=float)
    n = len(dem)

    # Wirkungsgrad symmetrisch auf Laden/Entladen aufteilen.
    eta = float(np.sqrt(params.roundtrip_eff))
    # Maximaler Energieumsatz pro Zeitschritt [kWh].
    e_limit = capacity_kwh * params.c_rate * dt_h

    soc = 0.0
    soc_arr = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    grid_import = np.zeros(n)
    grid_export = np.zeros(n)

    for i in range(n):
        surplus = gen[i] - dem[i]
        if surplus >= 0.0:
            # Freie Kapazität, ausgedrückt als aufnehmbare
            # Input-Energie (Ladeverluste berücksichtigt).
            room = (capacity_kwh - soc) / eta
            c = min(surplus, e_limit, room)
            soc += c * eta
            charge[i] = c
            grid_export[i] = surplus - c
        else:
            deficit = -surplus
            # Lieferbare Output-Energie (Entladeverluste
            # berücksichtigt).
            available = soc * eta
            d = min(deficit, e_limit, available)
            soc -= d / eta
            discharge[i] = d
            grid_import[i] = deficit - d
        # Rundungsfehler abfangen, damit der SoC exakt in den
        # physikalischen Grenzen bleibt.
        soc = min(max(soc, 0.0), capacity_kwh)
        soc_arr[i] = soc

    result = pd.DataFrame({
        'generation': gen,
        'load': dem,
        'grid_import': grid_import,
        'grid_export': grid_export,
        'charge': charge,
        'discharge': discharge,
        'soc': soc_arr,
    }, index=load.index)
    result.attrs['capacity_kwh'] = float(capacity_kwh)
    result.attrs['dt_h'] = dt_h
    return result


def compute_kpis(result: pd.DataFrame) -> dict:
    """Compute KPIs from a simulation result.

    Args:
        result (pd.DataFrame): Output of :func:`simulate_storage`.

    Returns:
        dict: KPIs with keys ``autarky``, ``self_consumption``,
            ``import_free_share`` (all 0-1), ``generation_kwh``,
            ``load_kwh``, ``grid_import_kwh``, ``grid_export_kwh``,
            ``discharged_kwh``.
    """
    generation_kwh = float(result['generation'].sum())
    load_kwh = float(result['load'].sum())
    grid_import_kwh = float(result['grid_import'].sum())
    grid_export_kwh = float(result['grid_export'].sum())

    autarky = (1.0 - grid_import_kwh / load_kwh
               if load_kwh > 0 else 0.0)
    self_consumption = (1.0 - grid_export_kwh / generation_kwh
                        if generation_kwh > 0 else 0.0)
    # Anteil der Zeitschritte ohne Netzbezug (Toleranz gegen
    # numerisches Rauschen).
    import_free_share = float(
        (result['grid_import'] <= 1e-9).mean()
    )

    return {
        'autarky': autarky,
        'self_consumption': self_consumption,
        'import_free_share': import_free_share,
        'generation_kwh': generation_kwh,
        'load_kwh': load_kwh,
        'grid_import_kwh': grid_import_kwh,
        'grid_export_kwh': grid_export_kwh,
        'discharged_kwh': float(result['discharge'].sum()),
    }


def storage_sweep(generation: pd.Series, load: pd.Series,
                  capacities: Iterable[float],
                  params: Optional[StorageParams] = None
                  ) -> pd.DataFrame:
    """Simulate a range of storage capacities and collect KPIs.

    Args:
        generation (pd.Series): Generation per timestep [kWh].
        load (pd.Series): Load per timestep [kWh].
        capacities (Iterable[float]): Storage capacities to
            simulate [kWh]. Include 0 to obtain the baseline
            without storage.
        params (StorageParams, optional): Technical storage
            parameters.

    Returns:
        pd.DataFrame: One row per capacity, sorted ascending, with
            column ``capacity_kwh`` plus all KPIs from
            :func:`compute_kpis`.
    """
    rows = []
    for cap in sorted(set(float(c) for c in capacities)):
        result = simulate_storage(generation, load, cap, params)
        rows.append({'capacity_kwh': cap, **compute_kpis(result)})
    return pd.DataFrame(rows)


def size_storage(sweep: pd.DataFrame,
                 target_autarky: Optional[float] = None,
                 target_self_consumption: Optional[float] = None,
                 target_import_free_share: Optional[float] = None,
                 max_grid_import_kwh: Optional[float] = None
                 ) -> Optional[dict]:
    """Find the smallest capacity in a sweep that meets all targets.

    All KPIs improve monotonically with capacity, so the first
    feasible row of the sorted sweep is the minimal storage size.

    Args:
        sweep (pd.DataFrame): Output of :func:`storage_sweep`.
        target_autarky (float, optional): Minimum self-sufficiency
            (0-1). ``None`` disables this criterion.
        target_self_consumption (float, optional): Minimum
            self-consumption ratio (0-1). ``None`` disables this
            criterion.
        target_import_free_share (float, optional): Minimum share
            of timesteps without grid import (0-1).
        max_grid_import_kwh (float, optional): Maximum allowed grid
            import over the whole period [kWh].

    Returns:
        dict or None: Row of the smallest feasible capacity as a
            dict, or ``None`` if no simulated capacity meets the
            targets.
    """
    feasible = pd.Series(True, index=sweep.index)
    if target_autarky is not None:
        feasible &= sweep['autarky'] >= target_autarky
    if target_self_consumption is not None:
        feasible &= (sweep['self_consumption']
                     >= target_self_consumption)
    if target_import_free_share is not None:
        feasible &= (sweep['import_free_share']
                     >= target_import_free_share)
    if max_grid_import_kwh is not None:
        feasible &= sweep['grid_import_kwh'] <= max_grid_import_kwh

    matches = sweep[feasible].sort_values('capacity_kwh')
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()
