import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable


@dataclass
class SimConfig:
    pv_scale: float = 1.0
    storage_kwh: float = 0.0
    c_rate: float = 1.0
    roundtrip_eff: float = 0.9
    allow_grid_charge: bool = False
    initial_soc: float = 0.0

@dataclass
class CostModel:
    """Simple configurable cost model for PV and storage.

    - __call__ implements a linear cost: pv_scale * pv_cost_per_unit + storage_kwh * storage_cost_per_kwh
    - .annualized(...) returns an annuitized annual cost using a simple annuity factor
    """
    pv_cost_per_unit: float = 1200
    storage_cost_per_kwh: float = 800
    ref_pv_kw: float = 50.0  # reference kWp corresponding to pv_scale==1

    def __call__(self, pv_scale: float, storage_kwh: float) -> float:
        return float(pv_scale * self.pv_cost_per_unit + storage_kwh * self.storage_cost_per_kwh)

    @staticmethod
    def _annuity_factor(r: float, n: int) -> float:
        return r / (1 - (1 + r) ** -n)

    def annualized(self, pv_scale: float, storage_kwh: float,
                   life_pv: int = 25, life_storage: int = 10, discount: float = 0.05) -> float:
        pv_kw = pv_scale * self.ref_pv_kw
        pv_capex = pv_kw * self.pv_cost_per_unit
        storage_capex = storage_kwh * self.storage_cost_per_kwh
        annual_pv = pv_capex * self._annuity_factor(discount, life_pv)
        annual_storage = storage_capex * self._annuity_factor(discount, life_storage)
        return float(annual_pv + annual_storage)

# default model instance
DEFAULT_COST_MODEL = CostModel()

class EEGModel:
    """Simulation model for an EEG (Erneuerbare-Energie-Gemeinschaft) with battery storage.

    One generation source is *scalable* (typically PV) – its output is multiplied
    by a scale factor in every simulation run so the optimizer can size it.
    All other sources (hydro, drinking-water turbines, …) are *fixed*: their
    profiles are added to total generation as-is.

    Quick start
    -----------
    >>> model = EEGModel(pv_profile, load_profile)
    >>> model = EEGModel(pv_profile, load_profile,
    ...                  fixed_sources={'hydro': hydro_profile,
    ...                                 'trinkwasser': tw_profile})
    >>> df  = model.simulate(pv_scale=2.0, storage_kwh=100)
    >>> res = model.optimize_grid(pv_scales=[1,2,3], storage_grid=[0,50,100])
    """

    def __init__(self, scalable_gen: pd.Series, load: pd.Series,
                 fixed_sources: Optional[Dict[str, pd.Series]] = None,
                 name: str = "EEG"):
        """
        Parameters
        ----------
        scalable_gen : pd.Series
            Hourly generation profile of the scalable source in kW
            (e.g. a 1-kWp-normalized PV profile from PVGIS).
            Multiplied by *pv_scale* in :meth:`simulate`.
        load : pd.Series
            Hourly community load profile in kW. Must share the same index.
        fixed_sources : dict[str, pd.Series], optional
            Further generation sources with fixed (non-scalable) output.
            Keys become column names in the result DataFrame.
            All series must share the same index as *load*.
        name : str
            Human-readable label used in plots and reports.
        """
        if not isinstance(scalable_gen, pd.Series) or not isinstance(load, pd.Series):
            raise TypeError("scalable_gen and load must be pandas Series")
        if len(scalable_gen) < 2 or len(load) < 2:
            raise ValueError("scalable_gen and load must contain at least two timesteps")
        if not scalable_gen.index.equals(load.index):
            raise ValueError("scalable_gen and load must have identical indices.")

        fixed_sources = fixed_sources or {}
        for src_name, src_series in fixed_sources.items():
            if not isinstance(src_series, pd.Series):
                raise TypeError(f"fixed_sources['{src_name}'] must be a pd.Series")
            if not src_series.index.equals(load.index):
                raise ValueError(f"fixed_sources['{src_name}'] index does not match load.")

        self.scalable_gen = scalable_gen.copy()
        self.fixed_sources: Dict[str, pd.Series] = {k: v.copy() for k, v in fixed_sources.items()}
        self.load = load.copy()
        self.name = name
        self.dt_h = (load.index[1] - load.index[0]).total_seconds() / 3600.0

        # results of the most recent simulation / optimization
        self.last_run: Optional[pd.DataFrame] = None
        self.last_summary: Optional[Dict[str, Any]] = None
        self.best_run: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.optim_results: Optional[pd.DataFrame] = None

        # summary-only cache  (pv_scale, storage_kwh, c_rate, eff, grid_charge, soc0) → summary
        self._cache: Dict[Tuple, Dict[str, Any]] = {}

    def _cache_key(self, pv_scale, storage_kwh, c_rate,
                   roundtrip_eff, allow_grid_charge, initial_soc):
        return (float(pv_scale), float(storage_kwh), float(c_rate),
                float(roundtrip_eff), bool(allow_grid_charge), float(initial_soc))

    def simulate(self, pv_scale: float = 1.0, storage_kwh: float = 0.0,
                 c_rate: float = 1.0, roundtrip_eff: float = 0.9,
                 allow_grid_charge: bool = False, initial_soc: float = 0.0,
                 keep_timeseries: bool = True) -> Optional[pd.DataFrame]:
        """Simulate one timestep-by-timestep energy balance and return results.

        The scalable generation source is multiplied by *pv_scale*; fixed
        sources are added unchanged.  A battery with capacity *storage_kwh*
        buffers surplus generation and covers deficits before grid import.

        Parameters
        ----------
        pv_scale : float
            Scale factor for the scalable generation source.
        storage_kwh : float
            Battery usable capacity in kWh.
        c_rate : float
            Battery C-rate [1/h]; limits per-timestep charge/discharge power.
        roundtrip_eff : float
            Round-trip efficiency (0–1); split symmetrically between charge
            and discharge (``eff = sqrt(roundtrip_eff)``).
        allow_grid_charge : bool
            Reserved for future use (grid charging not yet implemented).
        initial_soc : float
            Battery state-of-charge at simulation start [kWh].
        keep_timeseries : bool
            If False, only the summary dict is returned/cached (faster for
            large grid searches).  Returns ``None`` in that case.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns: ``pv``, *<fixed source names>*,
            ``total_gen``, ``load``, ``net``, ``grid_import``,
            ``grid_export``, ``soc``, ``battery_charge``,
            ``battery_discharge``.  ``None`` when *keep_timeseries* is False.
        """
        config_key = self._cache_key(pv_scale, storage_kwh, c_rate,
                                     roundtrip_eff, allow_grid_charge, initial_soc)

        if config_key in self._cache and not keep_timeseries:
            self.last_summary = self._cache[config_key]
            return None

        dt_h = self.dt_h
        pv = self.scalable_gen * pv_scale
        total_gen = pv.copy()
        for src in self.fixed_sources.values():
            total_gen = total_gen + src
        load = self.load
        n = len(load)

        soc = float(initial_soc)
        soc_arr        = np.zeros(n)
        grid_import    = np.zeros(n)
        grid_export    = np.zeros(n)
        batt_charge    = np.zeros(n)
        batt_discharge = np.zeros(n)

        eff = np.sqrt(roundtrip_eff) if roundtrip_eff > 0 else 1.0
        p_limit = storage_kwh * c_rate * dt_h   # max energy transfer per timestep [kWh]

        for i in range(n):
            gen  = float(total_gen.iloc[i])
            dem  = float(load.iloc[i])
            net  = gen - dem

            if net >= 0:
                charge = min(net, p_limit, max(0.0, storage_kwh - soc))
                soc += charge * eff
                batt_charge[i] = charge
                grid_export[i] = net - charge
            else:
                deficit = -net
                max_discharge = min(p_limit, soc * eff)
                discharge = min(deficit, max_discharge)
                if discharge > 0:
                    soc -= discharge / eff
                    batt_discharge[i] = discharge
                    deficit -= discharge
                grid_import[i] = deficit if deficit > 1e-12 else 0.0

            soc = max(0.0, min(soc, storage_kwh))
            soc_arr[i] = soc

        df = pd.DataFrame({
            'pv': pv.values,
            **{src_name: src.values for src_name, src in self.fixed_sources.items()},
            'total_gen':        total_gen.values,
            'load':             load.values,
            'net':              total_gen.values - load.values,
            'grid_import':      grid_import,
            'grid_export':      grid_export,
            'soc':              soc_arr,
            'battery_charge':   batt_charge,
            'battery_discharge':batt_discharge,
        }, index=load.index)

        summary = self.compute_metrics(df)
        df.attrs['summary'] = summary
        df.attrs['config'] = dict(pv_scale=pv_scale, storage_kwh=storage_kwh,
                                  c_rate=c_rate, roundtrip_eff=roundtrip_eff)

        self._cache[config_key] = summary
        self.last_summary = summary
        if keep_timeseries:
            self.last_run = df

        return df if keep_timeseries else None

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute KPIs from a simulation result DataFrame."""
        n = len(df)
        timesteps_with_import = int(np.sum(df['grid_import'] > 1e-9))

        daily_import = df['grid_import'].groupby(df.index.date).sum()
        days_total = len(daily_import)
        days_with_import = int((daily_import > 1e-9).sum()) if days_total else 0
        days_self_sufficient = float((days_total - days_with_import) / days_total) if days_total else 0.0

        total_gen   = float(df['total_gen'].sum() * self.dt_h)
        total_load  = float(df['load'].sum()      * self.dt_h)
        grid_import = float(df['grid_import'].sum()* self.dt_h)
        grid_export = float(df['grid_export'].sum()* self.dt_h)
        self_consumption = float((df['total_gen'] - df['grid_export']).clip(lower=0).sum() * self.dt_h)

        return {
            'total_gen_kwh':             total_gen,
            'total_load_kwh':            total_load,
            'grid_import_kwh':           grid_import,
            'grid_export_kwh':           grid_export,
            'self_consumption_kwh':      self_consumption,
            'self_sufficiency':          1.0 - grid_import / total_load if total_load else 0.0,
            'days_self_sufficient':      days_self_sufficient,
            'timesteps_with_import':     timesteps_with_import,
        }

    def plot_timeseries(self, df: Optional[pd.DataFrame] = None, start: Optional[str] = None,
                        end: Optional[str] = None, cols: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (12, 8)):
        """Plot simulation timeseries. Returns (fig, axes)."""
        if df is None:
            if self.last_run is None:
                raise ValueError("No timeseries available. Run simulate() first.")
            df = self.last_run
        if start or end:
            df = df.loc[start:end]

        default_cols = ['pv', *self.fixed_sources.keys(), 'total_gen',
                        'load', 'grid_import', 'grid_export', 'soc']
        cols = cols or [c for c in default_cols if c in df.columns]

        axes = df[cols].plot(subplots=True, figsize=figsize)
        for ax in axes:
            ax.set_ylabel('kW / kWh')
        fig = axes[0].get_figure()
        cfg = df.attrs.get('config', {})
        fig.suptitle(f"{self.name}  |  PV-Faktor: {cfg.get('pv_scale')}×  "
                     f"Speicher: {cfg.get('storage_kwh')} kWh")
        fig.tight_layout()
        return fig, axes

    def optimize_grid(self, pv_scales, storage_grid,
                      c_rate: float = 1.0, roundtrip_eff: float = 0.9,
                      metric: str = 'days_self_sufficient',
                      maximize: bool = True) -> pd.DataFrame:
        """Brute-force grid search over all (pv_scale, storage_kwh) combinations.

        Re-simulates the best point with a full timeseries at the end.
        """
        results = []
        for pv_s, stor in tqdm(list(itertools.product(pv_scales, storage_grid)),
                               desc='Grid search'):
            key = self._cache_key(pv_s, stor, c_rate, roundtrip_eff, False, 0.0)
            summary = self._cache.get(key)
            if summary is None:
                self.simulate(pv_scale=pv_s, storage_kwh=stor, c_rate=c_rate,
                              roundtrip_eff=roundtrip_eff, keep_timeseries=False)
                summary = self.last_summary
            results.append(dict(pv_scale=pv_s, storage_kwh=stor, **summary))

        res_df = (pd.DataFrame(results)
                  .sort_values(metric, ascending=not maximize)
                  .reset_index(drop=True))
        self.optim_results = res_df

        if not res_df.empty:
            best = res_df.iloc[0]
            self.best_params = best.to_dict()
            self.simulate(pv_scale=best['pv_scale'], storage_kwh=best['storage_kwh'],
                          c_rate=c_rate, roundtrip_eff=roundtrip_eff, keep_timeseries=True)
            self.best_run = self.last_run
        return res_df

    def optimize_min_cost(self, pv_scales, storage_grid,
                          threshold_metric: str = 'days_self_sufficient',
                          threshold: float = 0.75, op: str = '>=',
                          cost_model: Optional[CostModel] = None,
                          cost_fn: Optional[Callable[[float, float], float]] = None,
                          c_rate: float = 1.0, roundtrip_eff: float = 0.9) -> pd.DataFrame:
        """Find the cheapest (pv_scale, storage_kwh) that meets a threshold on any metric.

        Cost precedence: *cost_fn* > *cost_model* > ``DEFAULT_COST_MODEL``.

        Returns a DataFrame of feasible solutions sorted by ascending cost.
        """
        ops = {'>=': lambda v: v >= threshold, '<=': lambda v: v <= threshold,
               '>':  lambda v: v > threshold,  '<':  lambda v: v < threshold}
        if op not in ops:
            raise ValueError(f"op must be one of {list(ops)}")
        satisfies = ops[op]
        _cost_fn = cost_fn or (cost_model or DEFAULT_COST_MODEL)

        feasible = []
        for pv_s, stor in tqdm(list(itertools.product(pv_scales, storage_grid)),
                               desc='Cost-optimal search'):
            key = self._cache_key(pv_s, stor, c_rate, roundtrip_eff, False, 0.0)
            summary = self._cache.get(key)
            if summary is None:
                self.simulate(pv_scale=pv_s, storage_kwh=stor, c_rate=c_rate,
                              roundtrip_eff=roundtrip_eff, keep_timeseries=False)
                summary = self.last_summary
            val = (summary or {}).get(threshold_metric)
            if val is None or not satisfies(val):
                continue
            cost = float(_cost_fn(pv_s, stor))
            feasible.append(dict(pv_scale=pv_s, storage_kwh=stor, cost=cost, **summary))

        if not feasible:
            self.best_params = self.best_run = None
            return pd.DataFrame()

        feasible_df = (pd.DataFrame(feasible)
                       .sort_values(['cost', 'pv_scale', 'storage_kwh'])
                       .reset_index(drop=True))
        self.optim_results = feasible_df
        best = feasible_df.iloc[0]
        self.best_params = best.to_dict()
        self.simulate(pv_scale=best['pv_scale'], storage_kwh=best['storage_kwh'],
                      c_rate=c_rate, roundtrip_eff=roundtrip_eff, keep_timeseries=True)
        self.best_run = self.last_run

        print(f"Gefunden: {len(feasible_df)} Lösungen mit {threshold_metric} {op} {threshold}.")
        print(f"Beste Lösung: pv_scale={best['pv_scale']}, storage_kwh={best['storage_kwh']}, "
              f"Kosten={best['cost']:.0f}")
        return feasible_df