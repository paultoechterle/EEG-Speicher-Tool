"""
synthetic_profiles.py
------------
Synthetic generation profiles for renewable energy systems.

Class hierarchy (extensible):
    GenerationProfile          – abstract base class
        PVProfile              – PV via PVGIS seriescalc API
        HydroProfile           – (future) run-of-river hydro
        DrinkingWaterProfile   – (future) drinking-water turbine
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# PVGIS API base URL (v5_3)
# ---------------------------------------------------------------------------
PVGIS_BASE = "https://re.jrc.ec.europa.eu/api/v5_3"

# PVGIS liefert Daten nur bis Ende 2023
PVGIS_MAX_YEAR = 2023

# Lokaler Datei-Cache für PVGIS-Abrufe
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class GenerationProfile(ABC):
    """Abstract base class for renewable generation profiles.

    All subclasses must implement :meth:`fetch` and expose a
    :attr:`profile` property returning a ``pd.Series`` with a
    ``DatetimeIndex`` and values in **kW**.
    """

    _profile: Optional[pd.Series] = None

    @abstractmethod
    def fetch(self) -> pd.Series:
        """Fetch / compute the generation profile.

        Returns
        -------
        pd.Series
            Hourly (or sub-hourly) power values in kW,
            indexed by a timezone-aware or naive ``DatetimeIndex``.
        """

    @property
    def profile(self) -> pd.Series:
        """Cached generation profile in kW.  Calls :meth:`fetch` on first access."""
        if self._profile is None:
            self._profile = self.fetch()
        return self._profile

    def total_energy_kwh(self) -> float:
        """Total energy in kWh over the full period of the profile."""
        dt_h = (self.profile.index[1] - self.profile.index[0]).total_seconds() / 3600.0
        return float(self.profile.sum() * dt_h)

    def annual_energy_kwh(self) -> float:
        """Average annual energy in kWh.

        Divides the total energy by the number of calendar years spanned
        by the profile, so the result is correct for both single- and
        multi-year series (e.g. PVGIS with startyear / endyear).
        """
        dt_h = (self.profile.index[1] - self.profile.index[0]).total_seconds() / 3600.0
        total = float(self.profile.sum() * dt_h)
        n_years = (self.profile.index[-1] - self.profile.index[0]).days / 365.25
        return total / max(n_years, 1.0)

    def normalized(self) -> pd.Series:
        """Profile normalized to peak power (0–1 capacity factor)."""
        peak = self.profile.max()
        if peak == 0:
            return self.profile.copy()
        return self.profile / peak

    def scaled(self, peak_kw: float) -> pd.Series:
        """Return profile scaled to *peak_kw* kW peak power."""
        return self.normalized() * peak_kw


# ---------------------------------------------------------------------------
# PV profile via PVGIS seriescalc
# ---------------------------------------------------------------------------

@dataclass
class PVProfile(GenerationProfile):
    """Synthetic PV generation profile fetched from the PVGIS API.

    Uses the ``seriescalc`` endpoint with ``pvcalculation=1`` to obtain
    hourly AC power output.

    API docs:
    https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (south is negative).
    lon : float
        Longitude in decimal degrees (west is negative).
    peak_power_kw : float
        Installed PV peak power in kWp.
    loss : float
        System losses in percent (default 14 %).
    angle : float
        Tilt angle from horizontal in degrees (default 35°).
    azimuth : float
        Azimuth: 0 = south, 90 = west, -90 = east (default 0).
    optimal_angles : bool
        Let PVGIS optimise both tilt and azimuth (overrides *angle* / *azimuth*).
    optimal_inclination : bool
        Let PVGIS optimise tilt only (overrides *angle*).
    pvtech : str
        PV technology: ``"crystSi"``, ``"crystSi2025"``, ``"CIS"``,
        ``"CdTe"``, ``"Unknown"`` (default ``"crystSi"``).
    mounting : str
        ``"free"`` (free-standing) or ``"building"`` (BIPV).
    raddatabase : str | None
        Radiation database name.  ``None`` = PVGIS default for the location.
    startyear : int | None
        First year of the time series.  ``None`` = DB minimum.
    endyear : int | None
        Last year of the time series.  ``None`` = DB maximum.
    name : str
        Human-readable label for plots / reports.
    _retries : int
        Number of retries on HTTP 429 / 529 (overloaded server).
    _retry_delay : float
        Seconds to wait between retries.
    """

    lat: float
    lon: float
    peak_power_kw: float
    loss: float = 14.0
    angle: float = 25.0
    azimuth: float = 0.0
    optimal_angles: bool = False
    optimal_inclination: bool = False
    pvtech: str = "crystSi"
    mounting: str = "free"
    raddatabase: Optional[str] = None
    startyear: Optional[int] = None
    endyear: Optional[int] = None
    name: str = "PV"
    _retries: int = 3
    _retry_delay: float = 5.0

    # internal cache (not a dataclass field)
    _profile: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _metadata: dict = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> pd.Series:
        """Call the PVGIS API (or local cache) and return an hourly power series in kW."""
        # Clamp years to PVGIS availability and warn
        params = self._build_params()
        if self.startyear and self.startyear > PVGIS_MAX_YEAR:
            raise ValueError(
                f"startyear={self.startyear} liegt nach dem PVGIS-Datenlimit ({PVGIS_MAX_YEAR})."
            )
        if self.endyear and self.endyear > PVGIS_MAX_YEAR:
            import warnings
            warnings.warn(
                f"endyear={self.endyear} wurde auf {PVGIS_MAX_YEAR} (PVGIS-Limit) gedeckelt.",
                UserWarning, stacklevel=2,
            )
            params['endyear'] = PVGIS_MAX_YEAR

        data = self._request_cached(params)
        series = self._parse_response(data).resample('1h').sum()  # sicherstellen, dass Daten zur vollen Stunde vorliegen (PVGIS liefert manchmal offset)
        self._metadata = data.get("meta", {})
        return series
    
    @property
    def metadata(self) -> dict:
        """PVGIS metadata dict (populated after :meth:`fetch`)."""
        _ = self.profile  # trigger fetch if needed
        return self._metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_params(self) -> dict:
        params: dict = {
            "lat": self.lat,
            "lon": self.lon,
            "peakpower": self.peak_power_kw,
            "loss": self.loss,
            "pvcalculation": 1,
            "pvtechchoice": self.pvtech,
            "mountingplace": self.mounting,
            "outputformat": "json",
            "browser": 0,
        }

        if self.optimal_angles:
            params["optimalangles"] = 1
        elif self.optimal_inclination:
            params["optimalinclination"] = 1
            params["aspect"] = self.azimuth
        else:
            params["angle"] = self.angle
            params["aspect"] = self.azimuth

        if self.raddatabase:
            params["raddatabase"] = self.raddatabase
        if self.startyear is not None:
            params["startyear"] = self.startyear
        if self.endyear is not None:
            params["endyear"] = self.endyear

        return params

    def _request(self, url: str, params: dict) -> dict:
        """GET request with retry logic for rate-limit / overload responses."""
        for attempt in range(1, self._retries + 1):
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 529) and attempt < self._retries:
                time.sleep(self._retry_delay * attempt)
                continue
            response.raise_for_status()
        raise RuntimeError("PVGIS request failed after retries.")

    def _request_cached(self, params: dict) -> dict:
        """Wie _request, aber mit lokalem Datei-Cache (JSON unter cache/).

        Der Cache-Schlüssel ist ein SHA256-Hash der API-Parameter.
        Gecachte Abrufe werden mit ``self._from_cache = True`` markiert.
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        key = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_file = os.path.join(_CACHE_DIR, f"pvgis_{key}.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                self._from_cache = True
                return json.load(f)

        data = self._request(f"{PVGIS_BASE}/seriescalc", params)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        self._from_cache = False
        return data

    @staticmethod
    def _parse_response(data: dict) -> pd.Series:
        """Parse JSON response → hourly pd.Series in kW."""
        hourly = data["outputs"]["hourly"]
        df = pd.DataFrame(hourly)

        # timestamp column is named 'time' and formatted as 'YYYYDDMM:HHmm'
        # PVGIS format: "20160101:0010" → year=2016, day=01, month=01, HH=00, mm=10
        # Actual format observed: YYYYMMDDhhmm  e.g. "201601010010"
        # PVGIS actually uses  "20160101:0010"  → strip the colon
        df["datetime"] = pd.to_datetime(
            df["time"].str.replace(":", "", regex=False),
            format="%Y%m%d%H%M",
            utc=True,
        )
        df = df.set_index("datetime").sort_index()

        # 'P' column = AC power in W  → convert to kW
        series = df["P"].astype(float) / 1000.0
        series.name = "power_kw"
        return series


# ---------------------------------------------------------------------------
# Synthetic profiles based on monthly shape factors
# ---------------------------------------------------------------------------

# Default monthly capacity factors (Jan–Dec, index 0–11)
# Values represent typical relative output (0–1); will be normalized internally.

#: Typical run-of-river hydro in the Alpine region:
#: low in winter, peak during snowmelt (May/Jun), elevated summer.
_HYDRO_DEFAULT_MONTHLY = [0.30, 0.30, 0.45, 0.70, 1.00, 0.95,
                           0.80, 0.75, 0.65, 0.55, 0.40, 0.30]

#: Drinking-water turbines in Alpine tourist areas:
#: high summer consumption (irrigation, tourism) → more pressure/flow available.
_DRINKING_WATER_DEFAULT_MONTHLY = [0.55, 0.50, 0.55, 0.65, 0.80, 1.00,
                                    1.00, 0.95, 0.75, 0.65, 0.60, 0.55]


@dataclass
class MonthlyShapeProfile(GenerationProfile):
    """Synthetic annual generation profile driven by monthly capacity factors.

    The 12 monthly factors are interpolated to an 8760-hour time series and
    scaled so that the profile peak equals *installed_capacity_kw*.

    Parameters
    ----------
    installed_capacity_kw : float
        Installed (peak) power in kW.
    monthly_factors : list[float]
        Relative capacity factors for Jan–Dec (length 12).
        Values are normalized to their maximum internally, so the absolute
        scale does not matter – only the shape.
    year : int
        Reference year for the synthetic time series (default 2023).
    name : str
        Human-readable label.
    """

    installed_capacity_kw: float
    monthly_factors: list = field(default_factory=list)
    startyear: int = 2023
    endyear: int = 2023
    name: str = "synthetic"

    _profile: Optional[pd.Series] = field(default=None, init=False, repr=False)

    def fetch(self) -> pd.Series:
        """Build an hourly power series from monthly capacity factors.

        The series spans from *startyear* to *endyear* (inclusive), with the
        same monthly shape repeated each year.
        """
        if not self.monthly_factors or len(self.monthly_factors) != 12:
            raise ValueError("monthly_factors must be a list of exactly 12 values.")
        if self.endyear < self.startyear:
            raise ValueError("endyear must be >= startyear.")

        idx = pd.date_range(
            start=f"{self.startyear}-01-01",
            end=f"{self.endyear}-12-31 23:00",
            freq="h",
            tz="UTC",
        )

        # Assign the monthly factor to every hour – pattern repeats each year
        month_map = {m + 1: v for m, v in enumerate(self.monthly_factors)}
        factors = pd.Series(idx.month, index=idx).map(month_map).astype(float)

        # Normalize so peak = installed_capacity_kw
        peak = max(self.monthly_factors)
        series = (factors / peak) * self.installed_capacity_kw
        series.name = "power_kw"
        return series


@dataclass
class HydroProfile(MonthlyShapeProfile):
    """Synthetic run-of-river hydropower profile.

    Uses a typical Alpine seasonal pattern (snowmelt peak in May/Jun) by
    default.  Override *monthly_factors* to match a specific catchment.

    Parameters
    ----------
    installed_capacity_kw : float
        Installed turbine capacity in kW.
    monthly_factors : list[float]
        12 monthly capacity factors (Jan–Dec).  Defaults to a typical
        Alpine run-of-river pattern.
    startyear : int
        First year of the time series (default 2023).
    endyear : int
        Last year of the time series (default 2023).
    name : str
        Label for plots / reports.
    """

    name: str = "Wasserkraft"

    def __post_init__(self):
        if not self.monthly_factors:
            self.monthly_factors = list(_HYDRO_DEFAULT_MONTHLY)


@dataclass
class DrinkingWaterProfile(MonthlyShapeProfile):
    """Synthetic drinking-water pressure-reduction turbine profile.

    Models the generation potential from pressure reduction in a drinking-water
    network.  Flow (and thus power) follows the seasonal consumption pattern
    of the supply area.  Defaults to an Alpine tourist-area pattern with a
    summer peak.

    Parameters
    ----------
    installed_capacity_kw : float
        Installed turbine capacity in kW.
    monthly_factors : list[float]
        12 monthly capacity factors (Jan–Dec).  Defaults to a typical
        Alpine drinking-water consumption pattern.
    startyear : int
        First year of the time series (default 2023).
    endyear : int
        Last year of the time series (default 2023).
    name : str
        Label for plots / reports.
    """

    name: str = "Trinkwasserkraft"

    def __post_init__(self):
        if not self.monthly_factors:
            self.monthly_factors = list(_DRINKING_WATER_DEFAULT_MONTHLY)


# ---------------------------------------------------------------------------
# Synthetisches Lastprofil
# ---------------------------------------------------------------------------

# Monatliche Faktoren (Jan–Dec) für typische Verbrauchsprofile
# Werte werden intern normiert; nur die Form zählt.

_LOAD_MONTHLY: dict[str, list[float]] = {
    # Wohngebäude: höherer Verbrauch im Winter (Heizung, weniger Tageslicht)
    'Wohngebäude': [1.20, 1.15, 1.05, 0.95, 0.85, 0.75,
                    0.70, 0.75, 0.85, 1.00, 1.10, 1.20],
    # Gewerbe: relativ gleichmäßig, etwas höher im Sommer (Kühlung)
    'Gewerbe':     [0.90, 0.90, 0.95, 0.95, 1.00, 1.05,
                    1.10, 1.05, 1.00, 0.95, 0.90, 0.90],
    # Gemischt: Mittelwert aus Wohngebäude und Gewerbe
    'Gemischt':    [1.05, 1.02, 1.00, 0.95, 0.92, 0.90,
                    0.90, 0.90, 0.92, 0.97, 1.00, 1.05],
    # Tourismus: Sommer-Peak (Tirol)
    'Tourismus':   [0.60, 0.55, 0.65, 0.75, 0.90, 1.00,
                    1.00, 1.00, 0.85, 0.70, 0.65, 0.65],
}

# Stündliche Tagesganglinie (0–23 Uhr), normiert auf Summe = 1
# Typischer Haushalt: Morgen- und Abendspitze
_LOAD_HOURLY_DEFAULT: list[float] = [
    0.025, 0.020, 0.018, 0.017, 0.018, 0.022,  # 0–5
    0.032, 0.045, 0.050, 0.048, 0.042, 0.040,  # 6–11
    0.042, 0.040, 0.038, 0.038, 0.040, 0.048,  # 12–17
    0.058, 0.062, 0.060, 0.050, 0.040, 0.030,  # 18–23
]


@dataclass
class LoadProfile(GenerationProfile):
    """Synthetisches Lastprofil auf Basis von Jahresenergie + Saisonalität + Tagesgang.

    Das Profil kombiniert:
    - Einen **Jahresenergieverbrauch** (*annual_energy_kwh*)
    - **Monatliche Faktoren** (Saisonalität) – entweder vordefiniertes Profil
      oder eigene Liste
    - Eine **stündliche Tagesganglinie** (24 Stundenfaktoren, Summe = 1)

    Beide Dimensionen werden multiplikativ verknüpft und auf den gewünschten
    Jahresenergieverbrauch skaliert.

    Parameters
    ----------
    annual_energy_kwh : float
        Ziel-Jahresenergie in kWh.
    profile_type : str
        Vordefinierter Profiltyp: ``'Wohngebäude'``, ``'Gewerbe'``,
        ``'Gemischt'``, ``'Tourismus'``.  Wird ignoriert wenn
        *monthly_factors* explizit gesetzt ist.
    monthly_factors : list[float]
        12 monatliche Faktoren (Jan–Dec).  Überschreibt *profile_type*.
    hourly_factors : list[float]
        24 stündliche Faktoren (0–23 Uhr), Summe sollte 1 ergeben.
        Standard: typische Haushalt-Tagesganglinie.
    startyear : int
        Erstes Jahr der Zeitreihe.
    endyear : int
        Letztes Jahr der Zeitreihe.
    name : str
        Label für Plots / Reports.
    """

    annual_energy_kwh: float
    profile_type: str = 'Wohngebäude'
    monthly_factors: list = field(default_factory=list)
    hourly_factors: list = field(default_factory=list)
    startyear: int = 2023
    endyear: int = 2023
    name: str = "Last"

    _profile: Optional[pd.Series] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.monthly_factors:
            if self.profile_type not in _LOAD_MONTHLY:
                raise ValueError(
                    f"Unbekannter Profiltyp '{self.profile_type}'. "
                    f"Wähle einen aus: {list(_LOAD_MONTHLY.keys())}"
                )
            self.monthly_factors = list(_LOAD_MONTHLY[self.profile_type])
        if not self.hourly_factors:
            self.hourly_factors = list(_LOAD_HOURLY_DEFAULT)
        if len(self.monthly_factors) != 12:
            raise ValueError("monthly_factors muss genau 12 Werte enthalten.")
        if len(self.hourly_factors) != 24:
            raise ValueError("hourly_factors muss genau 24 Werte enthalten.")

    def fetch(self) -> pd.Series:
        """Erzeuge stündliche Lastzeitreihe durch Faltung von Monats- und Tagesgang."""
        if self.endyear < self.startyear:
            raise ValueError("endyear muss >= startyear sein.")

        idx = pd.date_range(
            start=f"{self.startyear}-01-01",
            end=f"{self.endyear}-12-31 23:00",
            freq="h",
            tz="UTC",
        )

        # Monatsfaktor × Tagesfaktor für jeden Zeitstempel
        month_map = {m + 1: v for m, v in enumerate(self.monthly_factors)}
        hour_map  = {h: v for h, v in enumerate(self.hourly_factors)}

        monthly = pd.Series(idx.month, index=idx).map(month_map).astype(float)
        hourly  = pd.Series(idx.hour,  index=idx).map(hour_map).astype(float)

        # Rohprofil = Produkt beider Dimensionen
        raw = monthly * hourly

        # Normieren: Summe über ein Jahr entspricht annual_energy_kwh
        hours_per_year = 8760
        n_years = max((self.endyear - self.startyear + 1), 1)
        raw_sum_per_year = raw.sum() / n_years
        scale = (self.annual_energy_kwh / raw_sum_per_year) if raw_sum_per_year > 0 else 0.0

        series = raw * scale
        series.name = "power_kw"
        return series
