"""Synthetic load and PV generation profiles.

Provides two profile generators returning hourly energy series in
kWh per timestep on a naive UTC ``DatetimeIndex``:

    pv_profile()    – PV generation via the PVGIS seriescalc API
                      (with a local file cache).
    load_profile()  – Synthetic community load from an annual energy
                      demand, a seasonal (monthly) shape and a daily
                      (hourly) shape.

Both series span the same calendar year, so they can be combined
directly in :mod:`core`.
"""

import hashlib
import json
import os
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# PVGIS
# ---------------------------------------------------------------------------

PVGIS_BASE = "https://re.jrc.ec.europa.eu/api/v5_3"

# PVGIS liefert derzeit Daten bis einschließlich 2023.
PVGIS_MAX_YEAR = 2023

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'cache'
)
_RETRIES = 3
_RETRY_DELAY_S = 5.0


def pv_profile(lat: float, lon: float, kwp: float,
               year: int = 2023, angle: float = 25.0,
               azimuth: float = 0.0, loss: float = 14.0) -> pd.Series:
    """Fetch an hourly PV generation profile from PVGIS.

    The API is queried for a 1-kWp system and scaled linearly to
    *kwp*, so repeated calls with different peak power hit the same
    cache entry.

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
        kwp (float): Installed PV peak power [kWp].
        year (int, optional): Calendar year of the time series.
            Defaults to 2023 (most recent PVGIS year).
        angle (float, optional): Tilt angle from horizontal [deg].
        azimuth (float, optional): Azimuth (0 = south, 90 = west,
            -90 = east) [deg].
        loss (float, optional): System losses [%]. Defaults to 14.

    Returns:
        pd.Series: PV generation [kWh per hour], naive UTC index.

    Raises:
        ValueError: If *year* is outside the PVGIS data range.
        requests.HTTPError: If the API request fails after retries.
    """
    if year > PVGIS_MAX_YEAR:
        raise ValueError(
            f"year={year} liegt nach dem PVGIS-Datenlimit "
            f"({PVGIS_MAX_YEAR})."
        )

    params = {
        'lat': lat,
        'lon': lon,
        'peakpower': 1.0,
        'loss': loss,
        'angle': angle,
        'aspect': azimuth,
        'startyear': year,
        'endyear': year,
        'pvcalculation': 1,
        'outputformat': 'json',
        'browser': 0,
    }
    data = _request_cached(params)

    hourly = pd.DataFrame(data['outputs']['hourly'])
    # PVGIS-Zeitstempel haben das Format "YYYYMMDD:HHMM" und liegen
    # bei Minute :10 → auf volle Stunde runden.
    timestamps = pd.to_datetime(
        hourly['time'].str.replace(':', '', regex=False),
        format='%Y%m%d%H%M',
    ).dt.floor('h')

    # Spalte 'P' ist AC-Leistung in W; über eine Stunde gemittelt
    # entspricht der Wert der Energie in Wh → /1000 = kWh.
    series = pd.Series(
        hourly['P'].to_numpy(dtype=float) / 1000.0 * kwp,
        index=pd.DatetimeIndex(timestamps),
        name='pv_kwh',
    )
    return series


def _request_cached(params: dict) -> dict:
    """Call the PVGIS API with a local JSON file cache.

    Args:
        params (dict): Query parameters for the seriescalc endpoint.

    Returns:
        dict: Parsed JSON response.

    Raises:
        requests.HTTPError: If the request fails after retries.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:16]
    cache_file = os.path.join(_CACHE_DIR, f"pvgis_{key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    url = f"{PVGIS_BASE}/seriescalc"
    for attempt in range(1, _RETRIES + 1):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            data = response.json()
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return data
        # Bei Überlastung (429/529) mit Wartezeit erneut versuchen.
        if response.status_code in (429, 529) and attempt < _RETRIES:
            time.sleep(_RETRY_DELAY_S * attempt)
            continue
        response.raise_for_status()
    raise RuntimeError("PVGIS request failed after retries.")


# ---------------------------------------------------------------------------
# Synthetisches Lastprofil
# ---------------------------------------------------------------------------

# Monatliche Faktoren (Jan-Dez) für typische Verbrauchsprofile.
# Nur die Form zählt; die Skalierung erfolgt über annual_kwh.
_LOAD_MONTHLY: dict = {
    # Wohngebäude: höherer Verbrauch im Winter.
    'Wohngebäude': [1.20, 1.15, 1.05, 0.95, 0.85, 0.75,
                    0.70, 0.75, 0.85, 1.00, 1.10, 1.20],
    # Gewerbe: relativ gleichmäßig, etwas höher im Sommer (Kühlung).
    'Gewerbe': [0.90, 0.90, 0.95, 0.95, 1.00, 1.05,
                1.10, 1.05, 1.00, 0.95, 0.90, 0.90],
    # Gemischt: Mittelwert aus Wohngebäude und Gewerbe.
    'Gemischt': [1.05, 1.02, 1.00, 0.95, 0.92, 0.90,
                 0.90, 0.90, 0.92, 0.97, 1.00, 1.05],
    # Tourismus: Sommer-Peak (Tirol).
    'Tourismus': [0.60, 0.55, 0.65, 0.75, 0.90, 1.00,
                  1.00, 1.00, 0.85, 0.70, 0.65, 0.65],
}

#: Verfügbare Lastprofil-Typen (für UI-Auswahl).
LOAD_PROFILE_TYPES = list(_LOAD_MONTHLY.keys())

# Stündliche Tagesganglinie (0-23 Uhr Lokalzeit): typischer Haushalt
# mit Morgen- und Abendspitze. Wird intern normiert.
_LOAD_HOURLY = [
    0.025, 0.020, 0.018, 0.017, 0.018, 0.022,   # 0-5
    0.032, 0.045, 0.050, 0.048, 0.042, 0.040,   # 6-11
    0.042, 0.040, 0.038, 0.038, 0.040, 0.048,   # 12-17
    0.058, 0.062, 0.060, 0.050, 0.040, 0.030,   # 18-23
]

# Das PV-Profil ist in UTC indiziert. Damit die Tagesganglinie dazu
# passt, wird sie um den CET-Versatz verschoben (DST vernachlässigt).
_UTC_OFFSET_H = 1


def load_profile(annual_kwh: float,
                 profile_type: str = 'Wohngebäude',
                 year: int = 2023) -> pd.Series:
    """Build a synthetic hourly load profile for one year.

    Combines a seasonal (monthly) shape with a daily (hourly) shape
    and scales the result to the requested annual energy.

    Args:
        annual_kwh (float): Annual energy demand [kWh].
        profile_type (str, optional): One of
            :data:`LOAD_PROFILE_TYPES`. Defaults to 'Wohngebäude'.
        year (int, optional): Calendar year of the time series.
            Defaults to 2023.

    Returns:
        pd.Series: Load [kWh per hour], naive UTC index matching
            :func:`pv_profile`.

    Raises:
        ValueError: If *profile_type* is unknown.
    """
    if profile_type not in _LOAD_MONTHLY:
        raise ValueError(
            f"Unbekannter Profiltyp '{profile_type}'. "
            f"Verfügbar: {LOAD_PROFILE_TYPES}"
        )

    idx = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:00",
        freq='h',
    )

    monthly = pd.Series(idx.month, index=idx).map(
        {m + 1: v for m, v in enumerate(_LOAD_MONTHLY[profile_type])}
    )
    local_hour = (idx.hour + _UTC_OFFSET_H) % 24
    hourly = pd.Series(local_hour, index=idx).map(
        {h: v for h, v in enumerate(_LOAD_HOURLY)}
    )

    raw = (monthly * hourly).astype(float)
    series = raw * (annual_kwh / raw.sum())
    series.name = 'load_kwh'
    return series
