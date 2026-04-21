"""
preprocessing.py
----------------
Ladelogik für EEG-Abrechnungsdaten (RC-CSV-Dateien des Netzbetreibers).

Hauptfunktion: load_eeg(data_dir) -> pd.DataFrame

Rückgabe-Spalten (immer gleich, unabhängig von der EEG):
    bez_ges   – Gesamtbezug aller Zählpunkte [kWh / Intervall]
    bez_rest  – Restbezug (Netzbezug außerhalb Gemeinschaft)
    bez_eff   – Effektiv aus der Gemeinschaft bezogen
    lief_ges  – Gesamtlieferung aller Zählpunkte [kWh / Intervall]
    lief_rest – Restlieferung (Netzeinspeisung außerhalb Gemeinschaft)
    lief_eff  – Effektiv an die Gemeinschaft geliefert

Bekannte EEGs (Unterordner unter data/):
    EEG Eben am Achensee
    EEG Terfens
"""

from __future__ import annotations

import os
from glob import glob
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'data')

# Name → Ordner unter DATA_ROOT
KNOWN_EEGS: dict[str, str] = {
    'EEG Eben am Achensee': 'EEG Eben am Achensee',
    'EEG Terfens':          'EEG Terfens',
}


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def list_eegs() -> list[str]:
    """Gibt die Namen aller bekannten EEGs zurück."""
    return list(KNOWN_EEGS.keys())


def load_eeg(eeg_name_or_dir: str) -> pd.DataFrame:
    """Lade und aggregiere alle RC-CSV-Dateien einer EEG.

    Parameters
    ----------
    eeg_name_or_dir : str
        Entweder ein bekannter EEG-Name (aus :func:`list_eegs`) oder ein
        absoluter Pfad zu einem Verzeichnis mit RC-CSV-Dateien.

    Returns
    -------
    pd.DataFrame
        15-Minuten-Zeitreihe mit den aggregierten Spalten
        ``bez_ges``, ``bez_rest``, ``bez_eff``,
        ``lief_ges``, ``lief_rest``, ``lief_eff``.
        Index: ``DatetimeIndex`` (lokale Zeit, nicht-DST-bewusst, wie vom
        Netzbetreiber geliefert).
    """
    # Verzeichnis auflösen
    if eeg_name_or_dir in KNOWN_EEGS:
        data_dir = os.path.join(DATA_ROOT, KNOWN_EEGS[eeg_name_or_dir])
    else:
        data_dir = eeg_name_or_dir

    files = sorted(glob(os.path.join(data_dir, 'RC*.csv')))
    if not files:
        raise FileNotFoundError(f'Keine RC-CSV-Dateien gefunden in: {data_dir}')

    dfs = []
    for file in files:
        df = _read_rc_csv(file)
        dfs.append(df)

    merged = pd.concat(dfs)

    # Duplikate entfernen (Überschneidungen zwischen Monatsdateien)
    merged = merged[~merged.index.duplicated(keep='first')].sort_index()

    return _aggregate(merged)


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _read_rc_csv(filepath: str) -> pd.DataFrame:
    """Lese eine einzelne RC-CSV-Datei und gib einen rohen DataFrame zurück.

    Die CSV-Dateien haben zwei Kopfzeilen:
        Zeile 0: Spaltenbezeichnungen (z.B. "Gesamtbezug", "Restbezug", …)
        Zeile 1: Zählpunkt-IDs (z.B. "AT003200...", oder leer für Metaspalten)
    Ab Zeile 2 folgen die eigentlichen Messwerte.
    """
    # Schritt 1: Kopfzeilen lesen
    raw = pd.read_csv(filepath, sep=';', header=None, nrows=2)
    headers    = raw.iloc[0].tolist()
    id_strings = raw.iloc[1].tolist()

    # Schritt 2: Spaltennamen zusammensetzen (Header + Zählpunkt-ID)
    col_names = []
    for header, zp_id in zip(headers, id_strings):
        if pd.notna(zp_id) and str(zp_id).strip():
            col_names.append(f"{header}_{zp_id}")
        else:
            col_names.append(str(header))

    # Schritt 3: Messdaten lesen
    df = pd.read_csv(filepath, sep=';', decimal=',', header=None, skiprows=2)
    df.columns = col_names

    # Schritt 4: Zeitstempel parsen und als Index setzen
    if 'Zeitpunkt' not in df.columns:
        raise ValueError(f"Keine 'Zeitpunkt'-Spalte in {filepath}")
    df['Zeitpunkt'] = pd.to_datetime(df['Zeitpunkt'], dayfirst=True)
    df = df.set_index('Zeitpunkt')

    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiere Rohspalten zu den sechs Standardspalten."""
    out = pd.DataFrame(index=df.index)

    def _sum_cols(mask_str: str) -> pd.Series:
        mask = df.columns.str.contains(mask_str)
        if not mask.any():
            return pd.Series(0.0, index=df.index)
        return df.loc[:, mask].sum(axis=1)

    out['bez_ges']  = _sum_cols('Gesamtbezug')
    out['bez_rest'] = _sum_cols('Restbezug')
    out['bez_eff']  = _sum_cols('bezogen')
    out['lief_ges'] = _sum_cols('Gesamtlieferung ')
    out['lief_rest']= _sum_cols('Restlieferung ')
    out['lief_eff'] = _sum_cols('geliefert ')

    return out.fillna(0.0)


if __name__ == '__main__':
    for name in list_eegs():
        df = load_eeg(name)
        print(f'{name}: {df.shape}, {df.index[0]} – {df.index[-1]}')
        print(df.head(2))
        print()
