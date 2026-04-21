"""
app.py
------
Streamlit-App für das EEG-Speicher-Tool.

Starten:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import EEGModel, CostModel
from preprocessing import list_eegs, load_eeg
from synthetic_profiles import (
    DrinkingWaterProfile,
    HydroProfile,
    LoadProfile,
    PVProfile,
)

# ---------------------------------------------------------------------------
# Seitenkonfiguration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EEG Speicher-Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _fmt_kwh(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f} GWh"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f} MWh"
    return f"{v:.0f} kWh"

def _fmt_pct(v: float) -> str:
    return f"{v*100:.1f} %"

def _metric_row(metrics: dict):
    cols = st.columns(4)
    labels = {
        'self_sufficiency':       ("Eigenversorgungsgrad",   _fmt_pct),
        'self_consumption_pct':   ("Eigenverbrauchsquote",   _fmt_pct),
        'grid_import_kwh':        ("Netzbezug",              _fmt_kwh),
        'grid_export_kwh':        ("Netzeinspeisung",        _fmt_kwh),
        'total_gen_kwh':          ("Gesamterzeugung",        _fmt_kwh),
        'total_load_kwh':         ("Gesamtlast",             _fmt_kwh),
        'days_self_sufficient':   ("Tage autark",            lambda v: f"{v:.0f} d"),
        'self_consumption_kwh':   ("Eigenverbrauch",         _fmt_kwh),
    }
    items = [(k, labels[k]) for k in labels if k in metrics]
    for i, (k, (label, fmt)) in enumerate(items[:8]):
        cols[i % 4].metric(label, fmt(metrics[k]))


@st.cache_data(show_spinner="Lade EEG-Daten …")
def _load_eeg_cached(name: str) -> pd.DataFrame:
    return load_eeg(name)


@st.cache_data(show_spinner="Hole PV-Daten von PVGIS …")
def _fetch_pv(lat, lon, peak_kw, loss, angle, azimuth, optimal_angles, optimal_inclination,
              pvtech, mounting, startyear, endyear) -> tuple[pd.Series, bool]:
    pv = PVProfile(
        lat=lat, lon=lon, peak_power_kw=peak_kw, loss=loss,
        angle=angle, azimuth=azimuth,
        optimal_angles=optimal_angles, optimal_inclination=optimal_inclination,
        pvtech=pvtech, mounting=mounting,
        startyear=startyear, endyear=endyear,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        profile = pv.profile
    # PVGIS liefert Zeitstempel mit Minute :10 (z.B. 00:10 UTC) → auf volle Stunde flooren
    profile.index = profile.index.floor('h')
    from_cache = getattr(pv, '_from_cache', False)
    return profile, from_cache


def _timeseries_fig(df: pd.DataFrame, title: str, zoom_days: int = 14) -> go.Figure:
    """Plotly-Zeitreihe der Simulationsergebnisse."""
    end = df.index[0] + pd.Timedelta(days=zoom_days)
    dfs = df[df.index <= end]

    fig = go.Figure()
    colors = {
        'total_gen':  '#2ca02c',
        'load':       '#d62728',
        'grid_import':'#ff7f0e',
        'grid_export':'#1f77b4',
        'soc':        '#9467bd',
    }
    names = {
        'total_gen':  'Erzeugung gesamt',
        'load':       'Last',
        'grid_import':'Netzbezug',
        'grid_export':'Netzeinspeisung',
        'soc':        'Speicher-SoC',
    }

    for col in ['total_gen', 'load', 'grid_import', 'grid_export']:
        if col in dfs.columns:
            fig.add_trace(go.Scatter(
                x=dfs.index, y=dfs[col],
                name=names[col], line=dict(color=colors[col]),
            ))
    if 'soc' in dfs.columns:
        fig.add_trace(go.Scatter(
            x=dfs.index, y=dfs['soc'],
            name=names['soc'], line=dict(color=colors['soc'], dash='dot'),
            yaxis='y2',
        ))
    fig.update_layout(
        title=title,
        yaxis=dict(title='Leistung [kW]'),
        yaxis2=dict(title='SoC [kWh]', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', y=-0.2),
        height=400,
        hovermode='x unified',
    )
    return fig


def _optim_heatmap(opt_df: pd.DataFrame, metric: str, metric_label: str) -> go.Figure:
    pivot = opt_df.pivot(index='storage_kwh', columns='pv_scale', values=metric)
    fig = px.imshow(
        pivot,
        labels=dict(x='PV-Skalierung', y='Speicher [kWh]', color=metric_label),
        color_continuous_scale='Viridis',
        aspect='auto',
        title=f"Optimierungsraster – {metric_label}",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar – globale Einstellungen
# ---------------------------------------------------------------------------

def _sidebar() -> dict:
    st.sidebar.title("⚡ EEG Speicher-Tool")
    st.sidebar.markdown("---")

    cfg = {}

    cfg['modus'] = st.sidebar.radio(
        "Modus",
        ["📂 EEG-Messdaten", "🔮 Synthetische Profile"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation")
    cfg['pv_scale'] = st.sidebar.slider("PV-Skalierung", 0.5, 5.0, 1.0, 0.5)
    cfg['storage_kwh'] = st.sidebar.slider("Speicher [kWh]", 0, 500, 100, 25)
    cfg['c_rate'] = st.sidebar.slider("C-Rate", 0.25, 2.0, 0.5, 0.25)
    cfg['roundtrip_eff'] = st.sidebar.slider("Wirkungsgrad Speicher", 0.70, 0.98, 0.90, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optimierungsraster")
    cfg['opt_pv_min'] = st.sidebar.number_input("PV min", 0.5, 10.0, 0.5, 0.5)
    cfg['opt_pv_max'] = st.sidebar.number_input("PV max", 1.0, 20.0, 5.0, 0.5)
    cfg['opt_pv_step'] = st.sidebar.number_input("PV Schritt", 0.25, 2.0, 0.5, 0.25)
    cfg['opt_stor_max'] = st.sidebar.number_input("Speicher max [kWh]", 0, 2000, 500, 50)
    cfg['opt_stor_step'] = st.sidebar.number_input("Speicher Schritt [kWh]", 25, 200, 50, 25)

    return cfg


# ===========================================================================
# MODUS A – EEG-Messdaten
# ===========================================================================

def _mode_real(cfg: dict):
    st.header("📂 EEG-Messdaten")

    col1, col2 = st.columns([2, 1])
    with col1:
        eeg_name = st.selectbox("EEG auswählen", list_eegs())
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("Daten laden", type="primary")

    if 'eeg_df' not in st.session_state or st.session_state.get('eeg_loaded') != eeg_name or load_btn:
        with st.spinner("Lade CSV-Dateien …"):
            try:
                df = _load_eeg_cached(eeg_name)
                st.session_state['eeg_df'] = df
                st.session_state['eeg_loaded'] = eeg_name
            except Exception as e:
                st.error(f"Fehler beim Laden: {e}")
                return

    if 'eeg_df' not in st.session_state:
        return

    df: pd.DataFrame = st.session_state['eeg_df']

    tab_data, tab_sim, tab_opt = st.tabs(["📊 Datenprofil", "⚙️ Simulation", "🔍 Optimierung"])

    # ---- Tab: Datenprofil -----------------------------------------------
    with tab_data:
        st.subheader("Rohdaten-Überblick")
        c1, c2, c3 = st.columns(3)
        c1.metric("Zeitraum von", str(df.index.min().date()))
        c2.metric("bis", str(df.index.max().date()))
        c3.metric("Datenpunkte", f"{len(df):,}")

        col_map = {
            'bez_ges':  'Gesamtbezug',
            'bez_rest': 'Restbezug',
            'bez_eff':  'Gemeinschaftsbezug',
            'lief_ges': 'Gesamtlieferung',
            'lief_rest':'Restlieferung',
            'lief_eff': 'Gemeinschaftslieferung',
        }
        avail = [c for c in col_map if c in df.columns]
        sel_cols = st.multiselect("Spalten anzeigen", avail,
                                  default=['bez_ges', 'lief_ges'],
                                  format_func=lambda x: col_map[x])
        if sel_cols:
            fig = px.line(df[sel_cols].rename(columns=col_map),
                          labels={'value': 'kWh', 'variable': 'Größe'},
                          title="Zeitreihe Messdaten")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Jahressummen [kWh]**")
        annual = df[avail].resample('YE').sum().rename(columns=col_map)
        st.dataframe(annual.style.format("{:.0f}"), use_container_width=True)

        with st.expander("Rohdaten (erste 500 Zeilen)"):
            st.dataframe(df.head(500), use_container_width=True)

    # ---- Tab: Simulation -------------------------------------------------
    with tab_sim:
        st.subheader("Simulation mit synthetischem PV-Profil")

        st.info("Hier wird ein synthetisches PV-Profil auf Basis der Koordinaten der EEG "
                "verwendet und mit den Messdaten kombiniert.")

        with st.expander("PV-Parameter", expanded=True):
            c1, c2, c3 = st.columns(3)
            lat = c1.number_input("Breitengrad", 46.0, 48.0, 47.4, 0.01, key="real_lat")
            lon = c2.number_input("Längengrad", 10.0, 13.0, 11.8, 0.01, key="real_lon")
            peak_kw = c3.number_input("Peak [kWp]", 1.0, 500.0, 50.0, 5.0, key="real_peak")
            opt_angles = st.checkbox("Optimale Ausrichtung (PVGIS)", value=True, key="real_opt")

        if st.button("Simulieren", key="real_sim_btn", type="primary"):
            try:
                pv_series, from_cache = _fetch_pv(
                    lat, lon, 1.0, 14, 25, 0, opt_angles, False,
                    "crystSi", "free", 2022, 2023,
                )
                if from_cache:
                    st.caption("✅ PV-Daten aus lokalem Cache")
                else:
                    st.caption("🌐 PV-Daten neu von PVGIS geladen")

                # Lokale Zeit → UTC, kWh/15min → kW, auf Stunden resamplen
                load_series = df['bez_ges'].copy() * 4  # kWh/15min → kW (Durchschnittsleistung)
                load_series = load_series.tz_localize('Europe/Vienna', ambiguous='NaT',
                                                       nonexistent='NaT')
                load_series = load_series.tz_convert('UTC').dropna()
                load_series = load_series.resample('h').mean().dropna()  # → stündliche kW

                # Zeitraum angleichen
                common = pv_series.index.intersection(load_series.index)
                if len(common) < 2:
                    st.error(
                        f"Kein gemeinsamer Zeitraum zwischen PV-Daten (PVGIS 2022–2023, UTC) "
                        f"und Messdaten ({load_series.index.min()} – {load_series.index.max()}). "
                        f"Bitte Zeitraum der Messdaten prüfen."
                    )
                    return
                pv_trim = pv_series.loc[common]
                load_trim = load_series.loc[common]

                pv_scaled = pv_trim * peak_kw
                model = EEGModel(pv_scaled, load_trim, name=eeg_name)
                result = model.simulate(
                    pv_scale=cfg['pv_scale'],
                    storage_kwh=cfg['storage_kwh'],
                    c_rate=cfg['c_rate'],
                    roundtrip_eff=cfg['roundtrip_eff'],
                )
                metrics = model.compute_metrics(result)

                _metric_row(metrics)
                fig = _timeseries_fig(result, "Simulation – erste 14 Tage")
                st.plotly_chart(fig, use_container_width=True)

                st.session_state['real_model'] = model
                st.session_state['real_result'] = result
                st.session_state['real_metrics'] = metrics

            except Exception as e:
                st.error(f"Simulation fehlgeschlagen: {e}")
                st.exception(e)

        # Download
        if 'real_result' in st.session_state:
            csv = st.session_state['real_result'].to_csv().encode('utf-8')
            st.download_button("⬇️ Ergebnis als CSV", csv, "simulation_ergebnis.csv", "text/csv")

    # ---- Tab: Optimierung ------------------------------------------------
    with tab_opt:
        _optimierung_tab(cfg, session_prefix='real')


# ===========================================================================
# MODUS B – Synthetische Profile
# ===========================================================================

def _mode_synthetic(cfg: dict):
    st.header("🔮 Synthetische Profile")

    with st.expander("⚡ PV-Anlage", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        lat = c1.number_input("Breitengrad", 46.0, 48.0, 47.4, 0.01)
        lon = c2.number_input("Längengrad", 10.0, 13.0, 11.8, 0.01)
        peak_kw = c3.number_input("Peak [kWp]", 1.0, 1000.0, 100.0, 10.0)
        loss = c4.number_input("Verluste [%]", 0.0, 30.0, 14.0, 1.0)
        c1b, c2b, c3b = st.columns(3)
        angle = c1b.slider("Neigung [°]", 0, 90, 25)
        azimuth = c2b.slider("Azimut [°]", -180, 180, 0)
        opt_angles = c3b.checkbox("Optimale Ausrichtung", False)
        c1c, c2c = st.columns(2)
        startyear = c1c.number_input("Von Jahr", 2005, 2023, 2020, 1)
        endyear = c2c.number_input("Bis Jahr", 2005, 2023, 2023, 1)

    with st.expander("💧 Wasserkraft (optional)"):
        hydro_on = st.checkbox("Wasserkraft aktivieren", False)
        hydro_kw = st.number_input("Installierte Leistung [kW]", 0.0, 5000.0, 50.0, 10.0,
                                    disabled=not hydro_on)

    with st.expander("🚿 Trinkwasserkraft (optional)"):
        tw_on = st.checkbox("Trinkwasserkraft aktivieren", False)
        tw_kw = st.number_input("Installierte Leistung [kW]", 0.0, 5000.0, 20.0, 5.0,
                                 disabled=not tw_on)

    with st.expander("🏘️ Lastprofil", expanded=True):
        profile_types = ['Wohngebäude', 'Gewerbe', 'Gemischt', 'Tourismus']
        profile_type = st.selectbox("Profiltyp", profile_types)
        annual_kwh = st.number_input("Jahresverbrauch [kWh]", 1000.0, 10_000_000.0, 500_000.0,
                                      10_000.0, format="%.0f")

    if st.button("Profile berechnen & simulieren", type="primary"):
        try:
            # PV
            with st.spinner("PVGIS-Daten …"):
                pv_norm, from_cache = _fetch_pv(
                    lat, lon, 1.0, loss, angle, azimuth, opt_angles, False,
                    "crystSi", "free", int(startyear), int(endyear),
                )
            if from_cache:
                st.caption("✅ PV aus Cache")
            else:
                st.caption("🌐 PV neu von PVGIS")

            pv_series = pv_norm * peak_kw

            # Lastprofil (auf den gleichen Zeitraum)
            yr0, yr1 = int(pv_series.index.year.min()), int(pv_series.index.year.max())
            load_profile = LoadProfile(
                annual_energy_kwh=annual_kwh,
                profile_type=profile_type,
                startyear=yr0, endyear=yr1,
            )
            load_series = load_profile.profile

            # Zeitraum angleichen
            common = pv_series.index.intersection(load_series.index)
            pv_trim = pv_series.loc[common]
            load_trim = load_series.loc[common]

            # Fixe Quellen
            fixed = {}
            if hydro_on:
                h = HydroProfile(installed_capacity_kw=hydro_kw, startyear=yr0, endyear=yr1)
                fixed['Wasserkraft'] = h.profile.loc[common]
            if tw_on:
                t = DrinkingWaterProfile(installed_capacity_kw=tw_kw, startyear=yr0, endyear=yr1)
                fixed['Trinkwasserkraft'] = t.profile.loc[common]

            model = EEGModel(pv_trim, load_trim, fixed_sources=fixed or None, name="Synthetisch")
            result = model.simulate(
                pv_scale=cfg['pv_scale'],
                storage_kwh=cfg['storage_kwh'],
                c_rate=cfg['c_rate'],
                roundtrip_eff=cfg['roundtrip_eff'],
            )
            metrics = model.compute_metrics(result)

            st.session_state['syn_model'] = model
            st.session_state['syn_result'] = result
            st.session_state['syn_metrics'] = metrics

        except Exception as e:
            st.error(f"Fehler: {e}")
            st.exception(e)
            return

    if 'syn_result' not in st.session_state:
        return

    result: pd.DataFrame = st.session_state['syn_result']
    metrics: dict = st.session_state['syn_metrics']
    model: EEGModel = st.session_state['syn_model']

    tab_res, tab_opt = st.tabs(["📊 Ergebnisse", "🔍 Optimierung"])

    with tab_res:
        _metric_row(metrics)

        fig = _timeseries_fig(result, "Simulation – erste 14 Tage")
        st.plotly_chart(fig, use_container_width=True)

        # Monatliche Auswertung
        monthly = result[['total_gen', 'load', 'grid_import', 'grid_export']].resample('ME').sum()
        monthly.columns = ['Erzeugung', 'Last', 'Netzbezug', 'Netzeinspeisung']
        fig2 = px.bar(monthly, barmode='group',
                      labels={'value': 'kWh', 'variable': 'Größe'},
                      title="Monatliche Energiebilanz")
        st.plotly_chart(fig2, use_container_width=True)

        csv = result.to_csv().encode('utf-8')
        st.download_button("⬇️ Ergebnis als CSV", csv, "simulation_synthetisch.csv", "text/csv")

    with tab_opt:
        _optimierung_tab(cfg, session_prefix='syn')


# ===========================================================================
# Optimierungsraster (gemeinsam für beide Modi)
# ===========================================================================

def _optimierung_tab(cfg: dict, session_prefix: str):
    st.subheader("Optimierungsraster")

    model_key = f'{session_prefix}_model'
    if model_key not in st.session_state:
        st.info("Bitte zuerst eine Simulation durchführen.")
        return

    model: EEGModel = st.session_state[model_key]

    metric_options = {
        'days_self_sufficient': 'Tage autark',
        'self_sufficiency':     'Eigenversorgungsgrad',
        'grid_import_kwh':      'Netzbezug [kWh]',
        'grid_export_kwh':      'Netzeinspeisung [kWh]',
    }
    metric = st.selectbox("Optimierungsziel", list(metric_options.keys()),
                          format_func=lambda x: metric_options[x],
                          key=f'{session_prefix}_opt_metric')

    if st.button("Optimierung starten", type="primary", key=f'{session_prefix}_opt_btn'):
        pv_scales = list(np.arange(cfg['opt_pv_min'], cfg['opt_pv_max'] + 0.01, cfg['opt_pv_step']))
        storage_grid = list(np.arange(0, cfg['opt_stor_max'] + 1, cfg['opt_stor_step']))

        with st.spinner(f"Berechne {len(pv_scales) * len(storage_grid)} Szenarien …"):
            try:
                opt_df = model.optimize_grid(
                    pv_scales=pv_scales,
                    storage_grid=storage_grid,
                    metric=metric,
                )
                best_params = model.best_params or {}
                best = best_params.get(metric, float('nan'))
                st.session_state[f'{session_prefix}_opt_df'] = opt_df
                st.session_state[f'{session_prefix}_best_params'] = best_params
                st.session_state[f'{session_prefix}_best'] = best
            except Exception as e:
                st.error(f"Optimierung fehlgeschlagen: {e}")
                st.exception(e)
                return

    if f'{session_prefix}_opt_df' not in st.session_state:
        return

    opt_df: pd.DataFrame = st.session_state[f'{session_prefix}_opt_df']
    best_params = st.session_state[f'{session_prefix}_best_params']
    best = st.session_state[f'{session_prefix}_best']

    st.success(
        f"**Optimum:** PV-Skalierung = {best_params.get('pv_scale', '–')}, "
        f"Speicher = {best_params.get('storage_kwh', '–')} kWh  →  "
        f"{metric_options[metric]} = {best:.2f}"
    )

    fig = _optim_heatmap(opt_df, metric, metric_options[metric])
    st.plotly_chart(fig, use_container_width=True)

    csv = opt_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Optimierungsraster als CSV", csv,
                       f"optimierung_{session_prefix}.csv", "text/csv",
                       key=f'{session_prefix}_opt_dl')


# ===========================================================================
# Hauptprogramm
# ===========================================================================

def main():
    cfg = _sidebar()

    if cfg['modus'] == "📂 EEG-Messdaten":
        _mode_real(cfg)
    else:
        _mode_synthetic(cfg)


if __name__ == "__main__":
    main()
