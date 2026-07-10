"""Streamlit app of the EEG storage sizing tool.

Linear workflow:
    1. Datengrundlage waehlen (synthetisch oder Messdaten-CSV)
    2. Ziele festlegen (Autarkie / Eigenverbrauch)
    3. Ergebnis: minimale Speichergroesse, die die Ziele erreicht

Starten:
    streamlit run app.py
"""

import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import (StorageParams, simulate_storage, size_storage,
                  storage_sweep, timestep_hours)
from preprocessing import list_eegs, load_eeg
from synthetic_profiles import (LOAD_PROFILE_TYPES, load_profile,
                                pv_profile)

# Feste Serienfarben (validierte Standard-Palette, Light Mode).
COLOR = {
    'autarky': '#2a78d6',           # blau
    'self_consumption': '#1baf7a',  # aqua
    'import_free': '#eda100',       # gelb
    'load': '#2a78d6',              # blau
    'generation': '#1baf7a',        # aqua
    'soc': '#4a3aa7',               # violett
    'target': '#898781',            # neutrales grau
    'recommend': '#008300',         # gruen
}

st.set_page_config(
    page_title="EEG Speicher-Tool",
    page_icon="🔋",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------

def _fmt_pct(value: float) -> str:
    """Format a 0-1 ratio as a percent string."""
    return f"{value * 100:.1f} %"


def _fmt_kwh(value: float) -> str:
    """Format an energy value with a sensible unit."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f} GWh"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.1f} MWh"
    return f"{value:,.0f} kWh"


# ---------------------------------------------------------------------------
# Gecachte Datenbeschaffung
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Hole PV-Profil von PVGIS …")
def _cached_pv(lat: float, lon: float, kwp: float) -> pd.Series:
    """Cached wrapper around :func:`synthetic_profiles.pv_profile`."""
    return pv_profile(lat=lat, lon=lon, kwp=kwp)


@st.cache_data(show_spinner="Erzeuge Lastprofil …")
def _cached_load(annual_kwh: float, profile_type: str) -> pd.Series:
    """Cached wrapper around :func:`synthetic_profiles.load_profile`."""
    return load_profile(annual_kwh=annual_kwh,
                        profile_type=profile_type)


@st.cache_data(show_spinner="Lade Messdaten …")
def _cached_eeg(name: str) -> pd.DataFrame:
    """Cached wrapper around :func:`preprocessing.load_eeg`."""
    return load_eeg(name)


@st.cache_data(show_spinner="Verarbeite hochgeladene CSV-Dateien …")
def _load_uploads(files: tuple) -> pd.DataFrame:
    """Parse uploaded RC-CSV files via :func:`preprocessing.load_eeg`.

    Args:
        files (tuple): Tuples of (filename, file content bytes).

    Returns:
        pd.DataFrame: Aggregated 15-min community time series.
    """
    tmpdir = tempfile.mkdtemp(prefix='eeg_upload_')
    for name, content in files:
        # load_eeg sucht nach dem Muster RC*.csv → Namen anpassen.
        fname = name if name.startswith('RC') else f"RC_{name}"
        with open(os.path.join(tmpdir, fname), 'wb') as f:
            f.write(content)
    return load_eeg(tmpdir)


@st.cache_data(show_spinner="Simuliere Speichergrößen …")
def _cached_sweep(generation: pd.Series, load: pd.Series,
                  capacities: tuple, c_rate: float,
                  roundtrip_eff: float) -> pd.DataFrame:
    """Cached wrapper around :func:`core.storage_sweep`."""
    params = StorageParams(c_rate=c_rate, roundtrip_eff=roundtrip_eff)
    return storage_sweep(generation, load, capacities, params)


# ---------------------------------------------------------------------------
# Diagramme
# ---------------------------------------------------------------------------

def _sweep_fig(sweep: pd.DataFrame, targets: dict,
               best_capacity: float) -> go.Figure:
    """Line chart: KPIs vs. storage capacity with target lines.

    Args:
        sweep (pd.DataFrame): Output of :func:`core.storage_sweep`.
        targets (dict): Active targets, KPI column -> value (0-1).
        best_capacity (float): Recommended capacity or ``nan``.

    Returns:
        go.Figure: Plotly figure.
    """
    series = [
        ('autarky', 'Autarkiegrad', COLOR['autarky']),
        ('self_consumption', 'Eigenverbrauchsquote',
         COLOR['self_consumption']),
        ('import_free_share', 'Zeit ohne Netzbezug',
         COLOR['import_free']),
    ]
    fig = go.Figure()
    for col, label, color in series:
        fig.add_trace(go.Scatter(
            x=sweep['capacity_kwh'], y=sweep[col] * 100,
            name=label, line=dict(color=color, width=2),
            hovertemplate='%{y:.1f} %<extra>' + label + '</extra>',
        ))

    for col, value in targets.items():
        fig.add_hline(
            y=value * 100, line_dash='dash', line_width=1,
            line_color=COLOR['target'],
            annotation_text=f"Ziel {value * 100:.0f} %",
            annotation_font_color=COLOR['target'],
        )
    if pd.notna(best_capacity):
        fig.add_vline(
            x=best_capacity, line_dash='dot', line_width=2,
            line_color=COLOR['recommend'],
            annotation_text=f"{best_capacity:.0f} kWh",
            annotation_font_color=COLOR['recommend'],
        )

    fig.update_layout(
        xaxis_title='Speicherkapazität [kWh]',
        yaxis_title='Anteil [%]',
        yaxis_range=[0, 100],
        legend=dict(orientation='h', y=-0.25),
        height=420,
        hovermode='x unified',
        margin=dict(t=30),
    )
    return fig


def _week_fig(result: pd.DataFrame, dt_h: float) -> go.Figure:
    """Example week: load/generation on top, storage SoC below.

    Args:
        result (pd.DataFrame): Output of
            :func:`core.simulate_storage`.
        dt_h (float): Timestep length in hours.

    Returns:
        go.Figure: Plotly figure with two stacked subplots.
    """
    # Woche in der Mitte des Zeitraums als repräsentatives Beispiel.
    start = result.index[len(result) // 2].normalize()
    week = result.loc[start:start + pd.Timedelta(days=7)]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(
        x=week.index, y=week['generation'] / dt_h,
        name='Erzeugung', line=dict(color=COLOR['generation'],
                                    width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=week.index, y=week['load'] / dt_h,
        name='Last', line=dict(color=COLOR['load'], width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=week.index, y=week['soc'],
        name='Speicherfüllstand', line=dict(color=COLOR['soc'],
                                            width=2),
    ), row=2, col=1)

    fig.update_yaxes(title_text='Leistung [kW]', row=1, col=1)
    fig.update_yaxes(title_text='SoC [kWh]', row=2, col=1)
    fig.update_layout(
        height=450,
        legend=dict(orientation='h', y=-0.15),
        hovermode='x unified',
        margin=dict(t=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Schritt 1 – Datengrundlage
# ---------------------------------------------------------------------------

def _step_data() -> tuple:
    """Render step 1 and return the community profiles.

    Returns:
        tuple: (generation, load) as pd.Series [kWh per timestep],
            or (None, None) while input is incomplete.
    """
    st.header("1️⃣ Datengrundlage")

    source = st.radio(
        "Woher kommen Last und Erzeugung?",
        ["Messdaten (CSV)", "Synthetische Profile"],
        horizontal=True,
    )

    if source == "Synthetische Profile":
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Breitengrad", 46.0, 49.0, 47.4,
                                  0.01)
            lon = st.number_input("Längengrad", 9.0, 17.0, 11.7,
                                  0.01)
            kwp = st.number_input("PV-Leistung [kWp]", 1.0, 5000.0,
                                  150.0, 10.0)
        with col2:
            annual_kwh = st.number_input(
                "Jahresverbrauch [kWh]", 1_000.0, 10_000_000.0,
                200_000.0, 10_000.0, format="%.0f",
            )
            profile_type = st.selectbox(
                "Verbrauchsprofil", LOAD_PROFILE_TYPES,
            ) or LOAD_PROFILE_TYPES[0]
        try:
            generation = _cached_pv(lat, lon, kwp)
            load = _cached_load(annual_kwh, profile_type)
        except Exception as exc:
            st.error(f"Profil konnte nicht erstellt werden: {exc}")
            return None, None
        return generation, load

    # --- Messdaten ------------------------------------------------
    example = st.selectbox(
        "Beispiel-EEG (oder eigene Dateien hochladen)",
        ["– eigene CSV-Dateien –", *list_eegs()],
    )
    if example != "– eigene CSV-Dateien –":
        try:
            df = _cached_eeg(example)
        except Exception as exc:
            st.error(f"Messdaten konnten nicht geladen werden: {exc}")
            return None, None
    else:
        uploads = st.file_uploader(
            "RC-CSV-Dateien des Netzbetreibers",
            type='csv', accept_multiple_files=True,
        )
        if not uploads:
            st.info("Bitte eine oder mehrere CSV-Dateien hochladen.")
            return None, None
        try:
            df = _load_uploads(
                tuple((u.name, u.getvalue()) for u in uploads)
            )
        except Exception as exc:
            st.error(f"CSV-Dateien konnten nicht gelesen werden: "
                     f"{exc}")
            return None, None

    st.caption(
        f"Zeitraum {df.index.min():%d.%m.%Y} – "
        f"{df.index.max():%d.%m.%Y}, {len(df):,} Zeitschritte"
    )
    # Gesamtbezug = Verbrauch der Gemeinschaft ab Zählpunkt,
    # Gesamtlieferung = Einspeisung (Erzeugungsüberschuss).
    return df['lief_ges'], df['bez_ges']


# ---------------------------------------------------------------------------
# Schritt 2 – Ziele
# ---------------------------------------------------------------------------

def _step_targets() -> tuple:
    """Render step 2 and return targets and storage settings.

    Returns:
        tuple: (targets, gen_scale, capacities, params) where
            *targets* maps KPI column names to 0-1 values.
    """
    st.header("2️⃣ Ziele festlegen")

    targets = {}
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.checkbox("Autarkiegrad", value=True,
                       help="Anteil des Verbrauchs, der ohne "
                            "Netzbezug gedeckt wird."):
            targets['autarky'] = st.slider(
                "Ziel Autarkiegrad [%]", 5, 100, 50, 5,
            ) / 100.0
    with col2:
        if st.checkbox("Eigenverbrauchsquote",
                       help="Anteil der Erzeugung, der in der "
                            "Gemeinschaft genutzt wird."):
            targets['self_consumption'] = st.slider(
                "Ziel Eigenverbrauch [%]", 5, 100, 70, 5,
            ) / 100.0
    with col3:
        if st.checkbox("Zeit ohne Netzbezug",
                       help="Anteil der Zeitschritte ganz ohne "
                            "Netzbezug."):
            targets['import_free_share'] = st.slider(
                "Ziel Zeit ohne Netzbezug [%]", 5, 100, 50, 5,
            ) / 100.0

    with st.expander("⚙️ Erweiterte Einstellungen"):
        col1, col2 = st.columns(2)
        with col1:
            gen_scale = st.slider(
                "PV-Ausbau (Skalierung der Erzeugung)",
                1.0, 5.0, 1.0, 0.25,
                help="Faktor auf das Erzeugungsprofil, um einen "
                     "PV-Ausbau mitzudenken.",
            )
            max_cap = st.number_input(
                "Größte untersuchte Kapazität [kWh]",
                50, 10_000, 1_000, 50,
            )
            cap_step = st.number_input(
                "Schrittweite [kWh]", 5, 500, 25, 5,
            )
        with col2:
            c_rate = st.slider(
                "C-Rate [1/h]", 0.25, 2.0, 0.5, 0.25,
                help="Verhältnis Lade-/Entladeleistung zu "
                     "Kapazität.",
            )
            roundtrip_eff = st.slider(
                "Round-Trip-Wirkungsgrad", 0.70, 0.98, 0.90, 0.01,
            )

    capacities = tuple(range(0, int(max_cap) + 1, int(cap_step)))
    params = (c_rate, roundtrip_eff)
    return targets, gen_scale, capacities, params


# ---------------------------------------------------------------------------
# Schritt 3 – Ergebnis
# ---------------------------------------------------------------------------

def _step_result(generation: pd.Series, load: pd.Series,
                 targets: dict, gen_scale: float,
                 capacities: tuple, params: tuple) -> None:
    """Render step 3: sizing result, KPIs and charts."""
    st.header("3️⃣ Ergebnis")

    c_rate, roundtrip_eff = params
    generation = generation * gen_scale

    sweep = _cached_sweep(generation, load, capacities, c_rate,
                          roundtrip_eff)
    baseline = sweep.iloc[0]

    st.subheader("Ausgangslage ohne Speicher")
    cols = st.columns(4)
    cols[0].metric("Verbrauch", _fmt_kwh(baseline['load_kwh']))
    cols[1].metric("Erzeugung",
                   _fmt_kwh(baseline['generation_kwh']))
    cols[2].metric("Autarkiegrad", _fmt_pct(baseline['autarky']))
    cols[3].metric("Eigenverbrauch",
                   _fmt_pct(baseline['self_consumption']))

    if not targets:
        st.info("Bitte in Schritt 2 mindestens ein Ziel "
                "auswählen.")
        return

    best = size_storage(
        sweep,
        target_autarky=targets.get('autarky'),
        target_self_consumption=targets.get('self_consumption'),
        target_import_free_share=targets.get('import_free_share'),
    )

    if best is None:
        max_row = sweep.iloc[-1]
        st.warning(
            f"Die Ziele sind selbst mit "
            f"{max_row['capacity_kwh']:.0f} kWh nicht erreichbar "
            f"(max. Autarkie {_fmt_pct(max_row['autarky'])}, "
            f"max. Eigenverbrauch "
            f"{_fmt_pct(max_row['self_consumption'])}). "
            f"Mögliche Hebel: PV-Ausbau erhöhen oder Ziele "
            f"anpassen."
        )
        best_capacity = float('nan')
    else:
        best_capacity = best['capacity_kwh']
        st.success(f"### Empfohlene Speichergröße: "
                   f"**{best_capacity:.0f} kWh**")
        d_autarky = (best['autarky'] - baseline['autarky']) * 100
        d_self = (best['self_consumption']
                  - baseline['self_consumption']) * 100
        cols = st.columns(3)
        cols[0].metric(
            "Autarkiegrad", _fmt_pct(best['autarky']),
            delta=f"+{d_autarky:.1f} %-Pkt.",
        )
        cols[1].metric(
            "Eigenverbrauch", _fmt_pct(best['self_consumption']),
            delta=f"+{d_self:.1f} %-Pkt.",
        )
        cols[2].metric(
            "Netzbezug",
            _fmt_kwh(best['grid_import_kwh']),
            delta=_fmt_kwh(best['grid_import_kwh']
                           - baseline['grid_import_kwh']),
            delta_color='inverse',
        )

    st.plotly_chart(
        _sweep_fig(sweep, targets, best_capacity),
        use_container_width=True,
    )

    with st.expander("📈 Beispielwoche im Detail"):
        cap = best_capacity if pd.notna(best_capacity) \
            else sweep['capacity_kwh'].iloc[-1]
        result = simulate_storage(
            generation, load, cap,
            StorageParams(c_rate=c_rate,
                          roundtrip_eff=roundtrip_eff),
        )
        st.caption(f"Simulation mit {cap:.0f} kWh Speicher")
        st.plotly_chart(
            _week_fig(result, timestep_hours(load.index)),
            use_container_width=True,
        )

    st.download_button(
        "⬇️ Alle simulierten Szenarien (CSV)",
        sweep.to_csv(index=False).encode('utf-8'),
        "speicher_szenarien.csv", "text/csv",
    )


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the linear three-step workflow."""
    st.title("🔋 EEG Speicher-Tool")
    st.markdown(
        "Wie groß muss ein Gemeinschaftsspeicher sein, um Ihre "
        "Ziele bei **Autarkie** und **Eigenverbrauch** zu "
        "erreichen?"
    )

    generation, load = _step_data()
    if generation is None or load is None:
        return

    targets, gen_scale, capacities, params = _step_targets()
    _step_result(generation, load, targets, gen_scale, capacities,
                 params)


if __name__ == "__main__":
    main()
