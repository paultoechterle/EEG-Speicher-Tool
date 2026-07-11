"""Streamlit app of the EEG storage sizing tool.

Linear workflow:
    1. Datengrundlage waehlen (synthetisch oder Messdaten-CSV)
    2. Ziele festlegen (Autarkie / Eigenverbrauch)
    3. Ergebnis: minimale Speichergroesse, die die Ziele erreicht

Starten:
    streamlit run app.py
"""

import io
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

# Zeitliche Aggregation für die Profil-Vorschau (Pandas-Resample-Regel;
# ``None`` = Rohdaten ohne Resampling).
_PROFILE_FREQ = {
    "Monatssumme": "MS",
    "Wochensumme": "W",
    "Tagessumme": "D",
    "Stundenwerte": "h",
    "Rohdaten (15 min)": None,
}

# Kennzahlen der Gegenüberstellung ohne/mit Speicher:
# (Label, Sweep-Spalte, Typ, kleiner_ist_besser).
_COMPARE_KPIS = [
    ("Autarkiegrad", 'autarky', 'pct', False),
    ("Eigenverbrauchsquote", 'self_consumption', 'pct', False),
    ("Zeit ohne Netzbezug", 'import_free_share', 'pct', False),
    ("Netzbezug", 'grid_import_kwh', 'kwh', True),
    ("Netzeinspeisung", 'grid_export_kwh', 'kwh', True),
]

# Deutsche Spaltennamen für den Zeitreihen-Export.
_EXPORT_COLS = {
    'generation': 'Erzeugung [kWh]',
    'load': 'Last [kWh]',
    'grid_import': 'Netzbezug [kWh]',
    'grid_export': 'Netzeinspeisung [kWh]',
    'charge': 'Speicher Ladung [kWh]',
    'discharge': 'Speicher Entladung [kWh]',
    'soc': 'Speicherfuellstand [kWh]',
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


def _kpi_value(row, col: str, kind: str) -> str:
    """Format a single KPI value for the comparison table.

    Args:
        row: Sweep row (pd.Series or dict) with KPI columns.
        col (str): KPI column name.
        kind (str): ``'pct'`` for a 0-1 ratio or ``'kwh'`` for an
            energy amount.

    Returns:
        str: Formatted value.
    """
    return _fmt_pct(row[col]) if kind == 'pct' else _fmt_kwh(row[col])


def _kpi_delta(base, scenario, col: str, kind: str) -> str:
    """Format the change of a KPI as a signed delta string.

    Args:
        base: Baseline sweep row (ohne Speicher).
        scenario: Scenario sweep row (mit Speicher).
        col (str): KPI column name.
        kind (str): ``'pct'`` or ``'kwh'`` (see :func:`_kpi_value`).

    Returns:
        str: Signed delta, e.g. ``+19.5 %-Pkt.`` or ``-40.0 MWh``.
    """
    if kind == 'pct':
        return f"{(scenario[col] - base[col]) * 100:+.1f} %-Pkt."
    diff = scenario[col] - base[col]
    # _fmt_kwh behält das Vorzeichen; Streamlit färbt daraus das
    # Delta (mit ``inverse`` ist eine Abnahme die Verbesserung).
    return _fmt_kwh(diff)


def _export_frame(result: pd.DataFrame) -> pd.DataFrame:
    """Rename simulation columns to German labels for export."""
    out = result.rename(columns=_EXPORT_COLS)
    out.index.name = 'Zeitpunkt'
    return out


def _to_excel(result: pd.DataFrame) -> bytes:
    """Serialise a simulation time series to an XLSX byte string.

    Args:
        result (pd.DataFrame): Output of
            :func:`core.simulate_storage`.

    Returns:
        bytes: In-memory XLSX file content.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        _export_frame(result).to_excel(writer, sheet_name='Zeitreihe')
    return buffer.getvalue()


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

def _profile_fig(generation: pd.Series, load: pd.Series,
                 freq_label: str) -> go.Figure:
    """Time series of generation and load at a chosen aggregation.

    Args:
        generation (pd.Series): Generation [kWh per timestep].
        load (pd.Series): Load [kWh per timestep].
        freq_label (str): Key of :data:`_PROFILE_FREQ`.

    Returns:
        go.Figure: Plotly line chart.
    """
    freq = _PROFILE_FREQ[freq_label]
    if freq is None:
        gen_plot, load_plot = generation, load
        y_title = 'Energie je Zeitschritt [kWh]'
    else:
        gen_plot = generation.resample(freq).sum()
        load_plot = load.resample(freq).sum()
        y_title = 'Energie je Periode [kWh]'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gen_plot.index, y=gen_plot.values, name='Erzeugung',
        line=dict(color=COLOR['generation'], width=2),
        hovertemplate='%{y:.0f} kWh<extra>Erzeugung</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=load_plot.index, y=load_plot.values, name='Last',
        line=dict(color=COLOR['load'], width=2),
        hovertemplate='%{y:.0f} kWh<extra>Last</extra>',
    ))
    fig.update_layout(
        yaxis_title=y_title,
        legend=dict(orientation='h', y=-0.2),
        height=380, hovermode='x unified', margin=dict(t=30),
    )
    return fig


def _profile_section(generation: pd.Series, load: pd.Series) -> None:
    """Expander to verify the input profiles as a time-series plot."""
    with st.expander("🔍 Datengrundlage prüfen"):
        freq_label = st.selectbox(
            "Zeitliche Auflösung", list(_PROFILE_FREQ), index=2,
        )
        st.plotly_chart(
            _profile_fig(generation, load, freq_label),
            use_container_width=True,
        )
        cols = st.columns(2)
        cols[0].metric("Erzeugung gesamt", _fmt_kwh(generation.sum()))
        cols[1].metric("Verbrauch gesamt", _fmt_kwh(load.sum()))


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
        ["Synthetische Profile", "Messdaten (CSV)"],
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

    # Nur ein Ziel gleichzeitig, damit die Optimierung eindeutig
    # bleibt. Pro Option: (KPI-Spalte, Slider-Label, Default, Hilfe).
    target_options = {
        "Autarkiegrad": (
            'autarky', "Ziel Autarkiegrad [%]", 50,
            "Anteil des Verbrauchs, der über den gesamten Zeitraum "
            "aus eigener Erzeugung und Speicher gedeckt wird – "
            "energiemengenbezogen (kWh). 100 % bedeutet: es wird "
            "kein Strom mehr aus dem Netz bezogen. Ein größerer "
            "Speicher hebt vor allem diesen Wert, weil "
            "Erzeugungsüberschüsse für Verbrauchsspitzen ohne "
            "Erzeugung zwischengespeichert werden.",
        ),
        "Eigenverbrauchsquote": (
            'self_consumption', "Ziel Eigenverbrauch [%]", 70,
            "Anteil der erzeugten Energie, der in der Gemeinschaft "
            "selbst genutzt statt ins Netz eingespeist wird "
            "(kWh-bezogen). 100 % bedeutet: keine Einspeisung von "
            "Überschüssen. Relevant vor allem bei viel PV – der "
            "Speicher nimmt Mittagsspitzen auf, die sonst "
            "eingespeist würden.",
        ),
        "Zeit ohne Netzbezug": (
            'import_free_share', "Ziel Zeit ohne Netzbezug [%]", 50,
            "Anteil der Zeitschritte (15-min-Intervalle), in denen "
            "die Gemeinschaft komplett ohne Netzbezug auskommt – "
            "zeit- statt energiebezogen. Ein anspruchsvolles Ziel, "
            "da schon geringer Netzbezug ein Intervall als "
            "'nicht autark' zählt; erfordert meist deutlich "
            "größere Speicher als der Autarkiegrad.",
        ),
    }
    choice = st.radio(
        "Optimierungsziel", list(target_options), horizontal=True,
        help="Welches Ziel soll mit dem Speicher erreicht werden?"
    )
    col, slider_label, default, option_help = target_options[choice]
    st.caption(option_help)
    targets = {col: st.slider(slider_label, 5, 100, default, 5)
               / 100.0}

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

def _kpi_comparison(baseline, scenario, cap_label: str) -> None:
    """Render a side-by-side KPI comparison ohne/mit Speicher.

    Lays out one row per KPI in three columns (Kennzahl, Ohne
    Speicher, Mit Speicher) so the effect of the storage is directly
    readable.

    Args:
        baseline: Sweep row at 0 kWh (ohne Speicher).
        scenario: Sweep row of the storage scenario (mit Speicher).
        cap_label (str): Human-readable capacity of the scenario.
    """
    st.subheader("Kennzahlen im Vergleich")
    st.caption(
        f"Verbrauch {_fmt_kwh(baseline['load_kwh'])} · "
        f"Erzeugung {_fmt_kwh(baseline['generation_kwh'])} "
        f"– unabhängig vom Speicher"
    )

    head = st.columns([3, 2, 2], vertical_alignment="bottom")
    head[0].markdown("**Kennzahl**")
    head[1].markdown("**Ohne Speicher**")
    head[2].markdown(f"**Mit Speicher**  \n{cap_label}")

    for label, col, kind, smaller_better in _COMPARE_KPIS:
        row = st.columns([3, 2, 2], vertical_alignment="center")
        row[0].markdown(label)
        row[1].metric(
            label, _kpi_value(baseline, col, kind),
            label_visibility="collapsed",
        )
        row[2].metric(
            label, _kpi_value(scenario, col, kind),
            delta=_kpi_delta(baseline, scenario, col, kind),
            delta_color='inverse' if smaller_better else 'normal',
            label_visibility="collapsed",
        )


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

    if not targets:
        st.info("Bitte in Schritt 2 mindestens ein Ziel "
                "auswählen, um eine Speichergröße zu empfehlen.")
        return

    best = size_storage(
        sweep,
        target_autarky=targets.get('autarky'),
        target_self_consumption=targets.get('self_consumption'),
        target_import_free_share=targets.get('import_free_share'),
    )

    if best is None:
        # Ziele nicht erreichbar → mit dem größten Speicher
        # vergleichen, damit das Potenzial dennoch sichtbar wird.
        scenario = sweep.iloc[-1]
        best_capacity = float('nan')
        st.warning(
            f"Die Ziele sind selbst mit "
            f"{scenario['capacity_kwh']:.0f} kWh nicht erreichbar "
            f"(max. Autarkie {_fmt_pct(scenario['autarky'])}, "
            f"max. Eigenverbrauch "
            f"{_fmt_pct(scenario['self_consumption'])}). "
            f"Mögliche Hebel: PV-Ausbau erhöhen oder Ziele "
            f"anpassen."
        )
        cap_label = f"größter Speicher · {scenario['capacity_kwh']:.0f} kWh"
    else:
        scenario = best
        best_capacity = best['capacity_kwh']

        st.plotly_chart(
            _sweep_fig(sweep, targets, best_capacity),
            use_container_width=True,
        )

        st.success(f"### Empfohlene Speichergröße: "
                   f"**{best_capacity:.0f} kWh**")
        cap_label = f"{best_capacity:.0f} kWh"

    _kpi_comparison(baseline, scenario, cap_label)


    # Zeitreihe des empfohlenen (bzw. größten) Szenarios – einmal
    # simulieren und für Detailgrafik und Download wiederverwenden.
    cap = best_capacity if pd.notna(best_capacity) \
        else sweep['capacity_kwh'].iloc[-1]
    result = simulate_storage(
        generation, load, cap,
        StorageParams(c_rate=c_rate, roundtrip_eff=roundtrip_eff),
    )

    with st.expander("📈 Beispielwoche im Detail"):
        st.caption(f"Simulation mit {cap:.0f} kWh Speicher")
        st.plotly_chart(
            _week_fig(result, timestep_hours(load.index)),
            use_container_width=True,
        )
    
    st.markdown("---")
    
    st.subheader("Downloads")
    st.caption(
        f"Zeitreihe des empfohlenen Szenarios ({cap:.0f} kWh) sowie "
        f"die Kennzahlen aller simulierten Speichergrößen."
    )
    tag = f"{cap:.0f}kWh"
    ts_csv = _export_frame(result).to_csv().encode('utf-8')
    cols = st.columns(3)
    cols[0].download_button(
        "⬇️ Zeitreihe (CSV)", ts_csv,
        f"speicher_{tag}_zeitreihe.csv", "text/csv",
    )
    cols[1].download_button(
        "⬇️ Zeitreihe (Excel)", _to_excel(result),
        f"speicher_{tag}_zeitreihe.xlsx",
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet",
    )
    cols[2].download_button(
        "⬇️ Alle Szenarien (CSV)",
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

    st.markdown("---")

    generation, load = _step_data()
    if generation is None or load is None:
        return
    _profile_section(generation, load)

    st.markdown("---")

    targets, gen_scale, capacities, params = _step_targets()

    st.markdown("---")
    
    _step_result(generation, load, targets, gen_scale, capacities,
                 params)


if __name__ == "__main__":
    main()
