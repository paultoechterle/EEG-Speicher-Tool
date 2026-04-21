"""Simple Flask web GUI for the PVBatteryModel simulation.

- Minimal vanilla HTML/CSS UI.
- Users can enter simulate() parameters and see a generated PNG plot.

Run:
  pip install -r requirements.txt
  python webapp.py

Open http://127.0.0.1:5000 in your browser.
"""

import os
import io
from glob import glob

from flask import Flask, render_template, request, send_file, abort
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core import EEGModel

# Optional import for interactive plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


def _fig_html_from_df(df: pd.DataFrame, max_points: int = 8000) -> str:
    """Construct a Plotly html fragment from a simulation dataframe.
    This function downsamples if the dataframe is very large to avoid browser OOM.
    Returns a HTML fragment (div+script) suitable for embedding with |safe.
    """
    # ensure plotly available here
    if not PLOTLY_AVAILABLE:
        raise RuntimeError('Plotly not available')

    n = len(df)
    if n > max_points:
        step = max(1, n // max_points)
        df_plot = df.iloc[::step].copy()
    else:
        df_plot = df

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("PV and Load", "State of Charge", "Grid Import / Export"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['pv'], name='PV', line=dict(color='#2ca02c')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['load'], name='Load', line=dict(color='#d62728')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['soc'], name='SoC', line=dict(color='#ff7f0e')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['grid_import'], name='Grid Import', line=dict(color="#0095ff")),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['grid_export'], name='Grid Export', line=dict(color="#8400ff")),
                  row=3, col=1)

    fig.update_yaxes(title_text='kWh', row=1, col=1)
    fig.update_yaxes(title_text='kWh', row=2, col=1)
    fig.update_yaxes(title_text='kWh', row=3, col=1)
    fig.update_layout(template='plotly_dark', height=600,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(t=100))

    return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'EEG Eben am Achensee', 'EEG Eben am Achensee')


def load_df_sum() -> pd.DataFrame:
    """Load and aggregate the CSV files into the same df_sum used in the notebooks.
    Result is cached after the first successful load.
    """
    if getattr(load_df_sum, '_cache', None) is not None:
        return load_df_sum._cache

    files = glob(os.path.join(DATA_DIR, '*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSV files found in {DATA_DIR}')

    dfs = []
    for file in sorted(files):
        # read header rows to construct column names, then read data
        raw_data = pd.read_csv(file, sep=';', header=None, nrows=2)
        original_headers = raw_data.iloc[0].tolist()
        id_strings = raw_data.iloc[1].tolist()

        new_column_names = []
        for header, id_string in zip(original_headers, id_strings):
            if pd.notna(id_string) and str(id_string).strip():
                new_col_name = f"{header}_{id_string}"
            else:
                new_col_name = str(header)
            new_column_names.append(new_col_name)

        df = pd.read_csv(file, sep=';', decimal=',', header=None, skiprows=2)
        df.columns = new_column_names
        # the original data uses a 'Zeitpunkt' column with format '%d.%m.%Y, %H:%M:%S'
        if 'Zeitpunkt' not in df.columns and 'timestamp' not in df.columns:
            raise ValueError(f"File {file} does not contain expected 'Zeitpunkt' column")
        # normalize timestamp column name
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['Zeitpunkt'], format='%d.%m.%Y, %H:%M:%S')
        df.set_index('timestamp', inplace=True)
        dfs.append(df)

    merged_df = pd.concat(dfs)

    # compute same aggregated columns as in the notebook
    gb_mask = merged_df.columns.str.contains('Gesamtbezug')
    merged_df['SUMME_Gesamtsbezug [kWh]'] = merged_df.loc[:, gb_mask].sum(axis=1)
    rb_mask = merged_df.columns.str.contains('Restbezug')
    merged_df['SUMME_Restbezug [kWh]'] = merged_df.loc[:, rb_mask].sum(axis=1)
    bez_mask = merged_df.columns.str.contains('bezogen')
    merged_df['SUMME_Effektiv aus Gemeinschaft bezogen [kWh]'] = merged_df.loc[:, bez_mask].sum(axis=1)
    gl_mask = merged_df.columns.str.contains('Gesamtlieferung ')
    merged_df['SUMME_Gesamtlieferung [kWh]'] = merged_df.loc[:, gl_mask].sum(axis=1)
    rl_mask = merged_df.columns.str.contains('Restlieferung ')
    merged_df['SUMME_Restlieferung [kWh]'] = merged_df.loc[:, rl_mask].sum(axis=1)
    gel_mask = merged_df.columns.str.contains('geliefert ')
    merged_df['SUMME_Effektiv an Gemeinschaft geliefert [kWh]'] = merged_df.loc[:, gel_mask].sum(axis=1)

    df_sum = merged_df.iloc[:, -6:].sort_index(axis=0)
    df_sum.rename(columns={
        'SUMME_Gesamtsbezug [kWh]': 'bez_ges',
        'SUMME_Restbezug [kWh]': 'bez_rest',
        'SUMME_Effektiv aus Gemeinschaft bezogen [kWh]': 'bez_eff',
        'SUMME_Gesamtlieferung [kWh]': 'lief_ges',
        'SUMME_Restlieferung [kWh]': 'lief_rest',
        'SUMME_Effektiv an Gemeinschaft geliefert [kWh]': 'lief_eff'
    }, inplace=True)

    load_df_sum._cache = df_sum
    return df_sum


# Try to pre-load dataset for faster first requests
try:
    DF_SUM = load_df_sum()
    DATA_LOAD_ERROR = None
except Exception as exc:  # pragma: no cover - simple app-level handling
    DF_SUM = None
    DATA_LOAD_ERROR = str(exc)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    params = {
        'pv_scale': request.args.get('pv_scale', '1.0'),
        'storage': request.args.get('storage', '0'),
        'c_rate': request.args.get('c_rate', '1.0'),
        'roundtrip_eff': request.args.get('roundtrip_eff', '0.9'),
        'initial_soc': request.args.get('initial_soc', '0'),
        'allow_grid_charge': request.args.get('allow_grid_charge', 'off'),
    }
    show_plot = 'pv_scale' in request.args or 'storage' in request.args

    fig_html = None
    if show_plot and DF_SUM is not None and PLOTLY_AVAILABLE:
        try:
            pv_scale = float(params['pv_scale'])
            storage = float(params['storage'])
            c_rate = float(params['c_rate'])
            roundtrip_eff = float(params['roundtrip_eff'])
            initial_soc = float(params['initial_soc'])
            allow_grid_charge = params['allow_grid_charge'] in ('on', '1', 'true', 'True')

            pv_base = DF_SUM['lief_ges'].fillna(0)
            load = DF_SUM['bez_ges'].fillna(0)
            model = PVBatteryModel(pv_base, load)
            df = model.simulate(pv_scale=pv_scale, storage=storage, c_rate=c_rate,
                                roundtrip_eff=roundtrip_eff, allow_grid_charge=allow_grid_charge,
                                initial_soc=initial_soc, keep_timeseries=True)
            fig_html = _fig_html_from_df(df)
        except Exception:
            # If anything fails building the interactive plot, leave fig_html None and let the template show a message.
            fig_html = None

    return render_template('index.html', params=params, show_plot=show_plot, fig_html=fig_html, data_ok=(DF_SUM is not None), load_error=DATA_LOAD_ERROR)


@app.route('/plot')
def plot():
    """Return a PNG image for the requested simulation parameters."""
    if DF_SUM is None:
        abort(500, DATA_LOAD_ERROR or 'Dataset not available')

    try:
        pv_scale = float(request.args.get('pv_scale', 1.0))
        storage = float(request.args.get('storage', 0.0))
        c_rate = float(request.args.get('c_rate', 1.0))
        roundtrip_eff = float(request.args.get('roundtrip_eff', 0.9))
        initial_soc = float(request.args.get('initial_soc', 0.0))
        allow_grid_charge = request.args.get('allow_grid_charge', 'off') in ('on', '1', 'true', 'True')
    except ValueError:
        abort(400, 'Invalid numeric parameter')

    # Build and run model
    pv_base = DF_SUM['lief_ges'].fillna(0)
    load = DF_SUM['bez_ges'].fillna(0)
    model = PVBatteryModel(pv_base, load)
    df = model.simulate(pv_scale=pv_scale, storage=storage, c_rate=c_rate,
                        roundtrip_eff=roundtrip_eff, allow_grid_charge=allow_grid_charge,
                        initial_soc=initial_soc, keep_timeseries=True)

    # build a simple figure (two subplots)
    # fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # ax1, ax2 = axes
    # ax1.plot(df.index, df['pv'], label='PV', color='tab:green')
    # ax1.plot(df.index, df['load'], label='Load', color='tab:red')
    # ax1.set_ylabel('kWh')
    # ax1.legend()

    # ax2.plot(df.index, df['soc'], label='State of charge', color='tab:orange')
    # ax2.set_ylabel('kWh')
    # ax2.set_xlabel('Time')
    fig, axes = model.plot_timeseries(df=df, figsize=(8, 6))
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


@app.route('/interactive')
def interactive():
    """Return an HTML page with an interactive Plotly plot for the requested parameters."""
    if DF_SUM is None:
        abort(500, DATA_LOAD_ERROR or 'Dataset not available')
    if not PLOTLY_AVAILABLE:
        abort(500, 'Plotly is not installed. Install it with `pip install plotly` or update requirements.')

    try:
        pv_scale = float(request.args.get('pv_scale', 1.0))
        storage = float(request.args.get('storage', 0.0))
        c_rate = float(request.args.get('c_rate', 1.0))
        roundtrip_eff = float(request.args.get('roundtrip_eff', 0.9))
        initial_soc = float(request.args.get('initial_soc', 0.0))
        allow_grid_charge = request.args.get('allow_grid_charge', 'off') in ('on', '1', 'true', 'True')
    except ValueError:
        abort(400, 'Invalid numeric parameter')

    # Build and run model
    pv_base = DF_SUM['lief_ges'].fillna(0)
    load = DF_SUM['bez_ges'].fillna(0)
    model = PVBatteryModel(pv_base, load)
    df = model.simulate(pv_scale=pv_scale, storage=storage, c_rate=c_rate,
                        roundtrip_eff=roundtrip_eff, allow_grid_charge=allow_grid_charge,
                        initial_soc=initial_soc, keep_timeseries=True)

    fig_html = _fig_html_from_df(df)

    params = {
        'pv_scale': request.args.get('pv_scale', '1.0'),
        'storage': request.args.get('storage', '0'),
        'c_rate': request.args.get('c_rate', '1.0'),
        'roundtrip_eff': request.args.get('roundtrip_eff', '0.9'),
        'initial_soc': request.args.get('initial_soc', '0'),
        'allow_grid_charge': request.args.get('allow_grid_charge', 'off'),
    }

    return render_template('interactive.html', fig_html=fig_html, params=params, data_ok=(DF_SUM is not None), load_error=DATA_LOAD_ERROR)


if __name__ == '__main__':  # pragma: no cover - run by developer
    app.run(debug=True)
