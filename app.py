"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from powerguard.data import load_hourly, validate_upload, fingerprint
from powerguard.inference import Engine
from powerguard import alerts

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Power Guard · Team T2', page_icon='⚡', layout='wide')
st.markdown('''<style>
.stApp {background: #0c1422;}
[data-testid="stSidebar"] {background: #101e30;}
.block-container {padding-top: 2.1rem; max-width: 1480px;}
h1,h2,h3 {letter-spacing: -.025em;}
[data-testid="stMetric"] {background:#142338;border:1px solid #26394e;border-radius:12px;padding:16px;}
[data-testid="stMetricLabel"] {color:#aabace;}
[data-testid="stMetricValue"] {color:#b6f279;}
.eyebrow {font-size:12px;letter-spacing:2px;color:#b6f279;font-weight:600;}
.subtitle {color:#aabace;font-size:16px;margin-top:-12px;margin-bottom:24px;}
.note {padding:14px 18px;background:#142338;border-left:3px solid #b6f279;border-radius:5px;}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return Engine(ROOT/'artifacts')


@st.cache_data
def assets():
    return (json.loads((ROOT/'artifacts'/'metadata.json').read_text()),
            json.loads((ROOT/'reports'/'metrics.json').read_text()),
            json.loads((ROOT/'data'/'provenance.json').read_text()))


@st.cache_data
def report_rows(name):
    df = pd.read_csv(ROOT/'reports'/name, parse_dates=['timestamp'])
    for col in ['observed','ae_alert','if_alert','injected']:
        if col in df: df[col] = df[col].astype(str).str.lower().eq('true')
    return df


@st.cache_data(max_entries=3)
def analyze_upload(content):
    return get_engine().predict_rows(validate_upload(content))


def csv_bytes(df):
    return df.to_csv(index=False).encode()


def chart(rows, detector, threshold, future=None):
    prefix = 'ae' if detector=='Autoencoder' else 'if'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows.timestamp,y=rows.load_kw,name='Observed load',
                            line=dict(color='#b6f279',width=2),connectgaps=False))
    fig.add_trace(go.Scatter(x=rows.timestamp,y=rows.forecast_kw,name='LSTM forecast',
                            line=dict(color='#67b7ff',width=2,dash='dot'),connectgaps=False))
    a = rows[rows[f'{prefix}_alert']]
    fig.add_trace(go.Scatter(x=a.timestamp,y=a.load_kw,name='Flagged for review',mode='markers',
                            marker=dict(color='#ff886b',size=10,symbol='diamond')))
    if 'injected' in rows:
        injected = rows[rows.injected]
        fig.add_trace(go.Scatter(x=injected.timestamp,y=injected.load_kw,name='Injected test event',
                                mode='markers',marker=dict(color='#fbc55f',size=15,symbol='circle-open')))
    if future:
        fig.add_trace(go.Scatter(x=[future['timestamp']],y=[future['forecast_kw']],mode='markers',
                                name='Next hour',marker=dict(color='#67b7ff',size=13,symbol='star')))
    fig.update_layout(height=390,margin=dict(l=10,r=10,t=20,b=20),paper_bgcolor='#0c1422',
                      plot_bgcolor='#0c1422',font_color='#d4deeb',hovermode='x unified',
                      legend=dict(orientation='h',y=1.15),yaxis_title='Mean load (kW)',
                      xaxis_title='Source local time',xaxis=dict(gridcolor='#233246'),yaxis=dict(gridcolor='#233246'))
    return fig


try:
    meta, metrics, provenance = assets()
    engine = get_engine()
except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
    st.error('Project artifacts are missing or incompatible. Install requirements, then run python scripts/train.py.')
    st.code(str(exc)); st.stop()

with st.sidebar:
    st.markdown('## ⚡ Power Guard')
    st.caption('ENERGY INTELLIGENCE · TEAM T2')
    page = st.radio('Workspace', ['Monitor & replay','Model results','Upload readings','Project review'])
    st.divider()
    detector = st.selectbox('Anomaly detector', ['Autoencoder','Isolation Forest'])
    st.caption('Each model uses its own threshold calibrated on validation readings.')
    st.divider()
    st.caption('AMC Engineering College (Autonomous)\n\nCSE (AI & ML) · Mini Project\n\nGuide: Dr. Nandeeswar S B')
    st.caption('Prototype v1 · Hourly advisory monitoring')

prefix = 'ae' if detector=='Autoencoder' else 'if'
threshold = meta[f'{prefix}_threshold']
st.markdown('<div class="eyebrow">POWER GUARD / REVIEW PROTOTYPE</div>', unsafe_allow_html=True)
st.title({'Monitor & replay':'Understand the next hour.','Model results':'Evidence behind the models.',
          'Upload readings':'Explore your hourly readings.','Project review':'From meter readings to reviewable alerts.'}[page])
st.markdown('<div class="subtitle">AI-driven load forecasting and anomaly detection for apartment energy monitoring.</div>', unsafe_allow_html=True)

if page=='Monitor & replay':
    scenario = st.radio('Data scenario', ['Original UCI readings','Injected event challenge'], horizontal=True)
    challenge = scenario=='Injected event challenge'
    st.info('Historical replay of one household in France. This is a household-data proxy for the proposed apartment application.')
    if challenge:
        st.warning('SIMULATED TEST EVENTS: known spikes, sustained increases and near-zero drops were added to held-out readings.')
    rows = report_rows('injection_challenge.csv' if challenge else 'test_predictions.csv')
    source = ('uci-injected-' if challenge else 'uci-original-')+meta['model_fingerprint']
    key = f'cursor_{source}'
    if key not in st.session_state: st.session_state[key] = 48+168
    c1,c2,c3,c4 = st.columns([1,1,1,2])
    with c1:
        if st.button('Reset replay', width='stretch'): st.session_state[key]=48
    with c2:
        if st.button('Advance 1 hour', width='stretch'): st.session_state[key]=min(len(rows),st.session_state[key]+1)
    with c3:
        if st.button('Advance 24 hours', width='stretch'): st.session_state[key]=min(len(rows),st.session_state[key]+24)
    with c4: playing=st.toggle('Auto replay · one hour every 2 seconds', key=f'play_{source}')
    window = st.select_slider('Chart window', options=[24,72,168,336], value=168, format_func=lambda n:f'{n} hours')

    @st.fragment(run_every=2 if playing else None)
    def replay_panel():
        if playing: st.session_state[key]=min(len(rows),st.session_state[key]+1)
        end=st.session_state[key]
        history=rows.iloc[:end]
        shown=history.tail(window)
        current=history.iloc[-1]
        try: future=engine.next_hour(history)
        except ValueError: future=None
        current_kw=f'{current.load_kw:.2f} kW' if pd.notna(current.load_kw) else 'Missing'
        a,b,c,d=st.columns(4)
        a.metric('Latest hourly load',current_kw)
        b.metric('Next-hour LSTM',f'{future["forecast_kw"]:.2f} kW' if future else 'Unavailable')
        c.metric('Alerts in chart',int(shown[f'{prefix}_alert'].sum()))
        # Mean kW * 1 h per observed hourly interval; incomplete hours are explicitly marked.
        d.metric('Observed energy estimate',f'{shown.loc[shown.observed,"load_kw"].sum():.1f} kWh')
        st.caption(f'Replay cursor: {current.timestamp} · Source local time · {int(shown.observed.sum())}/{len(shown)} observed hours in chart. Energy estimate excludes imputed/missing hours.')
        st.plotly_chart(chart(shown,detector,threshold,future),width='stretch')
        st.progress(max(0,(end-48)/(len(rows)-48)),text=f'{end-48:,} of {len(rows)-48:,} test-period hours replayed')
        score = go.Figure(go.Scatter(x=shown.timestamp,y=shown[f'{prefix}_score'],name='Anomaly score',line=dict(color='#a6a0ff')))
        score.add_hline(y=threshold,line_dash='dash',line_color='#ff886b',annotation_text='Validation threshold')
        score.update_layout(height=210,margin=dict(l=10,r=10,t=25,b=10),paper_bgcolor='#0c1422',plot_bgcolor='#0c1422',
                            font_color='#d4deeb',yaxis_title=f'{detector} score')
        st.plotly_chart(score,width='stretch')
        st.caption('An alert marks unusual consumption for investigation. It does not identify a confirmed fault or electricity theft.')
        alerts.record(ROOT/'runtime'/'alerts.db',history.iloc[48:],source,detector,threshold)
        log=alerts.read(ROOT/'runtime'/'alerts.db',source)
        st.subheader('Alert register')
        st.caption('Saved locally across restarts. The register can retain events from an earlier replay position or the other detector.')
        if log.empty: st.success('No flagged readings have been replayed for this scenario.')
        else:
            st.dataframe(log,hide_index=True,width='stretch')
            st.download_button('Download alert register',csv_bytes(log),'power_guard_alerts.csv','text/csv',key=f'dl_{source}')
        st.download_button('Download visible readings',csv_bytes(shown),'power_guard_visible_readings.csv','text/csv',key=f'vis_{source}')
    replay_panel()
    if st.button('Mark this scenario’s logged alerts as reviewed'):
        alerts.mark_reviewed(ROOT/'runtime'/'alerts.db',source); st.rerun()

elif page=='Model results':
    st.markdown('### Forecasting · held-out original readings')
    st.caption('All methods use the same valid test timestamps. Each prediction is one step ahead using earlier observed readings.')
    reg=pd.DataFrame(metrics['forecast'])
    st.dataframe(reg.rename(columns={'model':'Model','mae_kw':'MAE (kW)','rmse_kw':'RMSE (kW)','samples':'Test hours'}),hide_index=True,width='stretch')
    st.caption(f'Train ends {meta["train_last"]} · Validation ends {meta["validation_last"]} · Test: {meta["test_first"]} to {meta["test_last"]}')
    st.markdown('### Anomaly detection · injected-event recovery')
    st.warning(metrics['injection_label_scope'])
    det=pd.DataFrame(metrics['injected_event_detection']).T.reset_index(names='Detector')
    st.dataframe(det,hide_index=True,width='stretch')
    st.dataframe(pd.DataFrame(metrics['injected_event_types']).T.reset_index(names='Injected event'),hide_index=True,width='stretch')
    st.caption('Metrics are pointwise, not event-level. A missed injected hour is a false negative. An alert on unmodified background counts as a false positive for this controlled challenge.')
    st.markdown('### Measured processing latency')
    a,b=st.columns(2)
    a.metric('Median',f'{metrics["latency"]["median_ms"]:.1f} ms')
    b.metric('95th percentile',f'{metrics["latency"]["p95_ms"]:.1f} ms')
    st.caption(metrics['latency']['scope'])
    st.caption(f'{metrics["latency"]["iterations"]} warm runs on the build machine. Your laptop may differ. This measures processing after an hourly reading, not the wait for the next reading.')
    with st.expander('Training curves and calibration details'):
        for title,log in metrics['epoch_history'].items():
            st.write(title)
            st.line_chart(pd.DataFrame(log).set_index('epoch'))
        st.json({k:meta[k] for k in ['threshold_quantile','ae_threshold','if_threshold','screening','anomaly_training_samples','screened_out_training_samples']})
    st.download_button('Download measured results (JSON)',(ROOT/'reports'/'metrics.json').read_bytes(),'power_guard_metrics.json','application/json')

elif page=='Upload readings':
    st.info('Upload one site’s hourly mean load in kW. The included model was trained on a single UCI household; new sites need local retraining and calibration before operational use.')
    st.code('timestamp,load_kw\n2026-11-01 00:00:00,0.72\n2026-11-01 01:00:00,0.61',language='text')
    st.caption('At least 72 rows. Use exact hourly local timestamps without timezone offsets; no duplicates, negative loads or infinite values. Short gaps may use up to three earlier-value fills.')
    st.download_button('Download valid sample CSV',(ROOT/'data'/'sample_upload.csv').read_bytes(),'sample_upload.csv','text/csv')
    uploaded=st.file_uploader('Hourly consumption CSV',type=['csv'])
    if uploaded:
        try:
            analyzed=analyze_upload(uploaded.getvalue())
            st.success(f'{len(analyzed):,} hourly slots processed. {int((~analyzed.observed).sum()):,} missing input hours identified.')
            try:
                future=engine.next_hour(analyzed)
                st.metric('Next-hour forecast',f'{future["forecast_kw"]:.2f} kW',help=f'For {future["timestamp"]}')
            except ValueError as exc:
                future=None; st.warning(str(exc))
            st.plotly_chart(chart(analyzed.tail(336),detector,threshold,future),width='stretch')
            st.dataframe(analyzed.tail(100),hide_index=True,width='stretch')
            st.download_button('Download analyzed CSV',csv_bytes(analyzed),'power_guard_analysis.csv','text/csv')
            if st.button('Save flagged uploaded readings to alert register'):
                source='upload-'+fingerprint(analyzed)+'-'+meta['model_fingerprint']
                alerts.record(ROOT/'runtime'/'alerts.db',analyzed,source,detector,threshold)
                st.success('Flagged readings saved to the local register.')
                st.dataframe(alerts.read(ROOT/'runtime'/'alerts.db',source),hide_index=True)
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(str(exc))

else:
    st.markdown('### Review demonstration · 5 minutes')
    st.markdown('''1. **Explain the problem:** facility managers need expected demand and understandable alerts.
2. **Show original readings:** explain actual load, next-hour LSTM forecast and kW versus kWh.
3. **Select Injected event challenge:** compare both detectors, advance the replay and inspect flagged timestamps.
4. **Download the alert register:** show load, score, threshold and whether the reading was modified.
5. **Open Model results:** compare the LSTM with both simple baselines and discuss missed near-zero drops.
6. **Open Upload readings:** try the included sample CSV and explain why a new apartment needs local validation.''')
    st.markdown('### Implemented pipeline')
    st.graphviz_chart('''digraph {rankdir=TB; node [shape=box,style="rounded,filled",fillcolor="#142338",fontcolor="#dce7f4",color="#58708a"]; bgcolor="transparent"; edge [color="#b6f279"]; data [label="UCI readings / hourly CSV"]; prep [label="Validate, aggregate and causal gap handling"]; forecast [label="48-hour history → LSTM → next-hour kW"]; detect [label="Temporal features → autoencoder / Isolation Forest"]; view [label="Dashboard, replay and SQLite alert register"]; data -> prep; prep -> forecast; prep -> detect; forecast -> view; detect -> view;}''')
    st.markdown('### Scope you can defend')
    st.markdown('''- Working software prototype with trained models, measured results and local alert persistence.
- Public single-household data supports a proof of concept; it is not an Indian apartment dataset.
- Replay simulates the arrival of historical readings. No smart meter is connected.
- Near-zero drops can resemble genuinely low consumption and may be missed.
- No automatic switching, physical fault diagnosis, energy-saving claim or live notification delivery.
- Weather, GRU comparison and site-specific deployment remain later extensions.''')
    with st.expander('Dataset provenance and preprocessing'):
        st.json(provenance)
        st.link_button('UCI dataset and attribution','https://doi.org/10.24432/C58K54')
    st.markdown('### Team T2')
    st.write('P Vishal Ravi · Sagar J · Jeevan U Acharya · Varshith M')
    st.write('Guide: Dr. Nandeeswar S B · AMC Engineering College (Autonomous), CSE (AI & ML)')
    st.caption('The docs folder contains the setup guide, architecture, review script and viva preparation.')
