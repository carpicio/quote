import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet Dual Core v39", page_icon="🧠", layout="wide")
st.title("🧠 Calcolatore Strategico Dual Core (v39)")
st.markdown("---")

# --- FUNZIONI MATEMATICHE ---
def get_implicit_probs(elo_home, elo_away, hfa=100):
    try:
        exponent = (elo_away - (elo_home + hfa)) / 400
        p_elo_h = 1 / (1 + 10**exponent)
        p_elo_a = 1 - p_elo_h
        return p_elo_h, p_elo_a
    except:
        return 0, 0

def remove_margin(odd_1, odd_x, odd_2):
    try:
        if odd_1 <= 0 or odd_x <= 0 or odd_2 <= 0: return 0, 0, 0
        inv_sum = (1/odd_1) + (1/odd_x) + (1/odd_2)
        return (1/odd_1)/inv_sum, (1/odd_x)/inv_sum, (1/odd_2)/inv_sum
    except:
        return 0, 0, 0

def calculate_row(row, hfa=100):
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'Fair_1': 0, 'Fair_X': 0, 'Fair_2': 0, 'ELO_Diff': 0}
    try:
        elo_h = float(row.get('elohomeo', 1500))
        elo_a = float(row.get('eloawayo', 1500))
        o1 = float(row.get('cotaa', 0))
        ox = float(row.get('cotae', 0))
        o2 = float(row.get('cotad', 0))
    except:
        return pd.Series(res)
    
    if o1 > 0 and ox > 0 and o2 > 0:
        pf_1, pf_x, pf_2 = remove_margin(o1, ox, o2)
        p_elo_h, p_elo_a = get_implicit_probs(elo_h, elo_a, hfa)
        rem = 1 - pf_x
        p_fin_1 = rem * p_elo_h
        p_fin_2 = rem * p_elo_a
        
        res['Fair_1'] = 1/p_fin_1 if p_fin_1>0 else 0
        res['Fair_X'] = 1/pf_x if pf_x>0 else 0
        res['Fair_2'] = 1/p_fin_2 if p_fin_2>0 else 0
        res['EV_1'] = (o1 * p_fin_1) - 1
        res['EV_X'] = (ox * pf_x) - 1
        res['EV_2'] = (o2 * p_fin_2) - 1
        
    res['ELO_Diff'] = (elo_h + hfa) - elo_a
    return pd.Series(res)

# --- CARICAMENTO DATI ---
@st.cache_data(ttl=0)
def load_data(file):
    try:
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1')
            if len(df.columns) < 5: raise ValueError
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')

        df.columns = df.columns.str.strip().str.lower()
        
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2',
            'o2,5': 'cotao', 'u2,5': 'cotau',
            'data': 'datamecic', 'casa': 'txtechipa1', 'ospite': 'txtechipa2',
            'gg': 'cotagg', 'ng': 'cotang'
        }
        df = df.rename(columns=rename_map)
        
        cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2', 'cotagg', 'cotang']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        calc = df.apply(lambda r: calculate_row(r), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        # Gestione Risultati
        df['res_1x2'] = '-' 
        if 'scor1' in df.columns and 'scor2' in df.columns:
            mask = df['scor1'].notna() & df['scor2'].notna()
            df.loc[mask & (df['scor1'] > df['scor2']), 'res_1x2'] = '1'
            df.loc[mask & (df['scor1'] == df['scor2']), 'res_1x2'] = 'X'
            df.loc[mask & (df['scor1'] < df['scor2']), 'res_1x2'] = '2'
            
            df['goals_tot'] = df['scor1'] + df['scor2']
            if 'cotao' in df.columns:
                df['res_o25'] = np.nan
                df.loc[mask, 'res_o25'] = (df.loc[mask, 'goals_tot'] > 2.5).astype(int)
            if 'cotagg' in df.columns:
                df['res_gg'] = np.nan
                df.loc[mask, 'res_gg'] = ((df.loc[mask, 'scor1'] > 0) & (df.loc[mask, 'scor2'] > 0)).astype(int)
            
        return df, None
    except Exception as e:
        return None, f"Errore: {str(e)}"

# --- MOTORE ANALISI CLUSTER ---
def analyze_input(df_hist, odd, ev, market_type):
    if df_hist is None or df_hist.empty: return 0, 0, 0
    
    f_odd_min, f_odd_max = odd * 0.9, odd * 1.1
    f_ev_min = ev - 0.05 
    
    mask = pd.Series(True, index=df_hist.index)
    
    target, col_odd = None, None
    if market_type == "1": target, col_odd = '1', 'cotaa'
    elif market_type == "2": target, col_odd = '2', 'cotad'
    elif market_type == "X": target, col_odd = 'X', 'cotae'
    
    mask &= (df_hist[col_odd] >= f_odd_min) & (df_hist[col_odd] <= f_odd_max)
    col_ev = f"EV_{market_type}"
    mask &= (df_hist[col_ev] >= f_ev_min)
    
    cluster = df_hist[mask]
    
    if len(cluster) >= 5:
        wins = len(cluster[cluster['res_1x2'] == target])
        profit = (cluster[cluster['res_1x2'] == target][col_odd] - 1).sum() - (len(cluster) - wins)
        roi = (profit / len(cluster)) * 100
        return len(cluster), roi, profit
    return 0, 0, 0

# --- INTERFACCIA ---
st.sidebar.header("📂 Gestione File")
file_hist = st.sidebar.file_uploader("1. Carica STORICO (Cervello)", type=["csv"], key="u_hist")
file_fut = st.sidebar.file_uploader("2. Carica FUTURE (Target)", type=["csv"], key="u_fut")

df_history, df_future = None, None

if file_hist:
    df_history, err_h = load_data(file_hist)
    if err_h: st.sidebar.error(f"Err Storico: {err_h}")
    else: st.sidebar.success(f"🧠 Storico: {len(df_history)} partite")

if file_fut:
    df_future, err_f = load_data(file_fut)
    if err_f: st.sidebar.error(f"Err Future: {err_f}")
    else: st.sidebar.success(f"🎯 Future: {len(df_future)} partite")

tab1, tab2, tab3 = st.tabs(["🔮 Manuale + Cluster Check", "🚀 Analisi File Future", "📊 Report Storico"])

with tab1:
    st.header("Analisi Singola Partita")
    
    if df_history is None:
        st.warning("⚠️ Carica prima il file STORICO a sinistra per attivare l'Intelligenza Artificiale dei Cluster.")
    
    col_input, col_res = st.columns([1, 1])
    
    with col_input:
        team_h = st.text_input("Casa", "Home")
        team_a = st.text_input("Ospite", "Away")
        c1, c2 = st.columns(2)
        # FIX ERROR: Usiamo value=... esplicitamente
        elo_h = c1.number_input("ELO Casa", value=1500, min_value=0, step=10)
        elo_a = c2.number_input("ELO Ospite", value=1500, min_value=0, step=10)
        
        c3, c4, c5 = st.columns(3)
        # FIX ERROR: Usiamo value=... esplicitamente
        o1 = c3.number_input("Quota 1", value=2.00, min_value=1.01)
        ox = c4.number_input("Quota X", value=3.00, min_value=1.01)
        o2 = c5.number_input("Quota 2", value=3.50, min_value=1.01)
        
        btn_calc = st.button("Analizza Partita", type="primary")

    with col_res:
        if btn_calc:
            row = {'elohomeo': elo_h, 'eloawayo': elo_a, 'cotaa': o1, 'cotae': ox, 'cotad': o2}
            res = calculate_row(row)
            
            st.subheader(f"{team_h} vs {team_a}")
            
            def show_smart_card(label, odd, ev, market_type):
                bg = "#e9ecef"
                border = "gray"
                extra_msg = ""
                
                if df_history is not None:
                    n, roi, prof = analyze_input(df_history, odd, ev, market_type)
                    if n > 0 and roi > 5:
                        bg = "#d4edda"
                        border = "green"
                        extra_msg = f"✅ <b>CLUSTER VINCENTE!</b><br>Su {n} casi simili:<br>ROI: <b>+{roi:.1f}%</b>"
                    elif n > 0 and roi < -5:
                        bg = "#f8d7da"
                        border = "red"
                        extra_msg = f"❌ <b>CLUSTER PERDENTE</b><br>Su {n} casi simili:<br>ROI: <b>{roi:.1f}%</b>"
                    elif n > 0:
                        extra_msg = f"⚖️ Storico Neutro ({n} casi)"
                    else:
                        extra_msg = "⚪ Dati storici insufficienti"
                
                ev_color = "green" if ev > 0 else "red"
                
                st.markdown(f"""
                <div style="background-color:{bg}; padding:10px; border-radius:8px; border-left:5px solid {border}; margin-bottom:10px;">
                    <div style="font-size:20px; font-weight:bold;">Segno {label}</div>
                    <div style="font-size:30px; font-weight:800;">{odd:.2f}</div>
                    <div style="color:{ev_color}; font-weight:bold;">EV: {ev:+.1%}</div>
                    <div style="margin-top:5px; font-size:13px; line-height:1.2;">{extra_msg}</div>
                </div>
                """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            with k1: show_smart_card("1", o1, res['EV_1'], "1")
            with k2: show_smart_card("X", ox, res['EV_X'], "X")
            with k3: show_smart_card("2", o2, res['EV_2'], "2")

with tab2:
    st.header("🚀 Cacciatore di Occasioni (File Future)")
    
    if df_future is None:
        st.info("Carica un file di partite FUTURE per vedere le segnalazioni automatiche.")
    elif df_history is None:
        st.warning("⚠️ Carica ANCHE lo Storico per permettermi di filtrare le partite buone da quelle cattive.")
        st.dataframe(df_future[['datamecic', 'txtechipa1', 'txtechipa2', 'cotaa', 'cotad', 'EV_1', 'EV_2']])
    else:
        st.write("Analisi incrociata: Cerco nel file Future le partite che rientrano nei Cluster Vincenti dello Storico.")
        
        min_roi_target = st.slider("Mostra solo strategie con ROI Storico > %", 0, 50, 10)
        
        results = []
        progress = st.progress(0)
        for i, row in df_future.iterrows():
            n1, roi1, p1 = analyze_input(df_history, row['cotaa'], row['EV_1'], "1")
            if n1 >= 10 and roi1 >= min_roi_target and row['EV_1'] > 0:
                results.append(dict(row, **{'Bet': '1', 'Quota': row['cotaa'], 'EV': row['EV_1'], 'ROI_Storico': roi1, 'Casi_Simili': n1}))
            
            n2, roi2, p2 = analyze_input(df_history, row['cotad'], row['EV_2'], "2")
            if n2 >= 10 and roi2 >= min_roi_target and row['EV_2'] > 0:
                results.append(dict(row, **{'Bet': '2', 'Quota': row['cotad'], 'EV': row['EV_2'], 'ROI_Storico': roi2, 'Casi_Simili': n2}))
            
            progress.progress((i + 1) / len(df_future))
            
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            df_res = df_res.sort_values('ROI_Storico', ascending=False)
            st.success(f"Trovate **{len(df_res)}** Occasioni d'Oro!")
            
            cols_show = ['datamecic', 'txtechipa1', 'txtechipa2', 'Bet', 'Quota', 'EV', 'ROI_Storico', 'Casi_Simili']
            st.dataframe(
                df_res[cols_show].style.format({
                    'Quota': '{:.2f}', 'EV': '{:.1%}', 'ROI_Storico': '{:.1f}%'
                }).background_gradient(subset=['ROI_Storico'], cmap='Greens')
            )
        else:
            st.warning("Nessuna partita futura soddisfa i criteri di ROI storico richiesti.")

with tab3:
    st.header("📊 Report File Storico")
    if df_history is not None:
        df_played = df_history[df_history['res_1x2'] != '-']
        if not df_played.empty:
            pnl_1 = np.where(df_played['EV_1']>0, np.where(df_played['res_1x2']=='1', df_played['cotaa']-1, -1), 0).sum()
            pnl_2 = np.where(df_played['EV_2']>0, np.where(df_played['res_1x2']=='2', df_played['cotad']-1, -1), 0).sum()
            
            k1, k2 = st.columns(2)
            k1.metric("Totale Profitto 1", f"{pnl_1:.2f} u")
            k2.metric("Totale Profitto 2", f"{pnl_2:.2f} u")
            st.dataframe(df_played.head())
        else:
            st.info("Il file storico caricato non ha risultati.")
    else:
        st.info("Carica un file storico a sinistra.")
