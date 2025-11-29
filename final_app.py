import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet Pro", layout="wide")
st.title("⚽ Calcolatore Strategico: 1X2 & Goals")

# --- FUNZIONI DI CALCOLO ---
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
    elo_h = row.get('elohomeo', 1500)
    elo_a = row.get('eloawayo', 1500)
    o1 = row.get('cotaa', 0)
    ox = row.get('cotae', 0)
    o2 = row.get('cotad', 0)
    
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'Fair_1': 0, 'Fair_X': 0, 'Fair_2': 0, 'ELO_Diff': 0}
    
    if pd.notna(o1) and pd.notna(ox) and pd.notna(o2) and o1 > 0:
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
        
    res['ELO_Diff'] = abs((elo_h + hfa) - elo_a)
    return pd.Series(res)

# --- CARICAMENTO DATI ---
@st.cache_data(ttl=0)
def load_data(file):
    try:
        df = pd.read_csv(file, sep=';', encoding='latin1')
        if len(df.columns) < 5:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')
            
        df.columns = df.columns.str.strip().str.lower()
        df = df.loc[:, ~df.columns.duplicated()] 
        
        if 'res_1x2' not in df.columns: df['res_1x2'] = np.nan
        if 'res_o25' not in df.columns: df['res_o25'] = np.nan
        
        req_cols = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            return None, f"⚠️ Errore File: Mancano le colonne: {', '.join(missing)}"

        cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].apply(lambda x: str(x).replace(',', '.') if pd.notna(x) else x)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        calc = df.apply(lambda r: calculate_row(r), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        if 'scor1' in df.columns and 'scor2' in df.columns:
            df['goals_ft'] = df['scor1'] + df['scor2']
            conditions = [df['scor1'] > df['scor2'], df['scor1'] == df['scor2'], df['scor1'] < df['scor2']]
            df['res_1x2'] = np.select(conditions, ['1', 'X', '2'], default=np.nan)
            
            if 'cotao' in df.columns:
                df['res_o25'] = (df['goals_ft'] > 2.5).astype(int)
                df['res_u25'] = (df['goals_ft'] < 2.5).astype(int)
            
        return df, None
        
    except Exception as e:
        return None, f"Errore tecnico: {str(e)}"

# --- INTERFACCIA ---
tab1, tab2, tab3 = st.tabs(["🔮 Calcolatore", "📊 Report Generale", "🕵️ Analisi Avanzata"])

with tab1:
    st.header("Calcolatore 1X2")
    c1, c2, c3 = st.columns(3)
    elo_h = c1.number_input("ELO Casa", 1500, key="t1_eh")
    elo_a = c3.number_input("ELO Ospite", 1500, key="t1_ea")
    o1 = c1.number_input("Quota 1", 2.0, key="t1_o1")
    ox = c2.number_input("Quota X", 3.0, key="t1_ox")
    o2 = c3.number_input("Quota 2", 3.5, key="t1_o2")
    
    if st.button("Calcola", key="btn_calc"):
        row = {'elohomeo': elo_h, 'eloawayo': elo_a, 'cotaa': o1, 'cotae': ox, 'cotad': o2}
        res = calculate_row(row)
        st.write(f"**Differenza ELO:** {int(res['ELO_Diff'])}")
        k1, k2, k3 = st.columns(3)
        def show(col, lbl, odd, ev, k_suffix):
            color = "inverse" if ev > 0 else "normal"
            col.metric(lbl, f"{odd}", f"EV: {ev:.1%}", delta_color=color, key=f"res_metric_{k_suffix}")
        show(k1, "1", o1, res['EV_1'], "1")
        show(k2, "X", ox, res['EV_X'], "X")
        show(k3, "2", o2, res['EV_2'], "2")

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV (CGMBet)", type=["csv"], key="file_upl_v17")

if uploaded_file:
    df, error_msg = load_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        with tab2:
            st.header("Report Generale")
            if 'res_1x2' in df.columns and df['res_1x2'].notna().any():
                df_pnl = df.dropna(subset=['res_1x2'])
                pnl_1 = np.where(df_pnl['EV_1']>0, np.where(df_pnl['res_1x2']=='1', df_pnl['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_pnl['EV_2']>0, np.where(df_pnl['res_1x2']=='2', df_pnl['cotad']-1, -1), 0).sum()
                
                m1, m2 = st.columns(2)
                m1.metric("Totale Strategia CASA", f"{pnl_1:.2f} u", key="pnl_home_gen_v17")
                m2.metric("Totale Strategia OSPITE", f"{pnl_2:.2f} u", key="pnl_away_gen_v17")
            else:
                st.info("ℹ️ File senza risultati storici validi.")
            st.dataframe(df.head(10))

        with tab3:
            st.header("Analisi Cluster")
            # Inizializzo le variabili QUI all'inizio per evitare UnboundLocalError
            profit = 0.0
            roi = 0.0
            bets = 0
            
            if 'res_1x2' not in df.columns or df['res_1x2'].isna().all():
                st.warning("⚠️ Servono risultati storici per questa analisi.")
            else:
                mode = st.selectbox("Mercato", ["Casa (1)", "Ospite (2)", "Pare
