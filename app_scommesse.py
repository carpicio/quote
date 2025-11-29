import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet Calculator Pro", layout="wide")
st.title("⚽ Calcolatore Strategico: 1X2 & Goals")

# --- FUNZIONI ---
def get_implicit_probs(elo_home, elo_away, hfa=100):
    exponent = (elo_away - (elo_home + hfa)) / 400
    p_elo_h = 1 / (1 + 10**exponent)
    p_elo_a = 1 - p_elo_h
    return p_elo_h, p_elo_a

def remove_margin(odd_1, odd_x, odd_2):
    if odd_1 == 0 or odd_x == 0 or odd_2 == 0: return 0, 0, 0
    inv_sum = (1/odd_1) + (1/odd_x) + (1/odd_2)
    return (1/odd_1)/inv_sum, (1/odd_x)/inv_sum, (1/odd_2)/inv_sum

def calculate_row(row, hfa=100):
    # Dati base
    elo_h = row.get('elohomeo', 1500)
    elo_a = row.get('eloawayo', 1500)
    o1, ox, o2 = row.get('cotaa', 0), row.get('cotae', 0), row.get('cotad', 0)
    
    # 1. Calcoli 1X2
    res_1x2 = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'Fair_1': 0, 'Fair_X': 0, 'Fair_2': 0}
    if not (pd.isna(o1) or pd.isna(ox) or pd.isna(o2) or o1==0):
        pf_1, pf_x, pf_2 = remove_margin(o1, ox, o2)
        p_elo_h, p_elo_a = get_implicit_probs(elo_h, elo_a, hfa)
        
        # Mix ELO + Market Draw
        rem = 1 - pf_x
        p_fin_1 = rem * p_elo_h
        p_fin_2 = rem * p_elo_a
        
        res_1x2['Fair_1'] = 1/p_fin_1 if p_fin_1>0 else 0
        res_1x2['Fair_X'] = 1/pf_x if pf_x>0 else 0
        res_1x2['Fair_2'] = 1/p_fin_2 if p_fin_2>0 else 0
        res_1x2['EV_1'] = (o1 * p_fin_1) - 1
        res_1x2['EV_X'] = (ox * pf_x) - 1
        res_1x2['EV_2'] = (o2 * p_fin_2) - 1

    # 2. Dati Extra per Analisi Goal
    elo_diff = abs((elo_h + hfa) - elo_a)
    
    return pd.Series({**res_1x2, 'ELO_Diff': elo_diff})

# --- CARICAMENTO DATI (MODERNO) ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=';', encoding='latin1')
    # Pulizia colonne numeriche
    cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
    for c in cols_num:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
    
    # Applica calcoli
    calc = df.apply(lambda r: calculate_row(r), axis=1)
    df = pd.concat([df, calc], axis=1)
    
    # Determina Risultati
    if 'scor1' in df.columns:
        # 1X2
        conditions = [df['scor1'] > df['scor2'], df['scor1'] == df['scor2'], df['scor1'] < df['scor2']]
        df['res_1x2'] = np.select(conditions, ['1', 'X', '2'], default=np.nan)
        
        # Goal Totali (FT)
        df['goals_ft'] = df['scor1'] + df['scor2']
        if 'cotao' in df.columns: # Over 2.5
            df['res_o25'] = (df['goals_ft'] > 2.5).astype(int)
            df['res_u25'] = (df['goals_ft'] < 2.5).astype(int)
            
    return df

# --- INTERFACCIA ---
tab1, tab2, tab3 = st.tabs(["🔮 Calcolatore", "📊 Report Generale", "🕵️ Analisi Avanzata"])

with tab1:
    st.header("Calcolatore 1X2")
    c1, c2, c3 = st.columns(3)
    elo_h = c1.number_input("ELO Casa", 1500)
    elo_a = c3.number_input("ELO Ospite", 1500)
    o1 = c1.number_input("Quota 1", 2.0)
    ox = c2.number_input("Quota X", 3.0)
    o2 = c3.number_input("Quota 2", 3.5)
    
    if st.button("Calcola"):
        row = {'elohomeo': elo_h, 'eloawayo': elo_a, 'cotaa': o1, 'cotae': ox, 'cotad': o2}
        res = calculate_row(row)
        
        st.write(f"**Differenza ELO:** {int(res['ELO_Diff'])}")
        k1, k2, k3 = st.columns(3)
        
        def show(col, lbl, odd, ev):
            color = "inverse" if ev > 0 else "normal"
            col.metric(lbl, f"{odd}", f"EV: {ev:.1%}", delta_color=color)
            
        show(k1, "1", o1, res['EV_1'])
        show(k2, "X", ox, res['EV_X'])
        show(k3, "2", o2, res['EV_2'])

uploaded_file = st.sidebar.file_uploader("Carica CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    
    with tab2:
        st.header("Profitti Generali (Strategia Base)")
        if 'res_1x2' in df.columns:
            # Calcolo PnL solo dove c'è EV positivo
            df['pnl_1'] = np.where(df['EV_1']>0, np.where(df['res_1x2']=='1', df['cotaa']-1, -1), 0)
            df['pnl_2'] = np.where(df['EV_2']>0, np.where(df['res_1x2']=='2', df['cotad']-1, -1), 0)
            
            m1, m2 = st.columns(2)
            m1.metric("Totale Strategia CASA", f"{df['pnl_1'].sum():.2f} u")
            m2.metric("Totale Strategia OSPITE", f"{df['pnl_2'].sum():.2f} u")
        st.dataframe(df.head(10))

    with tab3:
        st.header("Analisi Cluster")
        mode = st.selectbox("Mercato", ["Casa (1)", "Ospite (2)", "Pareggio (X)", "Over 2.5", "Under 2.5"])
        
        # SLIDERS DINAMICI
        c1, c2 = st.columns(2)
        q_min, q_max = c1.slider("Range Quota", 1.0, 10.0, (1.5, 4.0))
        
        if "Over" in mode or "Under" in mode:
            # Per i Goal usiamo ELO DIFF invece di EV
            elo_min, elo_max = c2.slider("Differenza ELO (Equilibrio)", 0, 500, (0, 500))
            use_ev = False
        else:
            ev_min, ev_max = c2.slider("Range EV %", -10.0, 100.0, (0.0, 50.0))
            use_ev = True

        if 'res_1x2' in df.columns:
            # Logica Filtri
            mask = pd.Series(True, index=df.index)
            
            if mode == "Casa (1)":
                col_odd, col_res, target = 'cotaa', 'res_1x2', '1'
                mask &= (df['EV_1']*100 >= ev_min) & (df['EV_1']*100 <= ev_max)
            elif mode == "Ospite (2)":
                col_odd, col_res, target = 'cotad', 'res_1x2', '2'
                mask &= (df['EV_2']*100 >= ev_min) & (df['EV_2']*100 <= ev_max)
            elif mode == "Pareggio (X)":
                col_odd, col_res, target = 'cotae', 'res_1x2', 'X'
                mask &= (df['EV_X']*100 >= ev_min) & (df['EV_X']*100 <= ev_max)
            elif mode == "Over 2.5":
                col_odd, col_res, target = 'cotao', 'res_o25', 1
                mask &= (df['ELO_Diff'] >= elo_min) & (df['ELO_Diff'] <= elo_max)
            elif mode == "Under 2.5":
                col_odd, col_res, target = 'cotau', 'res_u25', 1
                mask &= (df['ELO_Diff'] >= elo_min) & (df['ELO_Diff'] <= elo_max)
            
            mask &= (df[col_odd] >= q_min) & (df[col_odd] <= q_max)
            
            # Calcolo Risultati
            df_filt = df[mask].copy()
            bets = len(df_filt)
            
            if bets > 0:
                wins = len(df_filt[df_filt[col_res] == target])
                profit = (df_filt[df_filt[col_res] == target][col_odd] - 1).sum() - (bets - wins)
                roi = (profit/bets)*100
                
                st.divider()
                k1, k2, k3 = st.columns(3)
                k1.metric("Bets", bets)
                k2.metric("Profitto", f"{profit:.2f} u")
                k3.metric("ROI", f"{roi:.2f}%", delta_color="normal" if roi>0 else "inverse")
                
                # Tabella Dati
                cols_view = ['datamecic', 'txtechipa1', 'txtechipa2', col_odd, 'ELO_Diff']
                if use_ev: cols_view.append('EV_1' if '1' in mode else 'EV_2' if '2' in mode else 'EV_X')
                cols_view.append(col_res)
                
                st.dataframe(df_filt[cols_view].style.format("{:.2f}", subset=[col_odd]))
            else:
                st.warning("Nessuna partita trovata con questi filtri.")
