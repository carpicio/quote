import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet Standard", page_icon="⚽", layout="wide")
st.title("⚽ Calcolatore Strategico (Standard File)")

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
        # Legge il file cercando di capire il separatore
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1')
            if len(df.columns) < 5: raise ValueError
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')

        # Normalizza i nomi (minuscolo)
        df.columns = df.columns.str.strip().str.lower()
        
        # --- MAPPA DI SICUREZZA ---
        # Se nel file hai lasciato i nomi vecchi, provo a correggerli io al volo
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2', 
            'data': 'datamecic', 'casa': 'txtechipa1', 'ospite': 'txtechipa2'
        }
        df = df.rename(columns=rename_map)

        # Controllo Colonne Fondamentali
        req_cols = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
        # Controlliamo solo quelle per il calcolo EV
        missing_ev = [c for c in ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'] if c not in df.columns]
        
        if missing_ev:
            return None, f"⚠️ Errore File: Mancano le colonne delle quote o ELO: {missing_ev}. Rinonimale in Excel come indicato."

        # Pulizia Numeri
        cols_num = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Calcoli
        calc = df.apply(lambda r: calculate_row(r), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        # Logica Risultati
        df['res_1x2'] = '-'
        
        # Se scor1 e scor2 sono validi, calcola il segno
        mask_valid = df['scor1'].notna() & df['scor2'].notna()
        
        df.loc[mask_valid & (df['scor1'] > df['scor2']), 'res_1x2'] = '1'
        df.loc[mask_valid & (df['scor1'] == df['scor2']), 'res_1x2'] = 'X'
        df.loc[mask_valid & (df['scor1'] < df['scor2']), 'res_1x2'] = '2'
        
        return df, None

    except Exception as e:
        return None, f"Errore Tecnico: {str(e)}"

# --- INTERFACCIA ---
tab1, tab2, tab3 = st.tabs(["🔮 Calcolatore", "📊 Report Storico", "🕵️ Analisi Cluster"])

with tab1:
    st.header("Calcolatore Manuale")
    c1, c2, c3 = st.columns(3)
    c_team1, c_mid, c_team2 = st.columns([2, 1, 2])
    with c_team1: team_h = st.text_input("Squadra Casa", "Home Team")
    with c_mid: st.markdown("<h2 style='text-align: center; margin-top: 10px;'>VS</h2>", unsafe_allow_html=True)
    with c_team2: team_a = st.text_input("Squadra Ospite", "Away Team")

    c1, c2, c3 = st.columns(3)
    elo_h = c1.number_input("ELO Casa", value=1500, min_value=0, step=10)
    elo_a = c3.number_input("ELO Ospite", value=1500, min_value=0, step=10)
    o1 = c1.number_input("Quota 1", value=2.00, min_value=1.01, step=0.01)
    ox = c2.number_input("Quota X", value=3.00, min_value=1.01, step=0.01)
    o2 = c3.number_input("Quota 2", value=2.50, min_value=1.01, step=0.01)
    
    if st.button("Calcola Previsione", type="primary", use_container_width=True):
        row = {'elohomeo': elo_h, 'eloawayo': elo_a, 'cotaa': o1, 'cotae': ox, 'cotad': o2}
        res = calculate_row(row)
        st.divider()
        diff = int(res['ELO_Diff'])
        diff_color = "green" if diff > 0 else "red"
        st.markdown(f"Differenza ELO: **:{diff_color}[{diff}]**")
        k1, k2, k3 = st.columns(3)
        def show_card(col, label, odd, ev, fair):
            bg = "#d4edda" if ev > 0 else "#f8d7da"
            txt = "#155724" if ev > 0 else "#721c24"
            col.markdown(f"""
            <div style="background-color:{bg}; padding:10px; border-radius:10px; text-align:center;">
                <h3 style="color:{txt}; margin:0;">{label}</h3>
                <h1 style="color:black; margin:0;">{odd:.2f}</h1>
                <p style="color:black; font-weight:bold; margin:5px;">Imp: {fair:.2f}</p>
                <p style="color:{txt}; margin:0;">EV: {ev:+.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        show_card(k1, "1", o1, res['EV_1'], res['Fair_1'])
        show_card(k2, "X", ox, res['EV_X'], res['Fair_X'])
        show_card(k3, "2", o2, res['EV_2'], res['Fair_2'])

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV Corretto", type=["csv"])

if uploaded_file:
    df, error_msg = load_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        # Filtra solo le partite che hanno un risultato (non è '-')
        df_played = df[df['res_1x2'] != '-'].copy()
        n_played = len(df_played)

        with tab2:
            st.header("Report Storico")
            if n_played > 0:
                st.success(f"Trovate **{n_played}** partite con risultati validi.")
                
                pnl_1 = np.where(df_played['EV_1']>0, np.where(df_played['res_1x2']=='1', df_played['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_played['EV_2']>0, np.where(df_played['res_1x2']=='2', df_played['cotad']-1, -1), 0).sum()
                
                m1, m2 = st.columns(2)
                m1.metric("Profitto Strategia CASA", f"{pnl_1:.2f} u")
                m2.metric("Profitto Strategia OSPITE", f"{pnl_2:.2f} u")
                
                st.dataframe(df_played.head(20))
            else:
                st.warning("⚠️ File letto, ma **0 partite** risultano giocate.")
                st.write("Verifica che le colonne **scor1** e **scor2** contengano numeri.")
                st.write("Anteprima dati caricati:")
                st.dataframe(df.head())

        with tab3:
            st.header("Analisi Cluster")
            if n_played == 0:
                st.warning("Servono risultati storici per questa analisi.")
            else:
                mode = st.selectbox("Mercato", ["Casa (1)", "Ospite (2)", "Pareggio (X)"])
                c1, c2 = st.columns(2)
                q_min, q_max = c1.slider("Quota", 1.0, 10.0, (1.5, 4.0))
                ev_min, ev_max = c2.slider("EV %", -10.0, 50.0, (0.0, 20.0))
                
                mask = pd.Series(True, index=df_played.index)
                target, col_odd = None, None
                
                if "Casa" in mode: target, col_odd, col_ev = '1', 'cotaa', 'EV_1'
                elif "Ospite" in mode: target, col_odd, col_ev = '2', 'cotad', 'EV_2'
                else: target, col_odd, col_ev = 'X', 'cotae', 'EV_X'
                
                mask &= (df_played[col_odd] >= q_min) & (df_played[col_odd] <= q_max)
                mask &= (df_played[col_ev]*100 >= ev_min) & (df_played[col_ev]*100 <= ev_max)
                
                df_filt = df_played[mask].copy()
                
                if not df_filt.empty:
                    wins = len(df_filt[df_filt['res_1x2'] == target])
                    profit = (df_filt[df_filt['res_1x2'] == target][col_odd] - 1).sum() - (len(df_filt) - wins)
                    roi = (profit/len(df_filt))*100
                    
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Bets", len(df_filt))
                    k2.metric("Profitto", f"{profit:.2f} u")
                    k3.metric("ROI", f"{roi:.2f}%")
                    st.dataframe(df_filt)
                else:
                    st.warning("Nessuna partita trovata.")
