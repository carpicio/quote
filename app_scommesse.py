import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet Calculator Pro", layout="wide")
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
    # Dati base con valori di default sicuri
    elo_h = row.get('elohomeo', 1500)
    elo_a = row.get('eloawayo', 1500)
    o1 = row.get('cotaa', 0)
    ox = row.get('cotae', 0)
    o2 = row.get('cotad', 0)
    
    # Inizializza risultati
    res = {'EV_1': -1, 'EV_X': -1, 'EV_2': -1, 'Fair_1': 0, 'Fair_X': 0, 'Fair_2': 0, 'ELO_Diff': 0}
    
    # Calcolo solo se i dati sono validi
    if pd.notna(o1) and pd.notna(ox) and pd.notna(o2) and o1 > 0:
        pf_1, pf_x, pf_2 = remove_margin(o1, ox, o2)
        p_elo_h, p_elo_a = get_implicit_probs(elo_h, elo_a, hfa)
        
        # Mix ELO + Market Draw
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

# --- CARICAMENTO DATI (BLINDATO) ---
@st.cache_data
def load_data(file):
    try:
        # Prova a leggere col separatore punto e virgola
        df = pd.read_csv(file, sep=';', encoding='latin1')
        
        # Se ha letto tutto in una colonna sola, riprova con la virgola
        if len(df.columns) < 5:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')
            
        # Normalizza nomi colonne (toglie spazi, minuscolo)
        df.columns = df.columns.str.strip().str.lower()
        
        # Verifica colonne essenziali
        req_cols = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo']
        missing = [c for c in req_cols if c not in df.columns]
        
        if missing:
            return None, f"⚠️ Errore nel file: mancano queste colonne obbligatorie: {', '.join(missing)}"
            
        # Pulizia numeri (virgola -> punto)
        cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Rimuove righe senza quote essenziali
        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        if df.empty:
            return None, "⚠️ Il file è vuoto o tutte le righe hanno dati mancanti."

        # Applica calcoli
        calc = df.apply(lambda r: calculate_row(r), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        # Determina Risultati (Solo se esistono le colonne punteggio)
        if 'scor1' in df.columns and 'scor2' in df.columns:
            conditions = [df['scor1'] > df['scor2'], df['scor1'] == df['scor2'], df['scor1'] < df['scor2']]
            df['res_1x2'] = np.select(conditions, ['1', 'X', '2'], default=np.nan)
            
            # Goal Totali
            df['goals_ft'] = df['scor1'] + df['scor2']
            if 'cotao' in df.columns:
                df['res_o25'] = (df['goals_ft'] > 2.5).astype(int)
                df['res_u25'] = (df['goals_ft'] < 2.5).astype(int)
                
        return df, None
        
    except Exception as e:
        return None, f"Errore tecnico durante la lettura del file: {str(e)}"

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

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV (CGMBet)", type=["csv"])

if uploaded_file:
    df, error_msg = load_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        # Se arriviamo qui, il file è valido!
        with tab2:
            st.header("Report Generale")
            
            # Mostra metriche solo se ci sono risultati
            if 'res_1x2' in df.columns and df['res_1x2'].notna().any():
                df_pnl = df.dropna(subset=['res_1x2'])
                pnl_1 = np.where(df_pnl['EV_1']>0, np.where(df_pnl['res_1x2']=='1', df_pnl['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_pnl['EV_2']>0, np.where(df_pnl['res_1x2']=='2', df_pnl['cotad']-1, -1), 0).sum()
                
                m1, m2 = st.columns(2)
                m1.metric("Totale Strategia CASA", f"{pnl_1:.2f} u")
                m2.metric("Totale Strategia OSPITE", f"{pnl_2:.2f} u")
            else:
                st.info("ℹ️ Il file caricato non contiene risultati (o colonne scor1/scor2 mancanti). Verranno mostrate solo le previsioni.")

            st.dataframe(df.head(10))

        with tab3:
            st.header("Analisi Cluster")
            if 'res_1x2' not in df.columns:
                st.warning("⚠️ Per fare l'analisi dei cluster servono i risultati storici (colonne scor1, scor2).")
            else:
                mode = st.selectbox("Mercato", ["Casa (1)", "Ospite (2)", "Pareggio (X)", "Over 2.5", "Under 2.5"])
                
                c1, c2 = st.columns(2)
                q_min, q_max = c1.slider("Range Quota", 1.0, 10.0, (1.5, 4.0))
                
                use_ev = True
                if "Over" in mode or "Under" in mode:
                    elo_min, elo_max = c2.slider("Differenza ELO", 0, 500, (0, 500))
                    use_ev = False
                else:
                    ev_min, ev_max = c2.slider("Range EV %", -10.0, 100.0, (0.0, 50.0))

                # Logica Filtri Sicura
                mask = pd.Series(True, index=df.index)
                target, col_odd, col_res = None, None, None

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
                    if 'cotao' in df.columns:
                        col_odd, col_res, target = 'cotao', 'res_o25', 1
                        mask &= (df['ELO_Diff'] >= elo_min) & (df['ELO_Diff'] <= elo_max)
                    else:
                        st.error("Il file non contiene la colonna 'cotao' per Over 2.5")
                elif mode == "Under 2.5":
                    if 'cotau' in df.columns:
                        col_odd, col_res, target = 'cotau', 'res_u25', 1
                        mask &= (df['ELO_Diff'] >= elo_min) & (df['ELO_Diff'] <= elo_max)
                    else:
                        st.error("Il file non contiene la colonna 'cotau' per Under 2.5")
                
                if col_odd and col_odd in df.columns:
                    mask &= (df[col_odd] >= q_min) & (df[col_odd] <= q_max)
                    df_filt = df[mask].copy()
                    
                    if len(df_filt) > 0:
                        wins = len(df_filt[df_filt[col_res] == target])
                        profit = (df_filt[df_filt[col_res] == target][col_odd] - 1).sum() - (len(df_filt) - wins)
                        roi = (profit/len(df_filt))*100
                        
                        st.divider()
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Bets", len(df_filt))
                        k2.metric("Profitto", f"{profit:.2f} u")
                        k3.metric("ROI", f"{roi:.2f}%", delta_color="normal" if roi>0 else "inverse")
                        
                        cols_view = ['datamecic', 'txtechipa1', 'txtechipa2', col_odd, 'ELO_Diff']
                        cols_view = [c for c in cols_view if c in df_filt.columns] # Sicurezza
                        st.dataframe(df_filt[cols_view])
                    else:
                        st.warning("Nessuna partita trovata.")
