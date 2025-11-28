import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Value Bet Calculator", layout="wide")

st.title("⚽ Calcolatore Quote Implicite & Value Bet")
st.markdown("""
Questa applicazione confronta le **Quote Reali** del bookmaker con le **Quote Implicite** derivate dal rating ELO.
Usa il menu laterale o le schede qui sotto per passare dal calcolatore singolo all'analisi storica.
""")

# --- FUNZIONI DI CALCOLO ---
def get_implicit_probs(elo_home, elo_away, hfa=100):
    """Calcola le probabilità implicite basate su ELO."""
    exponent = (elo_away - (elo_home + hfa)) / 400
    p_elo_h = 1 / (1 + 10**exponent)
    p_elo_a = 1 - p_elo_h
    return p_elo_h, p_elo_a

def remove_margin(odd_1, odd_x, odd_2):
    """Rimuove l'aggio del bookmaker."""
    if odd_1 == 0 or odd_x == 0 or odd_2 == 0:
        return 0, 0, 0
    inv_sum = (1/odd_1) + (1/odd_x) + (1/odd_2)
    p_fair_1 = (1/odd_1) / inv_sum
    p_fair_x = (1/odd_x) / inv_sum
    p_fair_2 = (1/odd_2) / inv_sum
    return p_fair_1, p_fair_x, p_fair_2

def calculate_value(row, hfa=100):
    """Calcola EV e quote implicite per una riga."""
    elo_h = row.get('elohomeo', 1500)
    elo_a = row.get('eloawayo', 1500)
    o1 = row.get('cotaa', 0)
    ox = row.get('cotae', 0)
    o2 = row.get('cotad', 0)
    
    # Se mancano le quote o sono zero, restituisce valori nulli
    if pd.isna(o1) or pd.isna(ox) or pd.isna(o2) or o1==0:
        return pd.Series({
            'Prob_Impl_1': 0, 'Prob_Impl_X': 0, 'Prob_Impl_2': 0,
            'Quota_Impl_1': 0, 'Quota_Impl_X': 0, 'Quota_Impl_2': 0,
            'EV_1': -1, 'EV_X': -1, 'EV_2': -1
        })

    # Step 1: Probabilità di mercato (per la X)
    pf_1, pf_x, pf_2 = remove_margin(o1, ox, o2)
    
    # Step 2: Probabilità ELO (per 1 e 2)
    p_elo_h_binary, p_elo_a_binary = get_implicit_probs(elo_h, elo_a, hfa)
    
    # Step 3: Mix (Usiamo la X del mercato, ridistribuiamo il resto su 1 e 2 in base all'ELO)
    remaining_prob = 1 - pf_x
    
    prob_final_1 = remaining_prob * p_elo_h_binary
    prob_final_2 = remaining_prob * p_elo_a_binary
    prob_final_x = pf_x
    
    # Quote Implicite
    imp_odd_1 = 1 / prob_final_1 if prob_final_1 > 0 else 0
    imp_odd_x = 1 / prob_final_x if prob_final_x > 0 else 0
    imp_odd_2 = 1 / prob_final_2 if prob_final_2 > 0 else 0
    
    # Calcolo EV
    ev_1 = (o1 * prob_final_1) - 1
    ev_x = (ox * prob_final_x) - 1
    ev_2 = (o2 * prob_final_2) - 1
    
    return pd.Series({
        'Prob_Impl_1': prob_final_1, 'Prob_Impl_X': prob_final_x, 'Prob_Impl_2': prob_final_2,
        'Quota_Impl_1': imp_odd_1, 'Quota_Impl_X': imp_odd_x, 'Quota_Impl_2': imp_odd_2,
        'EV_1': ev_1, 'EV_X': ev_x, 'EV_2': ev_2
    })

# --- INTERFACCIA ---
tab1, tab2 = st.tabs(["🔮 Calcolatore Partita Futura", "📊 Analisi CSV (Storico/Futuro)"])

# TAB 1: CALCOLATORE MANUALE
with tab1:
    st.header("Inserisci i dati della partita")
    col1, col2, col3 = st.columns(3)
    with col1:
        team_home = st.text_input("Squadra Casa", "Home Team")
        elo_h_input = st.number_input("ELO Casa", value=1500, step=10)
        odd_1_input = st.number_input("Quota 1 (Casa)", value=2.00, step=0.01)
    
    with col2:
        st.write("VS")
        hfa_input = st.number_input("Fattore Campo (HFA)", value=100, step=10)
        odd_x_input = st.number_input("Quota X (Pareggio)", value=3.00, step=0.01)
        
    with col3:
        team_away = st.text_input("Squadra Ospite", "Away Team")
        elo_a_input = st.number_input("ELO Ospite", value=1500, step=10)
        odd_2_input = st.number_input("Quota 2 (Ospite)", value=3.50, step=0.01)
        
    if st.button("Calcola Valore", type="primary"):
        dummy_row = {
            'elohomeo': elo_h_input, 'eloawayo': elo_a_input,
            'cotaa': odd_1_input, 'cotae': odd_x_input, 'cotad': odd_2_input
        }
        res = calculate_value(dummy_row, hfa=hfa_input)
        
        st.divider()
        st.subheader(f"Risultati: {team_home} vs {team_away}")
        c1, c2, c3 = st.columns(3)
        
        def show_metric(col, label, real_odd, impl_odd, ev):
            delta_color = "normal" if ev <= 0 else "inverse"
            col.metric(label=f"Segno {label}", value=f"{real_odd:.2f}", delta=f"EV: {ev:.1%}", delta_color=delta_color)
            col.markdown(f"Fair Odd: **{impl_odd:.2f}**")
            if ev > 0: col.success("✅ VALORE!")
            else: col.error("❌ No Valore")

        show_metric(c1, "1", odd_1_input, res['Quota_Impl_1'], res['EV_1'])
        show_metric(c2, "X", odd_x_input, res['Quota_Impl_X'], res['EV_X'])
        show_metric(c3, "2", odd_2_input, res['Quota_Impl_2'], res['EV_2'])

# TAB 2: ANALISI FILE CSV
with tab2:
    st.header("Analisi da File")
    uploaded_file = st.file_uploader("Carica il file CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
            
            # Pulizia colonne
            cols_to_convert = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
            for col in cols_to_convert:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
            
            # Calcoli
            st.write("⏳ Elaborazione in corso...")
            calc_results = df.apply(lambda row: calculate_value(row), axis=1)
            df = pd.concat([df, calc_results], axis=1)

            # --- GESTIONE RISULTATI (Versione Sicura) ---
            has_results = 'scor1' in df.columns and 'scor2' in df.columns and df['scor1'].notna().any()

            if has_results:
                # Funzione sicura per determinare il risultato riga per riga
                def get_result(row):
                    try:
                        s1, s2 = row['scor1'], row['scor2']
                        if pd.isna(s1) or pd.isna(s2): return np.nan
                        if s1 > s2: return '1'
                        elif s1 == s2: return 'X'
                        else: return '2'
                    except:
                        return np.nan

                df['result_1x2'] = df.apply(get_result, axis=1)
                
                # Calcolo PnL solo dove c'è un risultato valido
                df_valid = df.dropna(subset=['result_1x2'])
                
                if not df_valid.empty:
                    st.subheader("Risultati Simulazione Storica")
                    
                    df_valid['won_1'] = (df_valid['result_1x2'] == '1').astype(int)
                    df_valid['won_2'] = (df_valid['result_1x2'] == '2').astype(int)
                    
                    df_valid['pnl_1'] = np.where((df_valid['EV_1'] > 0), np.where(df_valid['won_1']==1, df_valid['cotaa']-1, -1), 0)
                    df_valid['pnl_2'] = np.where((df_valid['EV_2'] > 0), np.where(df_valid['won_2']==1, df_valid['cotad']-1, -1), 0)
                    
                    bets_1 = (df_valid['EV_1'] > 0).sum()
                    profit_1 = df_valid['pnl_1'].sum()
                    roi_1 = (profit_1 / bets_1 * 100) if bets_1 > 0 else 0
                    
                    bets_2 = (df_valid['EV_2'] > 0).sum()
                    profit_2 = df_valid['pnl_2'].sum()
                    roi_2 = (profit_2 / bets_2 * 100) if bets_2 > 0 else 0
                    
                    c1, c2 = st.columns(2)
                    c1.info("Strategia CASA (1)")
                    c1.metric("Scommesse", int(bets_1))
                    c1.metric("Profitto", f"{profit_1:.2f} u")
                    c1.metric("ROI", f"{roi_1:.2f}%")
                    
                    c2.info("Strategia TRASFERTA (2)")
                    c2.metric("Scommesse", int(bets_2))
                    c2.metric("Profitto", f"{profit_2:.2f} u")
                    c2.metric("ROI", f"{roi_2:.2f}%")
                else:
                    st.warning("⚠️ Ci sono le colonne dei risultati, ma sembrano vuote o invalide.")
            else:
                st.info("ℹ️ File senza risultati. Mostro solo le previsioni (Valore/EV).")

            st.divider()
            st.write("### Tabella Dati")
            cols = ['txtechipa1', 'txtechipa2', 'cotaa', 'cotad', 'Quota_Impl_1', 'Quota_Impl_2', 'EV_1', 'EV_2']
            if 'datamecic' in df.columns: cols.insert(0, 'datamecic')
            if 'result_1x2' in df.columns: cols.append('result_1x2')
            
            st.dataframe(df[cols].head(100))
            
            csv = df.to_csv(sep=';', decimal=',', index=False).encode('utf-8')
            st.download_button("Scarica CSV Completo", data=csv, file_name="analisi_quote.csv", mime="text/csv")
            
        except Exception as e:
            st.error(f"Errore tecnico: {e}")