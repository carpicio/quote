import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet v33", page_icon="⚽", layout="wide")
st.title("⚽ Calcolatore Strategico (v33 - Contrast Fix)")

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
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1')
            if len(df.columns) < 5: raise ValueError
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')

        df.columns = df.columns.str.strip().str.lower()
        
        # Mappa Completa per riconoscere i tuoi file
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2', # IMPORTANTISSIMO
            'o2,5': 'cotao', 'u2,5': 'cotau',
            'data': 'datamecic', 'casa': 'txtechipa1', 'ospite': 'txtechipa2'
        }
        df = df.rename(columns=rename_map)
        
        req_cols = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo']
        missing = [c for c in req_cols if c not in df.columns]
        if missing: return None, f"⚠️ Errore Colonne: {missing}"

        # Pulizia Numeri (inclusi i risultati scor1/scor2)
        cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        calc = df.apply(lambda r: calculate_row(r), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        # Gestione Risultati
        df['res_1x2'] = '-' 
        
        # Se le colonne scor1/scor2 esistono e hanno numeri
        if 'scor1' in df.columns and 'scor2' in df.columns:
            mask_played = df['scor1'].notna() & df['scor2'].notna()
            
            df.loc[mask_played & (df['scor1'] > df['scor2']), 'res_1x2'] = '1'
            df.loc[mask_played & (df['scor1'] == df['scor2']), 'res_1x2'] = 'X'
            df.loc[mask_played & (df['scor1'] < df['scor2']), 'res_1x2'] = '2'
            
            df['goals_ft'] = df['scor1'] + df['scor2']
            if 'cotao' in df.columns:
                df['res_o25'] = np.nan
                df.loc[mask_played, 'res_o25'] = (df.loc[mask_played, 'goals_ft'] > 2.5).astype(int)
            
        return df, None

    except Exception as e:
        return None, f"Errore Tecnico: {str(e)}"

# --- INTERFACCIA ---
tab1, tab2, tab3 = st.tabs(["🔮 Calcolatore", "📊 Report Storico", "🕵️ Analisi Cluster"])

with tab1:
    st.header("Calcolatore Manuale")
    
    # Input Nomi
    c_name1, c_vs, c_name2 = st.columns([3, 1, 3])
    with c_name1: team_h = st.text_input("Squadra Casa", "Home Team")
    with c_vs: st.markdown("<h3 style='text-align: center; margin-top: 20px;'>VS</h3>", unsafe_allow_html=True)
    with c_name2: team_a = st.text_input("Squadra Ospite", "Away Team")

    # Input Dati
    c1, c2, c3 = st.columns(3)
    elo_h = c1.number_input("ELO Casa", value=1500, min_value=0, step=10)
    o1 = c1.number_input("Quota 1", value=2.00, min_value=1.01, step=0.01)
    
    with c2:
        st.write("")
        st.write("") 
        ox = st.number_input("Quota X", value=3.00, min_value=1.01, step=0.01)
    
    elo_a = c3.number_input("ELO Ospite", value=1500, min_value=0, step=10)
    o2 = c3.number_input("Quota 2", value=2.50, min_value=1.01, step=0.01)
    
    if st.button("Calcola Previsione", type="primary", use_container_width=True):
        row = {'elohomeo': elo_h, 'eloawayo': elo_a, 'cotaa': o1, 'cotae': ox, 'cotad': o2}
        res = calculate_row(row)
        
        st.divider()
        st.subheader(f"Risultati: {team_h} vs {team_a}")
        
        diff = int(res['ELO_Diff'])
        diff_color = "green" if diff > 0 else "red"
        st.markdown(f"Differenza ELO (con fattore campo): **:{diff_color}[{diff}]**")
        
        k1, k2, k3 = st.columns(3)
        
        # --- FIX VISIVO: COLORE NERO PER IMPLICITA ---
        def show_card(col, label, odd, ev, fair):
            bg_color = "#d4edda" if ev > 0 else "#f8d7da"
            text_color = "#155724" if ev > 0 else "#721c24"
            border_color = "green" if ev > 0 else "red"
            
            with col:
                st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 5px solid {border_color};
                    text-align: center;
                    margin-bottom: 10px;
                ">
                    <h3 style="margin:0; color: {text_color};">Segno {label}</h3>
                    <h1 style="margin:0; font-size: 40px; color: black;">{odd:.2f}</h1>
                    <p style="margin:5px 0; font-weight: bold; color: black; background-color: rgba(255,255,255,0.5); border-radius: 5px;">Imp: {fair:.2f}</p>
                    <p style="margin:0; font-size: 18px; color: {text_color}; font-weight: bold;">EV: {ev:+.1%}</p>
                </div>
                """, unsafe_allow_html=True)

        show_card(k1, "1", o1, res['EV_1'], res['Fair_1'])
        show_card(k2, "X", ox, res['EV_X'], res['Fair_X'])
        show_card(k3, "2", o2, res['EV_2'], res['Fair_2'])

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV", type=["csv"])

if uploaded_file:
    df, error_msg = load_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        # Conta partite con risultato valido
        df_played = df[df['res_1x2'] != '-'].copy()
        n_played = len(df_played)

        with tab2:
            st.header("Report Storico")
            if n_played > 0:
                st.success(f"Analisi su **{n_played}** partite giocate.")
                
                # Calcolo PnL
                pnl_1 = np.where(df_played['EV_1']>0, np.where(df_played['res_1x2']=='1', df_played['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_played['EV_2']>0, np.where(df_played['res_1x2']=='2', df_played['cotad']-1, -1), 0).sum()
                
                m1, m2 = st.columns(2)
                m1.metric("Profitto Strategia CASA (1)", f"{pnl_1:.2f} u")
                m2.metric("Profitto Strategia OSPITE (2)", f"{pnl_2:.2f} u")
                
                st.write("### Dettaglio Ultime Partite")
                # Mostra solo colonne esistenti
                cols_view = ['datamecic', 'txtechipa1', 'txtechipa2', 'res_1x2', 'cotaa', 'cotad', 'EV_1', 'EV_2']
                cols_final = [c for c in cols_view if c in df_played.columns]
                st.dataframe(df_played[cols_final].head(20).style.format("{:.2f}", subset=['cotaa', 'cotad']))
            else:
                st.info("ℹ️ Nessun risultato trovato (le colonne GF/Scor sono vuote o mancano).")
                st.write("Ecco le quote caricate:")
                st.dataframe(df.head())

        with tab3:
            st.header("Analisi Cluster (Backtest)")
            
            if n_played == 0:
                st.warning("⚠️ Servono risultati storici per fare l'analisi dei cluster.")
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

                mask = pd.Series(True, index=df_played.index)
                target, col_odd, col_res = None, None, None

                if mode == "Casa (1)":
                    col_odd, col_res, target = 'cotaa', 'res_1x2', '1'
                    mask &= (df_played['EV_1']*100 >= ev_min) & (df_played['EV_1']*100 <= ev_max)
                elif mode == "Ospite (2)":
                    col_odd, col_res, target = 'cotad', 'res_1x2', '2'
                    mask &= (df_played['EV_2']*100 >= ev_min) & (df_played['EV_2']*100 <= ev_max)
                elif mode == "Pareggio (X)":
                    col_odd, col_res, target = 'cotae', 'res_1x2', 'X'
                    mask &= (df_played['EV_X']*100 >= ev_min) & (df_played['EV_X']*100 <= ev_max)
                elif mode == "Over 2.5":
                    if 'cotao' in df_played.columns:
                        col_odd, col_res, target = 'cotao', 'res_o25', 1
                        mask &= (df_played['ELO_Diff'] >= elo_min) & (df_played['ELO_Diff'] <= elo_max)
                    else: st.error("Manca quota Over (O2,5)")
                elif mode == "Under 2.5":
                    if 'cotau' in df_played.columns:
                        col_odd, col_res, target = 'cotau', 'res_u25', 1
                        mask &= (df_played['ELO_Diff'] >= elo_min) & (df_played['ELO_Diff'] <= elo_max)
                    else: st.error("Manca quota Under (U2,5)")
                
                if col_odd and col_odd in df_played.columns:
                    mask &= (df_played[col_odd] >= q_min) & (df_played[col_odd] <= q_max)
                    df_filt = df_played[mask].copy()
                    
                    if len(df_filt) > 0:
                        wins = len(df_filt[df_filt[col_res] == target])
                        profit = (df_filt[df_filt[col_res] == target][col_odd] - 1).sum() - (len(df_filt) - wins)
                        roi = (profit/len(df_filt))*100
                        bets = len(df_filt)
                        
                        st.divider()
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Bets", bets)
                        k2.metric("Profitto", f"{profit:.2f} u")
                        k3.metric("ROI", f"{roi:.2f}%", delta_color="normal" if roi>0 else "inverse")
                        
                        st.dataframe(df_filt)
                    else:
                        st.warning("Nessuna partita trovata con questi filtri.")
