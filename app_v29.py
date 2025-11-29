import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet v29", page_icon="⚽", layout="wide")
st.title("⚽ Calcolatore Strategico (v29 - Con Quote Implicite)")

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
        
        # Quote Implicite (Fair Odds)
        res['Fair_1'] = 1/p_fin_1 if p_fin_1>0 else 0
        res['Fair_X'] = 1/pf_x if pf_x>0 else 0
        res['Fair_2'] = 1/p_fin_2 if p_fin_2>0 else 0
        
        # EV
        res['EV_1'] = (o1 * p_fin_1) - 1
        res['EV_X'] = (ox * pf_x) - 1
        res['EV_2'] = (o2 * p_fin_2) - 1
        
    res['ELO_Diff'] = (elo_h + hfa) - elo_a # Differenza netta con HFA
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
            'data': 'datamecic', 'casa': 'txtechipa1', 'ospite': 'txtechipa2'
        }
        df = df.rename(columns=rename_map)
        
        req_cols = ['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo']
        missing = [c for c in req_cols if c not in df.columns]
        if missing: return None, f"⚠️ Errore Colonne: {missing}"

        cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        calc = df.apply(lambda r: calculate_row(r), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        if 'scor1' in df.columns and 'scor2' in df.columns:
            df['goals_ft'] = df['scor1'] + df['scor2']
            conditions = [df['scor1'] > df['scor2'], df['scor1'] == df['scor2'], df['scor1'] < df['scor2']]
            df['res_1x2'] = np.select(conditions, ['1', 'X', '2'], default='-')
            
            if 'cotao' in df.columns:
                df['res_o25'] = (df['goals_ft'] > 2.5).astype(int)
                df['res_u25'] = (df['goals_ft'] < 2.5).astype(int)
        else:
            df['res_1x2'] = '-'
            
        return df, None

    except Exception as e:
        return None, f"Errore Tecnico: {str(e)}"

# --- INTERFACCIA ---
tab1, tab2, tab3 = st.tabs(["🔮 Calcolatore", "📊 Report", "🕵️ Analisi"])

with tab1:
    st.header("Calcolatore Manuale")
    
    # Input Nomi Squadre
    c_name1, c_vs, c_name2 = st.columns([3, 1, 3])
    with c_name1: team_h = st.text_input("Squadra Casa", "Clermont")
    with c_vs: st.markdown("<h3 style='text-align: center; margin-top: 20px;'>VS</h3>", unsafe_allow_html=True)
    with c_name2: team_a = st.text_input("Squadra Ospite", "Troyes")

    # Input Dati
    c1, c2, c3 = st.columns(3)
    elo_h = c1.number_input("ELO Casa", value=1424, min_value=0, step=10)
    o1 = c1.number_input("Quota 1", value=2.76, min_value=1.01, step=0.01)
    
    with c2:
        st.write("")
        st.write("") 
        ox = st.number_input("Quota X", value=3.23, min_value=1.01, step=0.01)
    
    elo_a = c3.number_input("ELO Ospite", value=1543, min_value=0, step=10)
    o2 = c3.number_input("Quota 2", value=2.73, min_value=1.01, step=0.01)
    
    if st.button("Calcola Previsione", type="primary"):
        row = {'elohomeo': elo_h, 'eloawayo': elo_a, 'cotaa': o1, 'cotae': ox, 'cotad': o2}
        res = calculate_row(row)
        
        st.divider()
        st.subheader(f"Risultati: {team_h} vs {team_a}")
        st.info(f"**Analisi ELO:** {elo_h} (Casa) + 100 (HFA) vs {elo_a} (Ospite) = Differenza netta **{int(res['ELO_Diff'])}** punti.")
        
        k1, k2, k3 = st.columns(3)
        
        # --- NUOVA FUNZIONE VISUALIZZAZIONE CON QUOTE IMPLICITE ---
        def show(col, lbl, odd, ev, fair):
            color = "inverse" if ev > 0 else "normal"
            # Mostriamo la Quota Implicita nel titolo del cartellino
            label_text = f"Segno {lbl} (Imp: {fair:.2f})"
            col.metric(label=label_text, value=f"{odd}", delta=f"EV: {ev:.1%}", delta_color=color)
            
        show(k1, "1", o1, res['EV_1'], res['Fair_1'])
        show(k2, "X", ox, res['EV_X'], res['Fair_X'])
        show(k3, "2", o2, res['EV_2'], res['Fair_2'])

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV", type=["csv"])

if uploaded_file:
    df, error_msg = load_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        with tab2:
            st.header("Report Generale")
            df_valid = df[df['res_1x2'] != '-'].copy()
            if not df_valid.empty:
                pnl_1 = np.where(df_valid['EV_1']>0, np.where(df_valid['res_1x2']=='1', df_valid['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_valid['EV_2']>0, np.where(df_valid['res_1x2']=='2', df_valid['cotad']-1, -1), 0).sum()
                m1, m2 = st.columns(2)
                m1.metric("Totale Strategia CASA", f"{pnl_1:.2f} u")
                m2.metric("Totale Strategia OSPITE", f"{pnl_2:.2f} u")
            else:
                st.info("ℹ️ Nessun risultato storico trovato.")
            
            # Mostriamo anche le Quote Implicite nella tabella
            cols_hide = ['res_1x2', 'res_o25', 'res_u25', 'goals_ft']
            cols_base = [c for c in df.columns if c not in cols_hide and 'Fair' not in c]
            # Aggiungiamo colonne Fair alla visualizzazione
            cols_view = ['datamecic', 'txtechipa1', 'txtechipa2', 'cotaa', 'Fair_1', 'cotad', 'Fair_2', 'EV_1', 'EV_2']
            # Filtro colonne esistenti
            cols_final = [c for c in cols_view if c in df.columns]
            
            st.dataframe(df[cols_final].head(20).style.format("{:.2f}", subset=['cotaa', 'Fair_1', 'cotad', 'Fair_2']))

        with tab3:
            st.header("Analisi Cluster")
            my_profit, my_roi, my_bets = 0.0, 0.0, 0
            df_valid = df[df['res_1x2'] != '-'].copy()
            
            if df_valid.empty:
                st.warning("⚠️ Servono risultati storici per calcolare il ROI.")
                st.subheader("Filtra Partite Future")
                min_ev = st.slider("Minimo Valore EV %", 0, 50, 5)
                df_future = df[ (df['EV_1']*100 > min_ev) | (df['EV_2']*100 > min_ev) ]
                st.dataframe(df_future[['datamecic', 'txtechipa1', 'txtechipa2', 'cotaa', 'Fair_1', 'cotad', 'Fair_2', 'EV_1', 'EV_2']])
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

                mask = pd.Series(True, index=df_valid.index)
                target, col_odd, col_res = None, None, None

                if mode == "Casa (1)":
                    col_odd, col_res, target = 'cotaa', 'res_1x2', '1'
                    mask &= (df_valid['EV_1']*100 >= ev_min) & (df_valid['EV_1']*100 <= ev_max)
                elif mode == "Ospite (2)":
                    col_odd, col_res, target = 'cotad', 'res_1x2', '2'
                    mask &= (df_valid['EV_2']*100 >= ev_min) & (df_valid['EV_2']*100 <= ev_max)
                elif mode == "Pareggio (X)":
                    col_odd, col_res, target = 'cotae', 'res_1x2', 'X'
                    mask &= (df_valid['EV_X']*100 >= ev_min) & (df_valid['EV_X']*100 <= ev_max)
                elif mode == "Over 2.5":
                    if 'cotao' in df.columns:
                        col_odd, col_res, target = 'cotao', 'res_o25', 1
                        mask &= (df_valid['ELO_Diff'] >= elo_min) & (df_valid['ELO_Diff'] <= elo_max)
                    else: st.error("Manca quota Over (O2,5)")
                elif mode == "Under 2.5":
                    if 'cotau' in df.columns:
                        col_odd, col_res, target = 'cotau', 'res_u25', 1
                        mask &= (df_valid['ELO_Diff'] >= elo_min) & (df_valid['ELO_Diff'] <= elo_max)
                    else: st.error("Manca quota Under (U2,5)")
                
                if col_odd and col_odd in df.columns:
                    mask &= (df_valid[col_odd] >= q_min) & (df_valid[col_odd] <= q_max)
                    df_filt = df_valid[mask].copy()
                    
                    if len(df_filt) > 0:
                        wins = len(df_filt[df_filt[col_res] == target])
                        my_profit = (df_filt[df_filt[col_res] == target][col_odd] - 1).sum() - (len(df_filt) - wins)
                        my_roi = (my_profit/len(df_filt))*100
                        my_bets = len(df_filt)
                        
                        st.divider()
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Bets", my_bets)
                        k2.metric("Profitto", f"{my_profit:.2f} u")
                        k3.metric("ROI", f"{my_roi:.2f}%", delta_color="normal" if my_roi>0 else "inverse")
                        
                        cols_base = ['datamecic', 'txtechipa1', 'txtechipa2']
                        cols_view = [c for c in cols_base if c in df_filt.columns] + [col_odd, 'ELO_Diff']
                        st.dataframe(df_filt[cols_view])
                    else:
                        st.warning("Nessuna partita trovata.")
