import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet v37", page_icon="⚽", layout="wide")
st.title("⚽ Calcolatore Strategico (v37 - Future Ready)")

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

        # Drop solo se mancano le quote o ELO (i risultati possono mancare!)
        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        calc = df.apply(lambda r: calculate_row(r), axis=1)
        df = pd.concat([df, calc], axis=1)
        
        df['res_1x2'] = '-' 
        if 'scor1' in df.columns and 'scor2' in df.columns:
            # Maschera per capire quali righe hanno risultati
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
# Aggiunta scheda "Partite Future"
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Calcolatore Manuale", "📅 Partite Future", "📊 Report Storico", "🕵️ Analisi Cluster"])

with tab1:
    st.header("Calcolatore Manuale (Partita Singola)")
    c_name1, c_vs, c_name2 = st.columns([3, 1, 3])
    with c_name1: team_h = st.text_input("Squadra Casa", "Home Team")
    with c_vs: st.markdown("<h3 style='text-align: center; margin-top: 20px;'>VS</h3>", unsafe_allow_html=True)
    with c_name2: team_a = st.text_input("Squadra Ospite", "Away Team")

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
        diff = int(res['ELO_Diff'])
        diff_color = "green" if diff > 0 else "red"
        st.markdown(f"Differenza ELO: **:{diff_color}[{diff}]**")
        k1, k2, k3 = st.columns(3)
        def show_card(col, label, odd, ev, fair):
            bg = "#d4edda" if ev > 0 else "#f8d7da"
            txt = "#155724" if ev > 0 else "#721c24"
            col.markdown(f"""
            <div style="background-color:{bg}; padding:10px; border-radius:10px; text-align:center;">
                <h3 style="color:{txt}; margin:0;">Segno {label}</h3>
                <h1 style="color:black; margin:0;">{odd:.2f}</h1>
                <p style="color:black; font-weight:bold; margin:5px;">Imp: {fair:.2f}</p>
                <p style="color:{txt}; margin:0;">EV: {ev:+.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        show_card(k1, "1", o1, res['EV_1'], res['Fair_1'])
        show_card(k2, "X", ox, res['EV_X'], res['Fair_X'])
        show_card(k3, "2", o2, res['EV_2'], res['Fair_2'])

uploaded_file = st.sidebar.file_uploader("📂 Carica CSV per Analisi Multipla", type=["csv"])

if uploaded_file:
    df, error_msg = load_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        # Divido il file in due: Giocate vs Future
        df_future = df[df['res_1x2'] == '-'].copy()
        df_played = df[df['res_1x2'] != '-'].copy()

        # --- TAB 2: PARTITE FUTURE ---
        with tab2:
            st.header("📅 Lista Partite Future")
            if not df_future.empty:
                st.info(f"Trovate **{len(df_future)}** partite da giocare.")
                
                # Filtri
                min_ev = st.slider("Mostra solo EV > %", 0, 30, 5, key="ev_fut")
                
                # Creiamo una tabella facile da leggere
                # Selezioniamo le righe dove c'è valore
                mask_val = (df_future['EV_1']*100 >= min_ev) | (df_future['EV_2']*100 >= min_ev)
                df_opp = df_future[mask_val].copy()
                
                if not df_opp.empty:
                    # Aggiungiamo colonna "Consiglio"
                    def get_tip(r):
                        if r['EV_2']*100 >= min_ev: return f"2 (Quota {r['cotad']})"
                        if r['EV_1']*100 >= min_ev: return f"1 (Quota {r['cotaa']})"
                        return "-"
                    
                    df_opp['BET'] = df_opp.apply(get_tip, axis=1)
                    
                    # Colonne da mostrare
                    cols_show = ['datamecic', 'txtechipa1', 'txtechipa2', 'BET', 'EV_1', 'EV_2']
                    # Filtra solo colonne esistenti
                    cols_final = [c for c in cols_show if c in df_opp.columns]
                    
                    st.dataframe(
                        df_opp[cols_final].style.format({
                            'EV_1': "{:.1%}", 
                            'EV_2': "{:.1%}"
                        })
                    )
                else:
                    st.warning(f"Nessuna partita trovata con EV > {min_ev}%")
            else:
                st.warning("Il file non contiene partite future (tutte hanno un risultato).")

        # --- TAB 3: REPORT STORICO ---
        with tab3:
            st.header("Report Storico")
            if not df_played.empty:
                st.success(f"Analisi su **{len(df_played)}** partite giocate.")
                pnl_1 = np.where(df_played['EV_1']>0, np.where(df_played['res_1x2']=='1', df_played['cotaa']-1, -1), 0).sum()
                pnl_2 = np.where(df_played['EV_2']>0, np.where(df_played['res_1x2']=='2', df_played['cotad']-1, -1), 0).sum()
                m1, m2 = st.columns(2)
                m1.metric("Profitto Strategia CASA", f"{pnl_1:.2f} u")
                m2.metric("Profitto Strategia OSPITE", f"{pnl_2:.2f} u")
                st.dataframe(df_played.head(20))
            else:
                st.info("Nessuna partita storica trovata nel file.")

        # --- TAB 4: ANALISI CLUSTER ---
        with tab4:
            st.header("Analisi Cluster")
            if df_played.empty:
                st.warning("Servono risultati storici per questa analisi.")
            else:
                mode = st.selectbox("Mercato", ["Casa (1)", "Ospite (2)", "Pareggio (X)"])
                c1, c2 = st.columns(2)
                q_min, q_max = c1.slider("Quota", 1.0, 10.0, (1.5, 4.0))
                
                mask = pd.Series(True, index=df_played.index)
                target, col_odd = None, None
                
                if "Casa" in mode: target, col_odd = '1', 'cotaa'
                elif "Ospite" in mode: target, col_odd = '2', 'cotad'
                else: target, col_odd = 'X', 'cotae'
                
                mask &= (df_played[col_odd] >= q_min) & (df_played[col_odd] <= q_max)
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
