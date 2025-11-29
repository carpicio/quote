import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Value Bet Cluster v36", page_icon="💎", layout="wide")
st.title("💎 Cacciatore di Cluster (v36)")

# --- FUNZIONI CARICAMENTO ---
@st.cache_data(ttl=0)
def load_data(file):
    try:
        # Lettura
        try:
            df = pd.read_csv(file, sep=';', encoding='latin1')
            if len(df.columns) < 5: raise ValueError
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='latin1')

        df.columns = df.columns.str.strip().str.lower()
        
        # Mappa Nomi
        rename_map = {
            '1': 'cotaa', 'x': 'cotae', '2': 'cotad',
            'eloc': 'elohomeo', 'eloo': 'eloawayo',
            'gfinc': 'scor1', 'gfino': 'scor2',
            'o2,5': 'cotao', 'u2,5': 'cotau',
            'gg': 'cotagg', 'ng': 'cotang',
            'data': 'datamecic', 'casa': 'txtechipa1', 'ospite': 'txtechipa2'
        }
        df = df.rename(columns=rename_map)
        
        # Pulizia Numeri
        cols_num = ['cotaa', 'cotae', 'cotad', 'cotao', 'cotau', 'cotagg', 'cotang', 'elohomeo', 'eloawayo', 'scor1', 'scor2']
        for c in cols_num:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['cotaa', 'cotae', 'cotad', 'elohomeo', 'eloawayo'])
        
        # Calcolo ELO Diff Assoluto (per cluster Goal)
        df['elo_diff_abs'] = abs((df['elohomeo'] + 100) - df['eloawayo'])
        
        # Gestione Risultati
        df['res_1x2'] = '-' 
        if 'scor1' in df.columns and 'scor2' in df.columns:
            mask = df['scor1'].notna() & df['scor2'].notna()
            df.loc[mask & (df['scor1'] > df['scor2']), 'res_1x2'] = '1'
            df.loc[mask & (df['scor1'] == df['scor2']), 'res_1x2'] = 'X'
            df.loc[mask & (df['scor1'] < df['scor2']), 'res_1x2'] = '2'
            
            df['goals_tot'] = df['scor1'] + df['scor2']
            if 'cotao' in df.columns:
                df.loc[mask, 'res_o25'] = (df.loc[mask, 'goals_tot'] > 2.5).astype(int)
            if 'cotagg' in df.columns:
                df.loc[mask, 'res_gg'] = ((df.loc[mask, 'scor1'] > 0) & (df.loc[mask, 'scor2'] > 0)).astype(int)

        return df, None

    except Exception as e:
        return None, f"Errore Tecnico: {str(e)}"

# --- INTERFACCIA ---
uploaded_file = st.sidebar.file_uploader("📂 Carica CSV", type=["csv"])

if uploaded_file:
    df, error = load_data(uploaded_file)
    
    if error:
        st.error(error)
    else:
        st.success(f"File caricato! {len(df)} partite analizzate.")
        
        tab1, tab2 = st.tabs(["🚀 Ricerca Occasioni (Future)", "📊 Analisi Storica"])
        
        with tab1:
            st.header("Ricerca Partite Future nei Cluster Vincenti")
            st.info("Filtra le partite che non si sono ancora giocate basandoti sui cluster storici vincenti.")
            
            # Filtro Partite Future
            df_future = df[df['res_1x2'] == '-'].copy()
            
            cluster_choice = st.selectbox("Scegli Strategia Cluster", [
                "1. Sorpresa Casa (Quota 1 > 5.00)",
                "2. Sorpresa Ospite (Quota 2 > 5.00)",
                "3. Over 2.5 Alto Valore (Quota > 2.50, Match Equilibrato)",
                "4. Under 2.5 Controcorrente (Quota > 3.00)",
                "5. NG Alta Quota (Quota > 2.50)"
            ])
            
            df_filt = pd.DataFrame()
            
            if "Sorpresa Casa" in cluster_choice:
                df_filt = df_future[df_future['cotaa'] >= 5.00]
                cols = ['datamecic', 'txtechipa1', 'txtechipa2', 'cotaa']
            elif "Sorpresa Ospite" in cluster_choice:
                df_filt = df_future[df_future['cotad'] >= 5.00]
                cols = ['datamecic', 'txtechipa1', 'txtechipa2', 'cotad']
            elif "Over 2.5" in cluster_choice:
                if 'cotao' in df_future.columns:
                    df_filt = df_future[(df_future['cotao'] >= 2.50) & (df_future['elo_diff_abs'] < 100)]
                    cols = ['datamecic', 'txtechipa1', 'txtechipa2', 'cotao', 'elo_diff_abs']
                else: st.error("Quota Over mancante")
            elif "Under 2.5" in cluster_choice:
                if 'cotau' in df_future.columns:
                    df_filt = df_future[df_future['cotau'] >= 3.00]
                    cols = ['datamecic', 'txtechipa1', 'txtechipa2', 'cotau']
                else: st.error("Quota Under mancante")
            elif "NG" in cluster_choice:
                if 'cotang' in df_future.columns:
                    df_filt = df_future[df_future['cotang'] >= 2.50]
                    cols = ['datamecic', 'txtechipa1', 'txtechipa2', 'cotang']
                else: st.error("Quota NG mancante")
                
            if not df_filt.empty:
                st.write(f"Trovate **{len(df_filt)}** occasioni:")
                st.dataframe(df_filt[cols])
            else:
                st.warning("Nessuna partita futura rientra in questo cluster al momento.")

        with tab2:
            st.header("Verifica Storica (Backtest)")
            st.write("Qui puoi vedere come sono andati questi cluster nel passato.")
            
            # Filtro Partite Giocate
            df_hist = df[df['res_1x2'] != '-'].copy()
            
            if df_hist.empty:
                st.warning("Nessuna partita storica trovata nel file.")
            else:
                market = st.selectbox("Analizza Mercato", ["1", "X", "2", "Over 2.5", "Under 2.5", "GG", "NG"])
                
                c1, c2 = st.columns(2)
                q_min, q_max = c1.slider("Range Quota", 1.0, 10.0, (2.0, 5.0))
                
                # Logica filtri dinamica
                mask = pd.Series(True, index=df_hist.index)
                target, col_odd = None, None
                
                if market == "1": target, col_odd = '1', 'cotaa'
                elif market == "X": target, col_odd = 'X', 'cotae'
                elif market == "2": target, col_odd = '2', 'cotad'
                elif market == "Over 2.5" and 'cotao' in df_hist.columns: 
                    target, col_odd = 1, 'cotao'
                    # Per over/under uso colonna res_o25 calcolata prima (0/1)
                elif market == "Under 2.5" and 'cotau' in df_hist.columns:
                    target, col_odd = 1, 'cotau'
                
                if col_odd:
                    mask &= (df_hist[col_odd] >= q_min) & (df_hist[col_odd] <= q_max)
                    df_res = df_hist[mask]
                    
                    if not df_res.empty:
                        # Calcolo Win
                        if market in ["1", "X", "2"]:
                            wins = len(df_res[df_res['res_1x2'] == target])
                        elif market == "Over 2.5":
                            wins = df_res['res_o25'].sum()
                        elif market == "Under 2.5":
                            # Under è vinto se res_o25 è 0 (quindi gol < 2.5)
                            wins = (1 - df_res['res_o25']).sum()
                            
                        profit = (df_res[col_odd] * (1 if wins else 0) - 1).sum() if wins==0 else \
                                 (df_res[df_res['res_1x2']==target][col_odd]-1).sum() - (len(df_res)-wins) if market in ["1","X","2"] else \
                                 (df_res[df_res['res_o25']==target][col_odd]-1).sum() - (len(df_res)-wins) # Semplificato
                        
                        # Ricalcolo profitto preciso per O/U
                        if market == "Over 2.5":
                            p_win = df_res[df_res['res_o25']==1][col_odd].sum()
                            profit = p_win - len(df_res)
                        elif market == "Under 2.5":
                            p_win = df_res[df_res['res_o25']==0][col_odd].sum() # Se res_o25 è 0, è under
                            profit = p_win - len(df_res)
                        elif market in ["1", "X", "2"]:
                             p_win = df_res[df_res['res_1x2']==target][col_odd].sum()
                             profit = p_win - len(df_res)

                        roi = (profit / len(df_res)) * 100
                        
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Bets", len(df_res))
                        k2.metric("Profitto", f"{profit:.2f} u")
                        k3.metric("ROI", f"{roi:.2f}%")
                        st.dataframe(df_res)
                    else:
                        st.warning("Nessuna partita.")
