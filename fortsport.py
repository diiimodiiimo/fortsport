import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from io import StringIO

# -------------------- App config --------------------
st.set_page_config(page_title="FortSport", page_icon="🎮", layout="wide")
PASSCODE = "dimodimo"

FORTNITE_CSS = """
<style>
:root { --purple:#5b42f3; --purple2:#b01eff; --ink:#0f1020; --card:#171936cc; --glass:#1e2140cc; --accent:#00f0ff; }
html, body, [class*="css"] { background: linear-gradient(135deg, #0b0f2b 0%, #211a4b 60%, #2a1648 100%) !important; }
.block-container { padding-top: 0.8rem; }
h1,h2,h3 { color: #fff; text-shadow: 0 2px 8px rgba(0,0,0,.5); }
.sidebar .sidebar-content { background: var(--card) !important; }
.stButton>button, .stDownloadButton>button {
  border-radius: 12px; border: 0; padding: .6rem 1rem; font-weight: 700;
  background: linear-gradient(135deg, var(--purple) 0%, var(--purple2) 100%);
  color: white; box-shadow: 0 6px 16px rgba(0,0,0,.4);
}
[data-testid="stHeader"] { background: transparent; }
div[data-testid="stDataFrame"] div[role="table"] { background: var(--glass); border-radius: 12px; }
.stDataFrame thead th { background: #2d2a55 !important; color: #eaf0ff !important; }
.metric-card { background: var(--card); border:1px solid #ffffff22; border-radius:16px; padding:14px 16px; color:#eaf0ff;}
.metric-card h3 { margin:0; font-size:1rem; opacity:.85;}
.metric-card .big { font-size:1.6rem; font-weight:800; color:#ffffff; }
hr { border-color:#ffffff22; }
.badge { display:inline-block; padding:.15rem .5rem; border-radius:.5rem; background:#00f0ff22; color:#b6ffff; font-weight:700; }
.small { font-size:.9rem; color:#b6c2ff; }
</style>
"""
st.markdown(FORTNITE_CSS, unsafe_allow_html=True)

# -------------------- Helpers --------------------
def clean_odds(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return np.nan
    s = str(x).strip().replace("+", "").replace(",", "")
    try: return float(s)
    except Exception: return np.nan

def payout_multiple(odds: float) -> float:
    """Win multiple b. +200 -> 2.0 ; -250 -> 0.4"""
    if odds is None or (isinstance(odds, float) and math.isnan(odds)): return 0.0
    return odds/100.0 if odds > 0 else 100.0/abs(odds)

def implied_prob(odds: float) -> float:
    """Book implied probability (no vig)."""
    if odds is None or (isinstance(odds, float) and math.isnan(odds)): return 0.0
    return 100.0/(odds+100.0) if odds > 0 else abs(odds)/(abs(odds)+100.0)

def drink_change_row(odds, result, dagger: bool):
    """
    EV-neutral loss: loss = b * p / (1 - p)
    Dagger on a LOSS doubles the penalty.
    """
    b = payout_multiple(odds)
    p = implied_prob(odds)
    r = (str(result) if result is not None else "").strip().lower()
    if r == "win":
        return +b
    if r == "loss":
        L = b * p / max(1e-6, (1.0 - p))
        return -(2.0 * L if dagger else L)
    return 0.0

def example_df():
    # Matches your sheet structure
    return pd.DataFrame({
        "Name":  ["Jojo","Temp","Etan","Nick","Dimo","Jojo","Temp","Etan","Nick","Dimo","Jojo","Temp","Etan","Nick"],
        "Parlay #":[1,1,1,1,1,2,2,2,2,2,3,3,3,3],
        "Bet":   ["Mahomes 230+ Pass yards","Amon-Ra 70+ rec yards","Gibbs ATD","Travis 4+ Rec","Worthy ATD",
                  "Underwood 200+ pass yards","Iowa ML","Arch 25+ Rush yards","Oregon -6.5","Hosley ATD",
                  "JT ATD","Pickens ATD","DK ATD","Hassan Haskins ATD"],
        "Odds":  [-270,-115,-160,-350,105,117,-185,105,-120,-140,-280,120,180,150],
        "Result":["Win","Loss","Loss","Win","Win","Win","Win","Win","Loss","Win","Win","Win","Win","Loss"],
        "Dagger":[False,False,False,False,False,False,False,False,True,False,False,False,False,True],
        "Sport": ["NFL","NFL","NFL","NFL","NFL","CFB","CFB","CFB","CFB","CFB","NFL","NFL","NFL","NFL"]
    })

def compute_changes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Odds"] = out["Odds"].map(clean_odds)
    out["Result"] = out["Result"].fillna("").astype(str).str.title()
    out["Dagger"] = out["Dagger"].astype(bool)
    out["Drink Change"] = out.apply(lambda r: drink_change_row(r["Odds"], r["Result"], r["Dagger"]), axis=1)
    return out

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Name","Drink Count","Record","Cumulative Odds","Drinks Paid Out","Drinks Received"])
    tmp = df.copy()
    tmp["win"]  = (tmp["Result"].str.lower() == "win").astype(int)
    tmp["loss"] = (tmp["Result"].str.lower() == "loss").astype(int)
    # Dagger adds +1 loss to the record when that bet was a loss
    tmp["loss"] = tmp["loss"] + ((tmp["Dagger"]) & (tmp["Result"].str.lower() == "loss")).astype(int)

    if "Drinks Paid Out" not in tmp: tmp["Drinks Paid Out"] = 0.0
    if "Drinks Received" not in tmp: tmp["Drinks Received"] = 0.0

    agg = (tmp.groupby("Name", as_index=False)
             .agg(Drink_Count=("Drink Change","sum"),
                  Wins=("win","sum"),
                  Losses=("loss","sum"),
                  Cumulative_Odds=("Odds", lambda s: round(float(np.nanmean(s)),2) if len(s) else np.nan),
                  Drinks_Paid_Out=("Drinks Paid Out","sum"),
                  Drinks_Received=("Drinks Received","sum")))
    agg["Record"] = agg["Wins"].astype(str) + "-" + agg["Losses"].astype(str)
    agg = agg.drop(columns=["Wins","Losses"]).sort_values("Drink_Count", ascending=False, kind="mergesort")
    return agg.rename(columns={
        "Drink_Count":"Drink Count",
        "Cumulative_Odds":"Cumulative Odds",
        "Drinks_Paid_Out":"Drinks Paid Out",
        "Drinks_Received":"Drinks Received"
    })

# -------------------- Session state --------------------
if "bets" not in st.session_state:
    st.session_state.bets = example_df()   # start with sample data

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 🎮 FortSport")
    pw = st.text_input("Passcode to edit", type="password", placeholder="dimodimo")
    edit_mode = (pw.strip() == PASSCODE)
    st.markdown(f"**Mode:** {'🟢 Edit' if edit_mode else '🔵 View'}")
    st.caption("EV-neutral loss = b · p / (1−p). Dagger on a loss doubles the penalty.")

    st.markdown("---")
    # Save/Load local file (JSON)
    save_name = st.text_input("Save file name", value="fortsport_save.json")
    data_json = st.session_state.bets.to_json(orient="records")
    st.download_button("💾 Download Save", data=data_json, file_name=save_name, mime="application/json")

    up = st.file_uploader("Upload Save (.json)", type=["json"])
    if up and edit_mode:
        try:
            loaded = pd.read_json(up, orient="records")
            # Ensure required columns exist
            required = ["Name","Parlay #","Bet","Odds","Result","Dagger","Sport"]
            for c in required:
                if c not in loaded.columns: loaded[c] = ""
            st.session_state.bets = loaded[required].copy()
            st.success("Save loaded.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

# -------------------- Compute --------------------
bets = compute_changes(st.session_state.bets)
summary = summarize(bets)

# -------------------- UI --------------------
tab_dash, tab_bets = st.tabs(["🏆 Dashboard", "📋 Bets (Data)"])

with tab_dash:
    st.markdown("### Squad Leaderboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h3>Total Bets</h3>'
                    f'<div class="big">{len(bets)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>Participants</h3>'
                    f'<div class="big">{bets["Name"].nunique()}</div></div>', unsafe_allow_html=True)
    with c3:
        wins = (bets["Result"].str.lower()=="win").sum()
        st.markdown('<div class="metric-card"><h3>Total Wins</h3>'
                    f'<div class="big">{int(wins)}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h3>Last Updated</h3>'
                    f'<div class="big">{datetime.now().strftime("%m/%d %I:%M %p")}</div></div>', unsafe_allow_html=True)

    st.dataframe(summary, use_container_width=True, hide_index=True)

with tab_bets:
    st.markdown("### Bets (edit requires passcode)")

    # ---- Parlay batch entry ----
    with st.expander("➕ Add a batch (Parlay)"):
        colA, colB = st.columns([1,3])
        with colA:
            parlay_no = st.number_input("Parlay #", min_value=1, value=1, step=1)
            num_rows  = st.number_input("How many bets in this parlay?", min_value=1, value=2, step=1)
        with colB:
            st.caption("Fill the grid below, then click **Append Parlay**.")

        tmpl = pd.DataFrame({
            "Name": ["" for _ in range(num_rows)],
            "Parlay #": [parlay_no for _ in range(num_rows)],
            "Bet": ["" for _ in range(num_rows)],
            "Odds": ["" for _ in range(num_rows)],
            "Result": ["Win" for _ in range(num_rows)],
            "Dagger": [False for _ in range(num_rows)],
            "Sport": ["" for _ in range(num_rows)],
        })
        new_rows = st.data_editor(
            tmpl, key="new_parlay_editor",
            use_container_width=True, hide_index=True, num_rows="dynamic",
            column_config={
                "Dagger": st.column_config.CheckboxColumn("Dagger"),
                "Result": st.column_config.SelectboxColumn("Result", options=["Win","Loss","Void"])
            }
        )
        if st.button("Append Parlay", disabled=not edit_mode):
            add = new_rows.copy()
            add["Odds"] = add["Odds"].map(clean_odds)
            st.session_state.bets = pd.concat([st.session_state.bets, add], ignore_index=True)
            st.success("Parlay appended!")

    # ---- Main editable table ----
    edited = st.data_editor(
        st.session_state.bets,
        key="bets_table_editor",
        use_container_width=True,
        hide_index=True,
        disabled=not edit_mode,
        column_config={
            "Dagger": st.column_config.CheckboxColumn("Dagger"),
            "Result": st.column_config.SelectboxColumn("Result", options=["Win","Loss","Void"]),
            "Sport": st.column_config.SelectboxColumn("Sport", options=["NFL","CFB","NBA","MLB","NHL","Other"]),
        }
    )
    if edit_mode and st.button("Save changes to working copy"):
        st.session_state.bets = edited.copy()
        st.success("Saved to session.")

# Optional: show computed drink change
with st.expander("🔎 Computed Drink Change (read-only)"):
    st.dataframe(
        bets[["Name","Parlay #","Bet","Odds","Result","Dagger","Drink Change","Sport"]],
        use_container_width=True, hide_index=True
    )
