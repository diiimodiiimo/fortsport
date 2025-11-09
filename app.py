import os
import math
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --------------- App config ---------------
st.set_page_config(page_title="FortSport", page_icon="🎮", layout="wide")
PASSCODE = "dimodimo"
SEED_FILE = "seed.json"
AUTOSAVE_FILE = "fortsport_autosave.json"

# --------------- Global theme (BLACK + NEON) ---------------
NEON_CSS = """
<style>
:root{
  --bg:#0b120d;
  --panel:#283329;
  --panel-2:#1f2a21;
  --text:#e9ffe9;
  --muted:#bde8bd;
  --accent:#39ff14;
  --accent-dim:#2edc11;
  --grid:#3d4a3f;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }
/* Sidebar */
[data-testid="stSidebar"] { background: var(--panel-2) !important; color: var(--text) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] .stTextInput>div>div>input {
  background:#111 !important; color:var(--text) !important; border:1px solid #444 !important;
}
/* Cards / metrics */
.card { background: var(--panel); border: 1px solid var(--grid); border-radius: 16px; padding: 16px; color: var(--text); box-shadow: 0 8px 24px rgba(0,0,0,.35); }
.metric { background: var(--panel); border: 1px solid var(--grid); border-radius: 16px; padding: 14px 16px; color: var(--text); box-shadow: 0 8px 20px rgba(57,255,20,.2);
  height: 160px; display:flex; flex-direction:column; justify-content:space-between; }  /* fixed height for identical tiles */
.metric h4{ margin:0 0 6px 0; font-weight:800; color:var(--muted); }
.metric .big{ font-size: 1.9rem; font-weight: 900; color: var(--text); }

/* Uniform metric grid row */
.metric-row{
  display:grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 18px;
  align-items: stretch;
}
/* Responsive (optional) */
@media (max-width: 1200px){
  .metric-row{ grid-template-columns: repeat(3, minmax(0,1fr)); }
}
@media (max-width: 800px){
  .metric-row{ grid-template-columns: repeat(2, minmax(0,1fr)); }
}

/* Tabs */
.stTabs [role="tablist"] { gap: 14px; }
.stTabs [role="tab"] { background: var(--panel); color: var(--text); border: 1px solid var(--grid); border-radius: 14px; padding: 10px 18px; }
.stTabs [aria-selected="true"] { background: linear-gradient(180deg,var(--panel),var(--panel-2)); border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent) inset; }
/* Dataframes (base) */
div[data-testid="stDataFrame"] div[role="table"] { background: var(--panel); color: var(--text); border-radius: 12px; border: 1px solid var(--grid); }
div[data-testid="stDataFrame"] thead th { background: var(--panel-2) !important; color: var(--text) !important; border-bottom: 1px solid var(--grid) !important; }
div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th { border-bottom: 1px solid var(--grid) !important; }
/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 12px; border: 1px solid var(--accent-dim);
  padding:.55rem 1rem; font-weight:800;
  background: linear-gradient(180deg, var(--accent), var(--accent-dim));
  color:#021; text-shadow: 0 1px 0 rgba(255,255,255,.2);
  box-shadow: 0 8px 20px rgba(57,255,20,.2);
}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

def neon_style(df, highlight_col: str | None = None,
               fmt_map: dict[str, str] | None = None,
               default_decimals: int | None = None):
    """Neon styler with optional per-column or default numeric formatting."""
    sty = (df.style
           .set_properties(**{
               "background-color":"#283329",
               "color":"#e9ffe9",
               "border-color":"#3d4a3f",
               "border-width":"1px",
           })
           .set_table_styles([
               {"selector":"th", "props":[("background","#1f2a21"),("color","#e9ffe9"),
                                          ("border","1px solid #3d4a3f"),("font-weight","800")]},
               {"selector":"td, th", "props":[("padding","10px 12px")]}])
           .hide(axis="index"))

    if highlight_col and highlight_col in df.columns and not df.empty:
        vmin = pd.to_numeric(df[highlight_col], errors="coerce").min()
        vmax = pd.to_numeric(df[highlight_col], errors="coerce").max()
        sty = sty.bar(subset=[highlight_col], color="#39ff1422", vmin=vmin, vmax=vmax)

    if fmt_map:
        sty = sty.format(fmt_map)
    elif default_decimals is not None:
        num_cols = df.select_dtypes(include="number").columns
        sty = sty.format({c: f"{{:.{default_decimals}f}}" for c in num_cols})

    return sty

# --------------- Helpers ---------------
def clean_odds(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return np.nan
    s = str(x).strip().replace("+","").replace(",","")
    try: return float(s)
    except Exception: return np.nan

def implied_prob(odds: float) -> float:
    if odds is None or (isinstance(x := odds, float) and math.isnan(x)): return 0.0
    return 100.0/(odds+100.0) if odds > 0 else abs(odds)/(abs(odds)+100.0)

def payout_multiple(odds: float) -> float:
    if odds is None or (isinstance(x := odds, float) and math.isnan(x)): return 0.0
    return odds/100.0 if odds > 0 else 100.0/abs(odds)

# --- DRINKS: symmetric logic (wins +b, losses -b) ---
def base_drink_change(odds, result):
    b = payout_multiple(odds)
    r = (str(result) if result is not None else "").strip().lower()
    if r == "win":  return +b
    if r == "loss": return -b
    return 0.0

def example_df():
    df = pd.DataFrame({
        "Name":  ["Jojo","Temp","Etan","Nick","Dimo","Jojo","Temp","Etan","Nick","Dimo","Jojo","Temp","Etan","Nick"],
        "Parlay #":[1,1,1,1,1,2,2,2,2,2,3,3,3,3],
        "Bet":   ["Mahomes 230+ Pass yards","Amon-Ra 70+ rec yards","Gibbs ATD","Travis 4+ Rec","Worthy ATD",
                  "Underwood 200+ pass yards","Iowa ML","Arch 25+ Rush yards","Oregon -6.5","Hosley ATD",
                  "JT ATD","Pickens ATD","DK ATD","Hassan Haskins ATD"],
        "Odds":  [-270,-115,-160,-350,105,117,-185,105,-120,-140,-280,120,180,150],
        "Result":["Win","Loss","Loss","Win","Win","Win","Win","Win","Loss","Win","Win","Win","Win","Loss"],
        "Sport": ["NFL","NFL","NFL","NFL","NFL","CFB","CFB","CFB","CFB","CFB","NFL","NFL","NFL","NFL"],
    })
    df["Created"] = [datetime(2025,1,1,12,0,0).timestamp() + i for i in range(len(df))]
    return df

def ensure_columns(df):
    req = ["Name","Parlay #","Bet","Odds","Result","Sport","Created"]
    for c in req:
        if c not in df.columns:
            df[c] = (datetime.now().timestamp() if c=="Created" else "")
    return df[req]

def compute_daggers(df: pd.DataFrame) -> pd.Series:
    dag = pd.Series(False, index=df.index)
    for _, g in df.groupby("Parlay #"):
        if len(g) != 5:  # only on 5-leg parlays
            continue
        losers = g[g["Result"].str.lower() == "loss"]
        if losers.empty: continue
        idx = losers["Odds"].map(implied_prob).idxmax()  # biggest fave that lost
        dag.loc[idx] = True
    return dag

# --- OPTIONAL: all-miss parlay discount (drinks only) ---
def _parlay_all_miss_discounts(df: pd.DataFrame) -> pd.Series:
    """
    If every leg in a parlay is a LOSS, discount each loss in that parlay:
      multiplier = max(0, 1 - 0.10 * num_legs)
    Applies ONLY to Drink Change (NOT Dollars). Defaults to 1.0 elsewhere.
    """
    mult = pd.Series(1.0, index=df.index, dtype=float)
    if df.empty:
        return mult
    for pid, g in df.groupby("Parlay #"):
        res = g["Result"].astype(str).str.lower()
        if len(g) > 0 and (res == "loss").all():
            n = len(g)
            factor = max(0.0, 1.0 - 0.10 * n)  # cap at 100% discount
            mult.loc[g.index] = np.where(res == "loss", factor, 1.0)
    return mult

def apply_takeover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drink Change pipeline:
      - symmetric win/loss via base_drink_change
      - 'Takeover': after 3 straight wins, wins are doubled
      - all-miss parlay discount applied to LOSSES for drinks only
    """
    out = df.sort_values("Created").copy()
    out["Takeover"] = False
    out["Drink Change"] = 0.0

    # precompute all-miss discount per row (drinks only)
    discount = _parlay_all_miss_discounts(df)

    for _, g in out.groupby("Name", sort=False):
        streak = 0
        for idx, row in g.iterrows():
            takeover = streak >= 3
            out.at[idx,"Takeover"] = takeover

            val = base_drink_change(row["Odds"], row["Result"])

            # Takeover doubles ONLY wins
            if takeover and str(row["Result"]).lower() == "win":
                val *= 2.0

            # Apply all-miss discount to LOSSES (drinks only)
            if str(row["Result"]).lower() == "loss":
                val *= discount.get(idx, 1.0)

            out.at[idx,"Drink Change"] = val

            # update streak
            if str(row["Result"]).lower() == "win":
                streak += 1
            elif str(row["Result"]).lower() == "loss":
                streak = 0
    return out.sort_index()

# --- CASH P&L: Flat $100 stake per bet ---
def dollars_pnl_100(odds: float, result: str) -> float:
    """
    If staking $100 on every bet:
      - Win @ +odds:  profit = +odds
      - Win @ -odds:  profit = +100 * (100/|odds|)
      - Loss (any):   -100
      - Void/other:    0
    """
    r = (str(result) if result is not None else "").strip().lower()
    if r not in {"win", "loss"} or odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return 0.0
    if r == "loss":
        return -100.0
    if odds > 0:
        return float(odds)
    else:
        return 100.0 * (100.0 / abs(odds))

def compute_all(bets_raw: pd.DataFrame):
    df = bets_raw.copy()
    df["Odds"] = df["Odds"].map(clean_odds)
    df["Result"] = df["Result"].fillna("").astype(str).str.title()

    # Drinks (takeover + optional all-miss discount)
    df = apply_takeover(df)

    # Flags / extras
    df["Dagger"] = compute_daggers(df)
    df["Dollars"] = [dollars_pnl_100(o, r) for o, r in zip(df["Odds"], df["Result"])]

    return df

def summarise(df: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        base = pd.DataFrame(columns=[
            "Name","Drink Count","Record","Cumulative Odds","Parlays Won","Dollars Won",
            "Drinks Paid Out","Drinks Received"
        ])
    else:
        tmp = df.copy()
        tmp["win"]  = (tmp["Result"].str.lower()=="win").astype(int)
        tmp["loss"] = (tmp["Result"].str.lower()=="loss").astype(int)

        # parlay win = all legs win AND has at least 3 legs
        size = tmp.groupby("Parlay #")["Result"].count()
        all_win = tmp.groupby("Parlay #")["win"].apply(lambda s: (s == 1).all())
        parlay_win = ((size >= 3) & all_win)
        tmp = tmp.merge(parlay_win.rename("ParlayWon"), on="Parlay #", how="left").fillna({"ParlayWon":False})

        agg = (tmp.groupby("Name", as_index=False)
                 .agg(Drink_Sum=("Drink Change","sum"),
                      Wins=("win","sum"),
                      Losses=("loss","sum"),
                      Cumulative_Odds=("Odds", lambda s: float(np.nanmean(s)) if len(s) else np.nan),
                      Parlays_Won=("ParlayWon","sum"),
                      Dollars_Won=("Dollars","sum"),
                      Daggers=("Dagger","sum")))

        agg["Record"] = agg["Wins"].astype(str) + "-" + agg["Losses"].astype(str)

        base = agg[["Name","Drink_Sum","Daggers","Record","Cumulative_Odds","Parlays_Won","Dollars_Won"]].rename(
            columns={"Cumulative_Odds":"Cumulative Odds","Parlays_Won":"Parlays Won","Dollars_Won":"Dollars Won"}
        )

    # incorporate transfers (paid/received) for Drink Count:
    if transfers is None or transfers.empty:
        paid = pd.Series(0.0, index=base["Name"], name="Drinks Paid Out") if not base.empty \
               else pd.Series(name="Drinks Paid Out", dtype=float)
        recv = pd.Series(0.0, index=base["Name"], name="Drinks Received") if not base.empty \
               else pd.Series(name="Drinks Received", dtype=float)
    else:
        paid = transfers.groupby("From")["Amount"].sum().rename("Drinks Paid Out")
        recv = transfers.groupby("To")["Amount"].sum().rename("Drinks Received")

    base = base.merge(paid, left_on="Name", right_index=True, how="left")
    base = base.merge(recv, left_on="Name", right_index=True, how="left")
    base["Drinks Paid Out"] = base["Drinks Paid Out"].fillna(0.0)
    base["Drinks Received"] = base["Drinks Received"].fillna(0.0)

    base["Drink Count"] = (
        base.get("Drink_Sum", 0.0)
        - base.get("Daggers", 0.0)
        + base["Drinks Paid Out"]
        - base["Drinks Received"]
    )

    final = (base[["Name","Drink Count","Record","Cumulative Odds","Parlays Won","Dollars Won",
                   "Drinks Paid Out","Drinks Received"]]
             .sort_values("Drink Count", ascending=False, kind="mergesort"))
    return final

def total_parlays_won(df: pd.DataFrame) -> int:
    if df.empty: return 0
    size = df.groupby("Parlay #")["Result"].count()
    all_win = df.groupby("Parlay #")["Result"].apply(lambda s: (s.str.lower()=="win").all())
    return int(((size >= 3) & all_win).sum())

# --------------- Bundle save/load (bets + transfers) ---------------
def pack_state():
    return {
        "bets": ensure_columns(st.session_state.bets.copy()),
        "transfers": st.session_state.transfers.copy()
    }

def unpack_state(obj):
    if isinstance(obj, list):
        return ensure_columns(pd.DataFrame(obj)), pd.DataFrame(columns=["From","To","Amount","Created"])
    if isinstance(obj, dict) and "bets" in obj:
        bets = ensure_columns(pd.DataFrame(obj["bets"]))
        transfers = pd.DataFrame(obj.get("transfers", []))
        if transfers.empty:
            transfers = pd.DataFrame(columns=["From","To","Amount","Created"])
        return bets, transfers
    return ensure_columns(pd.DataFrame(obj)), pd.DataFrame(columns=["From","To","Amount","Created"])

def dump_json_bundle() -> str:
    bundle = pack_state()
    return json.dumps({
        "bets": json.loads(bundle["bets"].to_json(orient="records")),
        "transfers": json.loads(bundle["transfers"].to_json(orient="records"))
    })

# --------------- Seed I/O ---------------
@st.cache_resource
def load_seed_bundle():
    if os.path.exists(SEED_FILE):
        try:
            obj = json.loads(open(SEED_FILE, "r", encoding="utf-8").read())
            return unpack_state(obj)
        except Exception:
            pass
    return ensure_columns(example_df()), pd.DataFrame(columns=["From","To","Amount","Created"])

def save_seed_bundle():
    bets = ensure_columns(st.session_state.bets.copy())
    transfers = st.session_state.transfers.copy()
    with open(SEED_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "bets": json.loads(bets.to_json(orient="records")),
            "transfers": json.loads(transfers.to_json(orient="records"))
        }))

# --------------- Session init ---------------
if "bets" not in st.session_state or "transfers" not in st.session_state:
    if os.path.exists(AUTOSAVE_FILE):
        try:
            obj = json.loads(open(AUTOSAVE_FILE, "r", encoding="utf-8").read())
            bets, transfers = unpack_state(obj)
        except Exception:
            bets, transfers = load_seed_bundle()
    else:
        bets, transfers = load_seed_bundle()
    st.session_state.bets = bets
    st.session_state.transfers = transfers

# --------------- Sidebar ---------------
st.sidebar.markdown("## ⚙️ Settings")
pw = st.sidebar.text_input("Passcode to edit", type="password", placeholder="dimodimo")
edit_mode = (pw.strip() == PASSCODE)
st.sidebar.write(f"**Mode:** {'🟢 Edit' if edit_mode else '🔵 View'}")
st.sidebar.caption(
    "Drinks: symmetric ±payout • Dagger: worst loss (5-bet) −1 • "
    "Takeover: 2× wins after 3-win streak • "
    "All-miss parlay loss discount (drinks only): 10% × legs (capped at 100%) • "
    "Cash P&L: flat $100 stake per bet."
)

st.sidebar.divider()
save_name = st.sidebar.text_input("Save file name", value="fortsport_save.json")
st.sidebar.download_button("💾 Download Save", data=dump_json_bundle(),
                           file_name=save_name, mime="application/json")

up = st.sidebar.file_uploader("Upload Save (.json)", type=["json"])
if up and edit_mode:
    try:
        obj = json.loads(up.read().decode("utf-8"))
        bets, transfers = unpack_state(obj)
        st.session_state.bets = bets
        st.session_state.transfers = transfers
        st.success("Save loaded into working copy.")
    except Exception as e:
        st.error(f"Could not read file: {e}")

autosave = st.sidebar.toggle("Auto-save to disk", value=True)
if autosave:
    try:
        open(AUTOSAVE_FILE, "w", encoding="utf-8").write(dump_json_bundle())
    except Exception:
        pass
st.sidebar.caption(f"Autosave path: `{AUTOSAVE_FILE}`")

if edit_mode and st.sidebar.button("🚀 Publish current as default (seed)"):
    save_seed_bundle()
    st.success("Published! New visitors will see this data by default.")

# --------------- Compute ---------------
bets = compute_all(st.session_state.bets)
summary = summarise(bets, st.session_state.transfers)

# --------------- UI ---------------
tab_dash, tab_bets, tab_stats, tab_court = st.tabs(["🏆 Dashboard", "📋 Bets", "📈 Stats", "🍻 MyCourt"])

# ----- Dashboard -----
with tab_dash:
    dollars = float(bets["Dollars"].sum())
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric"><h4>Total Bets</h4><div class="big">{len(bets)}</div></div>
          <div class="metric"><h4>Parlays</h4><div class="big">{bets["Parlay #"].nunique()}</div></div>
          <div class="metric"><h4>Parlays Won</h4><div class="big">{total_parlays_won(bets)}</div></div>
          <div class="metric"><h4>Total $ Won</h4><div class="big">${dollars:,.2f}</div></div>
          <div class="metric"><h4>Last Updated</h4><div class="big">{datetime.now().strftime("%m/%d")}<br>{datetime.now().strftime("%I:%M %p")}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Squad Leaderboard")

    display = summary.copy()
    st.table(neon_style(
        display,
        highlight_col="Drink Count",
        fmt_map={
            "Drink Count":     "{:.2f}",
            "Cumulative Odds": "{:.2f}",
            "Parlays Won":     "{:.0f}",
            "Dollars Won":     "${:,.2f}",
            "Drinks Paid Out": "{:.2f}",
            "Drinks Received": "{:.2f}",
        }
    ))
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Bets -----
with tab_bets:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Add a Parlay")
    next_parlay = int(bets["Parlay #"].max() + 1) if not bets.empty else 1
    colA, colB = st.columns([1,3])
    with colA:
        parlay_no = st.number_input("Parlay #", min_value=1, value=next_parlay, step=1)
        num_rows  = st.number_input("How many bets in this parlay?", min_value=1, value=5, step=1)
    with colB:
        st.caption("Fill the grid then click **Append Parlay**.")

    tmpl = pd.DataFrame({
        "Name": ["" for _ in range(num_rows)],
        "Parlay #": [parlay_no for _ in range(num_rows)],
        "Bet": ["" for _ in range(num_rows)],
        "Odds": ["" for _ in range(num_rows)],
        "Result": ["Win" for _ in range(num_rows)],
        "Sport": ["" for _ in range(num_rows)],
    })
    new_rows = st.data_editor(
        tmpl, key="new_parlay_editor", use_container_width=True, hide_index=True,
        num_rows="dynamic",
        column_config={
            "Result": st.column_config.SelectboxColumn("Result", options=["Win","Loss","Void"]),
        }
    )
    st.write("")
    if st.button(f"Append Parlay{' (enter passcode to enable)' if not edit_mode else ''}", disabled=not edit_mode):
        add = new_rows.copy()
        add["Odds"] = add["Odds"].map(clean_odds)
        add["Created"] = datetime.now().timestamp()
        st.session_state.bets = pd.concat([st.session_state.bets, ensure_columns(add)], ignore_index=True)
        st.success("Parlay appended!")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### All Bets")
    edited = st.data_editor(
        st.session_state.bets, key="bets_table_editor", use_container_width=True, hide_index=True,
        disabled=not edit_mode,
        column_config={
            "Result": st.column_config.SelectboxColumn("Result", options=["Win","Loss","Void"]),
        }
    )
    if edit_mode and st.button("Save changes to working copy"):
        st.session_state.bets = ensure_columns(edited.copy())
        st.success("Saved to session.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Stats -----
with tab_stats:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Stat Explorer")
    col1, col2 = st.columns([2,1])
    with col1:
        names = ["All"] + sorted(list(bets["Name"].dropna().unique()))
        who = st.selectbox("Filter by Name", names, index=0)
    with col2:
        topn = st.number_input("Top N extremes to show", min_value=1, max_value=100, value=10, step=1)

    filt = bets.copy()
    if who != "All":
        filt = filt[filt["Name"] == who]

    # Worst/Best beats by implied prob
    losers = filt[filt["Result"].str.lower()=="loss"].copy()
    losers["_prob"] = losers["Odds"].map(implied_prob)
    worst = losers.sort_values("_prob", ascending=False).head(topn)[["Name","Parlay #","Bet","Odds","Result","Sport"]]

    winners = filt[filt["Result"].str.lower()=="win"].copy()
    winners["_prob"] = winners["Odds"].map(implied_prob)
    best = winners.sort_values("_prob", ascending=True).head(topn)[["Name","Parlay #","Bet","Odds","Result","Sport"]]

    st.markdown("#### 🟥 Worst Beats")
    wb = worst.copy()
    if not wb.empty: wb["Odds"] = pd.to_numeric(wb["Odds"], errors="coerce")
    st.table(neon_style(wb, fmt_map={"Odds": "{:.0f}"}))

    st.markdown("#### 🟩 Best Beats")
    bb = best.copy()
    if not bb.empty: bb["Odds"] = pd.to_numeric(bb["Odds"], errors="coerce")
    st.table(neon_style(bb, fmt_map={"Odds": "{:.0f}"}))

    # ---------- Bet History (name-filtered) ----------
    st.markdown("#### 📜 Bet History (Filtered)")
    history = filt.copy().sort_values("Created", ascending=False)
    if not history.empty:
        history["When"] = history["Created"].map(lambda t: datetime.fromtimestamp(t).strftime("%m/%d %I:%M %p"))
        hist_cols = ["When","Name","Parlay #","Bet","Odds","Result","Sport","Drink Change","Dollars"]
        hist_cols = [c for c in hist_cols if c in history.columns]
        st.table(neon_style(history[hist_cols], fmt_map={"Odds": "{:.0f}", "Drink Change":"{:.3f}", "Dollars":"${:,.2f}"}))
    else:
        st.info("No bets to show for this filter.")

    # ---------- Highlights for selected name ----------
    st.markdown("#### 🔦 Highlights")
    def pick_extreme(df, res, ascending):
        part = df[df["Result"].str.lower()==res].copy()
        if part.empty: return None
        part["_prob"] = part["Odds"].map(implied_prob)
        row = part.sort_values("_prob", ascending=ascending).iloc[0]
        return {
            "Parlay #": int(row["Parlay #"]),
            "Bet":      str(row["Bet"]),
            "Odds":     int(row["Odds"]) if pd.notna(row["Odds"]) else np.nan,
            "Prob%":    round(row["_prob"]*100, 2),
            "Result":   row["Result"]
        }

    risky_win   = pick_extreme(filt, "win",  ascending=True)    # lowest prob win
    safe_win    = pick_extreme(filt, "win",  ascending=False)   # highest prob win
    risky_loss  = pick_extreme(filt, "loss", ascending=True)    # lowest prob loss
    shock_loss  = pick_extreme(filt, "loss", ascending=False)   # highest prob loss

    highlight_rows = []
    if risky_win:  highlight_rows.append({"Type":"Riskiest Win", **risky_win})
    if safe_win:   highlight_rows.append({"Type":"Safest Win", **safe_win})
    if risky_loss: highlight_rows.append({"Type":"Riskiest Loss", **risky_loss})
    if shock_loss: highlight_rows.append({"Type":"Most Unexpected Loss", **shock_loss})

    if highlight_rows:
        hi_df = pd.DataFrame(highlight_rows)
        st.table(neon_style(hi_df, fmt_map={"Odds":"{:.0f}", "Prob%":"{:.2f}"}))
    else:
        st.info("No highlight stats available for this filter.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- 🍻 MyCourt (drinks paid/received ledger) -----
with tab_court:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Vlone Troops — MyCourt")
    st.caption("Transfers bring both people **toward zero**: receiver’s Drink Count goes **down** by the amount, payer’s goes **up** by the amount.")

    standings = summary.copy()
    positive = sorted(list(standings.loc[standings["Drink Count"] > 1, "Name"]))
    negative = sorted(list(standings.loc[standings["Drink Count"] <= -1, "Name"]))

    col1, col2, col3 = st.columns([1.4,1.4,1])
    with col1:
        receiver = st.selectbox("Receiver (positive — will go DOWN)", options=positive if positive else ["—"], index=0)
    with col2:
        payer = st.selectbox("Payer (negative — will go UP)", options=negative if negative else ["—"], index=0)
    with col3:
        amt = st.number_input("Drinks", min_value=0.5, max_value=50.0, value=1.0, step=0.5)

    valid = edit_mode and positive and negative and receiver != "—" and payer != "—" and receiver != payer
    if st.button(f"Record Transfer{' (enter passcode to enable)' if not edit_mode else ''}", disabled=not valid):
        new = pd.DataFrame([{
            "From": payer,
            "To": receiver,
            "Amount": float(amt),
            "Created": datetime.now().timestamp()
        }])
        st.session_state.transfers = pd.concat([st.session_state.transfers, new], ignore_index=True)
        st.success(f"Recorded: {payer} → {receiver} ({amt} drinks)")
        st.rerun()

    st.markdown("#### Transfer Ledger")
    if st.session_state.transfers.empty:
        st.info("No transfers yet.")
    else:
        ledger = st.session_state.transfers.copy().sort_values("Created", ascending=False)
        ledger["When"] = ledger["Created"].map(lambda t: datetime.fromtimestamp(t).strftime("%m/%d %I:%M %p"))
        st.table(neon_style(ledger[["When","From","To","Amount"]], fmt_map={"Amount": "{:.2f}"}))
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Debug / computed -----
with st.expander("🔎 Computed (read-only)"):
    debug = bets[["Name","Parlay #","Bet","Odds","Result","Sport","Takeover","Dagger","Drink Change","Dollars"]].copy()
    st.table(neon_style(debug, fmt_map={
        "Odds":         "{:.0f}",
        "Drink Change": "{:.5f}",
        "Dollars":      "${:,.2f}",
    }))
