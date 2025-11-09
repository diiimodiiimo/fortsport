import os
import math
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- App config ----------------
st.set_page_config(page_title="FortSport", page_icon="🎮", layout="wide")
PASSCODE = st.secrets.get("PASSCODE", "dimodimo")
SEED_FILE = "seed.json"
AUTOSAVE_FILE = "fortsport_autosave.json"
BASE_STAKE = 100.0  # baseline logic assumes $100 per bet; we scale from this

# ---------------- Global theme (BLACK + NEON) ----------------
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
  --grid:#e9ffe9; /* light borders so both tables match */
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--panel-2) !important; color: var(--text) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] .stTextInput>div>div>input {
  background:#0b0f0c !important; color:var(--text) !important; border:1px solid #444 !important;
}

/* Hide the dummy autofill trap cleanly */
.hide-trap input{ height:0 !important; padding:0 !important; border:0 !important; background:transparent !important; color:transparent !important; }

/* Cards / metrics */
.card { background: var(--panel); border: 1px solid var(--grid); border-radius: 16px; padding: 16px; color: var(--text); box-shadow: 0 8px 24px rgba(0,0,0,.35); }
.metric { background: var(--panel); border: 1px solid var(--grid); border-radius: 16px; padding: 14px 16px; color: var(--text); box-shadow: 0 8px 20px rgba(57,255,20,.2);
  height: 150px; display:flex; flex-direction:column; justify-content:space-between; }
.metric h4{ margin:0 0 4px 0; font-weight:800; color:var(--muted); font-size:0.95rem; }
.metric .big{ font-size: 1.7rem; font-weight: 900; color: var(--text); }

/* Uniform metric grid row */
.metric-row{
  display:grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 18px;
  align-items: stretch;
}
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

/* Dataframes: light borders + no-wrap headers (keeps "Name"/"Record" on one line) */
div[data-testid="stDataFrame"] div[role="table"] { background: var(--panel); color: var(--text); border-radius: 12px; border: 1px solid var(--grid); }
div[data-testid="stDataFrame"] thead th { background: var(--panel-2) !important; color: var(--text) !important; border-bottom: 1px solid var(--grid) !important; }
div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th { border-bottom: 1px solid var(--grid) !important; }
div[data-testid="stDataFrame"] th > div { white-space: nowrap; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 12px; border: 1px solid var(--accent-dim);
  padding:.55rem 1rem; font-weight:800;
  background: linear-gradient(180deg, var(--accent), var(--accent-dim));
  color:#021; text-shadow: 0 1px 0 rgba(255,255,255,.2);
  box-shadow: 0 8px 20px rgba(57,255,20,.2);
}

/* Stake preset buttons – compact */
.stake-buttons .stButton>button{
  padding:.35rem .7rem !important;
  font-size:.9rem !important;
  border-radius:10px !important;
}

/* Black input look (filters / numbers) */
div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput>div>div>input {
  background:#0b0f0c !important; color:#e9ffe9 !important; border:1px solid #3d4a3f !important;
}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------------- Neon table styler ----------------
def neon_style(df, highlight_col: str | None = None,
               fmt_map: dict[str, str] | None = None,
               default_decimals: int | None = None):
    df = df.reset_index(drop=True)

    sty = (df.style
           .set_properties(**{
               "background-color":"#283329",
               "color":"#e9ffe9",
               "border-color":"#e9ffe9",
               "border-width":"1px",
           })
           .set_table_styles([
               {"selector":"th", "props":[("background","#1f2a21"),
                                          ("color","#e9ffe9"),
                                          ("border","1px solid #e9ffe9"),
                                          ("font-weight","800")]},
               {"selector":"td, th", "props":[("padding","10px 12px")]},
               {"selector":"th > div", "props":[("white-space","nowrap")]}
           ])
           .hide(axis="index")
    )

    # Only apply width tweaks if those columns exist (prevents KeyError on other tables)
    if "Name" in df.columns:
        sty = sty.set_properties(subset=pd.IndexSlice[:, ["Name"]],
                                 **{"min-width":"160px", "width":"160px"})
    if "Record" in df.columns:
        sty = sty.set_properties(subset=pd.IndexSlice[:, ["Record"]],
                                 **{"min-width":"110px", "width":"110px"})
    if "Active Streak" in df.columns:
        sty = sty.set_properties(subset=pd.IndexSlice[:, ["Active Streak"]],
                                 **{"max-width":"80px", "min-width":"80px", "width":"80px"})

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

# ---------------- Helpers ----------------
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

# --- DRINKS: symmetric logic ---
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
        if len(g) != 5:
            continue
        losers = g[g["Result"].str.lower() == "loss"]
        if losers.empty: continue
        idx = losers["Odds"].map(implied_prob).idxmax()
        dag.loc[idx] = True
    return dag

def _parlay_all_miss_discounts(df: pd.DataFrame) -> pd.Series:
    mult = pd.Series(1.0, index=df.index, dtype=float)
    if df.empty: return mult
    for _, g in df.groupby("Parlay #"):
        res = g["Result"].astype(str).str.lower()
        if len(g) > 0 and (res == "loss").all():
            n = len(g)
            factor = max(0.0, 1.0 - 0.10 * n)
            mult.loc[g.index] = np.where(res == "loss", factor, 1.0)
    return mult

def apply_takeover(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("Created").copy()
    out["Takeover"] = False
    out["Drink Change"] = 0.0
    discount = _parlay_all_miss_discounts(df)

    for _, g in out.groupby("Name", sort=False):
        streak = 0
        for idx, row in g.iterrows():
            takeover = streak >= 3
            out.at[idx,"Takeover"] = takeover

            val = base_drink_change(row["Odds"], row["Result"])

            if takeover and str(row["Result"]).lower() == "win":
                val *= 2.0
            if str(row["Result"]).lower() == "loss":
                val *= discount.get(idx, 1.0)

            out.at[idx,"Drink Change"] = val

            if str(row["Result"]).lower() == "win":
                streak += 1
            elif str(row["Result"]).lower() == "loss":
                streak = 0
    return out.sort_index()

def dollars_pnl_100(odds: float, result: str) -> float:
    r = (str(result) if result is not None else "").strip().lower()
    if r not in {"win", "loss"} or odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return 0.0
    if r == "loss":
        return -BASE_STAKE
    if odds > 0:
        return float(odds)
    return BASE_STAKE * (100.0 / abs(odds))

def compute_all(bets_raw: pd.DataFrame):
    df = bets_raw.copy()
    df["Odds"] = df["Odds"].map(clean_odds)
    df["Result"] = df["Result"].fillna("").astype(str).str.title()

    df = apply_takeover(df)
    df["Dagger"] = compute_daggers(df)
    df["Dollars"] = [dollars_pnl_100(o, r) for o, r in zip(df["Odds"], df["Result"])]
    return df

def summarise(df: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        base = pd.DataFrame(columns=[
            "Name","Drink Count","Record","Active Streak","Cumulative Odds","Parlays Won","Dollars Won",
            "Drinks Paid Out","Drinks Received"
        ])
    else:
        tmp = df.copy()
        tmp["win"]  = (tmp["Result"].str.lower()=="win").astype(int)
        tmp["loss"] = (tmp["Result"].str.lower()=="loss").astype(int)

        size = tmp.groupby("Parlay #")["Result"].count()
        all_win = tmp.groupby("Parlay #")["win"].apply(lambda s: (s == 1).all())
        parlay_win = ((size >= 3) & all_win)
        tmp = tmp.merge(parlay_win.rename("ParlayWon"), on="Parlay #", how="left").fillna({"ParlayWon":False})

        # Active streak per Name
        streaks = []
        for name, g in tmp.sort_values("Created").groupby("Name"):
            s = 0
            for r in g["Result"].astype(str).str.lower():
                if r == "win": s = (s+1) if s >= 0 else 1
                elif r == "loss": s = (s-1) if s <= 0 else -1
            if s > 0: tag = f"W{s}"
            elif s < 0: tag = f"L{abs(s)}"
            else: tag = "L1"
            streaks.append((name, tag))
        streak_df = pd.DataFrame(streaks, columns=["Name","Active Streak"]) if streaks else pd.DataFrame(columns=["Name","Active Streak"])

        agg = (tmp.groupby("Name", as_index=False)
                 .agg(Drink_Sum=("Drink Change","sum"),
                      Wins=("win","sum"),
                      Losses=("loss","sum"),
                      Cumulative_Odds=("Odds", lambda s: float(np.nanmean(s)) if len(s) else np.nan),
                      Parlays_Won=("ParlayWon","sum"),
                      Dollars_Won=("Dollars","sum"),
                      Daggers=("Dagger","sum")))
        agg["Record"] = agg["Wins"].astype(str) + "-" + agg["Losses"].astype(str)
        base = agg[["Name","Drink_Sum","Daggers","Record","Cumulative_Odds","Parlays_Won","Dollars_Won"]]
        base = base.merge(streak_df, on="Name", how="left")
        base = base.rename(columns={"Cumulative_Odds":"Cumulative Odds","Parlays_Won":"Parlays Won","Dollars_Won":"Dollars Won"})

    # Transfers
    if transfers is None or transfers.empty:
        paid = pd.Series(0.0, index=base["Name"], name="Drinks Paid Out") if not base.empty else pd.Series(name="Drinks Paid Out", dtype=float)
        recv = pd.Series(0.0, index=base["Name"], name="Drinks Received") if not base.empty else pd.Series(name="Drinks Received", dtype=float)
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

    final = base[["Name","Drink Count","Record","Active Streak","Cumulative Odds","Parlays Won","Dollars Won",
                  "Drinks Paid Out","Drinks Received"]].sort_values("Drink Count", ascending=False, kind="mergesort")
    return final

def total_parlays_won(df: pd.DataFrame) -> int:
    if df.empty: return 0
    size = df.groupby("Parlay #")["Result"].count()
    all_win = df.groupby("Parlay #")["Result"].apply(lambda s: (s.str.lower()=="win").all())
    return int(((size >= 3) & all_win).sum())

# ---------------- Bundle save/load ----------------
def pack_state():
    return {"bets": ensure_columns(st.session_state.bets.copy()),
            "transfers": st.session_state.transfers.copy()}

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

# ---------------- Session init ----------------
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

if "stake_confirmed" not in st.session_state:
    st.session_state.stake_confirmed = BASE_STAKE  # default $100

# ---------------- Sidebar ----------------
st.sidebar.markdown("## ⚙️ Settings")


pw = st.sidebar.text_input("Passcode to edit", type="password", placeholder="Enter passcode", key="real_pw")
edit_mode = (pw.strip() == PASSCODE)
st.sidebar.write(f"**Mode:** {'🟢 Edit' if edit_mode else '🔵 View'}")
st.sidebar.caption(
    "Drinks: symmetric ±payout • Dagger: worst loss (5-bet) −1 • Takeover: 2× wins after 3-win streak • "
    "All-miss parlay loss discount (drinks only): 10% × legs (capped at 100%) • Cash P&L scales linearly with the chosen stake (baseline $100)."
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

# Save current working copy as the new base (seed.json)
if st.sidebar.button("📌 Make current version the base (overwrite seed.json)", disabled=not edit_mode):
    save_seed_bundle()
    st.sidebar.success("Base updated (wrote current bets & transfers to seed.json).")


# ---------------- Compute ----------------
bets = compute_all(st.session_state.bets)

def scaled_dollars(series: pd.Series) -> pd.Series:
    mult = float(st.session_state.stake_confirmed) / BASE_STAKE
    return series * mult

# ---------------- UI ----------------
tab_dash, tab_stats, tab_bets, tab_court = st.tabs(["🏆 Dashboard", "📈 Stats", "📋 Bets", "🍻 MyCourt"])

# ----- Dashboard -----
with tab_dash:
    dollars_total = float(scaled_dollars(bets["Dollars"]).sum())
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric"><h4>Total Bets</h4><div class="big">{len(bets)}</div></div>
          <div class="metric"><h4>Parlays</h4><div class="big">{bets["Parlay #"].nunique()}</div></div>
          <div class="metric"><h4>Parlays Won</h4><div class="big">{total_parlays_won(bets)}</div></div>
          <div class="metric"><h4>Total $ Won</h4><div class="big">${dollars_total:,.2f}</div></div>
          <div class="metric"><h4>Last Updated</h4><div class="big">{datetime.now().strftime("%m/%d")}<br>{datetime.now().strftime("%I:%M %p")}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Stake PRESET buttons (only on Dashboard)
    st.write("#### Stake per bet ($)")
    st.markdown('<div class="stake-buttons">', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("$5"):
        st.session_state.stake_confirmed = 5.0
        st.rerun()
    if b2.button("$10"):
        st.session_state.stake_confirmed = 10.0
        st.rerun()
    if b3.button("$25"):
        st.session_state.stake_confirmed = 25.0
        st.rerun()
    if b4.button("$100"):
        st.session_state.stake_confirmed = 100.0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"All dollar amounts scale to the selected stake. Current stake: **${st.session_state.stake_confirmed:.2f}**.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Squad Leaderboard")
    summary = summarise(bets, st.session_state.transfers).copy()
    if "Dollars Won" in summary.columns:
        summary["Dollars Won"] = scaled_dollars(summary["Dollars Won"])
    st.table(neon_style(
        summary,
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

# ----- Stats -----
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Stat Explorer")

# 1) Pick filters
sports_all = ["All"] + sorted(list(bets["Sport"].dropna().unique()))
sport_filter = st.selectbox("Select Sport", sports_all, index=0)

# (Optional) Make the name list aware of the chosen sport
if sport_filter == "All":
    names_all = ["All"] + sorted(list(bets["Name"].dropna().unique()))
else:
    names_all = ["All"] + sorted(list(bets.loc[bets["Sport"] == sport_filter, "Name"].dropna().unique()))

col1, col2 = st.columns([2,1])
with col1:
    who = st.selectbox("Filter by Name", names_all, index=0)
with col2:
    topn = st.number_input("Top N extremes to show", min_value=1, max_value=100, value=10, step=1)

# 2) Apply BOTH filters to a single dataframe
filt = bets.copy()
if sport_filter != "All":
    filt = filt[filt["Sport"] == sport_filter]
if who != "All":
    filt = filt[filt["Name"] == who]

# 3) Performance by Sport (now uses the same filt)
st.markdown("### 📊 Performance by Sport")
if not filt.empty:
    sport_stats = []
    for name in sorted(filt["Name"].dropna().unique()):
        person_bets = filt[filt["Name"] == name]
        wins = (person_bets["Result"].str.lower() == "win").sum()
        losses = (person_bets["Result"].str.lower() == "loss").sum()
        total = len(person_bets)
        win_pct = (wins / total * 100) if total > 0 else 0.0
        avg_odds = pd.to_numeric(person_bets["Odds"], errors="coerce").mean()
        drinks = person_bets["Drink Change"].sum()
        dollars = scaled_dollars(person_bets["Dollars"]).sum()
        sport_stats.append({
            "Name": name,
            "Record": f"{wins}-{losses}",
            "Win %": win_pct,
            "Total Bets": total,
            "Avg Odds": avg_odds,
            "Drinks": drinks,
            "Dollars": dollars
        })
    sport_df = pd.DataFrame(sport_stats).sort_values("Win %", ascending=False)
    st.table(neon_style(
        sport_df,
        highlight_col="Win %",
        fmt_map={
            "Win %": "{:.1f}%",
            "Total Bets": "{:.0f}",
            "Avg Odds": "{:.0f}",
            "Drinks": "{:.2f}",
            "Dollars": "${:,.2f}"
        }
    ))
else:
    st.info("No bets found for this filter.")

# 4) Worst / Best Beats (now use filt)
losers = filt[filt["Result"].str.lower() == "loss"].copy()
losers["_prob"] = losers["Odds"].map(implied_prob)
worst = losers.sort_values("_prob", ascending=False).head(topn)[["Name","Parlay #","Bet","Odds","Result","Sport"]]

winners = filt[filt["Result"].str.lower() == "win"].copy()
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

# 5) Filtered History (now uses filt)
st.markdown("#### 📜 Filtered History")
hist = filt.copy().sort_values("Created", ascending=False)
if not hist.empty:
    hist["When"] = hist["Created"].map(lambda t: datetime.fromtimestamp(t).strftime("%m/%d %I:%M %p"))
    view_cols = [c for c in ["When","Name","Parlay #","Bet","Odds","Result","Sport"] if c in hist.columns]
    if "Odds" in hist.columns:
        hist["Odds"] = pd.to_numeric(hist["Odds"], errors="coerce")
    st.table(neon_style(hist[view_cols], fmt_map={"Odds": "{:.0f}"}))
else:
    st.info("No bets match this filter yet.")

# 6) Spotlight — Wins & Losses (now uses filt)
st.markdown("### Spotlight — Wins & Losses")
fav_wins   = filt[(filt["Result"].str.lower()=="win")  & (pd.to_numeric(filt["Odds"], errors="coerce") < 0)].copy()
dog_wins   = filt[(filt["Result"].str.lower()=="win")  & (pd.to_numeric(filt["Odds"], errors="coerce") > 0)].copy()
fav_losses = filt[(filt["Result"].str.lower()=="loss") & (pd.to_numeric(filt["Odds"], errors="coerce") < 0)].copy()
dog_losses = filt[(filt["Result"].str.lower()=="loss") & (pd.to_numeric(filt["Odds"], errors="coerce") > 0)].copy()

for df_ in (fav_wins, dog_wins, fav_losses, dog_losses):
    if not df_.empty:
        df_["Odds"] = pd.to_numeric(df_["Odds"], errors="coerce")

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### ✅ Chalk **Wins** (favorite wins)")
    tbl = fav_wins.sort_values("Odds").head(topn)[["Name","Parlay #","Bet","Odds","Sport"]]
    st.table(neon_style(tbl, fmt_map={"Odds": "{:.0f}"}))
with c4:
    st.markdown("#### 🎲 Longshot **Misses** (largest +odds losses)")
    tbl = dog_losses.sort_values("Odds", ascending=False).head(topn)[["Name","Parlay #","Bet","Odds","Sport"]]
    st.table(neon_style(tbl, fmt_map={"Odds": "{:.0f}"}))

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
    if st.button("Append Parlay", disabled=not edit_mode):
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

# ----- 🍻 MyCourt -----
with tab_court:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Vlone Troops — MyCourt")
    st.caption("Transfers bring both people **toward zero**: receiver’s Drink Count goes **down** by the amount, payer’s goes **up** by the amount.")

    standings = summarise(bets, st.session_state.transfers).copy()
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
    if st.button("Record Transfer", disabled=not valid):
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
    debug["Dollars"] = scaled_dollars(debug["Dollars"])
    st.table(neon_style(debug, fmt_map={
        "Odds":         "{:.0f}",
        "Drink Change": "{:.5f}",
        "Dollars":      "${:,.2f}",
    }))
