"""
Swing-Screener Dashboard — Setup A (Trend-Pullback), Setup B (Mean Reversion), Flat-Top-Breakout
Gratis-Hosting: Streamlit Community Cloud (share.streamlit.io)
"""

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------- Watchlist ---------------------------------

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA",
    "AVGO", "CRM", "NFLX", "QCOM", "ORCL", "ADBE", "UBER", "SHOP",
    "PYPL", "ABNB", "PLTR", "CAT", "DE", "GS", "JPM", "XOM", "COCO", "VLY",
    "SAP.DE", "SIE.DE", "ALV.DE", "MUV2.DE", "RHM.DE", "IFX.DE",
    "ASML.AS", "AIR.PA", "MC.PA", "SU.PA",
]
BENCHMARK = "SPY"

# ----------------------------- Indikatoren -------------------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n):
    d = close.diff()
    ru = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    rd = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def prepare(df):
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI2"] = rsi(df["Close"], 2)
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["High20"] = df["High"].rolling(20).max()
    return df.dropna()

# ----------------------------- Setup-Erkennung ---------------------------

def detect(df, ticker, bench_perf3m):
    """Prueft die LETZTE Kerze auf alle drei Setups. Gibt Liste von Kandidaten zurueck."""
    out = []
    if len(df) < 220:
        return out
    r = df.iloc[-1]
    i = len(df) - 1
    atr_pct = r.ATR14 / r.Close * 100
    if not (1.0 <= atr_pct <= 8.0):
        return out
    perf3m = (r.Close / df.Close.iloc[-63] - 1) * 100 if len(df) > 63 else 0
    rs = perf3m - bench_perf3m

    trend_ok = (
        r.EMA50 > r.EMA200
        and df.EMA50.iloc[-1] > df.EMA50.iloc[-6]
        and df.EMA200.iloc[-1] > df.EMA200.iloc[-6]
    )

    # --- Setup A: Trend-Pullback ---
    if trend_ok and df.High.iloc[-20:].max() >= df.High20.iloc[-1] * 0.999:
        band_hi, band_lo = max(r.EMA20, r.EMA50), min(r.EMA20, r.EMA50)
        if r.Low <= band_hi and r.Close >= band_lo and 35 <= r.RSI14 <= 55:
            trigger = r.High * 1.001
            pull_low = df.Low.iloc[-8:].min()
            stop = min(pull_low, trigger - 1.5 * r.ATR14)
            target = df.High20.iloc[-1]
            crv = (target - trigger) / (trigger - stop) if trigger > stop else 0
            score = min(rs, 30) + crv * 10 + (10 if r.Close > r.Open else 0)
            out.append(dict(Ticker=ticker, Setup="A Trend-Pullback", Kurs=r.Close,
                            Trigger=trigger, Stop=stop, Ziel=target, CRV=crv,
                            ATRpct=atr_pct, RS3M=rs, Score=score))

    # --- Setup B: Mean Reversion ---
    if r.Close > r.EMA200 and r.RSI2 < 10 and (r.EMA20 - r.Close) >= 1.5 * r.ATR14:
        trigger = r.High * 1.001
        stop = trigger - 2.0 * r.ATR14
        target = r.EMA20
        crv = (target - trigger) / (trigger - stop) if trigger > stop else 0
        score = min(rs, 30) + (10 - r.RSI2) + (r.EMA20 - r.Close) / r.ATR14 * 5
        out.append(dict(Ticker=ticker, Setup="B Mean-Reversion", Kurs=r.Close,
                        Trigger=trigger, Stop=stop, Ziel=target, CRV=crv,
                        ATRpct=atr_pct, RS3M=rs, Score=score))

    # --- Flat-Top-Breakout: enge Konsolidierung unter 20T-Hoch ---
    if trend_ok:
        hi20 = df.High20.iloc[-1]
        touches = (df.High.iloc[-15:] >= hi20 * 0.985).sum()
        dist = (hi20 - r.Close) / r.Close * 100
        range15 = (df.High.iloc[-15:].max() - df.Low.iloc[-15:].min()) / r.Close * 100
        if touches >= 2 and 0 <= dist <= 3.0 and range15 <= 12:
            trigger = hi20 * 1.002
            stop = trigger - 1.5 * r.ATR14
            target = trigger + (hi20 - df.Low.iloc[-15:].min())  # Measured Move
            crv = (target - trigger) / (trigger - stop)
            score = min(rs, 30) + touches * 3 + (3 - dist) * 5
            out.append(dict(Ticker=ticker, Setup="Flat-Top-Breakout", Kurs=r.Close,
                            Trigger=trigger, Stop=stop, Ziel=target, CRV=crv,
                            ATRpct=atr_pct, RS3M=rs, Score=score))
    return out

# ----------------------------- Daten laden -------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker, period="2y"):
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ----------------------------- Chart -------------------------------------

def make_chart(df, row):
    import plotly.graph_objects as go
    d = df.iloc[-130:]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low,
                                 close=d.Close, name="Kurs", showlegend=False))
    for col, color in [("EMA20", "#E8A04A"), ("EMA50", "#d9534f"), ("EMA200", "#5bc0de")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col,
                                 line=dict(color=color, width=1.3)))
    levels = [("Trigger (Stop-Buy)", row["Trigger"], "#2ecc71", "solid"),
              ("Stop-Loss", row["Stop"], "#e74c3c", "dash"),
              ("Ziel", row["Ziel"], "#f1c40f", "dot")]
    for name, y, color, dash in levels:
        fig.add_hline(y=y, line_color=color, line_dash=dash,
                      annotation_text=f"{name}: {y:.2f}", annotation_font_color=color)
    fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, title=f"{row['Ticker']} — {row['Setup']}")
    return fig

# ----------------------------- UI ----------------------------------------

def main():
    st.set_page_config(page_title="Swing-Screener", layout="wide", page_icon="📈")
    st.title("📈 Swing-Screener — Stocks in Play")
    st.caption("Setup A: Trend-Pullback · Setup B: Mean Reversion · Flat-Top-Breakout — Daily, Long-only")

    with st.sidebar:
        st.header("Einstellungen")
        konto = st.number_input("Kontogröße €", 100, 1_000_000, 1000, step=100)
        risk_pct = st.number_input("Risiko pro Trade %", 0.25, 2.0, 1.0, step=0.25)
        fx = st.number_input("EUR/USD", 0.8, 1.6, 1.15, step=0.01)
        min_crv = st.slider("Min. CRV", 0.5, 3.0, 1.2, 0.1)
        tickers_txt = st.text_area("Watchlist (kommagetrennt)", ", ".join(DEFAULT_TICKERS), height=140)
        st.caption("⚠️ Earnings-Termine manuell prüfen — keine Trades <10 Handelstage vor Earnings!")

    tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]

    if st.button("🔍 Scan starten", type="primary", use_container_width=True):
        bench = load(BENCHMARK)
        bench_perf = (bench.Close.iloc[-1] / bench.Close.iloc[-63] - 1) * 100 if len(bench) > 63 else 0
        rows, frames = [], {}
        bar = st.progress(0.0, text="Lade Kursdaten …")
        for k, tk in enumerate(tickers):
            try:
                df = load(tk)
                if len(df) >= 220:
                    p = prepare(df)
                    frames[tk] = p
                    rows += detect(p, tk, bench_perf)
            except Exception:
                pass
            bar.progress((k + 1) / len(tickers), text=f"Scanne {tk} …")
        bar.empty()
        st.session_state["results"] = rows
        st.session_state["frames"] = frames

    rows = st.session_state.get("results")
    if rows is None:
        st.info("Watchlist links anpassen und **Scan starten** drücken.")
        return
    if not rows:
        st.warning("Heute keine Setups auf der Watchlist. Das ist okay — kein Setup heißt kein Trade.")
        return

    res = pd.DataFrame(rows)
    res = res[res.CRV >= min_crv].sort_values("Score", ascending=False).head(10).reset_index(drop=True)
    if res.empty:
        st.warning(f"Setups gefunden, aber keines erfüllt CRV ≥ {min_crv}.")
        return

    res["Risiko/Stk $"] = res.Trigger - res.Stop
    res["Stück"] = (konto * risk_pct / 100 / fx / res["Risiko/Stk $"]).round(2)
    res["Position $"] = (res["Stück"] * res.Trigger).round(0)

    st.subheader(f"Top {len(res)} Setups heute")
    show = res[["Ticker", "Setup", "Kurs", "Trigger", "Stop", "Ziel", "CRV",
                "ATRpct", "RS3M", "Stück", "Position $", "Score"]].copy()
    st.dataframe(show.style.format({"Kurs": "{:.2f}", "Trigger": "{:.2f}", "Stop": "{:.2f}",
                                    "Ziel": "{:.2f}", "CRV": "{:.2f}", "ATRpct": "{:.1f}%",
                                    "RS3M": "{:+.1f}%", "Stück": "{:.2f}", "Position $": "{:.0f}",
                                    "Score": "{:.0f}"}),
                 use_container_width=True, height=min(420, 60 + 38 * len(res)))

    sel = st.selectbox("Chart anzeigen für:",
                       [f"{r.Ticker} — {r.Setup}" for r in res.itertuples()])
    row = res.iloc[[f"{r.Ticker} — {r.Setup}" for r in res.itertuples()].index(sel)]
    st.plotly_chart(make_chart(st.session_state["frames"][row["Ticker"]], row),
                    use_container_width=True)
    st.caption("Levels: 🟢 Trigger = Stop-Buy-Order · 🔴 Stop-Loss · 🟡 Ziel. "
               "Entry NUR per Stop-Buy über dem Trigger — nie vorgreifen.")

if __name__ == "__main__":
    main()
