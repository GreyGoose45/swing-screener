"""
Swing-Screener Dashboard v2
Universum: S&P 500 / Dow Jones 30 / eigene Liste — gefiltert nach Plan-Kriterien
Setups: A Trend-Pullback · B Mean Reversion · Flat-Top-Breakout
"""

import numpy as np
import pandas as pd
import streamlit as st

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA", "AVGO",
    "CRM", "NFLX", "QCOM", "ORCL", "ADBE", "UBER", "PYPL", "ABNB", "PLTR",
    "CAT", "DE", "GS", "JPM", "XOM", "COCO", "VLY",
]
BENCHMARK = "SPY"
MIN_AVG_VOLUME = 1_000_000   # Plan-Kriterium: Liquidität
ATR_PCT_MIN, ATR_PCT_MAX = 2.0, 6.0  # Plan-Kriterium: Volatilitätsfenster

# ----------------------------- Universum ---------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_universe(name):
    try:
        if name == "S&P 500":
            t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
            return [s.replace(".", "-") for s in t["Symbol"].tolist()]
        if name == "Dow Jones 30":
            tables = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
            for t in tables:
                if "Symbol" in t.columns:
                    return [s.replace(".", "-") for s in t["Symbol"].tolist()]
    except Exception:
        pass
    return FALLBACK_TICKERS

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
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    return df.dropna()

# ----------------------------- Setup-Erkennung ---------------------------

def detect(df, ticker, bench_perf3m):
    out = []
    if len(df) < 200:
        return out
    r = df.iloc[-1]
    # Plan-Filter: Liquidität & Volatilitätsfenster
    if r.VolAvg20 < MIN_AVG_VOLUME:
        return out
    atr_pct = r.ATR14 / r.Close * 100
    if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX):
        return out

    perf3m = (r.Close / df.Close.iloc[-63] - 1) * 100 if len(df) > 63 else 0.0
    rs = float(np.nan_to_num(perf3m - bench_perf3m))
    vol_faktor = float(r.Volume / r.VolAvg20) if r.VolAvg20 > 0 else 0.0

    trend_ok = (
        r.EMA50 > r.EMA200
        and df.EMA50.iloc[-1] > df.EMA50.iloc[-6]
        and df.EMA200.iloc[-1] > df.EMA200.iloc[-6]
    )

    base = dict(Ticker=ticker, Kurs=float(r.Close), ATRpct=atr_pct,
                RS3M=rs, VolFaktor=vol_faktor)

    # --- Setup A: Trend-Pullback ---
    if trend_ok and df.High.iloc[-20:].max() >= df.High20.iloc[-1] * 0.999:
        band_hi, band_lo = max(r.EMA20, r.EMA50), min(r.EMA20, r.EMA50)
        # Pullback-Laenge: Tage seit letztem 20T-Hoch
        pull_tage = int(len(df) - 1 - np.argmax(df.High.iloc[-20:].values) - (len(df) - 20))
        if r.Low <= band_hi and r.Close >= band_lo and 35 <= r.RSI14 <= 55:
            trigger = float(r.High * 1.001)
            pull_low = float(df.Low.iloc[-8:].min())
            stop = float(min(pull_low, trigger - 1.5 * r.ATR14))
            target = float(df.High20.iloc[-1])
            crv = (target - trigger) / (trigger - stop) if trigger > stop else 0.0
            score = round(min(rs, 30) + crv * 10 + (10 if r.Close > r.Open else 0) + 20, 0)
            out.append(base | dict(Setup="A Trend-Pullback", Trigger=trigger, Stop=stop,
                                   Ziel=target, CRV=crv, Score=score,
                                   m_band_lo=float(band_lo), m_band_hi=float(band_hi),
                                   m_rsi14=float(r.RSI14), m_pull_tage=max(pull_tage, 1),
                                   m_pull_low=pull_low))

    # --- Setup B: Mean Reversion ---
    dist_atr = float((r.EMA20 - r.Close) / r.ATR14) if r.ATR14 > 0 else 0.0
    if r.Close > r.EMA200 and r.RSI2 < 10 and dist_atr >= 1.5:
        trigger = float(r.High * 1.001)
        stop = float(trigger - 2.0 * r.ATR14)
        target = float(r.EMA20)
        crv = (target - trigger) / (trigger - stop) if trigger > stop else 0.0
        score = round(min(rs, 30) + (10 - float(r.RSI2)) + dist_atr * 5 + 20, 0)
        out.append(base | dict(Setup="B Mean-Reversion", Trigger=trigger, Stop=stop,
                               Ziel=target, CRV=crv, Score=score,
                               m_rsi2=float(r.RSI2), m_dist_atr=dist_atr,
                               m_ema20=float(r.EMA20)))

    # --- Flat-Top-Breakout ---
    if trend_ok:
        hi20 = float(df.High20.iloc[-1])
        touches = int((df.High.iloc[-15:] >= hi20 * 0.985).sum())
        dist = (hi20 - r.Close) / r.Close * 100
        lo15 = float(df.Low.iloc[-15:].min())
        range15 = (df.High.iloc[-15:].max() - lo15) / r.Close * 100
        if touches >= 2 and 0 <= dist <= 3.0 and range15 <= 12:
            trigger = hi20 * 1.002
            stop = float(trigger - 1.5 * r.ATR14)
            target = float(trigger + (hi20 - lo15))
            crv = (target - trigger) / (trigger - stop)
            score = round(min(rs, 30) + touches * 3 + (3 - dist) * 5 + 15, 0)
            out.append(base | dict(Setup="Flat-Top-Breakout", Trigger=trigger, Stop=stop,
                                   Ziel=target, CRV=crv, Score=score,
                                   m_touches=touches, m_range15=float(range15),
                                   m_hi20=hi20, m_dist=float(dist)))
    return out

# ----------------------------- Begründung --------------------------------

def explain(row):
    s, t = row["Setup"], row["Ticker"]
    crv, rs, vf = row["CRV"], row["RS3M"], row["VolFaktor"]
    common = (f"\n\n**Risiko-Setup:** Stop-Buy-Order bei **{row['Trigger']:.2f}** — Entry nur, "
              f"wenn der Markt die Idee bestätigt, nie vorgreifen. Stop-Loss **{row['Stop']:.2f}**, "
              f"Ziel **{row['Ziel']:.2f}** → CRV **{crv:.2f}**. "
              f"Relative Stärke vs. S&P 500 (3M): **{rs:+.1f}%**. "
              f"Heutiges Volumen: **{vf:.1f}×** des 20-Tage-Schnitts."
              f"\n\n⚠️ Vor der Order: Earnings-Termin prüfen (mind. 10 Handelstage Abstand).")
    if s.startswith("A"):
        return (f"**Trade-Idee {t} — Trend-Fortsetzung nach Pullback:** Der Aufwärtstrend ist intakt "
                f"(EMA 50 über EMA 200, beide steigend) und die Aktie hat innerhalb der letzten "
                f"20 Tage ein neues Hoch markiert — Käufer haben die Kontrolle. Seit ~{row['m_pull_tage']} Tagen "
                f"läuft eine Korrektur, die heute das EMA-20/50-Band ({row['m_band_lo']:.2f}–{row['m_band_hi']:.2f}) "
                f"erreicht hat; der Schlusskurs hält über dem Band, d. h. die Zone wird verteidigt. "
                f"RSI(14) bei {row['m_rsi14']:.0f} zeigt einen gesunden Pullback, keinen Trendbruch. "
                f"Die Idee: Wiederaufnahme des Trends kaufen, sobald der Kurs das heutige Hoch überwindet. "
                f"Der Stop liegt unter dem Pullback-Tief ({row['m_pull_low']:.2f}) bzw. 1,5×ATR — dort wäre "
                f"die Pullback-These objektiv widerlegt. Erstes Ziel ist das alte Hoch; darüber Trailing per EMA 20."
                + common)
    if s.startswith("B"):
        return (f"**Trade-Idee {t} — Mean Reversion im Aufwärtstrend:** Die Aktie notiert über der "
                f"EMA 200 (übergeordneter Aufwärtstrend intakt), ist aber kurzfristig stark überverkauft: "
                f"RSI(2) bei {row['m_rsi2']:.1f} und der Kurs liegt {row['m_dist_atr']:.1f}×ATR unter der "
                f"EMA 20 ({row['m_ema20']:.2f}) — eine statistisch überdehnte Bewegung, die in intakten "
                f"Trends meist schnell zur EMA 20 zurückschnappt. Die Idee: den Rebound kaufen, sobald der "
                f"Kurs das heutige Hoch zurückerobert (erste Käufer-Bestätigung). Der Stop liegt bewusst "
                f"weit (2×ATR), weil Mean-Reversion-Trades Rauschen aushalten müssen. Exit am Ziel EMA 20, "
                f"bei RSI(2) > 70 — oder nach spätestens 5 Tagen (Zeit-Stop): was nicht schnell bounced, "
                f"ist gefährlich." + common)
    return (f"**Trade-Idee {t} — Flat-Top-Breakout:** Die Aktie konsolidiert in einem intakten "
            f"Aufwärtstrend seit ~3 Wochen in einer engen Range ({row['m_range15']:.1f}% Höhe) direkt "
            f"unter einem glatten Widerstand bei {row['m_hi20']:.2f}, der bereits {row['m_touches']}× "
            f"getestet wurde — Angebot wird dort sichtbar abgebaut, der Kurs steht nur noch "
            f"{row['m_dist']:.1f}% unter der Marke. Enge Konsolidierungen nach starken Anstiegen lösen "
            f"sich statistisch häufiger nach oben auf (Volatilitätskontraktion). Die Idee: den Ausbruch "
            f"über den Deckel kaufen — idealerweise bestätigt durch Volumen ≥1,5× des 20-Tage-Schnitts "
            f"(aktuell {row['VolFaktor']:.1f}×). Ziel ist der Measured Move (Range-Höhe auf den Ausbruch "
            f"projiziert); über dem alten Hoch gibt es keinen Widerstand mehr → Trailing per EMA 20 statt "
            f"fixem Gewinnmitnehmen." + common)

# ----------------------------- Daten laden -------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_chunk(tickers, period="2y"):
    import yfinance as yf
    df = yf.download(list(tickers), period=period, auto_adjust=True,
                     progress=False, group_by="ticker", threads=True)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_single(ticker, period="2y"):
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ----------------------------- Chart -------------------------------------

def make_chart(df, row):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    d = df.iloc[-130:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low,
                                 close=d.Close, name="Kurs", showlegend=False,
                                 increasing_line_color="#2ecc71",
                                 decreasing_line_color="#e74c3c"), row=1, col=1)
    for col, color in [("EMA20", "#E8A04A"), ("EMA50", "#d9534f"), ("EMA200", "#5bc0de")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col,
                                 line=dict(color=color, width=1.3)), row=1, col=1)
    for name, y, color, dash in [("Trigger", row["Trigger"], "#2ecc71", "solid"),
                                 ("Stop", row["Stop"], "#e74c3c", "dash"),
                                 ("Ziel", row["Ziel"], "#f1c40f", "dot")]:
        fig.add_hline(y=y, line_color=color, line_dash=dash, row=1, col=1,
                      annotation_text=f"{name}: {y:.2f}", annotation_font_color=color)
    vol_colors = np.where(d.Close >= d.Open, "#1e7e4e", "#922b21")
    fig.add_trace(go.Bar(x=d.index, y=d.Volume, marker_color=vol_colors,
                         name="Volumen", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.VolAvg20, name="Vol Ø20",
                             line=dict(color="#E8A04A", width=1.2)), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=640,
                      paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False,
                      title=f"{row['Ticker']} — {row['Setup']}",
                      legend=dict(orientation="h", y=1.05))
    return fig

# ----------------------------- UI ----------------------------------------

def main():
    st.set_page_config(page_title="Swing-Screener", layout="wide", page_icon="📈")
    st.title("📈 Swing-Screener — Stocks in Play")
    st.caption("Setup A: Trend-Pullback · Setup B: Mean Reversion · Flat-Top-Breakout — "
               "Daily, Long-only · Filter: Ø-Vol > 1 Mio · ATR 2–6%")

    with st.sidebar:
        st.header("Einstellungen")
        universe = st.radio("Universum", ["S&P 500", "Dow Jones 30", "Eigene Liste"])
        custom = ""
        if universe == "Eigene Liste":
            custom = st.text_area("Ticker (kommagetrennt)", ", ".join(FALLBACK_TICKERS), height=120)
        konto = st.number_input("Kontogröße €", 100, 1_000_000, 1000, step=100)
        risk_pct = st.number_input("Risiko pro Trade %", 0.25, 2.0, 1.0, step=0.25)
        fx = st.number_input("EUR/USD", 0.8, 1.6, 1.15, step=0.01)
        min_crv = st.slider("Min. CRV", 0.5, 3.0, 1.2, 0.1)
        st.caption("⚠️ Earnings-Termine manuell prüfen — keine Trades <10 Handelstage vor Earnings!")

    if st.button("🔍 Scan starten", type="primary", use_container_width=True):
        tickers = ([t.strip().upper() for t in custom.split(",") if t.strip()]
                   if universe == "Eigene Liste" else get_universe(universe))
        bench = load_single(BENCHMARK)
        bench_perf = float((bench.Close.iloc[-1] / bench.Close.iloc[-63] - 1) * 100) if len(bench) > 63 else 0.0

        rows, frames = [], {}
        chunks = [tickers[i:i + 50] for i in range(0, len(tickers), 50)]
        bar = st.progress(0.0, text=f"Scanne {len(tickers)} Aktien …")
        for k, chunk in enumerate(chunks):
            try:
                data = load_chunk(tuple(chunk))
                for tk in chunk:
                    try:
                        sub = data[tk].dropna(how="all") if len(chunk) > 1 else data.copy()
                        if len(sub) < 220:
                            continue
                        p = prepare(sub)
                        hits = detect(p, tk, bench_perf)
                        if hits:
                            frames[tk] = p
                            rows += hits
                    except Exception:
                        continue
            except Exception:
                pass
            bar.progress((k + 1) / len(chunks), text=f"Scanne … {min((k+1)*50, len(tickers))}/{len(tickers)}")
        bar.empty()
        st.session_state["results"] = rows
        st.session_state["frames"] = frames

    rows = st.session_state.get("results")
    if rows is None:
        st.info("Universum wählen und **Scan starten** drücken. Der erste S&P-500-Scan "
                "dauert 1–3 Minuten, danach greift der Cache (1 Stunde).")
        return
    if not rows:
        st.warning("Heute keine Setups gefunden. Das ist okay — kein Setup heißt kein Trade.")
        return

    res = pd.DataFrame(rows)
    res = res[res.CRV >= min_crv].sort_values("Score", ascending=False).head(10).reset_index(drop=True)
    if res.empty:
        st.warning(f"Setups gefunden, aber keines erfüllt CRV ≥ {min_crv}.")
        return

    res["Risiko/Stk $"] = res.Trigger - res.Stop
    res["Stück"] = (konto * risk_pct / 100 / fx / res["Risiko/Stk $"]).round(2)
    res["Position $"] = (res["Stück"] * res.Trigger).round(0)
    res.index = pd.RangeIndex(1, len(res) + 1, name="Rang")

    st.subheader(f"Top {len(res)} Setups heute")
    show = res[["Ticker", "Setup", "Kurs", "Trigger", "Stop", "Ziel", "CRV",
                "ATRpct", "RS3M", "VolFaktor", "Stück", "Position $", "Score"]]
    st.dataframe(show.style.format({"Kurs": "{:.2f}", "Trigger": "{:.2f}", "Stop": "{:.2f}",
                                    "Ziel": "{:.2f}", "CRV": "{:.2f}", "ATRpct": "{:.1f}%",
                                    "RS3M": "{:+.1f}%", "VolFaktor": "{:.1f}×",
                                    "Stück": "{:.2f}", "Position $": "{:.0f}",
                                    "Score": "{:.0f}"}),
                 use_container_width=True, height=min(430, 60 + 38 * len(res)))

    options = [f"#{i} {r.Ticker} — {r.Setup}" for i, r in zip(res.index, res.itertuples())]
    sel = st.selectbox("Setup im Detail:", options)
    row = res.iloc[options.index(sel)]
    st.plotly_chart(make_chart(st.session_state["frames"][row["Ticker"]], row),
                    use_container_width=True)
    st.markdown("### 📋 Begründung der Trade-Idee")
    st.markdown(explain(row))

if __name__ == "__main__":
    main()
