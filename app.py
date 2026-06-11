"""
Swing-Screener Dashboard v3
Neu: RSI-Panel · Support/Resistance-Zonen · Multi-Timeframe-Trendwidget ·
Einzelanalyse mit Suchfeld · Positions-Watchlist · optionale Claude-KI-Analyse
"""

import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA", "AVGO",
    "CRM", "NFLX", "QCOM", "ORCL", "ADBE", "UBER", "PYPL", "ABNB", "PLTR",
    "CAT", "DE", "GS", "JPM", "XOM", "COCO", "VLY",
]
BENCHMARK = "SPY"
MIN_AVG_VOLUME = 1_000_000
ATR_PCT_MIN, ATR_PCT_MAX = 2.0, 6.0
TREND_ICON = {"up": "🟢 ↑", "down": "🔴 ↓", "neutral": "⚪ →"}

# ----------------------------- Universum ---------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_universe(name):
    try:
        if name == "S&P 500":
            t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
            return [s.replace(".", "-") for s in t["Symbol"].tolist()]
        if name == "Dow Jones 30":
            for t in pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"):
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
    df["Volume"] = df["Volume"].fillna(0)  # Futures/Indizes liefern teils kein Volumen
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    return df.dropna(subset=["Close", "High", "Low", "ATR14", "High20", "EMA200"])

# --------------------- Support/Resistance-Zonen --------------------------

def find_levels(p, lookback=160, max_levels=5):
    """Pivot-Hochs/-Tiefs finden, nahe Level clustern, nach Touch-Anzahl ranken."""
    d = p.iloc[-lookback:]
    a = float(d.ATR14.iloc[-1])
    win = 5
    piv = []
    hi, lo = d.High.values, d.Low.values
    for i in range(win, len(d) - win):
        if hi[i] == hi[i - win:i + win + 1].max():
            piv.append(hi[i])
        if lo[i] == lo[i - win:i + win + 1].min():
            piv.append(lo[i])
    piv.sort()
    clusters = []
    for lvl in piv:
        if clusters and lvl - clusters[-1][-1] <= 0.6 * a:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    levels = [(float(np.mean(c)), len(c)) for c in clusters if len(c) >= 2]
    levels.sort(key=lambda x: -x[1])
    return [(lvl, n, 0.3 * a) for lvl, n in levels[:max_levels]]

# --------------------- Multi-Timeframe-Trend -----------------------------

def trend_state(close):
    if len(close) < 60:
        return "neutral"
    e20, e50 = ema(close, 20), ema(close, 50)
    c, a, b = float(close.iloc[-1]), float(e20.iloc[-1]), float(e50.iloc[-1])
    if c > a > b:
        return "up"
    if c < a < b:
        return "down"
    return "neutral"

@st.cache_data(ttl=3600, show_spinner=False)
def load_tf(ticker, interval):
    import yfinance as yf
    period = {"1wk": "5y", "1d": "2y", "1h": "3mo"}[interval]
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def mtf_trends(ticker):
    out = {}
    try:
        out["Weekly"] = trend_state(load_tf(ticker, "1wk").Close)
        out["Daily"] = trend_state(load_tf(ticker, "1d").Close)
        h1 = load_tf(ticker, "1h")
        out["4h"] = trend_state(h1.Close.resample("4h").last().dropna())
        out["1h"] = trend_state(h1.Close)
    except Exception:
        for k in ("Weekly", "Daily", "4h", "1h"):
            out.setdefault(k, "neutral")
    return out

def structural_up(p):
    """Uebergeordneter Daily-Trend nach STRUKTUR: EMA50>EMA200 (steigend), Kurs>EMA200.
    Bewusst ohne Kurs-vs-EMA20 — ein Pullback bricht den uebergeordneten Trend nicht."""
    if len(p) < 10:
        return False
    r = p.iloc[-1]
    return bool(r.EMA50 > r.EMA200 and p.EMA50.iloc[-1] > p.EMA50.iloc[-6]
                and r.Close > r.EMA200)

def mtf_verdict(t, daily_struct_up=False):
    """Bewertung nach der Praemisse: W+D muessen aufwaerts zeigen;
    1h/4h duerfen abwaerts zeigen (= Pullback-Konstellation)."""
    daily_ok = t["Daily"] == "up" or (t["Daily"] == "neutral" and daily_struct_up)
    if t["Weekly"] == "up" and daily_ok:
        if t["4h"] == "up" and t["1h"] == "up" and t["Daily"] == "up":
            return ("success", "✅ Alle Timeframes aufwärts ausgerichtet — "
                    "Trend-Continuation/Breakout-Modus.")
        return ("warning", "🟡 Übergeordnet (W+D) aufwärts, kurzfristig (4h/1h) Gegenbewegung — "
                "klassische Pullback-Konstellation: Long-Entry mit Bestätigung suchen.")
    if "down" in (t["Weekly"], t["Daily"]):
        return ("error", "🔴 Übergeordneter Trend (Weekly/Daily) zeigt nicht aufwärts — "
                "nach Plan KEIN Long-Trade. The trend is your friend.")
    return ("warning", "⚪ Übergeordneter Trend uneindeutig/seitwärts — abwarten, "
            "bis Weekly und Daily klar aufwärts zeigen.")

def render_mtf(t, daily_struct_up=False):
    cols = st.columns(4)
    for c, (label, state) in zip(cols, t.items()):
        c.metric(label, TREND_ICON[state])
    kind, msg = mtf_verdict(t, daily_struct_up)
    getattr(st, kind)(msg)

# ----------------------------- Setup-Erkennung ---------------------------

def detect(df, ticker, bench_perf3m, apply_universe_filter=True):
    out = []
    if len(df) < 200:
        return out
    r = df.iloc[-1]
    if apply_universe_filter:
        if r.VolAvg20 < MIN_AVG_VOLUME:
            return out
        atr_pct = r.ATR14 / r.Close * 100
        if not (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX):
            return out
    atr_pct = r.ATR14 / r.Close * 100
    perf3m = (r.Close / df.Close.iloc[-63] - 1) * 100 if len(df) > 63 else 0.0
    rs = float(np.nan_to_num(perf3m - bench_perf3m))
    vol_faktor = float(r.Volume / r.VolAvg20) if r.VolAvg20 > 0 else 0.0
    trend_ok = (r.EMA50 > r.EMA200
                and df.EMA50.iloc[-1] > df.EMA50.iloc[-6]
                and df.EMA200.iloc[-1] > df.EMA200.iloc[-6])
    base = dict(Ticker=ticker, Kurs=float(r.Close), ATRpct=float(atr_pct),
                RS3M=rs, VolFaktor=vol_faktor)

    if trend_ok and df.High.iloc[-20:].max() >= df.High20.iloc[-1] * 0.999:
        band_hi, band_lo = max(r.EMA20, r.EMA50), min(r.EMA20, r.EMA50)
        pull_tage = int(19 - np.argmax(df.High.iloc[-20:].values))
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

def no_setup_reasons(p):
    """Erklaert regelbasiert, warum KEIN Setup vorliegt."""
    if len(p) == 0:
        return ["Keine auswertbaren Kursdaten (z. B. Futures/Index ohne saubere Historie)."]
    r = p.iloc[-1]
    reasons = []
    atr_pct = r.ATR14 / r.Close * 100
    if r.VolAvg20 < MIN_AVG_VOLUME:
        reasons.append(f"Liquidität zu gering (Ø-Volumen {r.VolAvg20/1e6:.1f} Mio < 1 Mio)")
    if atr_pct < ATR_PCT_MIN:
        reasons.append(f"ATR {atr_pct:.1f}% — zu träge (< {ATR_PCT_MIN}%), Ziele werden kaum erreicht")
    if atr_pct > ATR_PCT_MAX:
        reasons.append(f"ATR {atr_pct:.1f}% — zu volatil (> {ATR_PCT_MAX}%), Stops werden ausgenommen")
    if not (r.EMA50 > r.EMA200):
        reasons.append("EMA 50 unter EMA 200 — kein etablierter Aufwärtstrend")
    elif p.EMA50.iloc[-1] <= p.EMA50.iloc[-6]:
        reasons.append("EMA 50 fällt — Trendqualität fraglich")
    if r.Close < r.EMA200:
        reasons.append("Kurs unter EMA 200 — übergeordneter Abwärtsdruck")
    if r.EMA50 > r.EMA200 and r.Close > r.EMA200:
        if r.RSI14 > 55:
            reasons.append(f"RSI(14) bei {r.RSI14:.0f} — kein Pullback, Aktie läuft bereits (nicht hinterherkaufen)")
        if r.Low > max(r.EMA20, r.EMA50):
            reasons.append("Kurs weit über dem EMA-20/50-Band — auf Rücksetzer in die Zone warten")
        if r.RSI2 >= 10:
            reasons.append("Keine Überdehnung nach unten (RSI(2) ≥ 10) — kein Mean-Reversion-Fall")
    return reasons or ["Kein Muster trifft die Regelkriterien — beobachten statt erzwingen."]

# ----------------------------- Playbook-Checkliste -----------------------

def build_checklist(p, trends, rs, row=None, min_crv=1.5):
    """Bewertet alle objektiven Plan-Kriterien. True=erfuellt, False=nicht, None=manuell."""
    r = p.iloc[-1]
    atr_pct = float(r.ATR14 / r.Close * 100)
    items = []
    add = lambda ok, txt: items.append((ok, txt))

    add(bool(r.VolAvg20 >= MIN_AVG_VOLUME),
        f"Liquidität: Ø-Volumen {r.VolAvg20/1e6:.1f} Mio (≥ 1 Mio)")
    add(ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX,
        f"Volatilitätsfenster: ATR {atr_pct:.1f}% (Soll 2–6%)")
    add(bool(r.Close > r.EMA200), f"Kurs über EMA 200 ({r.EMA200:.2f})")
    add(bool(r.EMA50 > r.EMA200), "EMA 50 über EMA 200")
    add(bool(len(p) > 6 and p.EMA50.iloc[-1] > p.EMA50.iloc[-6]), "EMA 50 steigend")
    add(bool(rs > 0), f"Relative Stärke vs. S&P 500 (3M): {rs:+.1f}%")
    daily_struct = structural_up(p)
    add(trends.get("Weekly") == "up" and daily_struct,
        "Übergeordneter Trend aufwärts (Weekly-Ampel + Daily-Struktur "
        "EMA50>EMA200 steigend, Kurs>EMA200)")

    setup = row["Setup"] if row is not None else None
    if setup and setup.startswith("A"):
        add(bool(p.High.iloc[-20:].max() >= p.High20.iloc[-1] * 0.999),
            "Neues 20-Tage-Hoch innerhalb der letzten 20 Tage")
        band_hi, band_lo = max(r.EMA20, r.EMA50), min(r.EMA20, r.EMA50)
        add(bool(r.Low <= band_hi and r.Close >= band_lo),
            f"Pullback ins EMA-20/50-Band, Schluss hält darüber ({band_lo:.2f}–{band_hi:.2f})")
        add(35 <= float(r.RSI14) <= 55, f"RSI(14) im Pullback-Fenster 35–55 (ist {r.RSI14:.0f})")
        add(trends.get("4h") != "up" or trends.get("1h") != "up",
            "Kurzfristige Gegenbewegung (4h/1h) = Pullback-Konstellation")
        add(bool(row["CRV"] >= min_crv), f"CRV ≥ {min_crv} (ist {row['CRV']:.2f})")
    elif setup and setup.startswith("B"):
        add(float(r.RSI2) < 10, f"RSI(2) < 10 (ist {r.RSI2:.1f})")
        add(bool((r.EMA20 - r.Close) >= 1.5 * r.ATR14),
            f"Überdehnung: ≥ 1,5×ATR unter EMA 20 (ist {(r.EMA20-r.Close)/r.ATR14:.1f}×)")
        add(None, "Kein fundamentaler Grund für den Abverkauf (News manuell prüfen!)")
    elif setup == "Flat-Top-Breakout":
        add(bool(row.get("m_touches", 0) >= 2),
            f"Widerstand mehrfach getestet ({row.get('m_touches', 0)}× ≥ 2)")
        add(bool(row.get("m_dist", 99) <= 3.0),
            f"Kurs nah am Trigger ({row.get('m_dist', 0):.1f}% ≤ 3%)")
        add(bool(row.get("m_range15", 99) <= 12),
            f"Enge Konsolidierung (Range {row.get('m_range15', 0):.1f}% ≤ 12%)")
        add(None, f"Volumen am AUSBRUCHSTAG ≥ 1,5× Ø20 (heute {row['VolFaktor']:.1f}× — "
                  "erst bei Ausführung bewertbar)")
        add(bool(row["CRV"] >= min_crv), f"CRV ≥ {min_crv} (ist {row['CRV']:.2f})")

    add(None, "Earnings ≥ 10 Handelstage entfernt (manuell prüfen — TradingView 'E')")
    add(None, "Max. 3 offene Positionen / 1 pro Sektor (selbst prüfen)")
    return items

def render_checklist(items):
    erfuellt = sum(1 for ok, _ in items if ok is True)
    bewertbar = sum(1 for ok, _ in items if ok is not None)
    offen = sum(1 for ok, _ in items if ok is None)
    st.markdown(f"### ✅ Playbook-Checkliste — {erfuellt}/{bewertbar} objektive Kriterien erfüllt"
                + (f" · {offen} manuell zu prüfen" if offen else ""))
    for ok, txt in items:
        icon = "✅" if ok is True else ("❌" if ok is False else "⚠️")
        st.markdown(f"{icon} {txt}")
    if erfuellt < bewertbar:
        st.error("Mindestens ein objektives Kriterium ist NICHT erfüllt — nach Playbook "
                 "ist das kein vollwertiger Trade. Kein Kriterium wegdiskutieren.")



def explain(row):
    s, t = row["Setup"], row["Ticker"]
    crv, rs, vf = row["CRV"], row["RS3M"], row["VolFaktor"]
    common = (f"\n\n**Risiko-Setup:** Stop-Buy-Order bei **{row['Trigger']:.2f}** — Entry nur, "
              f"wenn der Markt die Idee bestätigt. Stop-Loss **{row['Stop']:.2f}**, "
              f"Ziel **{row['Ziel']:.2f}** → CRV **{crv:.2f}**. "
              f"Relative Stärke vs. S&P 500 (3M): **{rs:+.1f}%**. "
              f"Heutiges Volumen: **{vf:.1f}×** des 20-Tage-Schnitts."
              f"\n\n⚠️ Vor der Order: Earnings-Termin prüfen (mind. 10 Handelstage Abstand).")
    if s.startswith("A"):
        return (f"**Trade-Idee {t} — Trend-Fortsetzung nach Pullback:** Aufwärtstrend intakt "
                f"(EMA 50 über EMA 200, beide steigend), neues 20-Tage-Hoch vorhanden. Seit "
                f"~{row['m_pull_tage']} Tagen Korrektur bis ins EMA-20/50-Band "
                f"({row['m_band_lo']:.2f}–{row['m_band_hi']:.2f}); der Schlusskurs hält darüber — "
                f"die Zone wird verteidigt. RSI(14) {row['m_rsi14']:.0f} = gesunder Pullback. "
                f"Idee: Trendwiederaufnahme kaufen, sobald das heutige Hoch überwunden wird. Stop "
                f"unter Pullback-Tief ({row['m_pull_low']:.2f}) — dort ist die These widerlegt. "
                f"Ziel altes Hoch, darüber Trailing per EMA 20." + common)
    if s.startswith("B"):
        return (f"**Trade-Idee {t} — Mean Reversion im Aufwärtstrend:** Kurs über EMA 200 "
                f"(Trend intakt), aber kurzfristig überverkauft: RSI(2) {row['m_rsi2']:.1f}, "
                f"Abstand {row['m_dist_atr']:.1f}×ATR unter der EMA 20 ({row['m_ema20']:.2f}). "
                f"Solche Überdehnungen schnappen in intakten Trends meist schnell zurück. Idee: "
                f"Rebound kaufen, sobald das heutige Hoch zurückerobert wird. Stop bewusst weit "
                f"(2×ATR). Exit an der EMA 20, bei RSI(2) > 70 oder nach max. 5 Tagen (Zeit-Stop)."
                + common)
    return (f"**Trade-Idee {t} — Flat-Top-Breakout:** Enge Konsolidierung "
            f"({row['m_range15']:.1f}% Range) im Aufwärtstrend, direkt unter glattem Widerstand "
            f"bei {row['m_hi20']:.2f} ({row['m_touches']}× getestet, nur noch {row['m_dist']:.1f}% "
            f"entfernt). Enge Ranges nach Anstiegen lösen sich statistisch häufiger nach oben auf. "
            f"Idee: Ausbruch über den Deckel kaufen — Bestätigung durch Volumen ≥1,5× Schnitt "
            f"(aktuell {row['VolFaktor']:.1f}×). Ziel = Measured Move; darüber kein Widerstand → "
            f"Trailing per EMA 20." + common)

# ----------------------------- KI-Analyse (Claude API) -------------------

@st.cache_data(ttl=43200, show_spinner=False)
def ki_analyse(payload, cache_key, api_key):
    prompt = (
        "Du bist ein nüchterner, kritischer Swing-Trading-Analyst. Bewerte das folgende, "
        "regelbasiert erkannte Long-Setup auf Tagesbasis. Antworte auf Deutsch, max. 250 Wörter, "
        "strukturiert in: 1) Stärken, 2) Schwächen/Risiken, 3) Worauf vor dem Entry achten, "
        "4) Fazit (eine Zeile). Keine Anlageberatung, keine Kursziele erfinden, nur die "
        "gelieferten Daten bewerten. Sei kritisch — wenn das Setup schwach ist, sag es klar.\n\n"
        + payload
    )
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": "claude-haiku-4-5-20251001", "max_tokens": 700,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=90,
    )
    resp.raise_for_status()
    return "".join(b.get("text", "") for b in resp.json().get("content", []))

def build_payload(row, p, trends, levels):
    d = p.iloc[-10:]
    lines = [f"{i.date()}: O {r.Open:.2f} H {r.High:.2f} L {r.Low:.2f} C {r.Close:.2f} Vol {r.Volume/1e6:.1f}M"
             for i, r in d.iterrows()]
    lvl_txt = ", ".join(f"{l:.2f} ({n} Touches)" for l, n, _ in levels) or "keine markanten"
    return (f"Ticker: {row['Ticker']} | Setup: {row['Setup']}\n"
            f"Kurs: {row['Kurs']:.2f} | Trigger: {row['Trigger']:.2f} | Stop: {row['Stop']:.2f} | "
            f"Ziel: {row['Ziel']:.2f} | CRV: {row['CRV']:.2f}\n"
            f"ATR%: {row['ATRpct']:.1f} | RS vs SPY 3M: {row['RS3M']:+.1f}% | "
            f"Volumen heute: {row['VolFaktor']:.1f}x Ø20\n"
            f"RSI14: {p.RSI14.iloc[-1]:.0f} | RSI2: {p.RSI2.iloc[-1]:.0f} | "
            f"EMA20 {p.EMA20.iloc[-1]:.2f} / EMA50 {p.EMA50.iloc[-1]:.2f} / EMA200 {p.EMA200.iloc[-1]:.2f}\n"
            f"Trends: Weekly {trends['Weekly']}, Daily {trends['Daily']}, 4h {trends['4h']}, 1h {trends['1h']}\n"
            f"Support/Resistance-Level: {lvl_txt}\n"
            f"Letzte 10 Tageskerzen:\n" + "\n".join(lines))

# ----------------------------- Daten laden -------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_chunk(tickers, period="2y"):
    import yfinance as yf
    return yf.download(list(tickers), period=period, auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)

# ----------------------------- Chart -------------------------------------

def make_chart(p, row=None, levels=None, title=""):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    d = p.iloc[-130:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.62, 0.19, 0.19])
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low,
                                 close=d.Close, name="Kurs", showlegend=False,
                                 increasing_line_color="#2ecc71",
                                 decreasing_line_color="#e74c3c"), row=1, col=1)
    for col, color in [("EMA20", "#E8A04A"), ("EMA50", "#d9534f"), ("EMA200", "#5bc0de")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col,
                                 line=dict(color=color, width=1.3)), row=1, col=1)
    for lvl, touches, halb in (levels or []):
        fig.add_hrect(y0=lvl - halb, y1=lvl + halb, fillcolor="#9b59b6",
                      opacity=0.18, line_width=0, row=1, col=1)
        fig.add_annotation(x=d.index[-1], y=lvl, text=f"S/R {lvl:.2f} ({touches}×)",
                           font=dict(color="#bb8fce", size=10), showarrow=False,
                           xanchor="left", row=1, col=1)
    if row is not None:
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
    fig.add_trace(go.Scatter(x=d.index, y=d.RSI14, name="RSI 14",
                             line=dict(color="#af7ac5", width=1.4)), row=3, col=1)
    for y in (30, 50, 70):
        fig.add_hline(y=y, line_color="#566573", line_dash="dot", line_width=1, row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_layout(template="plotly_dark", height=760,
                      paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, title=title,
                      legend=dict(orientation="h", y=1.04))
    return fig

# ------------------------- Detail-Ansicht (gemeinsam) ---------------------

def render_detail(row, p, api_key, min_crv=1.5):
    trends = mtf_trends(row["Ticker"])
    st.markdown("##### Multi-Timeframe-Trend")
    render_mtf(trends, structural_up(p))
    levels = find_levels(p)
    st.plotly_chart(make_chart(p, row, levels, f"{row['Ticker']} — {row['Setup']}"),
                    use_container_width=True)
    render_checklist(build_checklist(p, trends, row["RS3M"], row, min_crv))
    st.markdown("### 📋 Begründung der Trade-Idee")
    st.markdown(explain(row))
    st.markdown("### 🤖 KI-Zweitmeinung (optional)")
    if not api_key:
        st.info("Für die KI-Analyse einen Anthropic-API-Key in den Streamlit-Secrets "
                "hinterlegen (`ANTHROPIC_API_KEY`) oder in der Seitenleiste eingeben. "
                "Kosten: < 1 Cent pro Analyse (Haiku-Modell, 12h-Cache).")
    elif st.button(f"🤖 KI-Analyse für {row['Ticker']} abrufen", key=f"ki_{row['Ticker']}_{row['Setup']}"):
        with st.spinner("Claude analysiert …"):
            try:
                payload = build_payload(row, p, trends, levels)
                key = f"{row['Ticker']}|{row['Setup']}|{dt.date.today()}"
                st.markdown(ki_analyse(payload, key, api_key))
                st.caption("KI-generierte Einschätzung auf Basis der übermittelten Kennzahlen — "
                           "keine Anlageberatung. Ergebnis wird 12 h gecacht.")
            except Exception as ex:
                st.error(f"KI-Analyse fehlgeschlagen: {ex}")

# ----------------------------- UI ----------------------------------------

def main():
    st.set_page_config(page_title="Swing-Screener", layout="wide", page_icon="📈")
    st.title("📈 Swing-Screener — Stocks in Play")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("Einstellungen")
        universe = st.radio("Universum", ["S&P 500", "Dow Jones 30", "Eigene Liste"])
        custom = ""
        if universe == "Eigene Liste":
            custom = st.text_area("Ticker (kommagetrennt)", ", ".join(FALLBACK_TICKERS), height=110)
        konto = st.number_input("Kontogröße €", 100, 1_000_000, 1000, step=100)
        risk_pct = st.number_input("Risiko pro Trade %", 0.25, 2.0, 1.0, step=0.25)
        fx = st.number_input("EUR/USD", 0.8, 1.6, 1.15, step=0.01)
        min_crv = st.slider("Min. CRV", 0.5, 3.0, 1.2, 0.1)

        api_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
        if not api_key:
            api_key = st.text_input("Anthropic API-Key (optional, für KI-Analyse)",
                                    type="password")

        # ---------- Positions-Watchlist (in URL gespeichert) ----------
        st.divider()
        st.subheader("📌 Meine Positionen")
        if "positions" not in st.session_state:
            raw = st.query_params.get("pos", "")
            st.session_state.positions = [t for t in raw.split(",") if t]
        c1, c2 = st.columns([3, 1])
        new_pos = c1.text_input("Ticker", label_visibility="collapsed",
                                placeholder="z. B. COCO")
        if c2.button("➕") and new_pos.strip():
            tk = new_pos.strip().upper()
            if tk not in st.session_state.positions:
                st.session_state.positions.append(tk)
                st.query_params["pos"] = ",".join(st.session_state.positions)
                st.rerun()
        for tk in list(st.session_state.positions):
            p1, p2 = st.columns([3, 1])
            p1.write(f"**{tk}**")
            if p2.button("✖", key=f"del_{tk}"):
                st.session_state.positions.remove(tk)
                st.query_params["pos"] = ",".join(st.session_state.positions)
                st.rerun()
        if st.session_state.positions:
            st.caption("💡 Lesezeichen aktualisieren — die Liste steckt in der URL.")
        st.caption("⚠️ Earnings manuell prüfen — keine Trades <10 Handelstage vor Earnings!")

    tab_scan, tab_single, tab_pos = st.tabs(["🔍 Screener", "🔎 Einzelanalyse", "📌 Positionen"])

    # ---------- Tab 1: Screener ----------
    with tab_scan:
        st.caption("Setup A: Trend-Pullback · Setup B: Mean Reversion · Flat-Top-Breakout — "
                   "Filter: Ø-Vol > 1 Mio · ATR 2–6%")
        if st.button("🔍 Scan starten", type="primary", use_container_width=True):
            tickers = ([t.strip().upper() for t in custom.split(",") if t.strip()]
                       if universe == "Eigene Liste" else get_universe(universe))
            bench = load_tf(BENCHMARK, "1d")
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
                bar.progress((k + 1) / len(chunks),
                             text=f"Scanne … {min((k+1)*50, len(tickers))}/{len(tickers)}")
            bar.empty()
            st.session_state["results"] = rows
            st.session_state["frames"] = frames

        rows = st.session_state.get("results")
        if rows is None:
            st.info("Universum wählen und **Scan starten**. Erster S&P-500-Scan: 1–3 Min, "
                    "danach Cache (1 Std.).")
        elif not rows:
            st.warning("Heute keine Setups. Das ist okay — kein Setup heißt kein Trade.")
        else:
            res = pd.DataFrame(rows)
            res = res[res.CRV >= min_crv].sort_values("Score", ascending=False).head(10).reset_index(drop=True)
            if res.empty:
                st.warning(f"Setups gefunden, aber keines erfüllt CRV ≥ {min_crv}.")
            else:
                res["Risiko/Stk $"] = res.Trigger - res.Stop
                res["Stück"] = (konto * risk_pct / 100 / fx / res["Risiko/Stk $"]).round(2)
                res["Position $"] = (res["Stück"] * res.Trigger).round(0)
                res.index = pd.RangeIndex(1, len(res) + 1, name="Rang")
                st.subheader(f"Top {len(res)} Setups heute")
                show = res[["Ticker", "Setup", "Kurs", "Trigger", "Stop", "Ziel", "CRV",
                            "ATRpct", "RS3M", "VolFaktor", "Stück", "Position $", "Score"]]
                st.dataframe(show.style.format({"Kurs": "{:.2f}", "Trigger": "{:.2f}",
                                                "Stop": "{:.2f}", "Ziel": "{:.2f}", "CRV": "{:.2f}",
                                                "ATRpct": "{:.1f}%", "RS3M": "{:+.1f}%",
                                                "VolFaktor": "{:.1f}×", "Stück": "{:.2f}",
                                                "Position $": "{:.0f}", "Score": "{:.0f}"}),
                             use_container_width=True, height=min(430, 60 + 38 * len(res)))
                options = [f"#{i} {r.Ticker} — {r.Setup}" for i, r in zip(res.index, res.itertuples())]
                sel = st.selectbox("Setup im Detail:", options)
                row = res.iloc[options.index(sel)]
                render_detail(row, st.session_state["frames"][row["Ticker"]], api_key, min_crv)

    # ---------- Tab 2: Einzelanalyse ----------
    with tab_single:
        st.caption("Beliebige Aktie analysieren — auch wenn sie im Screener nicht auftaucht.")
        c1, c2 = st.columns([3, 1])
        tk = c1.text_input("Ticker (Yahoo-Format: AAPL, SAP.DE, ASML.AS …)",
                           placeholder="z. B. COCO").strip().upper()
        if c2.button("Analysieren", type="primary") and tk:
            st.session_state["single_tk"] = tk
        tk = st.session_state.get("single_tk")
        if tk:
            try:
                df = load_tf(tk, "1d")
                if len(df) < 220:
                    st.error(f"Zu wenig Kurshistorie für {tk} (mind. ~1 Jahr nötig).")
                else:
                    p = prepare(df)
                    if len(p) < 200:
                        st.error(f"{tk}: zu wenig auswertbare Daten nach Aufbereitung — "
                                 "vermutlich Futures/Index ohne saubere Historie. "
                                 "Das Tool ist für Aktien gebaut (Trading212 Invest "
                                 "handelt ohnehin keine Futures).")
                        st.stop()
                    bench = load_tf(BENCHMARK, "1d")
                    bench_perf = float((bench.Close.iloc[-1] / bench.Close.iloc[-63] - 1) * 100)
                    hits = detect(p, tk, bench_perf, apply_universe_filter=False)
                    if hits:
                        best = max(hits, key=lambda h: h["Score"])
                        best["Risiko/Stk $"] = best["Trigger"] - best["Stop"]
                        st.success(f"✅ Regelkonformes Setup erkannt: **{best['Setup']}**")
                        stk = konto * risk_pct / 100 / fx / best["Risiko/Stk $"]
                        st.markdown(f"**Positionsgröße:** {stk:.2f} Stück "
                                    f"(~{stk * best['Trigger']:.0f} $) bei {risk_pct}% Risiko")
                        render_detail(pd.Series(best), p, api_key, min_crv)
                    else:
                        st.warning("**Kein regelkonformes Setup — aktuell keine sinnvolle "
                                   "Handelsmöglichkeit nach Plan.**")
                        for grund in no_setup_reasons(p):
                            st.markdown(f"- {grund}")
                        st.markdown("##### Multi-Timeframe-Trend")
                        trends = mtf_trends(tk)
                        render_mtf(trends, structural_up(p))
                        st.plotly_chart(make_chart(p, None, find_levels(p), f"{tk} — Übersicht"),
                                        use_container_width=True)
                        rs_single = float((p.Close.iloc[-1] / p.Close.iloc[-63] - 1) * 100
                                          - bench_perf) if len(p) > 63 else 0.0
                        render_checklist(build_checklist(p, trends, rs_single))
            except Exception as ex:
                st.error(f"Konnte {tk} nicht laden: {ex}")

    # ---------- Tab 3: Positionen ----------
    with tab_pos:
        if not st.session_state.positions:
            st.info("Keine Positionen markiert. Links in der Seitenleiste Ticker hinzufügen, "
                    "sobald du einen Trade platziert hast.")
        else:
            for tk in st.session_state.positions:
                try:
                    df = load_tf(tk, "1d")
                    p = prepare(df)
                    r = p.iloc[-1]
                    chg = (r.Close / p.Close.iloc[-2] - 1) * 100
                    trends = mtf_trends(tk)
                    c = st.columns([1.2, 1, 1, 2.6, 2.2])
                    c[0].markdown(f"### {tk}")
                    c[1].metric("Kurs", f"{r.Close:.2f}", f"{chg:+.2f}%")
                    ema20_status = "✅ über EMA 20" if r.Close > r.EMA20 else "⚠️ UNTER EMA 20"
                    c[2].markdown(f"**{ema20_status}**\n\nEMA20: {r.EMA20:.2f}")
                    c[3].markdown("  ".join(f"{k}: {TREND_ICON[v]}" for k, v in trends.items()))
                    warn = []
                    if r.Close < r.EMA20:
                        warn.append("Schluss unter EMA 20 → Trailing-Exit-Regel prüfen!")
                    if r.RSI2 > 90:
                        warn.append("RSI(2) > 90 — kurzfristig stark überkauft")
                    c[4].markdown("\n".join(f"🔔 {w}" for w in warn) or "—")
                    st.divider()
                except Exception:
                    st.warning(f"{tk}: Daten nicht ladbar")

if __name__ == "__main__":
    main()
