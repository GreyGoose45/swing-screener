# Swing-Screener Dashboard

Tägliches Screening-Dashboard für den Swing-Trading-Plan (Setup A Trend-Pullback,
Setup B Mean Reversion, Flat-Top-Breakout). Zeigt die Top-10-Setups mit Trigger-,
Stop- und Ziel-Level direkt im Chart, plus Positionsgrößen-Berechnung.

## Gratis online hosten (einmalig ~10 Minuten)

1. **GitHub-Account** anlegen (github.com), falls noch nicht vorhanden
2. **Neues Repository** erstellen (z. B. `swing-screener`, public)
3. Diese drei Dateien hochladen: `app.py`, `requirements.txt`, `README.md`
   (auf GitHub: "Add file" → "Upload files")
4. Auf **share.streamlit.io** gehen, mit GitHub anmelden
5. "Create app" → Repository `swing-screener` wählen, Main file: `app.py` → **Deploy**
6. Fertig — du bekommst eine feste URL wie `https://swing-screener.streamlit.app`,
   die du am Handy als Lesezeichen/Homescreen-Icon speicherst

## Tägliche Nutzung (Abendroutine)

1. URL öffnen (App "schläft" bei Inaktivität, wacht in ~30 Sek. auf)
2. **Scan starten** drücken → Top-10-Tabelle erscheint, sortiert nach Score
3. Kandidaten anklicken → Chart mit eingezeichnetem Trigger (grün),
   Stop (rot) und Ziel (gelb)
4. **Earnings manuell prüfen** (TradingView "E"-Symbol) — keine Trades
   <10 Handelstage vor Earnings
5. Für max. 1–2 Kandidaten Stop-Buy-Order in Trading212 platzieren —
   Stückzahl steht bereits in der Tabelle

## Anpassen

- **Watchlist:** direkt in der Seitenleiste der App ändern (kommagetrennt,
  Yahoo-Finance-Ticker: US ohne Suffix, Xetra mit `.DE`, Amsterdam `.AS`, Paris `.PA`)
- **Kontogröße / Risiko / EUR-USD:** Seitenleiste
- Dauerhaft andere Standard-Watchlist: `DEFAULT_TICKERS` in `app.py` editieren

## Grenzen (ehrlich)

- Kursdaten: Yahoo Finance, End-of-Day bzw. ~15 Min. verzögert — für
  Daily-Swing-Setups ausreichend, nicht für Intraday
- Earnings-Termine werden NICHT automatisch geprüft (Yahoo-Daten unzuverlässig) —
  bleibt ein manueller Pflichtcheck
- Der Score ist eine Heuristik zum Sortieren, kein Profitabilitäts-Versprechen.
  Die Entscheidung triffst du am Chart, nicht die Tabelle.
