# Swing-Screener Dashboard v3

## Neu in v3
- **3-Panel-Chart:** Kurs + EMAs + S/R-Zonen | Volumen mit Ø20-Linie | RSI(14) mit 30/50/70
- **Support/Resistance-Zonen** (lila): Pivot-Hochs/-Tiefs der letzten ~160 Tage,
  geclustert und nach Anzahl der Berührungen gerankt
- **Multi-Timeframe-Trendwidget:** Weekly / Daily / 4h / 1h mit Ampel-Bewertung
  nach der Regel: Kurs > EMA20 > EMA50 = aufwärts (umgekehrt = abwärts, sonst neutral).
  Verdikt: W+D aufwärts + 4h/1h abwärts = Pullback-Konstellation ✅
- **Einzelanalyse-Tab:** beliebigen Ticker eingeben → Setup-Prüfung, Levels, Trends.
  Wenn kein Setup: regelbasierte Begründung, warum kein Trade sinnvoll ist
- **Positionen:** In der Seitenleiste Ticker markieren, in denen du aktiv bist.
  Eigener Tab zeigt Kurs, Tagesveränderung, EMA-20-Status (Trailing-Warnung!) und
  alle Timeframe-Trends. Die Liste wird in der URL gespeichert → nach dem Hinzufügen
  das Lesezeichen aktualisieren.
- **Optionale KI-Zweitmeinung (Claude):** Button unter jedem Detail-Chart

## KI-Analyse einrichten (optional)
1. Auf console.anthropic.com einen Account + API-Key erstellen
   (Pay-per-Use, UNABHÄNGIG vom Claude.ai-Abo — kleines Guthaben aufladen)
2. Auf share.streamlit.io: App → Settings → Secrets → eintragen:
   ANTHROPIC_API_KEY = "sk-ant-..."
3. Fertig. Kosten pro Analyse: < 1 Cent (Haiku-Modell, ~1.000 Zeichen Payload,
   Ergebnis wird 12 h gecacht). Die Analyse läuft NUR auf Knopfdruck für den
   ausgewählten Titel — nie automatisch für alle.

## Deployment / Update
Dateien im GitHub-Repo ersetzen (entpackt!, inkl. .streamlit/config.toml) →
Streamlit deployt automatisch neu. Bei Problemen: Manage app → Reboot.

## Dateien
app.py · requirements.txt · README.md · .streamlit/config.toml

## Grenzen (ehrlich)
- Kursdaten: Yahoo Finance, End-of-Day / verzögert — für Daily-Setups gedacht
- Earnings werden NICHT automatisch geprüft — manueller Pflichtcheck
- S/R-Zonen und Trends sind regelbasierte Heuristiken; die KI-Zweitmeinung ist
  eine Einschätzung, keine Anlageberatung. Entscheidung triffst du am Chart.
