# Grundsatzfrage: Netzdienliches Verhalten in der Speichersimulation

## Kontext

Die aktuelle Speicherlogik in `simulate_storage()` (`core.py:55-149`) ist
bewusst **nicht netzdienlich**, sondern rein eigenverbrauchsoptimiert:
Der Speicher lädt bei lokalem Erzeugungsüberschuss und entlädt bei
lokalem Bedarfsüberschuss (`core.py:114-136`). Es gibt keinerlei Bezug
zu einem externen Signal (Preis, Netzauslastung, Redispatch).

Diese Notiz hält fest, was grundsätzlich nötig wäre, um netzdienliches
Verhalten zu simulieren — als spätere Erweiterung, aktuell keine
Priorität (siehe Projekt-CLAUDE.md, Abschnitt "Speicherlogik").

## Was "netzdienlich" bedeuten kann

Der Begriff ist nicht eindeutig — mögliche Auslegungen:

1. **Preisorientiert**: Laden bei niedrigen, Entladen bei hohen
   Spotmarktpreisen (Day-Ahead), unabhängig vom lokalen Über-/Unterschuss.
2. **Netzauslastungsorientiert**: Reaktion auf Signale des
   Netzbetreibers (z.B. Trafo-/Leitungsauslastung, Redispatch-Aufruf).
3. **Peak-Shaving**: Begrenzung von Netzbezugs-/Einspeisespitzen auf
   ein vorgegebenes Limit, unabhängig vom Preis.

Diese drei Varianten erfordern unterschiedliche Eingangsdaten und
Steuerlogiken — die Entscheidung, welche(s) davon relevant ist, sollte
vor einer Umsetzung getroffen werden.

## Was sich am Modell ändern müsste

### 1. Zusätzliches externes Signal als Input
Aktuell bekommt `simulate_storage()` nur `generation` und `load`. Für
Netzdienlichkeit braucht es eine dritte Zeitreihe (Preis- oder
Netzsignal), die im Datenmodell bisher nicht existiert — weder in den
Messdaten noch in den synthetischen Profilen.

### 2. Steuerungslogik von reaktiv auf vorausschauend
Die jetzige Schleife (`core.py:114-136`) ist myopisch: Sie kennt nur
den aktuellen Zeitschritt. Netzdienliches Verhalten braucht oft
Vorausschau (z.B. "lade jetzt, weil in 3h ein Preis-Peak kommt"). Zwei
Optionen:
- **Regelbasiert** mit Schwellenwerten (einfach, aber suboptimal).
- **Horizont-Optimierung** (z.B. Day-Ahead-Planung der SoC-Trajektorie
  per `scipy.optimize.linprog`), die Kapazitäts- und Leistungsgrenzen
  als Nebenbedingungen berücksichtigt.

Architektonisch würde das bedeuten, `simulate_storage()` von einer
festen Regel auf eine **austauschbare Steuerungsstrategie**
umzubauen: bestehende Regel bleibt als Modus `self_consumption`
erhalten, neue Modi (`price_optimized`, `peak_shaving`, ...) kommen
hinzu.

### 3. Zielkonflikt mit dem bestehenden Optimierungsziel
Eigenverbrauchsoptimierung und Netzdienlichkeit stehen teilweise im
Widerspruch: Der Speicher müsste sich unter Umständen entladen, obwohl
gerade lokaler Überschuss besteht, weil das externe Signal es
verlangt. Es bräuchte also entweder:
- ein neues Optimierungsziel/KPI (z.B. "minimierte Netzbezugsspitze",
  "Preiskosten") zusätzlich zu den bestehenden drei Targets
  (Autarkie kWh, Autarkie %, Eigenverbrauch %), oder
- eine gewichtete Kombination aus lokalem und netzseitigem Ziel.

## Fazit

Der größte strukturelle Eingriff wäre der Umbau der
Lade-/Entladelogik von einer festen Regel auf eine austauschbare
Steuerungsstrategie, plus die Notwendigkeit einer zusätzlichen
Preis-/Signal-Zeitreihe als Dateninput. Vor einer Umsetzung sollte
geklärt werden, welche Auslegung von "netzdienlich" (preis-,
netzauslastungs- oder peak-shaving-orientiert) für die EEGs relevant
ist.

Reine Grundsatzfrage — keine Implementierung in diesem Schritt.
