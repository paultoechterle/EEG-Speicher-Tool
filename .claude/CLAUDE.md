# Hintergrund
Wir entwickeln ein Optimierungs-Tool für Erneuerbare Energie Gemeinschaften (EEGs).
Die meisten EEGs haben keine eigenen Stromspeicher. Um den Autarkiegrad und
den Eigenverbrauch zu optimieren, sind solche Speicher aber sinnvoll. Das Tool
soll es ermöglichen die optimale Speichergröße zu dimensionieren.

# Datengrundlage
## Messdaten
Die Grundlage bilden reale Messwerte aus EEGs, die als CSV files hochgeladen 
werden. Als Beispieldaten liegen von zwei EEGs (Eben am Achensee, Terfens) Last- und 
Erzeugungsprofile (15-min) der einzelnen Zählerpunkte vor. Die Zeitreihen 
umfassen die Variablen
* Zeitpunkt
* Gesamtbezug [kWh]
* Effektiv aus Gemeinschaft bezogen [kWh]
* Restbezug [kWh]
* Gesamtlieferung [kWh]
* Effektiv an Gemeinschaft geliefert [kWh]
* Restlieferung [kWh]

Alle Daten haben die selbe Datenstruktur wie die Beispieldaten, nur mit 
unterschiedlichen Zeiträumen und unterschiedlich vielen Zählpunkten.

**Achtung**: Die gemessenen Daten erfassen nur den Anteil von Erzeugung und Verbauch ab dem
Zählpunkt. Eigenverbrauch innerhalb eines Hauses wird beispielsweise nicht 
erfasst. Die Synthetischen Last- und Erzeugungsprofile sollen helfen hier ein
vollständigeres Bild zu schaffen.

## Synthetische Werte
Zusätzlich sollen synthetische Last- und Erzeugungsprofile erstellt werden können,
um auch fiktive Szenarien simulieren zu können.

# Speicherlogik
Der Kern des Modells is eine Lade- und Entlade Logik für einen zu simulierenden 
Stromspeicher. Dieser Speicher hat eine gewisse Kapazität und Leistung (sinnvolle 
Standardwerte für Batteriespeicher verwenden) und lädt sich auf, sobald mehr
Strom erzeugt als verbraucht wird, bis der Speicher voll ist. Wenn weniger Strom
erzeugt als verbraucht wird, entlädt sich der Speicher wieder.

Diese Speicherlogik ist bewusst nicht "netzdienlich". Netzdienliches Verhalten
kann als späterer Erweiterungsschritt angedacht werden, hat aber erstmal keine 
Priorität.

# Optimierung (Grid-search)
Basierend auf den Last- und Erzeugungsprofilen sollen diverse Szenarien mit
unterschiedlichen Speichergrößen simuliert werden. Als zweite
Optimierungs-variable kommt der Ausbau der Erzeugung hinzu (beschränkt auf 
PV-Ausbau).

Es soll verschiedene Optimierungs-targets geben.
* Autarkie: Netzbezug (in kWh), Schwellenwert
* Autarkie: Zeitschritte ohne Netzbezug (in %), Schwellenwert
* Eigenverbrauch: (in %), Schwellenwert.

# User Interface
Es soll mit Streamlit ein User Interface erstellt werden, wobei vor allem die 
Optimierung im Fordergrund stehen soll.

# Python Environment
Nutze das **`hydrology_env`** conda environment für dieses Projekt. Do NOT use the 
bare `python` command — it resolves to the Microsoft Store stub and fails.

Interpreter path:

    C:\Users\paul.toechterle\.conda\envs\hydrology_env\python.exe

Run a script:

    & C:\Users\paul.toechterle\.conda\envs\hydrology_env\python.exe path\to\script.py

Frühre ein `requirements.txt` file um zu tracken, was du alles für packages 
brauchst.

---