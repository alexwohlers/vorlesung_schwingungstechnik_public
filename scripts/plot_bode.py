#!/usr/bin/env python3
"""
Plot Bode-Diagramm: Visualisierung der Übertragungsfunktion.

Erstellt ein Bode-Diagramm mit Amplitudengang (in dB) und Phasengang aus
den von generate_bode.py berechneten Daten.

Beispiele:
  - Einfaches Bode-Diagramm:
      python plot_bode.py bode_demo
  - Mit Frequenzbereich 0-10 Hz:
      python plot_bode.py signal --fmax 10
  - Mit Titel:
      python plot_bode.py data --title "Übertragungsfunktion"
  - Als PNG speichern:
      python plot_bode.py signal --save bode_plot.png
  - Explizite Datei:
      python plot_bode.py --file bode/custom.csv --save output.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Tuple, List, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Fehler: Benötigte Bibliothek nicht installiert: {e}")
    print("Bitte installieren mit: pip install matplotlib numpy")
    sys.exit(1)


def read_bode_data(file_path: str, delimiter: str = ",") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Liest Bode-Daten aus CSV.
    
    Returns:
        frequencies: Frequenzen in Hz
        magnitude: Betrag der Übertragungsfunktion
        phase: Phase in Grad
    """
    frequencies = []
    magnitudes = []
    phases = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        if header is None:
            raise ValueError("CSV-Datei hat keine Kopfzeile")
        
        for row in reader:
            try:
                if len(row) >= 3:
                    frequencies.append(float(row[0]))
                    magnitudes.append(float(row[1]))
                    phases.append(float(row[2]))
            except (ValueError, IndexError):
                continue
    
    if len(frequencies) == 0:
        raise ValueError("Keine Daten in der CSV-Datei gefunden")
    
    return np.array(frequencies), np.array(magnitudes), np.array(phases)


def plot_bode_diagram(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    phase: np.ndarray,
    title: str = "Bode-Diagramm",
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Erstellt ein Bode-Diagramm mit Amplituden- und Phasengang.
    
    Args:
        frequencies: Frequenzen in Hz
        magnitude: Betrag der Übertragungsfunktion
        phase: Phase in Grad
        title: Titel des Plots
        fmin: Minimale Frequenz zum Anzeigen
        fmax: Maximale Frequenz zum Anzeigen (None = alle)
        save_path: Pfad zum Speichern (None = anzeigen)
    """
    # Frequenzbereich filtern
    mask = frequencies >= fmin
    if fmax is not None:
        mask &= frequencies <= fmax
    
    freq_filtered = frequencies[mask]
    mag_filtered = magnitude[mask]
    phase_filtered = phase[mask]
    
    if len(freq_filtered) == 0:
        print("Warnung: Keine Daten im angegebenen Frequenzbereich")
        return
    
    # Magnitude in dB umrechnen
    # Vermeide log(0) durch Hinzufügen eines kleinen epsilon
    epsilon = 1e-10
    mag_db = 20 * np.log10(mag_filtered + epsilon)
    
    # Plot erstellen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Amplitudengang (in dB)
    ax1.semilogx(freq_filtered, mag_db, 'b-', linewidth=2)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("Magnitude [dB]", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # Phasengang (in Grad)
    ax2.semilogx(freq_filtered, phase_filtered, 'r-', linewidth=2)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel("Frequenz [Hz]", fontsize=12)
    ax2.set_ylabel("Phase [°]", fontsize=12)
    ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    
    # Y-Achsen-Bereich für Phase setzen
    ax2.set_ylim([-180, 180])
    
    plt.tight_layout()
    
    # Speichern oder anzeigen
    if save_path:
        # Verzeichnis erstellen falls nötig
        out_dir = os.path.dirname(save_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {save_path}")
        plt.close()
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Erstellt ein Bode-Diagramm aus Übertragungsfunktions-Daten.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Dateiname (ohne Endung, liest aus bode/, speichert nach plot_bode/)",
    )
    p.add_argument(
        "--file",
        dest="input_file",
        help="Eingabedatei (überschreibt automatische Pfaderstellung)",
    )
    p.add_argument(
        "--title",
        default="Bode-Diagramm",
        help="Titel des Plots",
    )
    p.add_argument(
        "--fmin",
        type=float,
        default=0.1,
        help="Minimale Frequenz zum Anzeigen [Hz]",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Maximale Frequenz zum Anzeigen [Hz]",
    )
    p.add_argument(
        "--save",
        dest="save_path",
        default=None,
        help="Speicherpfad für Plot (z.B. plot.png). Ohne Angabe wird angezeigt.",
    )
    p.add_argument(
        "--delimiter",
        default=",",
        help="CSV-Trennzeichen",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Eingabepfad bestimmen
    if args.input_file:
        input_path = args.input_file
    elif args.name:
        filename = args.name if args.name.endswith(".csv") else f"{args.name}.csv"
        input_path = os.path.join("..", "data", "bode", filename)
    else:
        print("Fehler: Entweder 'name' oder '--file' muss angegeben werden")
        sys.exit(1)
    
    # Ausgabepfad bestimmen (falls --save ohne Verzeichnis)
    save_path = args.save_path
    if save_path and not os.path.dirname(save_path):
        save_path = os.path.join("..", "bilder", "plot_bode", save_path)
    
    # Automatischer Speicherpfad wenn nur Name angegeben
    if args.name and args.save_path is None:
        base_name = args.name if not args.name.endswith(".csv") else args.name[:-4]
        save_path = os.path.join("..", "bilder", "plot_bode", f"{base_name}.png")
    
    # Prüfe ob Eingabedatei existiert
    if not os.path.exists(input_path):
        print(f"Fehler: Eingabedatei nicht gefunden: {input_path}")
        print(f"Tipp: Erst 'python generate_bode.py {args.name}' ausführen")
        sys.exit(1)
    
    try:
        # Daten einlesen
        print(f"Lese Bode-Daten: {input_path}")
        frequencies, magnitude, phase = read_bode_data(input_path, args.delimiter)
        
        print(f"Anzahl Frequenzpunkte: {len(frequencies)}")
        print(f"Frequenzbereich: {frequencies[0]:.2f} Hz bis {frequencies[-1]:.2f} Hz")
        
        # Bode-Diagramm erstellen
        plot_bode_diagram(
            frequencies,
            magnitude,
            phase,
            title=args.title,
            fmin=args.fmin,
            fmax=args.fmax,
            save_path=save_path,
        )
        
    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
