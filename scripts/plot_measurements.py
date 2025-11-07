#!/usr/bin/env python3
"""
Plot-Messdaten: Visualisierung von CSV-Messdaten aus dem measurements/ Verzeichnis.

Plots werden automatisch als PNG im plot_measurements/ Verzeichnis gespeichert.

Beispiele:
  - Einfacher Plot (speichert als plot_measurements/sample.png):
      python plot_measurements.py sample
  - Mit Titel und Beschriftungen:
      python plot_measurements.py sinus --title "Sinus 1 Hz" --xlabel "Zeit [s]" --ylabel "Amplitude"
  - Mehrere Dateien überlagert:
      python plot_measurements.py sinus cosinus --title "Vergleich"
  - Mit eigenem Dateinamen speichern:
      python plot_measurements.py data --save-as mein_plot.png
  - Mit Grid und Marker:
      python plot_measurements.py sample --marker o
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Fehler: matplotlib ist nicht installiert.")
    print("Bitte installieren mit: pip install matplotlib")
    sys.exit(1)


def detect_csv_format(file_path: str, delimiter: str = ",") -> str:
    """
    Erkennt das CSV-Format anhand der Spaltenüberschriften.
    Rückgabe: 'relative', 'iso', 'epoch', 'none', 'multi'
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        if header is None:
            return "none"  # Keine Kopfzeile
        
        header_lower = [col.lower() for col in header]
        
        # Prüfe auf mehrere Kanäle (value_1, value_2, ...)
        value_cols = [col for col in header_lower if col.startswith("value_")]
        if len(value_cols) > 1:
            return "multi"
        
        if "timestamp_iso" in header_lower:
            return "iso"
        elif "timestamp_epoch" in header_lower:
            return "epoch"
        elif "t" in header_lower:
            return "relative"
        elif len(header) == 2 and "index" in header_lower and "value" in header_lower:
            return "none"
        else:
            # Fallback: versuche zu raten
            if len(header) >= 3:
                return "relative"
            else:
                return "none"


def read_csv_data(file_path: str, delimiter: str = ",") -> Tuple[List[float], List[List[float]], str, List[str]]:
    """
    Liest CSV-Daten und gibt (x_values, y_values_list, format, channel_names) zurück.
    x_values: Zeit oder Index
    y_values_list: Liste von Listen mit Messwerten (ein Eintrag pro Kanal)
    format: 'relative', 'iso', 'epoch', 'none', 'multi'
    channel_names: Liste mit Namen der Kanäle
    """
    csv_format = detect_csv_format(file_path, delimiter)
    
    x_values = []
    y_values_list = []  # Liste von Listen für mehrere Kanäle
    channel_names = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        # Wenn keine Kopfzeile oder Format 'none', verwende Index als x
        if header is None or csv_format == "none":
            # Zurücksetzen wenn header gelesen wurde
            f.seek(0)
            if header is not None and "index" in [h.lower() for h in header]:
                next(reader)  # Header überspringen
            
            # Ein Kanal
            y_values_list.append([])
            channel_names.append("value")
            
            for idx, row in enumerate(reader):
                try:
                    if len(row) >= 2:
                        x_values.append(float(row[0]))  # index
                        y_values_list[0].append(float(row[-1]))  # letzter Wert ist value
                    elif len(row) == 1:
                        x_values.append(idx)
                        y_values_list[0].append(float(row[0]))
                except (ValueError, IndexError):
                    continue
        
        elif csv_format == "multi":
            # Mehrere Kanäle (value_1, value_2, ...)
            header_lower = [col.lower() for col in header]
            
            # Finde Zeit-Spalte und Wert-Spalten
            time_idx = None
            value_indices = []
            
            for idx, col in enumerate(header_lower):
                if col == "t":
                    time_idx = idx
                elif col.startswith("value_"):
                    value_indices.append(idx)
                    channel_names.append(header[idx])  # Original-Name
            
            # Initialisiere Listen für jeden Kanal
            for _ in value_indices:
                y_values_list.append([])
            
            # Daten lesen
            for row in reader:
                try:
                    if time_idx is not None:
                        x_values.append(float(row[time_idx]))
                    else:
                        x_values.append(len(x_values))
                    
                    for i, val_idx in enumerate(value_indices):
                        y_values_list[i].append(float(row[val_idx]))
                except (ValueError, IndexError):
                    continue
        
        elif csv_format == "relative":
            # Ein Kanal mit relativer Zeit
            y_values_list.append([])
            channel_names.append("value")
            
            for row in reader:
                try:
                    if len(row) >= 3:
                        x_values.append(float(row[1]))  # t (Zeit in Sekunden)
                        y_values_list[0].append(float(row[2]))  # value
                except (ValueError, IndexError):
                    continue
        
        elif csv_format == "iso":
            # Ein Kanal mit ISO-Zeitstempel
            y_values_list.append([])
            channel_names.append("value")
            
            for row in reader:
                try:
                    if len(row) >= 3:
                        # ISO-Zeitstempel in datetime konvertieren
                        ts_str = row[1]
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        x_values.append(dt)
                        y_values_list[0].append(float(row[2]))
                except (ValueError, IndexError):
                    continue
        
        elif csv_format == "epoch":
            # Ein Kanal mit Epoch-Zeitstempel
            y_values_list.append([])
            channel_names.append("value")
            
            for row in reader:
                try:
                    if len(row) >= 3:
                        # Epoch zu datetime konvertieren
                        epoch = float(row[1])
                        dt = datetime.fromtimestamp(epoch)
                        x_values.append(dt)
                        y_values_list[0].append(float(row[2]))
                except (ValueError, IndexError):
                    continue
    
    return x_values, y_values_list, csv_format, channel_names


def plot_data(
    files: List[str],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = True,
    marker: Optional[str] = None,
    linestyle: str = "-",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    delimiter: str = ",",
) -> None:
    """
    Plottet Messdaten aus einer oder mehreren CSV-Dateien.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    has_datetime = False
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0
    
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warnung: Datei '{file_path}' nicht gefunden, überspringe...")
            continue
        
        try:
            x_values, y_values_list, csv_format, channel_names = read_csv_data(file_path, delimiter)
            
            if len(x_values) == 0:
                print(f"Warnung: Keine Daten in '{file_path}' gefunden.")
                continue
            
            # Dateiname als Basis-Label
            file_label = Path(file_path).stem
            
            # Jeden Kanal plotten
            for i, (y_values, channel_name) in enumerate(zip(y_values_list, channel_names)):
                # Label erstellen
                if len(y_values_list) > 1:
                    # Mehrere Kanäle: zeige Kanal-Name
                    if len(files) > 1:
                        label = f"{file_label} - {channel_name}"
                    else:
                        label = channel_name
                else:
                    # Ein Kanal: nur Dateiname
                    label = file_label
                
                # Plot-Optionen
                plot_kwargs = {
                    "label": label,
                    "linestyle": linestyle,
                    "color": color_cycle[color_idx % len(color_cycle)],
                }
                if marker:
                    plot_kwargs["marker"] = marker
                    plot_kwargs["markersize"] = 4
                
                # Prüfen ob datetime-Objekte
                if csv_format in ("iso", "epoch"):
                    has_datetime = True
                
                ax.plot(x_values, y_values, **plot_kwargs)
                color_idx += 1
        
        except Exception as e:
            print(f"Fehler beim Lesen von '{file_path}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Achsenbeschriftungen
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    else:
        # Auto-Label basierend auf Format
        if has_datetime:
            ax.set_xlabel("Zeit", fontsize=12)
        else:
            ax.set_xlabel("Zeit [s]", fontsize=12)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel("Wert", fontsize=12)
    
    # Titel
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        if len(files) == 1:
            ax.set_title(f"Messdaten: {Path(files[0]).name}", fontsize=14)
        else:
            ax.set_title("Messdaten-Vergleich", fontsize=14)
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--")
    
    # Legende wenn mehrere Dateien oder mehrere Kanäle
    # Prüfe ob es überhaupt mehrere Linien gibt
    lines = ax.get_lines()
    if len(lines) > 1:
        ax.legend(loc="best", framealpha=0.9)
    
    # Datetime-Formatierung für X-Achse
    if has_datetime:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()  # Automatische Drehung der Zeitstempel
    
    plt.tight_layout()
    
    # Speichern oder anzeigen
    if save_path:
        # Sicherstellen, dass Ausgabeverzeichnis existiert
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plottet Messdaten aus CSV-Dateien im measurements/ Verzeichnis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "files",
        nargs="+",
        help="Dateiname(n) ohne Verzeichnis und Endung (z.B. 'sinus' oder 'data1 data2')",
    )
    p.add_argument("--title", help="Titel des Plots")
    p.add_argument("--xlabel", help="Beschriftung der X-Achse")
    p.add_argument("--ylabel", help="Beschriftung der Y-Achse")
    p.add_argument("--grid", action="store_true", default=True, help="Gitter anzeigen")
    p.add_argument("--no-grid", action="store_true", help="Gitter ausblenden")
    p.add_argument("--marker", help="Marker-Stil (z.B. 'o', 's', '^', 'x')")
    p.add_argument("--linestyle", default="-", help="Linien-Stil (z.B. '-', '--', '-.', ':')")
    p.add_argument("--figsize", nargs=2, type=int, default=[12, 6], help="Größe der Figur (Breite Höhe)")
    p.add_argument("--save-as", dest="save_path", help="Eigener Dateiname für Plot (wird in plot_measurements/ gespeichert)")
    p.add_argument("--delimiter", default=",", help="CSV-Trennzeichen")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Grid-Behandlung
    grid = args.grid and not args.no_grid
    
    # Pfade zu Dateien verarbeiten
    # Namen in vollständige Pfade umwandeln
    files = []
    for name in args.files:
        # Wenn bereits vollständiger Pfad, verwende ihn direkt
        if os.path.exists(name):
            files.append(name)
            continue
        
        # Sonst: Name ohne Endung -> measurements/<name>.csv
        # Endung entfernen falls vorhanden
        if name.endswith('.csv'):
            base_name = name[:-4]
        else:
            base_name = name
        
        # Pfad erstellen
        file_path = os.path.join("..", "data", "measurements", f"{base_name}.csv")
        
        if not os.path.exists(file_path):
            print(f"Warnung: Datei '{file_path}' nicht gefunden!")
            sys.exit(1)
        
        files.append(file_path)
    
    # Speicherpfad verarbeiten - immer speichern in plot_measurements/
    if args.save_path:
        # Eigener Dateiname angegeben
        if not os.path.dirname(args.save_path):
            save_path = os.path.join("..", "bilder", "plot_measurements", args.save_path)
        else:
            save_path = args.save_path
    else:
        # Standard: plot_measurements/<erster_dateiname>.png
        base_name = Path(files[0]).stem
        save_path = os.path.join("..", "bilder", "plot_measurements", f"{base_name}.png")
    
    try:
        plot_data(
            files=files,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            grid=grid,
            marker=args.marker,
            linestyle=args.linestyle,
            figsize=tuple(args.figsize),
            save_path=save_path,
            delimiter=args.delimiter,
        )
    except Exception as exc:
        print(f"Fehler beim Plotten: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
