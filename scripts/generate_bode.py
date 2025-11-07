#!/usr/bin/env python3
"""
Bode-Diagramm-Generator: Berechnet die Übertragungsfunktion aus zwei Kanälen.

Liest CSV-Dateien mit mindestens zwei Wertspalten (value_1, value_2),
berechnet die Übertragungsfunktion H(f) = FFT(output) / FFT(input)
und speichert Frequenz, Magnitude und Phase.

Beispiele:
  - Einfache Übertragungsfunktion:
      python generate_bode.py bode_demo
  - Mit Fenster-Funktion:
      python generate_bode.py signal --window hann
  - Explizite Ein-/Ausgabe:
      python generate_bode.py --in measurements/data.csv --out bode/result.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Tuple, Optional

try:
    import numpy as np
except ImportError:
    print("Fehler: numpy ist nicht installiert.")
    print("Bitte installieren mit: pip install numpy")
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
            return "none"
        
        header_lower = [col.lower() for col in header]
        
        # Prüfe auf mehrere Kanäle
        value_cols = [col for col in header_lower if col.startswith("value_")]
        if len(value_cols) >= 2:
            return "multi"
        
        if "timestamp_iso" in header_lower:
            return "iso"
        elif "timestamp_epoch" in header_lower:
            return "epoch"
        elif "t" in header_lower:
            return "relative"
        else:
            return "none"


def read_csv_multi_channel(
    file_path: str, delimiter: str = ","
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Liest CSV-Datei mit zwei Kanälen (value_1, value_2).
    
    Returns:
        channel1: numpy array mit Eingangssignal
        channel2: numpy array mit Ausgangssignal
        fs: Abtastrate in Hz
    """
    csv_format = detect_csv_format(file_path, delimiter)
    
    if csv_format != "multi":
        raise ValueError(
            f"CSV-Datei muss mindestens zwei Kanäle (value_1, value_2) enthalten. "
            f"Erkanntes Format: {csv_format}"
        )
    
    time_values = []
    channel1_values = []
    channel2_values = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        
        if header is None:
            raise ValueError("CSV-Datei hat keine Kopfzeile")
        
        header_lower = [col.lower() for col in header]
        
        # Finde relevante Spalten-Indizes
        time_idx = None
        value1_idx = None
        value2_idx = None
        
        for idx, col in enumerate(header_lower):
            if col == "t":
                time_idx = idx
            elif col == "value_1":
                value1_idx = idx
            elif col == "value_2":
                value2_idx = idx
        
        if value1_idx is None or value2_idx is None:
            raise ValueError("Spalten value_1 und value_2 nicht gefunden")
        
        # Daten einlesen
        for row in reader:
            try:
                if time_idx is not None:
                    time_values.append(float(row[time_idx]))
                else:
                    time_values.append(len(time_values))
                
                channel1_values.append(float(row[value1_idx]))
                channel2_values.append(float(row[value2_idx]))
            except (ValueError, IndexError):
                continue
    
    if len(time_values) < 2:
        raise ValueError("Zu wenige Datenpunkte in der CSV-Datei")
    
    # Abtastrate berechnen
    time_arr = np.array(time_values)
    if time_idx is not None:
        dt = np.mean(np.diff(time_arr))
        fs = 1.0 / dt if dt > 0 else 1.0
    else:
        fs = 1.0
    
    return np.array(channel1_values), np.array(channel2_values), fs


def compute_transfer_function(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    fs: float,
    window: str = "none",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet die Übertragungsfunktion H(f) = FFT(output) / FFT(input).
    
    Args:
        input_signal: Eingangssignal (z.B. Anregung)
        output_signal: Ausgangssignal (z.B. Antwort)
        fs: Abtastrate in Hz
        window: Fensterfunktion ('none', 'hann', 'hamming', 'blackman')
    
    Returns:
        frequencies: Frequenzen in Hz (nur positive)
        magnitude: Betrag der Übertragungsfunktion
        phase: Phase in Grad
    """
    N = len(input_signal)
    
    if len(output_signal) != N:
        raise ValueError("Input- und Output-Signal müssen gleiche Länge haben")
    
    # Fenster anwenden
    if window.lower() == "hann":
        window_func = np.hanning(N)
    elif window.lower() == "hamming":
        window_func = np.hamming(N)
    elif window.lower() == "blackman":
        window_func = np.blackman(N)
    else:
        window_func = np.ones(N)
    
    input_windowed = input_signal * window_func
    output_windowed = output_signal * window_func
    
    # FFT berechnen
    input_fft = np.fft.fft(input_windowed)
    output_fft = np.fft.fft(output_windowed)
    
    # Übertragungsfunktion: H(f) = Y(f) / X(f)
    # Nur positive Frequenzen (einseitig)
    n_pos = N // 2 + 1
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    input_fft_pos = input_fft[:n_pos]
    output_fft_pos = output_fft[:n_pos]
    
    # Division mit kleinem epsilon zur Vermeidung von Division durch Null
    epsilon = 1e-10
    H = output_fft_pos / (input_fft_pos + epsilon)
    
    # Magnitude und Phase extrahieren
    magnitude = np.abs(H)
    phase = np.angle(H, deg=True)
    
    return freqs, magnitude, phase


def save_bode_csv(
    file_path: str,
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    phase: np.ndarray,
    delimiter: str = ",",
) -> None:
    """Speichert Bode-Daten als CSV."""
    out_dir = os.path.dirname(file_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["frequency_hz", "magnitude", "phase_deg"])
        
        for freq, mag, ph in zip(frequencies, magnitude, phase):
            writer.writerow([freq, mag, ph])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Berechnet die Übertragungsfunktion aus zwei Kanälen einer Messdatei.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Dateiname (ohne Endung, liest aus measurements/, schreibt nach bode/)",
    )
    p.add_argument(
        "--in",
        dest="input_file",
        help="Eingabedatei (überschreibt automatische Pfaderstellung)",
    )
    p.add_argument(
        "--out",
        dest="output_file",
        help="Ausgabedatei (überschreibt automatische Pfaderstellung)",
    )
    p.add_argument(
        "--window",
        choices=["none", "hann", "hamming", "blackman"],
        default="none",
        help="Fensterfunktion für FFT",
    )
    p.add_argument(
        "--delimiter",
        default=",",
        help="CSV-Trennzeichen",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Eingabe- und Ausgabepfade bestimmen
    if args.input_file:
        input_path = args.input_file
    elif args.name:
        filename = args.name if args.name.endswith(".csv") else f"{args.name}.csv"
        input_path = os.path.join("..", "data", "measurements", filename)
    else:
        print("Fehler: Entweder 'name' oder '--in' muss angegeben werden")
        sys.exit(1)
    
    if args.output_file:
        output_path = args.output_file
    elif args.name:
        filename = args.name if args.name.endswith(".csv") else f"{args.name}.csv"
        output_path = os.path.join("..", "data", "bode", filename)
    else:
        # Fallback: aus input_path ableiten
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join("..", "data", "bode", f"{base}.csv")
    
    # Prüfe ob Eingabedatei existiert
    if not os.path.exists(input_path):
        print(f"Fehler: Eingabedatei nicht gefunden: {input_path}")
        sys.exit(1)
    
    try:
        # Daten einlesen
        print(f"Lese Datei: {input_path}")
        channel1, channel2, fs = read_csv_multi_channel(input_path, args.delimiter)
        
        print(f"Signal-Länge: {len(channel1)} Samples")
        print(f"Abtastrate: {fs:.2f} Hz")
        print(f"Dauer: {len(channel1)/fs:.3f} s")
        
        # Übertragungsfunktion berechnen
        print("Berechne Übertragungsfunktion...")
        print(f"  Fenster: {args.window}")
        
        freqs, magnitude, phase = compute_transfer_function(
            channel1, channel2, fs, window=args.window
        )
        
        print(f"\nÜbertragungsfunktion berechnet:")
        print(f"  Frequenzauflösung: {freqs[1] - freqs[0]:.4f} Hz")
        print(f"  Frequenzbereich: {freqs[0]:.2f} Hz bis {freqs[-1]:.2f} Hz")
        print(f"  Anzahl Frequenz-Bins: {len(freqs)}")
        
        # Ergebnisse speichern
        save_bode_csv(output_path, freqs, magnitude, phase, delimiter=args.delimiter)
        
        print(f"\nErgebnisse gespeichert: {output_path}")
        
    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
