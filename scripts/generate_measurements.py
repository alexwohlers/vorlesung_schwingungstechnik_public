#!/usr/bin/env python3
"""
Messdaten-Generator: Werte einer Funktion über der Zeit erzeugen und als CSV speichern.

Beispiele:
  - Sinus 1 Hz, 5 s bei 100 Hz:
      python generate_measurements.py sinus --func "sin(2*pi*1*t)" --fs 100 --duration 5
  - Mit Rauschen (σ=0.05), ISO-Zeitstempel:
      python generate_measurements.py mixed --func "0.5*sin(2*pi*0.2*t) + 0.2*cos(2*pi*1.5*t)" --fs 50 --duration 10 --noise 0.05 --timestamp iso
  - Rechteck über sign-Funktion (Vorzeichen von sin):
      python generate_measurements.py rechteck --func "sign(sin(2*pi*2*t))" --fs 200 --duration 2
  - Expliziter Ausgabepfad (überschreibt automatische Benennung):
      python generate_measurements.py --func "sin(2*pi*t)" --fs 100 --duration 1 --out custom/path.csv

Die Funktions-Variable ist t (Zeit in Sekunden ab 0) und n (Abtastindex).
Verfügbare Funktionen/Konstanten: sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh,
exp, log, log10, sqrt, floor, ceil, abs, min, max, clamp, sign, pi, tau, e.

Sicherheitshinweis: Ausdrücke werden in einer stark eingeschränkten Umgebung ausgewertet
(ohne Builtins), aber übergeben Sie nur vertrauenswürdige Ausdrücke.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Optional


def _build_eval_env(t: float, n: int) -> Dict[str, Any]:
    """Erlaubte Umgebung für den Funktionsausdruck."""
    # Mathematische Funktionen bereitstellen
    env = {
        # Variablen
        "t": t,
        "n": n,
        # Konstanten
        "pi": math.pi,
        "tau": math.tau,
        "e": math.e,
        # Standardfunktionen
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "floor": math.floor,
        "ceil": math.ceil,
        "abs": abs,
        "min": min,
        "max": max,
        # Nützliche Helfer
        "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
        "sign": lambda x: (0 if x == 0 else (1 if x > 0 else -1)),
    }
    return env


def _evaluate_func(expr: str, t: float, n: int) -> float:
    """Bewertet den Ausdruck expr mit Variablen t und n in einer eingeschränkten Umgebung."""
    try:
        # Keine Builtins zulassen, nur unsere env
        val = eval(expr, {"__builtins__": {}}, _build_eval_env(t, n))
    except Exception as exc:
        raise ValueError(f"Fehler beim Auswerten des Ausdrucks bei t={t}, n={n}: {exc}") from exc
    try:
        return float(val)
    except Exception as exc:
        raise ValueError(f"Ausdruck ergab keinen numerischen Wert (Typ {type(val)}): {val}") from exc


def generate_csv(
    expr: str,
    fs: float,
    duration: Optional[float],
    samples: Optional[int],
    out_path: str,
    amplitude: float = 1.0,
    offset: float = 0.0,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
    timestamp_mode: str = "relative",
    delimiter: str = ",",
    header: bool = True,
    encoding: str = "utf-8",
) -> None:
    if fs <= 0:
        raise ValueError("Abtastrate fs muss > 0 sein.")
    
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    import os
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if samples is None:
        if duration is None:
            raise ValueError("Entweder --duration oder --samples angeben.")
        if duration < 0:
            raise ValueError("duration muss >= 0 sein.")
        samples = int(round(duration * fs))
    else:
        if samples < 0:
            raise ValueError("samples muss >= 0 sein.")

    if seed is not None:
        random.seed(seed)

    base_time = datetime.now(timezone.utc)

    # CSV schreiben
    with open(out_path, "w", newline="", encoding=encoding) as f:
        writer = csv.writer(f, delimiter=delimiter)

        # Kopfzeile
        if header:
            if timestamp_mode == "none":
                writer.writerow(["index", "value"])  # ohne Zeit
            elif timestamp_mode == "relative":
                writer.writerow(["index", "t", "value"])  # t in Sekunden
            elif timestamp_mode in ("iso", "epoch"):
                col = "timestamp_iso" if timestamp_mode == "iso" else "timestamp_epoch"
                writer.writerow(["index", col, "value"])  # absolute Zeit
            else:
                raise ValueError("Ungültiger timestamp_mode. Erlaubt: none|relative|iso|epoch")

        # Datenzeilen
        for n in range(samples):
            t = n / fs
            y = _evaluate_func(expr, t, n)
            y = offset + amplitude * y
            if noise_std > 0:
                y += random.gauss(0.0, noise_std)

            if timestamp_mode == "none":
                row = [n, y]
            elif timestamp_mode == "relative":
                row = [n, t, y]
            elif timestamp_mode == "iso":
                ts = (base_time).astimezone(timezone.utc)  # Basiszeit UTC
                # pro Sample versetzt
                ts = ts.timestamp() + t
                ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                row = [n, ts_iso, y]
            elif timestamp_mode == "epoch":
                row = [n, base_time.timestamp() + t, y]
            else:
                raise ValueError("Ungültiger timestamp_mode. Erlaubt: none|relative|iso|epoch")

            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generiert Messdaten aus einem Funktionsausdruck und speichert sie als CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("name", nargs="?", default="data", help="Dateiname (ohne Endung, wird automatisch in measurements/ gespeichert)")
    p.add_argument("--func", action="append", required=False, help="Funktionsausdruck in t (und optional n). Mehrfach verwendbar für mehrere Kanäle.")
    p.add_argument("--fs", type=float, default=100.0, help="Abtastrate in Hz")
    p.add_argument("--duration", type=float, help="Dauer in Sekunden (alternativ --samples)")
    p.add_argument("--samples", type=int, help="Anzahl Samples (überschreibt --duration)")
    p.add_argument("--amplitude", type=float, default=1.0, help="Amplitude (multiplikativ)")
    p.add_argument("--offset", type=float, default=0.0, help="Offset (additiv)")
    p.add_argument("--noise", dest="noise_std", type=float, default=0.0, help="Rausch-Std.-Abweichung (Gaussian)")
    p.add_argument("--seed", type=int, help="Zufalls-Seed für Reproduzierbarkeit")
    p.add_argument(
        "--timestamp",
        choices=["none", "relative", "iso", "epoch"],
        default="relative",
        help="Zeitstempelspalte: keine, relative Sekunden, ISO 8601, oder Epoch-seconds",
    )
    p.add_argument("--delimiter", default=",", help="CSV-Trennzeichen, z.B. ',' oder ';'")
    p.add_argument("--no-header", action="store_true", help="Keine Kopfzeile schreiben")
    p.add_argument("--out", help="Ausgabedatei (überschreibt automatische Pfaderstellung)")
    p.add_argument("--encoding", default="utf-8", help="Datei-Encoding")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Ausgabepfad bestimmen
    import os
    if args.out:
        # Explizit angegebener Pfad
        out_path = args.out
    else:
        # Automatisch aus Name erstellen
        # Sicherstellen dass .csv Endung hinzugefügt wird
        filename = args.name if args.name.endswith(".csv") else f"{args.name}.csv"
        out_path = os.path.join("..", "data", "measurements", filename)

    try:
        func_list = args.func if args.func else ["sin(2*pi*1*t)"]
        generate_csv_multi(
            expr_list=func_list,
            fs=args.fs,
            duration=args.duration,
            samples=args.samples,
            out_path=out_path,
            amplitude=args.amplitude,
            offset=args.offset,
            noise_std=args.noise_std,
            seed=args.seed,
            timestamp_mode=args.timestamp,
            delimiter=args.delimiter,
            header=(not args.no_header),
            encoding=args.encoding,
        )
    except Exception as exc:
        print(f"Fehler: {exc}")
        raise SystemExit(1)

    print(
        f"Fertig. Datei '{out_path}' geschrieben. Ausdrücke={func_list}, fs={args.fs}, "
        f"samples={args.samples if args.samples is not None else ('~'+str(int(round((args.duration or 0)*args.fs))))}, "
        f"timestamp={args.timestamp}"
    )


def generate_csv_multi(
        expr_list: list[str],
        fs: float,
        duration: Optional[float],
        samples: Optional[int],
        out_path: str,
        amplitude: float = 1.0,
        offset: float = 0.0,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        timestamp_mode: str = "relative",
        delimiter: str = ",",
        header: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        if fs <= 0:
            raise ValueError("Abtastrate fs muss > 0 sein.")
        import os
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        if samples is None:
            if duration is None:
                raise ValueError("Entweder --duration oder --samples angeben.")
            if duration < 0:
                raise ValueError("duration muss >= 0 sein.")
            samples = int(round(duration * fs))
        else:
            if samples < 0:
                raise ValueError("samples muss >= 0 sein.")

        if seed is not None:
            random.seed(seed)

        base_time = datetime.now(timezone.utc)

        # CSV schreiben
        with open(out_path, "w", newline="", encoding=encoding) as f:
            writer = csv.writer(f, delimiter=delimiter)

            # Kopfzeile
            if header:
                header_row = ["index"]
                if timestamp_mode == "relative":
                    header_row.append("t")
                elif timestamp_mode == "iso":
                    header_row.append("timestamp_iso")
                elif timestamp_mode == "epoch":
                    header_row.append("timestamp_epoch")
                # Kanäle
                for i in range(len(expr_list)):
                    header_row.append(f"value_{i+1}")
                writer.writerow(header_row)

            # Datenzeilen
            for n in range(samples):
                t = n / fs
                values = []
                for expr in expr_list:
                    y = _evaluate_func(expr, t, n)
                    y = offset + amplitude * y
                    if noise_std > 0:
                        y += random.gauss(0.0, noise_std)
                    values.append(y)

                row = [n]
                if timestamp_mode == "relative":
                    row.append(t)
                elif timestamp_mode == "iso":
                    ts = (base_time).astimezone(timezone.utc)
                    ts = ts.timestamp() + t
                    ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    row.append(ts_iso)
                elif timestamp_mode == "epoch":
                    row.append(base_time.timestamp() + t)
                row.extend(values)
                writer.writerow(row)


if __name__ == "__main__":
    main()
