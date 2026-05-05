"""Build a stress-strain curve database from Tau.evol files.

The current project tree groups curves by morphology and crystallinity, e.g.:

    /data/hlobato/cryst/lattice/ipp_10/in.def_7.4e-8/Tau.evol
    /data/hlobato/cryst/random/ipp_90/in.def_7.4e-7/Tau.evol

When executed as a script, this module recursively scans the input root,
extracts curve metadata and summary properties, and writes:

- curves_summary.json: one entry per curve with the processed arrays and metrics
- curves_summary.csv: a tabular summary for downstream ML work
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import uniform_filter1d
except ImportError:  # pragma: no cover - optional dependency fallback
    uniform_filter1d = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:  # pragma: no cover - optional dependency fallback
    lowess = None

MAX_STRAIN = 3.0
SMOOTH_SPAN = 0.05
MIN_WINDOW_SIZE = 10
WINDOW_FRACTION = 0.02
SLOPE_METHOD_THRESHOLD = 0.3
DEFAULT_INPUT_ROOT = "/data/hlobato/cryst"


class CurveMetadata:
    """Path-derived metadata for a single Tau.evol file."""

    def __init__(
        self,
        curve_id,
        path,
        relative_path,
        file_name,
        structure,
        crystallinity_pct,
        crystallinity,
        deformation_rate_label,
        deformation_rate,
    ):
        self.curve_id = curve_id
        self.path = path
        self.relative_path = relative_path
        self.file_name = file_name
        self.structure = structure
        self.crystallinity_pct = crystallinity_pct
        self.crystallinity = crystallinity
        self.deformation_rate_label = deformation_rate_label
        self.deformation_rate = deformation_rate


class CurveSummary:
    """Processed data and summary metrics for a curve."""

    def __init__(
        self,
        curve_id,
        path,
        relative_path,
        file_name,
        structure,
        crystallinity_pct,
        crystallinity,
        deformation_rate_label,
        deformation_rate,
        n_raw,
        n_original,
        n_smoothed,
        n_cut,
        raw_lambda,
        raw_stress_atm,
        strain,
        stress_mpa,
        smoothed_strain,
        smoothed_stress_mpa,
        yield_strain,
        yield_stress,
        youngs_modulus,
        end_linear_strain,
        end_linear_stress,
        ultimate_strain,
        ultimate_stress,
    ):
        self.curve_id = curve_id
        self.path = path
        self.relative_path = relative_path
        self.file_name = file_name
        self.structure = structure
        self.crystallinity_pct = crystallinity_pct
        self.crystallinity = crystallinity
        self.deformation_rate_label = deformation_rate_label
        self.deformation_rate = deformation_rate
        self.n_raw = n_raw
        self.n_original = n_original
        self.n_smoothed = n_smoothed
        self.n_cut = n_cut
        self.raw_lambda = raw_lambda
        self.raw_stress_atm = raw_stress_atm
        self.strain = strain
        self.stress_mpa = stress_mpa
        self.smoothed_strain = smoothed_strain
        self.smoothed_stress_mpa = smoothed_stress_mpa
        self.yield_strain = yield_strain
        self.yield_stress = yield_stress
        self.youngs_modulus = youngs_modulus
        self.end_linear_strain = end_linear_strain
        self.end_linear_stress = end_linear_stress
        self.ultimate_strain = ultimate_strain
        self.ultimate_stress = ultimate_stress

    def to_json_dict(self):
        return {
            "curve_id": self.curve_id,
            "path": self.path,
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "structure": self.structure,
            "crystallinity_pct": self.crystallinity_pct,
            "crystallinity": self.crystallinity,
            "deformation_rate_label": self.deformation_rate_label,
            "deformation_rate": self.deformation_rate,
            "n_raw": self.n_raw,
            "n_original": self.n_original,
            "n_smoothed": self.n_smoothed,
            "n_cut": self.n_cut,
            "raw_lambda": self.raw_lambda,
            "raw_stress_atm": self.raw_stress_atm,
            "strain": self.strain,
            "stress_mpa": self.stress_mpa,
            "smoothed_strain": self.smoothed_strain,
            "smoothed_stress_mpa": self.smoothed_stress_mpa,
            "yield_strain": self.yield_strain,
            "yield_stress": self.yield_stress,
            "youngs_modulus": self.youngs_modulus,
            "end_linear_strain": self.end_linear_strain,
            "end_linear_stress": self.end_linear_stress,
            "ultimate_strain": self.ultimate_strain,
            "ultimate_stress": self.ultimate_stress,
        }

    def to_csv_row(self):
        return {
            "curve_id": self.curve_id,
            "path": self.path,
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "structure": self.structure,
            "crystallinity_pct": self.crystallinity_pct,
            "crystallinity": self.crystallinity,
            "deformation_rate_label": self.deformation_rate_label,
            "deformation_rate": self.deformation_rate,
            "n_raw": self.n_raw,
            "n_original": self.n_original,
            "n_smoothed": self.n_smoothed,
            "n_cut": self.n_cut,
            "yield_strain": self.yield_strain,
            "yield_stress": self.yield_stress,
            "youngs_modulus": self.youngs_modulus,
            "end_linear_strain": self.end_linear_strain,
            "end_linear_stress": self.end_linear_stress,
            "ultimate_strain": self.ultimate_strain,
            "ultimate_stress": self.ultimate_stress,
        }


def _find_structure(parts: Tuple[str, ...]):
    for part in parts:
        lowered = part.lower()
        if lowered in {"lattice", "random"}:
            return lowered
    return None


def _extract_crystallinity(parts: Tuple[str, ...]):
    for part in parts:
        match = re.fullmatch(r"ipp[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            crystallinity_pct = int(match.group(1))
            return crystallinity_pct, crystallinity_pct / 100.0
    return None, None


def _extract_deformation_rate(parts: Tuple[str, ...]):
    for part in parts:
        match = re.search(r"in\.def_([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", part, flags=re.IGNORECASE)
        if match:
            rate_label = match.group(1)
            try:
                return rate_label, float(rate_label)
            except ValueError:
                return rate_label, None
    return None, None


def _curve_id_from_relative_path(relative_path: Path) -> str:
    return relative_path.with_suffix("").as_posix().replace("/", "_")


def discover_tau_evol_files(input_root):
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    return sorted(root.rglob("Tau.evol"))


def extract_curve_metadata(path, input_root):
    file_path = Path(path)
    root = Path(input_root)
    relative_path = file_path.relative_to(root)
    parts = relative_path.parts

    structure = _find_structure(parts)
    crystallinity_pct, crystallinity = _extract_crystallinity(parts)
    rate_label, deformation_rate = _extract_deformation_rate(parts)

    if structure is None:
        raise ValueError(f"Could not infer structure from path: {file_path}")
    if crystallinity_pct is None or crystallinity is None:
        raise ValueError(f"Could not infer crystallinity from path: {file_path}")
    if rate_label is None:
        raise ValueError(f"Could not infer deformation rate from path: {file_path}")
    if deformation_rate is None:
        raise ValueError(f"Could not parse deformation rate from path: {file_path}")

    return CurveMetadata(
        curve_id=_curve_id_from_relative_path(relative_path),
        path=str(file_path),
        relative_path=relative_path.as_posix(),
        file_name=file_path.name,
        structure=structure,
        crystallinity_pct=crystallinity_pct,
        crystallinity=crystallinity,
        deformation_rate_label=rate_label,
        deformation_rate=deformation_rate,
    )


def load_tau_evol_curve(path, max_strain=MAX_STRAIN):
    """Load a Tau.evol file and return raw and engineering stress-strain arrays.

    Returns:
        raw_lambda: original stretch ratio column
        raw_stress_atm: original stress column in atm
        strain: engineering strain trimmed to max_strain, with a leading zero point
        stress_mpa: stress in MPa trimmed to max_strain, with a leading zero point
    """
    file_path = Path(path)
    data = np.loadtxt(file_path, comments="#")

    if data.ndim == 1:
        if data.size < 4:
            raise ValueError(f"File '{file_path}' has fewer than 4 columns")
        raw_lambda = np.array([data[1]], dtype=float)
        raw_stress_atm = np.array([data[3]], dtype=float)
    else:
        if data.shape[1] < 4:
            raise ValueError(f"File '{file_path}' has fewer than 4 columns")
        raw_lambda = np.asarray(data[:, 1], dtype=float)
        raw_stress_atm = np.asarray(data[:, 3], dtype=float)

    strain = raw_lambda - 1.0
    stress_mpa = raw_stress_atm * 0.101325

    strain = np.insert(strain, 0, 0.0)
    stress_mpa = np.insert(stress_mpa, 0, 0.0)

    valid_mask = strain <= max_strain
    if np.any(valid_mask):
        last_valid_idx = int(np.where(valid_mask)[0][-1])
        strain = strain[: last_valid_idx + 1]
        stress_mpa = stress_mpa[: last_valid_idx + 1]
    else:
        strain = np.array([0.0], dtype=float)
        stress_mpa = np.array([0.0], dtype=float)

    return raw_lambda, raw_stress_atm, strain, stress_mpa


def smooth_stress_strain(
    strain,
    stress,
    span=SMOOTH_SPAN,
    min_window_size=MIN_WINDOW_SIZE,
    window_fraction=WINDOW_FRACTION,
):
    """Return a smoothed copy of the curve for metric extraction."""
    if strain.size == 0 or stress.size == 0:
        return np.array([]), np.array([])

    if lowess is not None:
        smoothed = lowess(stress, strain, frac=span, return_sorted=True)
        smoothed_strain = np.asarray(smoothed[:, 0], dtype=float)
        smoothed_stress = np.asarray(smoothed[:, 1], dtype=float)
    else:  # pragma: no cover - fallback when statsmodels is unavailable
        if uniform_filter1d is None:
            return strain.copy(), stress.copy()
        window = max(min_window_size, int(len(stress) * window_fraction))
        smoothed_strain = strain.copy()
        smoothed_stress = uniform_filter1d(stress, size=window)

    order = np.argsort(smoothed_strain)
    return smoothed_strain[order], smoothed_stress[order]


def compute_yield_point(strain, stress):
    """Compute a simple yield point from the first stress drop."""
    if len(strain) < 2 or len(stress) < 2:
        return 0.0, 0.0

    for index in range(len(stress) - 1):
        if stress[index] > stress[index + 1]:
            return float(strain[index]), float(stress[index])

    max_index = int(np.argmax(stress))
    return float(strain[max_index]), float(stress[max_index])


def analyze_linear_region(
    strain,
    stress,
    threshold=SLOPE_METHOD_THRESHOLD,
    min_window_size=MIN_WINDOW_SIZE,
    window_fraction=WINDOW_FRACTION,
):
    """Estimate Young's modulus and the end of the linear regime.

    The implementation is intentionally compact and quiet because this module is
    meant to generate a database, not to print a lot of per-curve diagnostics.
    """
    if len(strain) < 20 or len(stress) < 20:
        return 0.0, float(strain[-1]) if len(strain) else 0.0, float(stress[-1]) if len(stress) else 0.0

    window_size = max(min_window_size, int(len(strain) * window_fraction))
    window_size = min(window_size, max(2, len(strain) // 3))

    slopes = []
    end_linear_idx = None

    for index in range(window_size, len(strain) - window_size):
        x_window = strain[index - window_size : index + window_size]
        y_window = stress[index - window_size : index + window_size]
        x_var = float(np.var(x_window))
        if x_var == 0.0:
            continue
        slope = float(np.cov(x_window, y_window)[0, 1] / x_var)
        slopes.append(slope)
        if len(slopes) >= 10:
            initial_slope = float(np.mean(slopes[:10]))
            if initial_slope != 0.0:
                current_change = abs((slope - initial_slope) / initial_slope)
                if current_change > threshold:
                    end_linear_idx = index
                    break

    if end_linear_idx is None:
        end_linear_idx = min(len(strain) // 2, int(0.8 * len(strain)))

    end_linear_idx = max(1, min(end_linear_idx, len(strain) - 1))
    youngs_modulus = float(np.polyfit(strain[:end_linear_idx], stress[:end_linear_idx], 1)[0])
    return youngs_modulus, float(strain[end_linear_idx]), float(stress[end_linear_idx])


def build_curve_summary(
    path,
    input_root,
    max_strain=MAX_STRAIN,
):
    metadata = extract_curve_metadata(path, input_root)
    raw_lambda, raw_stress_atm, strain, stress_mpa = load_tau_evol_curve(path, max_strain=max_strain)
    smoothed_strain, smoothed_stress = smooth_stress_strain(strain, stress_mpa)

    yield_strain, yield_stress = compute_yield_point(smoothed_strain, smoothed_stress)
    youngs_modulus, end_linear_strain, end_linear_stress = analyze_linear_region(strain, stress_mpa)

    if len(smoothed_stress) > 0:
        cut_index = int(np.argmax(smoothed_stress))
    elif len(stress_mpa) > 0:
        cut_index = int(np.argmax(stress_mpa))
    else:
        cut_index = 0

    cut_index = max(0, min(cut_index, len(strain) - 1))
    cut_strain = strain[: cut_index + 1]
    cut_stress = stress_mpa[: cut_index + 1]

    if len(cut_strain) > 0:
        ultimate_strain = float(cut_strain[-1])
        ultimate_stress = float(cut_stress[-1])
    else:
        ultimate_strain = 0.0
        ultimate_stress = 0.0

    if yield_strain == 0.0 and yield_stress == 0.0:
        yield_strain = ultimate_strain
        yield_stress = ultimate_stress

    return CurveSummary(
        curve_id=metadata.curve_id,
        path=metadata.path,
        relative_path=metadata.relative_path,
        file_name=metadata.file_name,
        structure=metadata.structure,
        crystallinity_pct=metadata.crystallinity_pct,
        crystallinity=metadata.crystallinity,
        deformation_rate_label=metadata.deformation_rate_label,
        deformation_rate=metadata.deformation_rate,
        n_raw=int(len(raw_lambda)),
        n_original=int(len(strain)),
        n_smoothed=int(len(smoothed_strain)),
        n_cut=int(len(cut_strain)),
        raw_lambda=np.asarray(raw_lambda, dtype=float).tolist(),
        raw_stress_atm=np.asarray(raw_stress_atm, dtype=float).tolist(),
        strain=np.asarray(cut_strain, dtype=float).tolist(),
        stress_mpa=np.asarray(cut_stress, dtype=float).tolist(),
        smoothed_strain=np.asarray(smoothed_strain, dtype=float).tolist(),
        smoothed_stress_mpa=np.asarray(smoothed_stress, dtype=float).tolist(),
        yield_strain=float(yield_strain),
        yield_stress=float(yield_stress),
        youngs_modulus=float(youngs_modulus),
        end_linear_strain=float(end_linear_strain),
        end_linear_stress=float(end_linear_stress),
        ultimate_strain=float(ultimate_strain),
        ultimate_stress=float(ultimate_stress),
    )


def build_curve_database(
    input_root,
    output_csv,
    output_json=None,
    max_strain=MAX_STRAIN,
    verbose=False,
):
    """Scan input_root for Tau.evol files and write summary CSV/JSON outputs."""
    input_root = Path(input_root)
    output_csv = Path(output_csv)
    records = {}

    curve_files = discover_tau_evol_files(input_root)
    for file_path in curve_files:
        try:
            summary = build_curve_summary(file_path, input_root=input_root, max_strain=max_strain)
        except Exception as exc:
            if verbose:
                print(f"Skipping {file_path}: {exc}")
            continue
        records[summary.curve_id] = summary
        if verbose:
            print(
                f"Added {summary.curve_id} | structure={summary.structure} | "
                f"Xc={summary.crystallinity:.2f} | rate={summary.deformation_rate:.2e} | "
                f"yield={summary.yield_stress:.2f} MPa"
            )

    sorted_records = dict(sorted(records.items(), key=lambda item: item[0]))
    csv_rows = [summary.to_csv_row() for summary in sorted_records.values()]

    csv_fieldnames = [
        "curve_id",
        "path",
        "relative_path",
        "file_name",
        "structure",
        "crystallinity_pct",
        "crystallinity",
        "deformation_rate_label",
        "deformation_rate",
        "n_raw",
        "n_original",
        "n_smoothed",
        "n_cut",
        "yield_strain",
        "yield_stress",
        "youngs_modulus",
        "end_linear_strain",
        "end_linear_stress",
        "ultimate_strain",
        "ultimate_stress",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        json_payload = {curve_id: summary.to_json_dict() for curve_id, summary in sorted_records.items()}
        with output_json.open("w", encoding="utf-8") as jsonfile:
            json.dump(json_payload, jsonfile, indent=2)

    return sorted_records


def _default_output_paths(script_path):
    return script_path.with_name("curves_summary.csv"), script_path.with_name("curves_summary.json")


def main(argv=None):
    script_path = Path(__file__).resolve()
    default_csv, default_json = _default_output_paths(script_path)

    parser = argparse.ArgumentParser(
        description="Build a curve database from Tau.evol files under a lattice/random project tree.",
    )
    parser.add_argument(
        "--input-root",
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing lattice/random curve folders.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(default_csv),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-json",
        default=str(default_json),
        help="Output JSON path (set to empty string to disable).",
    )
    parser.add_argument(
        "--max-strain",
        type=float,
        default=MAX_STRAIN,
        help="Maximum engineering strain to keep per curve.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-curve processing details.",
    )

    args = parser.parse_args(argv)
    json_path = Path(args.output_json) if args.output_json else None

    records = build_curve_database(
        input_root=args.input_root,
        output_csv=args.output_csv,
        output_json=json_path,
        max_strain=args.max_strain,
        verbose=args.verbose,
    )

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] Database created with {len(records)} curves")
    print(f"CSV: {Path(args.output_csv).resolve()}")
    if json_path is not None:
        print(f"JSON: {json_path.resolve()}")

    # Attempt to generate a stress_strain_plots.png using the local plotting utility.
    try:
        import plotting_function as _plotmod
        import numpy as _np

        # Prepare curves_summary mapping (JSON-serializable) from in-memory records
        curves_summary = {k: v.to_json_dict() for k, v in records.items()}

        out_png = script_path.with_name('stress_strain_plots.png')
        out_path = _plotmod.plot_stress_curve_predictions(
            curves_summary,
            strain_array='strain',
            stress_array='stress_mpa',
            max_strain=args.max_strain,
            output_filename=str(out_png),
        )
        print(f"Saved stress-strain plot to: {out_path}")
    except Exception as _plot_exc:
        print(f"Could not generate stress-strain plot: {_plot_exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
