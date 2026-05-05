"""Delegate to the original `plot_stress_strain_curves` when available.

This module wraps the project's canonical plotting function from
`/home/hlobato/iPP-MD-ML/collect_curves.py`. If that module cannot be
imported, it falls back to a compatible local implementation.
"""

from pathlib import Path
import sys
import importlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _try_import_collect_curves():
    # Prefer importing from the user's iPP-MD-ML folder
    try:
        return importlib.import_module('collect_curves')
    except Exception:
        # Try adding the known path
        try:
            sys.path.insert(0, '/home/hlobato/iPP-MD-ML')
            return importlib.import_module('collect_curves')
        except Exception:
            return None


def plot_stress_curve_predictions(*args: Any, **kwargs: Any) -> Tuple[Path, Any]:
    """Produce a multi-panel stress-strain figure using the project's function.

    Accepted call patterns:
    - plot_stress_curve_predictions(curves_summary, ...)
    - plot_stress_curve_predictions(results, curves_summary, ...)

    The first form is what `simulation_curve_processing.py` currently uses.
    The second is kept for compatibility with older call sites.
    """
    results = None
    curves_summary = None

    if len(args) == 1:
        curves_summary = args[0]
    elif len(args) >= 2:
        results = args[0]
        curves_summary = args[1]
    else:
        curves_summary = kwargs.pop('curves_summary', None)
        results = kwargs.pop('results', None)

    strain_array = kwargs.pop('strain_array', 'strain')
    stress_array = kwargs.pop('stress_array', 'stress_mpa')
    max_strain = kwargs.pop('max_strain', 3.0)
    n_cols = kwargs.pop('n_cols', 6)
    figsize = kwargs.pop('figsize', (5 * 6, 4 * 6))
    save = kwargs.pop('save', True)
    filename = kwargs.pop('filename', 'stress_strain_plots.png')
    output_filename = kwargs.pop('output_filename', None)

    if output_filename is not None:
        filename = output_filename

    if kwargs:
        # Preserve future compatibility by ignoring unknown keywords, but keep
        # a breadcrumb in case the caller is drifting from the supported API.
        pass

    cc_mod = _try_import_collect_curves()
    # Build a data_dict compatible with collect_curves.plot_stress_strain_curves
    data_dict = {}
    for k, v in (curves_summary or {}).items():
        rec = {}
        # Map arrays to the names expected by collect_curves.plot_stress_strain_curves.
        # Keep the raw/cut curve separate from the smoothed curve so the plotter
        # can render both layers.
        cut_strain = np.asarray(v.get('strain', v.get('cut_strain_array', [])), dtype=float)
        cut_stress = np.asarray(v.get('stress_mpa', v.get('cut_stress_array', [])), dtype=float)
        smoothed_strain = np.asarray(
            v.get('smoothed_strain', v.get('smoothed_strain_array', cut_strain)),
            dtype=float,
        )
        smoothed_stress = np.asarray(
            v.get('smoothed_stress', v.get('smoothed_stress_mpa', v.get('smoothed_stress_array', cut_stress))),
            dtype=float,
        )
        if cut_strain.size and np.isfinite(max_strain):
            mask = cut_strain <= float(max_strain)
            cut_strain = cut_strain[mask]
            cut_stress = cut_stress[mask]
        rec['cut_strain_array'] = cut_strain.tolist()
        rec['cut_stress_array'] = cut_stress.tolist()
        rec['smoothed_strain'] = smoothed_strain.tolist()
        rec['smoothed_stress'] = smoothed_stress.tolist()
        # copy scalar metrics if present
        for fld in ('yield_strain', 'yield_stress', 'end_linear_strain', 'end_linear_stress', 'ultimate_strain', 'ultimate_stress', 'youngs_modulus'):
            if fld in v:
                rec[fld] = v.get(fld)
        data_dict[k] = rec

    if cc_mod is not None and hasattr(cc_mod, 'plot_stress_strain_curves'):
        try:
            fig, axes = cc_mod.plot_stress_strain_curves(data_dict, n_cols=n_cols)
            out_path = Path(filename)
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return out_path, axes
        except Exception as e:
            print(f'Failed to use collect_curves.plot_stress_strain_curves: {e}')

    # Fallback: simple plotting similar to the original but implemented here
    items = list(data_dict.items())
    n_plots = len(items)
    if n_plots == 0:
        raise ValueError('No data to plot')

    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for idx, (key, rec) in enumerate(items):
        ax = axes_flat[idx]
        cut_strain = np.asarray(rec.get('cut_strain_array', []))
        cut_stress = np.asarray(rec.get('cut_stress_array', []))
        sm_strain = np.asarray(rec.get('smoothed_strain', []))
        sm_stress = np.asarray(rec.get('smoothed_stress', []))

        if cut_strain.size and cut_stress.size:
            ax.scatter(cut_strain, cut_stress, s=1, alpha=0.5, color='blue', label='Original data')
        if sm_strain.size and sm_stress.size:
            ax.plot(sm_strain, sm_stress, 'r-', linewidth=1.5, label='Smoothed')

        try:
            if rec.get('end_linear_strain') is not None and rec.get('end_linear_stress') is not None:
                ax.plot(rec['end_linear_strain'], rec['end_linear_stress'], 'go', markersize=6, markeredgecolor='black')
            if rec.get('yield_strain') is not None and rec.get('yield_stress') is not None:
                ax.plot(rec['yield_strain'], rec['yield_stress'], 'ro', markersize=6, markeredgecolor='black')
            if rec.get('ultimate_strain') is not None and rec.get('ultimate_stress') is not None:
                ax.plot(rec['ultimate_strain'], rec['ultimate_stress'], 'mo', markersize=6, markeredgecolor='black')
        except Exception:
            pass

        try:
            if rec.get('youngs_modulus') is not None and cut_strain.size:
                x_max = float(np.max(cut_strain))
                x_young = np.linspace(0, x_max, 50)
                y_young = float(rec['youngs_modulus']) * x_young
                ax.plot(x_young, y_young, 'k--', alpha=0.7)
        except Exception:
            pass

        if cut_strain.size:
            ax.set_xlim(0, float(np.max(cut_strain)))
        if cut_stress.size:
            ax.set_ylim(0, float(np.max(cut_stress)))

        ax.set_xlabel('Strain (-)')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title(f'{key}')
        ax.grid(True, alpha=0.3)

    for idx in range(n_plots, len(axes_flat)):
        try:
            axes_flat[idx].set_visible(False)
        except Exception:
            pass

    plt.tight_layout()
    out_path = Path(filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path, axes
