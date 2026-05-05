import numpy as np
from scipy.ndimage import uniform_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, List, Dict, Any, Callable, Optional
from itertools import combinations, islice


def compute_molecular_weights(chain_info: Iterable[tuple], monomer_mass: float = 42.08) -> Tuple[float, float, float]:
    """Compute Mn, Mw and PDI from chain-length/count data.

    Args:
        chain_info: iterable of (length, count) pairs where `length` is number of
            repeat units in a chain and `count` is how many such chains are present.
        monomer_mass: mass of a single repeat unit (default 42.08 g/mol).

    Returns:
        (Mn, Mw, PDI)

    Notes:
        Mn = sum(cj * Mj) / sum(cj)
        Mw = sum(cj * Mj^2) / sum(cj * Mj)
    """
    # Convert to numpy arrays for vectorized computation. Handle empty input.
    data = np.array(list(chain_info), dtype=float)
    if data.size == 0:
        return 0.0, 0.0, 0.0

    lengths = data[:, 0]
    counts = data[:, 1]

    masses = lengths * monomer_mass  # Mj

    total_chains = counts.sum()
    if total_chains == 0:
        return 0.0, 0.0, 0.0

    Mn = (counts * masses).sum() / total_chains

    denom = (counts * masses).sum()
    if denom == 0:
        Mw = 0.0
    else:
        Mw = (counts * masses**2).sum() / denom

    PDI = Mw / Mn if Mn != 0 else 0.0
    return float(Mn), float(Mw), float(PDI)


def smooth_stress_strain(strain, stress, span: float = 0.05,
                         min_window_size: int = 10, window_fraction: float = 0.02):
    """Return LOESS-smoothed (strain, stress) arrays.

    Args:
        strain: strain array (engineering strain, dimensionless)
        stress: stress array (MPa)
        span: fraction of data used for each local regression (frac in lowess).

    Returns:
        (strain_smooth, stress_smooth) as np.ndarrays.
    """
    if strain is None or stress is None or strain.size == 0 or stress.size == 0:
        return np.array([]), np.array([])

    # perform lowess if available; if unavailable, fall back to a simple moving average
    try:
        loess_result = lowess(stress, strain, frac=span)
        strain_smooth = np.asarray(loess_result[:, 0], dtype=float)
        stress_smooth = np.asarray(loess_result[:, 1], dtype=float)
    except ImportError:
        # Fallback smoothing: moving average via uniform_filter1d.
        # Choose window proportional to span but at least min_window_size points.
        n = strain.size
        # window based on fraction of data but at least min_window_size
        frac_win = int(max(1, int(n * window_fraction)))
        win = max(min_window_size, frac_win)
        stress_smooth = uniform_filter1d(stress, size=win)
        strain_smooth = strain.copy()  # Keep original strain values

    # ensure sorted by strain
    order = np.argsort(strain_smooth)
    strain_smooth = strain_smooth[order]
    stress_smooth = stress_smooth[order]

    return strain_smooth, stress_smooth


def compute_yield(smoothed_strain, smoothed_stress):
    """Compute yield point (strain and stress) as the point just before the first stress drop.
    
    Args:
        smoothed_strain: smoothed strain array
        smoothed_stress: smoothed stress array (MPa)
        
    Returns:
        tuple: (yield_strain, yield_stress) or (None, None) if no data
    """
    if len(smoothed_stress) < 2:
        return None, None
        
    strain = smoothed_strain
    stress = smoothed_stress
    
    # Find where stress first drops (stress[i] > stress[i+1])
    for i in range(len(stress) - 1):
        if stress[i] > stress[i + 1]:
            print(f"Yield point found: {stress[i]:.2f} MPa at strain {strain[i]:.3f}")
            return strain[i], stress[i]
    
    # If no drop found, use the maximum stress point
    max_idx = np.argmax(stress)
    print(f"No stress drop found, using max stress as yield: {stress[max_idx]:.2f} MPa at strain {strain[max_idx]:.3f}")
    return strain[max_idx], stress[max_idx]


def analyze_linear_region(
    strain,
    stress,
    threshold=0.2,
    min_window_size: int = 10,
    window_fraction: float = 0.02,
):
    """
    Detect linear region based on slope change.
    
    Parameters:
    strain, stress: np.arrays 
    threshold: float - relative change in slope (0.3 = 30% change)
    min_window_size: minimum window size for slope calculation
    window_fraction: fraction of data points to use for window size
    
    Returns:
    youngs_modulus, end_linear_strain, end_linear_stress, slopes, strain_points
    """
    if len(strain) < 20: 
        print("Not enough data points for linear region analysis.")
        return None, None, None, np.array([]), np.array([])
    
    # Calculate local slopes using sliding window
    window_size = max(min_window_size, int(len(strain) * window_fraction))
    slopes = []
    strain_points = []
    
    # Stop calculating slopes once we find the end of linear region
    end_of_linear_idx = None
    
    for i in range(window_size, len(strain) - window_size):
        # Calculate slope over a local window
        x_window = strain[i-window_size:i+window_size]
        y_window = stress[i-window_size:i+window_size]
        
        # Linear fit over window using numpy (more efficient)
        slope = np.cov(x_window, y_window)[0, 1] / np.var(x_window)
        
        slopes.append(slope)
        strain_points.append(strain[i])
        
        # Check if we've exceeded threshold (need at least 10 slopes for initial reference)
        if len(slopes) >= 10:
            initial_slope = np.mean(slopes[:10])
            current_change = np.abs((slope - initial_slope) / initial_slope)
            
            if current_change > threshold and end_of_linear_idx is None:
                end_of_linear_idx = i
                break  # Stop calculating slopes once threshold is met
    
    slopes = np.array(slopes)
    strain_points = np.array(strain_points)
    
    # If no significant change found, use a reasonable fraction of the data
    if end_of_linear_idx is None:
        end_of_linear_idx = min(len(strain) // 2, int(0.8 * len(strain)))
    
    # Calculate results
    youngs_modulus = None
    end_linear_strain = strain[end_of_linear_idx]
    # Determine end_linear_stress by the function of the line y = Ex
    end_linear_stress = youngs_modulus * end_linear_strain if youngs_modulus is not None else stress[end_of_linear_idx]
    
    if end_of_linear_idx > 10:
        # Fit Young's modulus to the linear region using polyfit
        youngs_modulus = np.polyfit(strain[:end_of_linear_idx], 
                                   stress[:end_of_linear_idx], 1)[0]
    
    return youngs_modulus, end_linear_strain, end_linear_stress, slopes, strain_points


def plot_linear_region_analysis(raw_strain, raw_stress, smoothed_strain, smoothed_stress, 
                              youngs_modulus, end_linear_strain, end_linear_stress,
                              yield_strain, yield_stress,
                              slopes, slope_strain_points):
    """
    Plot the linear region analysis results.
    
    Parameters:
    raw_strain, raw_stress: original unsmoothed arrays
    smoothed_strain, smoothed_stress: smoothed arrays for curve
    youngs_modulus: calculated Young's modulus
    end_linear_strain: strain at end of linear region
    end_linear_stress: stress at end of linear regionRg_by_type_over_t_df
    yield_strain: strain at yield point
    yield_stress: stress at yield point
    slopes: slopes array from analysis
    slope_strain_points: strain points corresponding to slopes
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Calculate plot limits based on yield point
    if yield_strain is not None and yield_stress is not None:
        xlim_max = yield_strain * 1.25
        ylim_max = yield_stress * 1.25
    else:
        xlim_max = np.max(smoothed_strain) * 1.1 if len(smoothed_strain) > 0 else 1.0
        ylim_max = np.max(smoothed_stress) * 1.1 if len(smoothed_stress) > 0 else 1.0
    
    # Main stress-strain plot
    # Plot non-smoothed data as scatter
    ax1.scatter(raw_strain, raw_stress, color='red', alpha=0.3, s=10, label='Raw Data')
    
    # Plot smoothed curve
    ax1.plot(smoothed_strain, smoothed_stress, 'r-', linewidth=2, label='Smoothed Curve')
    
    # Plot linear fit extended to plot limits
    if youngs_modulus is not None:
        # Extend linear fit to xlim
        linear_strain = np.linspace(0, xlim_max, 100)
        linear_fit = youngs_modulus * linear_strain
        
        # But don't extend beyond ylim
        max_linear_strain = min(xlim_max, ylim_max / youngs_modulus) if youngs_modulus > 0 else xlim_max
        linear_strain = np.linspace(0, max_linear_strain, 100)
        linear_fit = youngs_modulus * linear_strain
        
        ax1.plot(linear_strain, linear_fit, 'g-', linewidth=2, 
                label=f'Linear Fit (E={youngs_modulus:.2f} MPa)')
    
    # Mark end of linear region
    if end_linear_strain is not None and end_linear_stress is not None:
        # Vertical line at end of linear strain
        ax1.axvline(x=end_linear_strain, color='orange', linestyle='--', 
                   alpha=0.8, label='End Linear Region')
        
        # Horizontal line at end of linear stress
        ax1.axhline(y=end_linear_stress, color='orange', linestyle='--', 
                   alpha=0.8)
        
        # Point where they intersect
        ax1.plot(end_linear_strain, end_linear_stress, 'ro', markersize=8, 
                label=f'End Linear Point ({end_linear_stress:.2f} MPa)')
    
    # Mark yield point
    if yield_strain is not None and yield_stress is not None:
        ax1.axvline(x=yield_strain, color='purple', linestyle=':', 
                   alpha=0.8, label='Yield Strain')
        
        ax1.axhline(y=yield_stress, color='purple', linestyle=':', 
                   alpha=0.8, label='Yield Stress')
        
        ax1.plot(yield_strain, yield_stress, 'mo', markersize=8, 
                label=f'Yield Point ({yield_stress:.2f} MPa)')
    
    # Set axis limits to focus on linear and yield region
    ax1.set_xlim(0, xlim_max)
    ax1.set_ylim(0, ylim_max)
    
    ax1.set_xlabel('Strain')
    ax1.set_ylabel('Stress (MPa)')
    ax1.set_title('Linear Region and Yield Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Slope evolution plot 
    if len(slopes) > 0 and end_linear_strain is not None:
        mask = slope_strain_points <= end_linear_strain * 1.1  # 10% beyond for context
        filtered_strain_points = slope_strain_points[mask]
        filtered_slopes = slopes[mask]
        
        ax2.plot(filtered_strain_points, filtered_slopes, 'b-', linewidth=2, label='Local Slope')
        
        # Mark end of linear region in slope plot
        if end_linear_strain is not None:
            ax2.axvline(x=end_linear_strain, color='orange', linestyle='--', 
                       label='End Linear Region')
                        
        # Set xlim for slope plot to match main plot
        ax2.set_xlim(0, xlim_max)
        ax2.set_ylim(0, 1000)
        
        ax2.set_xlabel('Strain')
        ax2.set_ylabel('Local Slope (MPa)')
        ax2.set_title('Slope Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Slope data not available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Slope Evolution')
    
    plt.tight_layout()
    plt.show()


