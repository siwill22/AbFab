#!/usr/bin/env python
"""
AbFab Tiling Comparison Script

Demonstrates 4 methods for generating synthetic abyssal hill bathymetry:
1. Fixed parameters (baseline)
2. Median spreading rate parameters
3. von Kármán filter
4. Spatially varying spreading rate (with global binning)

Generates: tiling_comparison_4methods.png
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import xarray as xr
import time
import pygmt
import AbFab as af

# ============================================================================
# USER PARAMETERS - Modify these as needed
# ============================================================================

# Region selection (longitude, latitude bounds)
#XMIN, XMAX = 63, 74
#YMIN, YMAX = -30, -23
XMIN, XMAX = 0, 50  # Southern Africa test region
YMIN, YMAX = -40, -20

# Grid parameters
SPACING = '5m'  # Grid spacing (e.g., '2m' = 2 arcmin)

# Fixed abyssal hill parameters (baseline)
PARAMS_FIXED = {
    'H': 50.0,       # RMS height (m)
    'lambda_n': 3,  # Perpendicular wavelength (km)
    'lambda_s': 30, # Parallel wavelength (km)
    'D': 2.2          # Fractal dimension
}

# Chunking parameters for parallel processing
CHUNKSIZE = 80      # Chunk size in pixels
CHUNKPAD = 20        # Padding to avoid edge effects
NUM_CPUS = 4         # Number of parallel workers

# Optimization settings
USE_OPTIMIZATION = True   # Use filter bank (50x speedup)
AZIMUTH_BINS = 36        # Azimuth discretization (18-72 typical)
SEDIMENT_BINS = 8        # Sediment discretization (3-10 typical)
SPREADING_RATE_BINS = 10  # Spreading rate discretization for Method 4

# Random seed for reproducibility
RANDOM_SEED = 42

# Output
OUTPUT_FILE = 'tiling_comparison_4methods.png'
NETCDF_FILE = 'method4_bathymetry.nc'
DPI = 300

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_bathymetry_chunk(coord, age_dataarray, sed_dataarray, rand_dataarray,
                             chunksize, chunkpad, params, lon_spacing_deg, filter_type='gaussian',
                             optimize=True, azimuth_bins=36, sediment_bins=5,
                             spreading_rate_bins=1, base_params=None,
                             sediment_range=None, spreading_rate_range=None):
    """
    Process a single chunk of bathymetry with optional global binning.

    Parameters
    ----------
    coord : tuple
        (row, col) starting coordinates
    age_dataarray : xarray.DataArray
        Seafloor age data
    sed_dataarray : xarray.DataArray
        Sediment thickness data
    rand_dataarray : xarray.DataArray
        Random field (global, shared across chunks)
    chunksize : int
        Size of chunk in pixels
    chunkpad : int
        Padding pixels to handle edge effects
    params : dict
        Abyssal hill parameters
    lon_spacing_deg : float
        Longitude spacing in degrees (for spherical correction)
    filter_type : str
        'gaussian' or 'von_karman'
    optimize : bool
        Use optimized filter bank
    azimuth_bins : int
        Azimuth discretization
    sediment_bins : int
        Sediment discretization
    spreading_rate_bins : int
        Spreading rate discretization (1 = disabled)
    base_params : dict or None
        Base parameters for spreading rate scaling
    sediment_range : tuple or None
        Global (min, max) for sediment binning
    spreading_rate_range : tuple or None
        Global (min, max) for spreading rate binning

    Returns
    -------
    xarray.DataArray
        Processed chunk with padding trimmed
    """
    # Extract chunk WITH padding
    chunk_age = age_dataarray[coord[0]:coord[0]+chunksize+chunkpad,
                               coord[1]:coord[1]+chunksize+chunkpad]
    chunk_sed = sed_dataarray[coord[0]:coord[0]+chunksize+chunkpad,
                               coord[1]:coord[1]+chunksize+chunkpad]
    chunk_random = rand_dataarray[coord[0]:coord[0]+chunksize+chunkpad,
                                   coord[1]:coord[1]+chunksize+chunkpad]

    # Skip empty chunks
    if np.all(np.isnan(chunk_age.data)):
        return chunk_age

    # Calculate grid spacing at chunk's mean latitude (Phase 2: spherical correction)
    chunk_lat_coords = chunk_age.lat.values
    mean_lat = float(np.mean(chunk_lat_coords))
    grid_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(mean_lat))

    # Generate synthetic bathymetry
    synthetic_bathymetry = af.generate_bathymetry_spatial_filter(
        chunk_age.data,
        chunk_sed.data,
        params,
        grid_spacing_km,
        chunk_random.data,
        filter_type=filter_type,
        optimize=optimize,
        azimuth_bins=azimuth_bins,
        sediment_bins=sediment_bins,
        spreading_rate_bins=spreading_rate_bins,
        base_params=base_params,
        sediment_range=sediment_range,
        spreading_rate_range=spreading_rate_range,
        lat_coords=chunk_lat_coords  # Phase 1: pass latitude for gradient corrections
    )

    # Trim padding and return
    pad_half = chunkpad // 2
    return xr.DataArray(
        synthetic_bathymetry,
        coords=chunk_age.coords,
        name='z'
    )[pad_half:-pad_half, pad_half:-pad_half]


def process_method(method_name, age_da, sed_da, rand_da, coords, chunksize, chunkpad,
                   params, lon_spacing_deg, filter_type='gaussian',
                   spreading_rate_bins=1, base_params=None,
                   sediment_range=None, spreading_rate_range=None):
    """
    Process all chunks for a given method.

    Parameters
    ----------
    lon_spacing_deg : float
        Longitude spacing in degrees (for spherical correction)

    Returns
    -------
    list of xarray.DataArray
        Processed chunks
    float
        Elapsed time in seconds
    """
    print(f"\n{'='*70}")
    print(f"{method_name}")
    print(f"{'='*70}")

    start = time.time()
    results = Parallel(n_jobs=NUM_CPUS, timeout=600)(delayed(process_bathymetry_chunk)(
        coord, age_da, sed_da, rand_da, chunksize, chunkpad, params, lon_spacing_deg,
        filter_type, USE_OPTIMIZATION, AZIMUTH_BINS, SEDIMENT_BINS,
        spreading_rate_bins, base_params, sediment_range, spreading_rate_range
    ) for coord in coords)

    elapsed = time.time() - start

    # Filter out empty chunks
    results = [result for result in results if 0 not in result.shape]

    print(f"Completed in {elapsed:.1f} seconds ({len(results)} valid chunks)")
    print(f"  → {elapsed/len(results):.2f} seconds per chunk")

    return results, elapsed


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*70)
    print("AbFab Tiling Comparison - 4 Methods")
    print("="*70)

    # Load data
    print("\nLoading seafloor age and sediment data...")
    age_da = pygmt.grdsample('/Users/simon/Data/AgeGrids/2020/age.2020.1.GeeK2007.6m.nc',
                             region='g', spacing=SPACING)
    sed_da = pygmt.grdsample('/Users/simon/GIT/pyBacktrack/pybacktrack/bundle_data/sediment_thickness/GlobSed.nc',
                             region='g', spacing=SPACING)

    # Extend longitude range for continuity
    age_da = af.extend_longitude_range(age_da).sel(lon=slice(-190, 190))
    sed_da = af.extend_longitude_range(sed_da).sel(lon=slice(-190, 190))

    # Clean sediment data
    sed_da = sed_da.where(np.isfinite(sed_da), 1.)
    sed_da = sed_da.where(sed_da < 1000., 1000.)

    # Select region
    age_da = age_da.sel(lon=slice(XMIN, XMAX), lat=slice(YMIN, YMAX))
    sed_da = sed_da.sel(lon=slice(XMIN, XMAX), lat=slice(YMIN, YMAX))

    print(f"Region: {XMIN}° to {XMAX}° E, {YMIN}° to {YMAX}° N")
    print(f"Grid shape: {age_da.shape}")
    print(f"Age range: {np.nanmin(age_da.data):.1f} - {np.nanmax(age_da.data):.1f} Myr")
    print(f"Sediment range: {np.nanmin(sed_da.data):.1f} - {np.nanmax(sed_da.data):.1f} m")

    # Generate global random field
    print("\nGenerating global random field...")
    np.random.seed(RANDOM_SEED)
    rand_da = age_da.copy()
    rand_da.data = af.generate_random_field(rand_da.data.shape)

    # Calculate longitude spacing for spherical corrections
    lon_spacing_deg = float(np.abs(age_da.lon.values[1] - age_da.lon.values[0]))
    mean_lat = float(np.mean(age_da.lat.values))
    grid_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(mean_lat))
    print(f"Longitude spacing: {lon_spacing_deg:.4f}°")
    print(f"Grid spacing at mean latitude ({mean_lat:.1f}°): {grid_spacing_km:.3f} km/pixel")
    print(f"  (Note: Per-chunk spacing will vary with latitude for spherical correction)")

    # Generate chunk coordinates
    full_ny, full_nx = age_da.shape
    chunkpad = int(2 * np.round(CHUNKPAD / 2))  # Ensure even
    coords_y, coords_x = np.meshgrid(np.arange(0, full_ny-1, CHUNKSIZE),
                                      np.arange(0, full_nx-1, CHUNKSIZE))
    coords = list(zip(coords_y.flatten(), coords_x.flatten()))

    print(f"\nProcessing {len(coords)} chunks (size={CHUNKSIZE}, pad={chunkpad}, CPUs={NUM_CPUS})")
    print(f"Optimization: {'ENABLED' if USE_OPTIMIZATION else 'DISABLED'}")
    if USE_OPTIMIZATION:
        print(f"  • Azimuth bins: {AZIMUTH_BINS}")
        print(f"  • Sediment bins: {SEDIMENT_BINS}")

    # ========================================================================
    # METHOD 1: Fixed Parameters
    # ========================================================================
    results_method1, time1 = process_method(
        "METHOD 1: Fixed Parameters + Gaussian Filter",
        age_da, sed_da, rand_da, coords, CHUNKSIZE, chunkpad,
        PARAMS_FIXED, lon_spacing_deg,
        filter_type='gaussian',
        spreading_rate_bins=1,
        base_params=None,
        sediment_range=None,
        spreading_rate_range=None
    )

    # ========================================================================
    # METHOD 2: Median Spreading Rate Parameters
    # ========================================================================
    print("\n" + "="*70)
    print("METHOD 2: Median Spreading Rate Parameters + Gaussian Filter")
    print("="*70)

    # Calculate median spreading rate with spherical correction
    spreading_rate = af.calculate_spreading_rate_from_age(
        age_da.data, grid_spacing_km, lat_coords=age_da.lat.values
    )
    median_rate = np.nanmedian(spreading_rate)
    print(f"Median spreading rate: {median_rate:.1f} mm/yr (with spherical correction)")

    # Derive parameters
    params_derived = af.spreading_rate_to_params(median_rate, base_params=PARAMS_FIXED)
    print(f"Derived parameters: H={params_derived['H']:.1f}m, "
          f"λ_n={params_derived['lambda_n']:.1f}km, λ_s={params_derived['lambda_s']:.1f}km")

    results_method2, time2 = process_method(
        "",  # Already printed header
        age_da, sed_da, rand_da, coords, CHUNKSIZE, chunkpad,
        params_derived, lon_spacing_deg,
        filter_type='gaussian',
        spreading_rate_bins=1,
        base_params=None,
        sediment_range=None,
        spreading_rate_range=None
    )

    # ========================================================================
    # METHOD 3: von Kármán Filter
    # ========================================================================
    results_method3, time3 = process_method(
        "METHOD 3: Fixed Parameters + von Kármán Filter",
        age_da, sed_da, rand_da, coords, CHUNKSIZE, chunkpad,
        PARAMS_FIXED, lon_spacing_deg,
        filter_type='von_karman',
        spreading_rate_bins=1,
        base_params=None,
        sediment_range=None,
        spreading_rate_range=None
    )

    # ========================================================================
    # METHOD 4: Spatially Varying Spreading Rate (with Global Binning)
    # ========================================================================
    print("\n" + "="*70)
    print("METHOD 4: Spatially Varying Spreading Rate + Global Binning")
    print("="*70)

    # Calculate global ranges for consistent binning with spherical correction
    print("Calculating global bin ranges...")
    spreading_rate_global = af.calculate_spreading_rate_from_age(
        age_da.data, grid_spacing_km, lat_coords=age_da.lat.values
    )
    spreading_rate_global = np.where(np.isnan(spreading_rate_global),
                                      np.nanmedian(spreading_rate_global),
                                      spreading_rate_global)
    sr_min_global = float(np.min(spreading_rate_global))
    sr_max_global = float(np.max(spreading_rate_global))

    sed_min_global = float(np.min(sed_da.data))
    sed_max_global = float(np.max(sed_da.data))

    print(f"  Global spreading rate range: {sr_min_global:.1f} - {sr_max_global:.1f} mm/yr (spherical corrected)")
    print(f"  Global sediment range: {sed_min_global:.1f} - {sed_max_global:.1f} m")
    print(f"  Spreading rate bins: {SPREADING_RATE_BINS}")

    results_method4, time4 = process_method(
        "",  # Already printed header
        age_da, sed_da, rand_da, coords, CHUNKSIZE, chunkpad,
        PARAMS_FIXED, lon_spacing_deg,
        filter_type='gaussian',
        spreading_rate_bins=SPREADING_RATE_BINS,
        base_params=PARAMS_FIXED,
        sediment_range=(sed_min_global, sed_max_global),
        spreading_rate_range=(sr_min_global, sr_max_global)
    )

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("Generating comparison figure...")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(30, 24))

    axes = axes.flatten()
    vmin, vmax = -1, 1

    # Method 1
    ax = axes[0]
    for res in results_method1:
        ax.pcolormesh(res.lon, res.lat, res.data, vmin=vmin, vmax=vmax, cmap='seismic')
    ax.set_title(f'Method 1: Fixed Parameters (H={PARAMS_FIXED["H"]:.0f}m) + Gaussian Filter',
                 fontweight='bold', fontsize=16)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_aspect('equal')

    # Method 2
    ax = axes[1]
    for res in results_method2:
        ax.pcolormesh(res.lon, res.lat, res.data, vmin=vmin, vmax=vmax, cmap='seismic')
    ax.set_title(f'Method 2: Median Spreading Rate (H={params_derived["H"]:.0f}m from {median_rate:.0f} mm/yr) + Gaussian',
                 fontweight='bold', fontsize=16)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_aspect('equal')

    # Method 3
    ax = axes[2]
    for res in results_method3:
        ax.pcolormesh(res.lon, res.lat, res.data, vmin=vmin, vmax=vmax, cmap='seismic')
    ax.set_title(f'Method 3: Fixed Parameters (H={PARAMS_FIXED["H"]:.0f}m) + von Kármán Filter',
                 fontweight='bold', fontsize=16)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_aspect('equal')

    # Method 4
    ax = axes[3]
    for res in results_method4:
        ax.pcolormesh(res.lon, res.lat, res.data, vmin=vmin, vmax=vmax, cmap='seismic')
    ax.set_title(f'Method 4: Spatially Varying Spreading Rate (base H={PARAMS_FIXED["H"]:.0f}m, {SPREADING_RATE_BINS} bins) + Global Binning',
                 fontweight='bold', fontsize=16)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_FILE}")

    # ========================================================================
    # SAVE METHOD 4 TO NETCDF
    # ========================================================================
    print("\n" + "="*70)
    print("Saving Method 4 grid to NetCDF...")
    print("="*70)

    # Combine chunks into a single DataArray
    # Get the full coordinate arrays from the original age data
    output_grid = xr.DataArray(
        np.full((full_ny, full_nx), np.nan),
        coords={'lat': age_da.lat, 'lon': age_da.lon},
        dims=['lat', 'lon'],
        name='synthetic_bathymetry',
        attrs={
            'long_name': 'Synthetic abyssal hill bathymetry',
            'units': 'm',
            'method': 'Spatially varying spreading rate with global binning',
            'base_H': PARAMS_FIXED['H'],
            'base_lambda_n': PARAMS_FIXED['lambda_n'],
            'base_lambda_s': PARAMS_FIXED['lambda_s'],
            'base_D': PARAMS_FIXED['D'],
            'spreading_rate_bins': SPREADING_RATE_BINS,
            'azimuth_bins': AZIMUTH_BINS,
            'sediment_bins': SEDIMENT_BINS,
            'spherical_correction': 'enabled',
            'random_seed': RANDOM_SEED
        }
    )

    # Fill in the chunks
    for res in results_method4:
        lat_slice = slice(res.lat.values[0], res.lat.values[-1])
        lon_slice = slice(res.lon.values[0], res.lon.values[-1])
        output_grid.loc[{'lat': lat_slice, 'lon': lon_slice}] = res.data

    # Add CF-compliant coordinate metadata
    output_grid = af.add_cf_compliant_coordinate_attrs(output_grid)

    # Save to NetCDF
    output_grid.to_netcdf(NETCDF_FILE)
    print(f"✓ Saved Method 4 grid to: {NETCDF_FILE}")
    print(f"  Grid shape: {output_grid.shape}")
    print(f"  Lat range: {float(output_grid.lat.min()):.2f}° to {float(output_grid.lat.max()):.2f}°")
    print(f"  Lon range: {float(output_grid.lon.min()):.2f}° to {float(output_grid.lon.max()):.2f}°")
    print(f"  Valid pixels: {np.sum(~np.isnan(output_grid.data))}/{output_grid.size}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nMethod 1: {time1:.1f}s ({len(results_method1)} chunks)")
    print(f"Method 2: {time2:.1f}s ({len(results_method2)} chunks)")
    print(f"Method 3: {time3:.1f}s ({len(results_method3)} chunks)")
    print(f"Method 4: {time4:.1f}s ({len(results_method4)} chunks) - {time4/time1:.1f}x slower (3D filter bank)")

    print("\nKey features:")
    print("  • Method 1: Baseline with fixed parameters")
    print("  • Method 2: Parameters from median spreading rate")
    print("  • Method 3: Theoretically rigorous von Kármán filter")
    print("  • Method 4: Spatially varying with global binning (smooth across chunks)")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
