#!/usr/bin/env python
"""
Complete Bathymetry Generation Script

Generates realistic synthetic ocean floor bathymetry by combining:
1. Long-wavelength thermal subsidence (from seafloor age)
2. Short-wavelength abyssal hill fabric (spatially varying with spreading rate)
3. Sediment drape effects

Uses Method 4: Spatially varying spreading rate with global binning and spherical corrections
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed
import xarray as xr
import time
import pygmt
import AbFab as af

# ============================================================================
# USER PARAMETERS - Modify these as needed
# ============================================================================

# Region selection (longitude, latitude bounds)
XMIN, XMAX = -30, 30
YMIN, YMAX = -70, 10
#XMIN, XMAX = 63, 74
#YMIN, YMAX = -30, -23

# Grid parameters
SPACING = '1m'  # Grid spacing (e.g., '2m' = 2 arcmin)

# Fixed abyssal hill parameters (baseline for spreading rate scaling)
PARAMS_BASE = {
    'H': 250.0,       # RMS height (m)
    'lambda_n': 3,  # Perpendicular wavelength (km)
    'lambda_s': 30.0, # Parallel wavelength (km)
    'D': 2.2          # Fractal dimension
}

# Thermal subsidence model
SUBSIDENCE_MODEL = 'GDH1'  # Options: 'GDH1', 'half_space', 'plate'

# Sediment treatment
SEDIMENT_MODE = 'fill'  # Options: 'none', 'drape', 'fill'
SEDIMENT_DIFFUSION = 0.3  # Diffusion coefficient for 'fill' mode (0-1, higher = more ponding)

# Chunking parameters for parallel processing
CHUNKSIZE = 100      # Chunk size in pixels
CHUNKPAD = 20        # Padding to avoid edge effects
NUM_CPUS = 6         # Number of parallel workers

# Optimization settings
USE_OPTIMIZATION = True   # Use filter bank (50x speedup)
AZIMUTH_BINS = 36        # Azimuth discretization (18-72 typical)
SEDIMENT_BINS = 8        # Sediment discretization (3-10 typical)
SPREADING_RATE_BINS = 10  # Spreading rate discretization

# Random seed for reproducibility
RANDOM_SEED = 42

# Output files
OUTPUT_FIGURE = 'complete_bathymetry.png'
OUTPUT_NETCDF = 'complete_bathymetry.nc'
DPI = 300

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_complete_bathymetry_chunk(coord, age_dataarray, sed_dataarray, rand_dataarray,
                                       chunksize, chunkpad, params, lon_spacing_deg,
                                       subsidence_model, sediment_mode, sediment_diffusion,
                                       sediment_range=None, spreading_rate_range=None):
    """
    Process a single chunk of complete bathymetry.

    Combines thermal subsidence + abyssal hills + sediment drape.
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

    # Calculate grid spacing at chunk's mean latitude (spherical correction)
    chunk_lat_coords = chunk_age.lat.values
    mean_lat = float(np.mean(chunk_lat_coords))
    grid_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(mean_lat))

    # Generate complete bathymetry (subsidence + hills + sediment)
    complete_bathy = af.generate_complete_bathymetry(
        chunk_age.data,
        chunk_sed.data,
        params,
        grid_spacing_km,
        random_field=chunk_random.data,
        subsidence_model=subsidence_model,
        sediment_mode=sediment_mode,
        sediment_diffusion=sediment_diffusion,
        filter_type='gaussian',
        optimize=USE_OPTIMIZATION,
        azimuth_bins=AZIMUTH_BINS,
        sediment_bins=SEDIMENT_BINS,
        spreading_rate_bins=SPREADING_RATE_BINS,
        base_params=params,
        sediment_range=sediment_range,
        spreading_rate_range=spreading_rate_range,
        lat_coords=chunk_lat_coords
    )

    # Trim padding and return
    pad_half = chunkpad // 2
    return xr.DataArray(
        complete_bathy,
        coords=chunk_age.coords,
        name='bathymetry'
    )[pad_half:-pad_half, pad_half:-pad_half]


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*70)
    print("Complete Bathymetry Generation")
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

    # Generate chunk coordinates
    full_ny, full_nx = age_da.shape
    chunkpad = int(2 * np.round(CHUNKPAD / 2))  # Ensure even
    coords_y, coords_x = np.meshgrid(np.arange(0, full_ny-1, CHUNKSIZE),
                                      np.arange(0, full_nx-1, CHUNKSIZE))
    coords = list(zip(coords_y.flatten(), coords_x.flatten()))

    print(f"\nProcessing {len(coords)} chunks (size={CHUNKSIZE}, pad={chunkpad}, CPUs={NUM_CPUS})")
    print(f"Optimization: {'ENABLED' if USE_OPTIMIZATION else 'DISABLED'}")
    print(f"  • Azimuth bins: {AZIMUTH_BINS}")
    print(f"  • Sediment bins: {SEDIMENT_BINS}")
    print(f"  • Spreading rate bins: {SPREADING_RATE_BINS}")
    print(f"  • Subsidence model: {SUBSIDENCE_MODEL}")
    print(f"  • Sediment mode: {SEDIMENT_MODE}")
    if SEDIMENT_MODE == 'fill':
        print(f"  • Sediment diffusion: {SEDIMENT_DIFFUSION}")

    # Calculate global ranges for consistent binning
    print("\nCalculating global bin ranges...")
    spreading_rate_global = af.calculate_spreading_rate_from_age(
        age_da.data, grid_spacing_km, lat_coords=age_da.lat.values
    )
    spreading_rate_global = np.where(np.isnan(spreading_rate_global),
                                      np.nanmedian(spreading_rate_global),
                                      spreading_rate_global)
    sr_min_global = float(np.nanmin(spreading_rate_global))
    sr_max_global = float(np.nanmax(spreading_rate_global))

    sed_min_global = float(np.nanmin(sed_da.data))
    sed_max_global = float(np.nanmax(sed_da.data))

    print(f"  Global spreading rate range: {sr_min_global:.1f} - {sr_max_global:.1f} mm/yr")
    print(f"  Global sediment range: {sed_min_global:.1f} - {sed_max_global:.1f} m")

    # Process all chunks in parallel
    print("\n" + "="*70)
    print("Generating Complete Bathymetry...")
    print("="*70)

    start_time = time.time()
    results = Parallel(n_jobs=NUM_CPUS, timeout=600)(delayed(process_complete_bathymetry_chunk)(
        coord, age_da, sed_da, rand_da, CHUNKSIZE, chunkpad, PARAMS_BASE, lon_spacing_deg,
        SUBSIDENCE_MODEL, SEDIMENT_MODE, SEDIMENT_DIFFUSION,
        sediment_range=(sed_min_global, sed_max_global),
        spreading_rate_range=(sr_min_global, sr_max_global)
    ) for coord in coords)

    elapsed = time.time() - start_time

    # Filter out empty chunks
    results = [result for result in results if 0 not in result.shape]

    print(f"Completed in {elapsed:.1f} seconds ({len(results)} valid chunks)")
    print(f"  → {elapsed/len(results):.2f} seconds per chunk")

    # ========================================================================
    # ASSEMBLE COMPLETE GRID
    # ========================================================================
    print("\n" + "="*70)
    print("Assembling complete grid...")
    print("="*70)

    complete_grid = xr.DataArray(
        np.full((full_ny, full_nx), np.nan),
        coords={'lat': age_da.lat, 'lon': age_da.lon},
        dims=['lat', 'lon'],
        name='bathymetry',
        attrs={
            'long_name': 'Complete synthetic bathymetry',
            'units': 'm',
            'description': 'Thermal subsidence + abyssal hills + sediment',
            'subsidence_model': SUBSIDENCE_MODEL,
            'sediment_mode': SEDIMENT_MODE,
            'sediment_diffusion': SEDIMENT_DIFFUSION if SEDIMENT_MODE == 'fill' else 'N/A',
            'base_H': PARAMS_BASE['H'],
            'base_lambda_n': PARAMS_BASE['lambda_n'],
            'base_lambda_s': PARAMS_BASE['lambda_s'],
            'base_D': PARAMS_BASE['D'],
            'spreading_rate_bins': SPREADING_RATE_BINS,
            'azimuth_bins': AZIMUTH_BINS,
            'sediment_bins': SEDIMENT_BINS,
            'spherical_correction': 'enabled',
            'random_seed': RANDOM_SEED
        }
    )

    # Fill in the chunks
    for res in results:
        lat_slice = slice(res.lat.values[0], res.lat.values[-1])
        lon_slice = slice(res.lon.values[0], res.lon.values[-1])
        complete_grid.loc[{'lat': lat_slice, 'lon': lon_slice}] = res.data

    # ========================================================================
    # SAVE TO NETCDF
    # ========================================================================
    print("\nSaving to NetCDF...")
    complete_grid.to_netcdf(OUTPUT_NETCDF)
    print(f"✓ Saved: {OUTPUT_NETCDF}")
    print(f"  Grid shape: {complete_grid.shape}")
    print(f"  Lat range: {float(complete_grid.lat.min()):.2f}° to {float(complete_grid.lat.max()):.2f}°")
    print(f"  Lon range: {float(complete_grid.lon.min()):.2f}° to {float(complete_grid.lon.max()):.2f}°")
    print(f"  Valid pixels: {np.sum(~np.isnan(complete_grid.data))}/{complete_grid.size}")
    print(f"  Depth range: {float(np.nanmin(complete_grid.data)):.0f} to {float(np.nanmax(complete_grid.data)):.0f} m")

    # ========================================================================
    # GENERATE COMPONENTS FOR VISUALIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("Generating component grids for visualization...")
    print("="*70)

    # Calculate subsidence component
    print("  Calculating thermal subsidence...")
    subsidence_grid = af.calculate_thermal_subsidence(age_da.data, model=SUBSIDENCE_MODEL)

    # Extract a representative profile for detailed analysis
    profile_idx = full_ny // 2
    lon_profile = complete_grid.lon.values
    complete_profile = complete_grid.data[profile_idx, :]
    subsidence_profile = subsidence_grid[profile_idx, :]
    age_profile = age_da.data[profile_idx, :]
    sediment_profile = sed_da.data[profile_idx, :]

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("Creating visualization...")
    print("="*70)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Complete bathymetry (large, top)
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.pcolormesh(complete_grid.lon, complete_grid.lat, complete_grid.data,
                         cmap='terrain', shading='auto')
    sediment_label = f"Sediment [{SEDIMENT_MODE}]"
    ax1.set_title(f'Complete Synthetic Bathymetry\n(Subsidence [{SUBSIDENCE_MODEL}] + Abyssal Hills + {sediment_label})',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude (°E)', fontsize=12)
    ax1.set_ylabel('Latitude (°N)', fontsize=12)
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Depth (m)', orientation='horizontal', pad=0.05)

    # Plot 2: Regional subsidence only
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.pcolormesh(complete_grid.lon, complete_grid.lat, subsidence_grid,
                         cmap='terrain', shading='auto')
    ax2.set_title(f'Regional Subsidence Only\n({SUBSIDENCE_MODEL} Model)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude (°E)', fontsize=11)
    ax2.set_ylabel('Latitude (°N)', fontsize=11)
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Depth (m)')

    # Plot 3: Abyssal hills component (residual)
    ax3 = fig.add_subplot(gs[1, 1])
    hills_component = complete_grid.data - subsidence_grid
    if SEDIMENT_MODE == 'drape':
        hills_component = hills_component - sed_da.data
    elif SEDIMENT_MODE == 'fill':
        # For 'fill' mode, sediment is already incorporated in the bathymetry
        # The residual shows the combined effect of hills + sediment redistribution
        pass
    im3 = ax3.pcolormesh(complete_grid.lon, complete_grid.lat, hills_component,
                         cmap='seismic', shading='auto', vmin=-500, vmax=500)
    residual_label = 'Complete - Subsidence' + (' - Sediment' if SEDIMENT_MODE == 'drape' else '')
    ax3.set_title(f'Abyssal Hills Component\n({residual_label})',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude (°E)', fontsize=11)
    ax3.set_ylabel('Latitude (°N)', fontsize=11)
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='Height (m)')

    # Plot 4: Cross-section profile
    ax4 = fig.add_subplot(gs[2, :])

    # Plot bathymetry
    ax4.plot(lon_profile, complete_profile, 'k-', linewidth=2, label='Complete Bathymetry', alpha=0.8)
    ax4.plot(lon_profile, subsidence_profile, 'b--', linewidth=2, label='Subsidence Only', alpha=0.8)

    # Shade sediment layer if mode is drape or fill
    if SEDIMENT_MODE in ['drape', 'fill']:
        sediment_top = complete_profile
        sediment_bottom = complete_profile - sediment_profile
        sediment_label = 'Sediment Layer' if SEDIMENT_MODE == 'drape' else 'Sediment (diffused)'
        ax4.fill_between(lon_profile, sediment_top, sediment_bottom,
                        color='brown', alpha=0.3, label=sediment_label)

    # Add RMS height reference
    ax4.fill_between(lon_profile,
                    subsidence_profile - PARAMS_BASE['H'],
                    subsidence_profile + PARAMS_BASE['H'],
                    color='red', alpha=0.15, label=f"±{PARAMS_BASE['H']:.0f}m (Base H)")

    ax4.set_xlabel('Longitude (°E)', fontsize=12)
    ax4.set_ylabel('Depth (m)', fontsize=12)
    ax4.set_title(f'Cross-Section at {age_da.lat.values[profile_idx]:.1f}°N',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10, loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()

    # Add age as secondary x-axis
    ax4_age = ax4.twiny()
    ax4_age.plot(age_profile, complete_profile, alpha=0)  # Invisible plot for alignment
    ax4_age.set_xlabel('Seafloor Age (Myr)', fontsize=12)
    ax4_age.set_xlim(ax4.get_xlim())

    plt.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_FIGURE}")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print(f"\nComplete bathymetry:")
    print(f"  Min depth: {np.nanmin(complete_grid.data):.0f} m")
    print(f"  Max depth: {np.nanmax(complete_grid.data):.0f} m")
    print(f"  Mean depth: {np.nanmean(complete_grid.data):.0f} m")
    print(f"  Depth range: {np.nanmax(complete_grid.data) - np.nanmin(complete_grid.data):.0f} m")

    print(f"\nRegional subsidence:")
    print(f"  Min depth: {np.nanmin(subsidence_grid):.0f} m")
    print(f"  Max depth: {np.nanmax(subsidence_grid):.0f} m")
    print(f"  Depth range: {np.nanmax(subsidence_grid) - np.nanmin(subsidence_grid):.0f} m")

    if SEDIMENT_MODE != 'none':
        print(f"\nSediment ({SEDIMENT_MODE} mode):")
        print(f"  Mean thickness: {np.nanmean(sed_da.data):.0f} m")
        print(f"  Max thickness: {np.nanmax(sed_da.data):.0f} m")
        if SEDIMENT_MODE == 'fill':
            print(f"  Diffusion coefficient: {SEDIMENT_DIFFUSION}")

    print(f"\nAbyssal hills (estimated from residual):")
    print(f"  RMS amplitude: {np.nanstd(hills_component):.0f} m")
    print(f"  Target base H: {PARAMS_BASE['H']:.0f} m")
    print(f"  (Varies with spreading rate and sediment)")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  • {OUTPUT_NETCDF} - Complete bathymetry grid")
    print(f"  • {OUTPUT_FIGURE} - Visualization")


if __name__ == "__main__":
    main()
