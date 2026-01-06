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
#XMIN, XMAX = -30, 30
#YMIN, YMAX = -70, 10
#XMIN, XMAX = 63, 74
#YMIN, YMAX = -30, -23
XMIN, XMAX = -180, 180
YMIN, YMAX = -70, 70
#XMIN, XMAX = 0, 50  # Southern Africa test region
#YMIN, YMAX = -40, -20

# Grid parameters
SPACING = '5m'  # Grid spacing (e.g., '2m' = 2 arcmin, '5m' = 5 arcmin for testing)

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
SEDIMENT_BINS = 10        # Sediment discretization (3-10 typical)
SPREADING_RATE_BINS = 20  # Spreading rate discretization

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
                                       sediment_range=None, spreading_rate_range=None,
                                       sediment_levels=None, spreading_rate_levels=None,
                                       spreading_rate_fill_value=None,
                                       azimuth_dataarray=None, spreading_rate_dataarray=None):
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

    # Extract azimuth and spreading rate chunks if provided (for consistency)
    chunk_azimuth = None
    chunk_spreading_rate = None
    if azimuth_dataarray is not None:
        chunk_azimuth = azimuth_dataarray[coord[0]:coord[0]+chunksize+chunkpad,
                                          coord[1]:coord[1]+chunksize+chunkpad]
    if spreading_rate_dataarray is not None:
        chunk_spreading_rate = spreading_rate_dataarray[coord[0]:coord[0]+chunksize+chunkpad,
                                                        coord[1]:coord[1]+chunksize+chunkpad]

    # Skip empty chunks
    if np.all(np.isnan(chunk_age.data)):
        return chunk_age

    # Calculate grid spacing at chunk's mean latitude (spherical correction)
    chunk_lat_coords = chunk_age.lat.values
    mean_lat = float(np.mean(chunk_lat_coords))
    grid_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(mean_lat))

    # Generate basement bathymetry ONLY (subsidence + hills + simple sediment drape)
    # DO NOT apply diffusive infill here - it will be done globally after assembly
    basement_bathy = af.generate_complete_bathymetry(
        chunk_age.data,
        chunk_sed.data,
        params,
        grid_spacing_km,
        random_field=chunk_random.data,
        subsidence_model=subsidence_model,
        sediment_mode='drape',  # Use simple drape per chunk, diffusion applied globally later
        sediment_diffusion=sediment_diffusion,
        filter_type='gaussian',
        optimize=USE_OPTIMIZATION,
        azimuth_bins=AZIMUTH_BINS,
        sediment_bins=SEDIMENT_BINS,
        spreading_rate_bins=SPREADING_RATE_BINS,
        base_params=params,
        sediment_range=sediment_range,
        spreading_rate_range=spreading_rate_range,
        sediment_levels=sediment_levels,
        spreading_rate_levels=spreading_rate_levels,
        spreading_rate_fill_value=spreading_rate_fill_value,
        lat_coords=chunk_lat_coords,
        azimuth_field=chunk_azimuth.data if chunk_azimuth is not None else None,
        spreading_rate_field=chunk_spreading_rate.data if chunk_spreading_rate is not None else None
    )

    # Trim padding and return
    pad_half = chunkpad // 2
    trimmed_bathy = basement_bathy[pad_half:-pad_half, pad_half:-pad_half]

    # Get trimmed coordinates
    trimmed_lat = chunk_age.lat.values[pad_half:-pad_half]
    trimmed_lon = chunk_age.lon.values[pad_half:-pad_half]

    return xr.DataArray(
        trimmed_bathy,
        coords={'lat': trimmed_lat, 'lon': trimmed_lon},
        dims=['lat', 'lon'],
        name='bathymetry'
    )


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

    # Clean sediment data
    sed_da = sed_da.where(np.isfinite(sed_da), 1.)
    sed_da = sed_da.where(sed_da < 1000., 1000.)

    # Check if the requested region is global in longitude
    is_global = af.is_global_longitude(age_da.lon.values)

    # Calculate required margin for periodic boundary handling
    lon_spacing_deg_initial = float(np.abs(age_da.lon.values[1] - age_da.lon.values[0]))
    grid_spacing_km_equator = lon_spacing_deg_initial * 111.32
    required_margin = af.calculate_required_longitude_margin(
        PARAMS_BASE, grid_spacing_km_equator, CHUNKPAD
    )

    print(f"\nLongitude coverage: {'GLOBAL' if is_global else 'REGIONAL'}")
    if is_global:
        print(f"Periodic boundary handling: ENABLED")
        print(f"  Required margin: {required_margin:.1f}° (based on λ_s={PARAMS_BASE['lambda_s']}km, pad={CHUNKPAD}px)")

        # Extend longitude range for periodic boundary
        age_da = af.extend_longitude_range(age_da)
        sed_da = af.extend_longitude_range(sed_da)

        # Select extended region with margins
        margin_range = 180 + required_margin
        age_da = age_da.sel(lon=slice(-margin_range, margin_range))
        sed_da = sed_da.sel(lon=slice(-margin_range, margin_range))

        # Store original extent for trimming later
        original_lon_min, original_lon_max = XMIN, XMAX
    else:
        print(f"Periodic boundary handling: DISABLED (regional grid)")
        # For regional grids, just select the requested region
        age_da = age_da.sel(lon=slice(XMIN, XMAX), lat=slice(YMIN, YMAX))
        sed_da = sed_da.sel(lon=slice(XMIN, XMAX), lat=slice(YMIN, YMAX))
        original_lon_min, original_lon_max = None, None

    # Select latitude range (same for both global and regional)
    if is_global:
        age_da = age_da.sel(lat=slice(YMIN, YMAX))
        sed_da = sed_da.sel(lat=slice(YMIN, YMAX))

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

    # Calculate global azimuth and spreading rate for perfect consistency across chunks
    print("\nCalculating global azimuth and spreading rate fields...")

    # Calculate azimuth globally (ensures gradient is computed from full context)
    azimuth_global = af.calculate_azimuth_from_age(
        age_da.data, lat_coords=age_da.lat.values
    )
    print(f"  Azimuth field calculated: shape {azimuth_global.shape}")

    # Calculate spreading rate globally (ensures gradient is computed from full context)
    spreading_rate_global = af.calculate_spreading_rate_from_age(
        age_da.data, grid_spacing_km, lat_coords=age_da.lat.values
    )

    # Compute global median BEFORE filling NaNs (for consistent NaN filling across chunks)
    sr_median_global = float(np.nanmedian(spreading_rate_global))

    spreading_rate_global = np.where(np.isnan(spreading_rate_global),
                                      sr_median_global,
                                      spreading_rate_global)
    sr_min_global = float(np.nanmin(spreading_rate_global))
    sr_max_global = float(np.nanmax(spreading_rate_global))

    sed_min_global = float(np.nanmin(sed_da.data))
    sed_max_global = float(np.nanmax(sed_da.data))

    print(f"  Global spreading rate range: {sr_min_global:.1f} - {sr_max_global:.1f} mm/yr")
    print(f"  Global spreading rate median (for NaN fill): {sr_median_global:.1f} mm/yr")
    print(f"  Global sediment range: {sed_min_global:.1f} - {sed_max_global:.1f} m")

    # Compute bin levels ONCE globally to ensure consistency across all chunks
    print("\nComputing global bin levels for consistency...")
    sr_range = sr_max_global - sr_min_global
    if sr_range < 1e-6 or SPREADING_RATE_BINS == 1:
        spreading_rate_levels_global = np.array([np.mean([sr_min_global, sr_max_global])])
    else:
        spreading_rate_levels_global = np.linspace(sr_min_global, sr_max_global, SPREADING_RATE_BINS)

    sed_range = sed_max_global - sed_min_global
    if sed_range < 1e-6:
        sediment_levels_global = np.array([sed_min_global])
    else:
        sediment_levels_global = np.linspace(sed_min_global, sed_max_global, SEDIMENT_BINS)

    print(f"  Spreading rate bin levels (n={len(spreading_rate_levels_global)}): {spreading_rate_levels_global[:3]}...")
    print(f"  Sediment bin levels (n={len(sediment_levels_global)}): {sediment_levels_global[:3]}...")

    # Create DataArrays for azimuth and spreading rate (for consistent chunking)
    print("\nCreating DataArrays for global azimuth and spreading rate fields...")
    azimuth_da = xr.DataArray(
        azimuth_global,
        coords={'lat': age_da.lat, 'lon': age_da.lon},
        dims=['lat', 'lon'],
        name='azimuth'
    )
    spreading_rate_da = xr.DataArray(
        spreading_rate_global,
        coords={'lat': age_da.lat, 'lon': age_da.lon},
        dims=['lat', 'lon'],
        name='spreading_rate'
    )

    # Process all chunks in parallel
    print("\n" + "="*70)
    print("Generating Complete Bathymetry...")
    print("="*70)

    start_time = time.time()
    results = Parallel(n_jobs=NUM_CPUS, timeout=600)(delayed(process_complete_bathymetry_chunk)(
        coord, age_da, sed_da, rand_da, CHUNKSIZE, chunkpad, PARAMS_BASE, lon_spacing_deg,
        SUBSIDENCE_MODEL, SEDIMENT_MODE, SEDIMENT_DIFFUSION,
        sediment_range=(sed_min_global, sed_max_global),
        spreading_rate_range=(sr_min_global, sr_max_global),
        sediment_levels=sediment_levels_global,
        spreading_rate_levels=spreading_rate_levels_global,
        spreading_rate_fill_value=sr_median_global,
        azimuth_dataarray=azimuth_da,
        spreading_rate_dataarray=spreading_rate_da
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
    # APPLY DIFFUSIVE SEDIMENT INFILL (if requested)
    # This MUST be done globally AFTER assembly to avoid chunk discontinuities!
    # ========================================================================
    if SEDIMENT_MODE == 'fill':
        print("\n" + "="*70)
        print("Applying global diffusive sediment infill...")
        print("="*70)
        print(f"  Diffusion coefficient: {SEDIMENT_DIFFUSION}")

        # Calculate grid spacing at mean latitude
        mean_lat_global = float(complete_grid.lat.mean())
        grid_spacing_global = lon_spacing_deg * 111.32 * np.cos(np.radians(mean_lat_global))
        print(f"  Grid spacing (at {mean_lat_global:.1f}°N): {grid_spacing_global:.2f} km")

        # Get sediment thickness data matching complete_grid extent
        # Use simple nearest neighbor interpolation to avoid coordinate issues
        sed_trimmed_data = np.zeros_like(complete_grid.data)
        for i, lat in enumerate(complete_grid.lat.values):
            for j, lon in enumerate(complete_grid.lon.values):
                lat_idx = np.argmin(np.abs(sed_da.lat.values - lat))
                lon_idx = np.argmin(np.abs(sed_da.lon.values - lon))
                sed_trimmed_data[i, j] = sed_da.data[lat_idx, lon_idx]

        # Apply diffusive infill to the complete grid
        # Note: complete_grid currently has subsidence + hills + simple drape
        # We need to subtract the drape, apply diffusion, then it adds back
        # Actually, the diffusive infill function expects basement (without sediment drape)
        # But we already added drape in chunks. So we need to:
        # 1. Subtract sediment (to get basement + hills)
        # 2. Apply diffusive infill

        basement_grid = complete_grid.data - sed_trimmed_data  # Remove the drape

        print(f"  Applying diffusive infill to {complete_grid.shape} grid...")
        final_grid = af.apply_diffusive_sediment_infill(
            basement_grid,
            sed_trimmed_data,
            grid_spacing_global,
            SEDIMENT_DIFFUSION
        )

        complete_grid.data[:] = final_grid
        print(f"  ✓ Diffusive infill applied globally")

    # ========================================================================
    # TRIM TO ORIGINAL EXTENT AND REMOVE DUPLICATES (for global grids)
    # ========================================================================
    if is_global and original_lon_min is not None:
        print("\nTrimming to original longitude extent...")
        print(f"  Extended range: {float(complete_grid.lon.min()):.1f}° to {float(complete_grid.lon.max()):.1f}°")

        # Find unique longitude values and their indices
        lon_vals = complete_grid.lon.values
        _, unique_indices = np.unique(lon_vals, return_index=True)
        unique_indices = np.sort(unique_indices)  # Keep original order

        # Select only the first occurrence of each unique longitude value
        complete_grid = complete_grid.isel(lon=unique_indices)

        # Now trim to original extent, excluding +180° endpoint
        lon_max_exclusive = original_lon_max - lon_spacing_deg / 2
        complete_grid = complete_grid.sel(lon=slice(original_lon_min, lon_max_exclusive))

        print(f"  Trimmed range: {float(complete_grid.lon.min()):.1f}° to {float(complete_grid.lon.max()):.1f}°")
        print(f"  Unique longitudes: {len(np.unique(complete_grid.lon.values))} out of {len(complete_grid.lon.values)}")
        print(f"  Note: Excluded +180° to avoid duplicate with -180° (same point on sphere)")

    # ========================================================================
    # ADD CF-COMPLIANT METADATA
    # ========================================================================
    print("\nAdding CF-compliant coordinate metadata...")
    complete_grid = af.add_cf_compliant_coordinate_attrs(complete_grid)

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

    # For visualization, we need age and sediment grids that match complete_grid
    # Reload and process with clean coordinates
    print("  Reloading data for visualization...")
    age_vis = pygmt.grdsample('/Users/simon/Data/AgeGrids/2020/age.2020.1.GeeK2007.6m.nc',
                              region=[float(complete_grid.lon.min()), float(complete_grid.lon.max()),
                                     float(complete_grid.lat.min()), float(complete_grid.lat.max())],
                              spacing=SPACING)
    sed_vis = pygmt.grdsample('/Users/simon/GIT/pyBacktrack/pybacktrack/bundle_data/sediment_thickness/GlobSed.nc',
                              region=[float(complete_grid.lon.min()), float(complete_grid.lon.max()),
                                     float(complete_grid.lat.min()), float(complete_grid.lat.max())],
                              spacing=SPACING)
    sed_vis = sed_vis.where(np.isfinite(sed_vis), 1.)
    sed_vis = sed_vis.where(sed_vis < 1000., 1000.)

    # Calculate subsidence component (now matching trimmed dimensions)
    print("  Calculating thermal subsidence...")
    subsidence_grid = af.calculate_thermal_subsidence(age_vis.data, model=SUBSIDENCE_MODEL)

    # Extract a representative profile for detailed analysis
    profile_idx = len(complete_grid.lat) // 2
    lon_profile = complete_grid.lon.values
    complete_profile = complete_grid.data[profile_idx, :]
    subsidence_profile = subsidence_grid[profile_idx, :]
    age_profile = age_vis.data[profile_idx, :]
    sediment_profile = sed_vis.data[profile_idx, :]

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
        hills_component = hills_component - sed_vis.data
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
        print(f"  Mean thickness: {np.nanmean(sed_vis.data):.0f} m")
        print(f"  Max thickness: {np.nanmax(sed_vis.data):.0f} m")
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
