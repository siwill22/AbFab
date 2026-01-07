#!/usr/bin/env python
"""
GPU-Accelerated Complete Bathymetry Generation

This script provides GPU-accelerated bathymetry generation using PyTorch MPS
backend for Apple Silicon Macs (M1/M2/M3/M4).

Uses the same YAML configuration format as generate_complete_bathymetry.py

Usage:
    python generate_complete_bathymetry_gpu.py [config.yaml]

Requirements:
    pip install torch>=2.0  # For MPS support

Performance Notes:
    - M2 with 96GB unified memory can process large grids without chunking
    - Expected speedup: 5-20× depending on grid size
    - Best performance for grids > 500×500 pixels

Author: AbFab project
"""

import sys
import os
import yaml
import numpy as np
import time

# Import GPU module
import AbFab_gpu as af_gpu


def load_config(config_file='config_default.yaml'):
    """Load configuration from YAML file with defaults."""
    default_config = {
        'input': {
            'age_file': '/Users/simon/Data/AgeGrids/2020/age.2020.1.GeeK2007.6m.nc',
            'sediment_file': '/Users/simon/GIT/pyBacktrack/pybacktrack/bundle_data/sediment_thickness/GlobSed.nc',
            'constant_sediment': None
        },
        'region': {
            'lon_min': -180,
            'lon_max': 180,
            'lat_min': -70,
            'lat_max': 70,
            'spacing': '5m'
        },
        'abyssal_hills': {
            'H': 250.0,
            'lambda_n': 3.0,
            'lambda_s': 30.0,
            'D': 2.2,
            'filter_type': 'gaussian'
        },
        'subsidence': {
            'model': 'GDH1'
        },
        'sediment': {
            'mode': 'fill',
            'diffusion': 0.3
        },
        'optimization': {
            'enabled': True,
            'azimuth_bins': 36,
            'sediment_bins': 10,
            'spreading_rate_bins': 20
        },
        'parallel': {
            'num_cpus': 8,
            'chunk_size': 100,
            'chunk_pad': 20
        },
        'output': {
            'netcdf': 'complete_bathymetry_gpu.nc',
            'figure': 'complete_bathymetry_gpu.png',
            'dpi': 300,
            'verbose': True
        },
        'advanced': {
            'random_seed': 42,
            'timeout': 600
        },
        'gpu': {
            'tile_size': 500  # Internal tile size for GPU memory management
        }
    }

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)

        def merge_configs(default, user):
            for key, value in user.items():
                if isinstance(value, dict) and key in default:
                    merge_configs(default[key], value)
                else:
                    default[key] = value
            return default

        config = merge_configs(default_config, user_config)
        print(f"Loaded configuration from: {config_file}")
    else:
        if config_file != 'config_default.yaml':
            print(f"Warning: Config file '{config_file}' not found, using defaults")
        config = default_config

    return config


def run_gpu_workflow(config):
    """
    Run the complete GPU-accelerated bathymetry workflow.
    
    This handles the full pipeline including:
    - Loading and preprocessing data
    - GPU-accelerated generation
    - Global diffusive sediment infill (if mode='fill')
    - Saving results
    """
    import pygmt
    import xarray as xr
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    verbose = config['output'].get('verbose', True)
    
    if verbose:
        print("="*70)
        print("GPU-Accelerated Bathymetry Generation")
        print(f"Device: {af_gpu.DEVICE}")
        print("="*70)
    
    total_start = time.time()
    
    # ========================================================================
    # LOAD INPUT DATA
    # ========================================================================
    if verbose:
        print("\nLoading input data...")
    
    age_file = config['input']['age_file']
    sediment_file = config['input'].get('sediment_file')
    constant_sediment = config['input'].get('constant_sediment')
    spacing = config['region']['spacing']
    
    t_load = time.time()
    age_da = pygmt.grdsample(age_file, region='g', spacing=spacing)
    
    if sediment_file is not None:
        sed_da = pygmt.grdsample(sediment_file, region='g', spacing=spacing)
        sed_da = sed_da.where(np.isfinite(sed_da), 1.)
        sed_da = sed_da.where(sed_da < 1000., 1000.)
        if verbose:
            print(f"  Loaded sediment from: {sediment_file}")
    elif constant_sediment is not None:
        sed_da = age_da.copy()
        sed_da.data = np.full_like(age_da.data, constant_sediment)
        sed_da = sed_da.where(~np.isnan(age_da.data), np.nan)
        if verbose:
            print(f"  Using constant sediment: {constant_sediment} m")
    else:
        sed_da = age_da.copy()
        sed_da.data = np.zeros_like(age_da.data)
        if verbose:
            print("  No sediment data")
    
    if verbose:
        print(f"  Data loading: {time.time() - t_load:.2f}s")
    
    # ========================================================================
    # REGION SELECTION AND PERIODIC BOUNDARY HANDLING
    # ========================================================================
    lon_min = config['region']['lon_min']
    lon_max = config['region']['lon_max']
    lat_min = config['region']['lat_min']
    lat_max = config['region']['lat_max']
    
    # Validate region
    if lat_min >= lat_max:
        raise ValueError(f"Invalid latitude range: {lat_min} >= {lat_max}")
    if lon_min >= lon_max:
        raise ValueError(f"Invalid longitude range: {lon_min} >= {lon_max}")
    
    # Import AbFab for helper functions
    import AbFab as af
    
    # Check if this is a global grid
    is_global = af.is_global_longitude(age_da.lon.values)
    
    if verbose:
        print(f"\nLongitude coverage: {'GLOBAL' if is_global else 'REGIONAL'}")
    
    # Handle global grids with periodic boundaries
    if is_global:
        # Get grid spacing for margin calculation
        lon_spacing_deg = float(np.abs(age_da.lon.values[1] - age_da.lon.values[0]))
        grid_spacing_km_equator = lon_spacing_deg * 111.32
        
        params_base = config['abyssal_hills']
        required_margin = af.calculate_required_longitude_margin(
            params_base, grid_spacing_km_equator, 
            config.get('gpu', {}).get('tile_size', 500) // 10  # Approximate padding
        )
        
        if verbose:
            print(f"Periodic boundary handling: ENABLED")
            print(f"  Required margin: {required_margin:.1f}°")
        
        # Extend longitude range for continuity
        age_da = af.extend_longitude_range(age_da)
        sed_da = af.extend_longitude_range(sed_da)
        
        # Select with margin
        margin_range = 180 + required_margin
        age_da = age_da.sel(lon=slice(-margin_range, margin_range))
        sed_da = sed_da.sel(lon=slice(-margin_range, margin_range))
        
        # Store original bounds for later trimming
        original_lon_min, original_lon_max = lon_min, lon_max
        
        # Select latitude
        age_da = age_da.sel(lat=slice(lat_min, lat_max))
        sed_da = sed_da.sel(lat=slice(lat_min, lat_max))
    else:
        # Regional grid - simple selection
        if verbose:
            print(f"Periodic boundary handling: DISABLED (regional grid)")
        age_da = age_da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        sed_da = sed_da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        original_lon_min, original_lon_max = None, None
    
    ny, nx = age_da.shape
    
    if verbose:
        print(f"\nRegion: {lon_min}° to {lon_max}°E, {lat_min}° to {lat_max}°N")
        print(f"Grid shape: {ny} × {nx} = {ny*nx:,} pixels")
        print(f"Age range: {np.nanmin(age_da.data):.1f} - {np.nanmax(age_da.data):.1f} Myr")
        if sediment_file or constant_sediment:
            print(f"Sediment range: {np.nanmin(sed_da.data):.1f} - {np.nanmax(sed_da.data):.1f} m")
    
    # ========================================================================
    # CALCULATE GRID SPACING
    # ========================================================================
    lon_spacing_deg = float(np.abs(age_da.lon.values[1] - age_da.lon.values[0]))
    mean_lat = float(np.mean(age_da.lat.values))
    grid_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(mean_lat))
    
    if verbose:
        print(f"\nGrid spacing: {grid_spacing_km:.3f} km/pixel at {mean_lat:.1f}°N")
    
    # ========================================================================
    # GENERATE RANDOM FIELD
    # ========================================================================
    random_seed = config['advanced'].get('random_seed')
    if random_seed is not None:
        np.random.seed(random_seed)
    random_field = np.random.randn(ny, nx).astype(np.float32)
    
    if verbose:
        print(f"Random seed: {random_seed}")
    
    # ========================================================================
    # GPU BATHYMETRY GENERATION
    # ========================================================================
    params = config['abyssal_hills']
    
    if verbose:
        print(f"\n{'='*70}")
        print("GPU Processing")
        print('='*70)
        print(f"Parameters: H={params['H']}m, λ_n={params['lambda_n']}km, λ_s={params['lambda_s']}km")
        print(f"Filter bank: {config['optimization']['azimuth_bins']}×"
              f"{config['optimization']['sediment_bins']}×"
              f"{config['optimization']['spreading_rate_bins']} bins")
    
    t_gpu_start = time.time()
    
    # Use 'drape' mode for GPU generation; apply diffusion globally after
    gpu_sediment_mode = 'drape' if config['sediment']['mode'] == 'fill' else config['sediment']['mode']
    
    # Get tile_size for GPU memory management
    tile_size = config.get('gpu', {}).get('tile_size', 500)
    if verbose:
        print(f"  GPU tile size: {tile_size}")
    
    bathymetry = af_gpu.generate_complete_bathymetry_gpu(
        age_da.data.astype(np.float32),
        sed_da.data.astype(np.float32),
        params,
        grid_spacing_km,
        random_field,
        lat_coords=age_da.lat.values.astype(np.float32),
        subsidence_model=config['subsidence']['model'],
        sediment_mode=gpu_sediment_mode,
        azimuth_bins=config['optimization']['azimuth_bins'],
        sediment_bins=config['optimization']['sediment_bins'],
        spreading_rate_bins=config['optimization']['spreading_rate_bins'],
        tile_size=tile_size,
        verbose=verbose
    )
    
    t_gpu_end = time.time()
    
    if verbose:
        print(f"\nGPU generation complete: {t_gpu_end - t_gpu_start:.2f}s")
    
    # ========================================================================
    # GLOBAL DIFFUSIVE SEDIMENT INFILL (if needed)
    # ========================================================================
    if config['sediment']['mode'] == 'fill' and (sediment_file or constant_sediment):
        if verbose:
            print(f"\n{'='*70}")
            print("Applying Global Diffusive Sediment Infill")
            print('='*70)
        
        import AbFab as af
        
        t_diff_start = time.time()
        
        # Calculate basement (bathymetry without sediment)
        basement = bathymetry - sed_da.data
        
        # Apply diffusive infill globally
        final_bathymetry = af.apply_diffusive_sediment_infill(
            basement,
            sed_da.data,
            grid_spacing_km,
            config['sediment']['diffusion']
        )
        
        bathymetry = final_bathymetry
        
        t_diff_end = time.time()
        if verbose:
            print(f"Diffusive infill complete: {t_diff_end - t_diff_start:.2f}s")
    
    # ========================================================================
    # CREATE OUTPUT DATAARRAY
    # ========================================================================
    complete_grid = xr.DataArray(
        bathymetry,
        coords={'lat': age_da.lat, 'lon': age_da.lon},
        dims=['lat', 'lon'],
        name='bathymetry',
        attrs={
            'long_name': 'Complete synthetic bathymetry (GPU)',
            'units': 'm',
            'description': 'Thermal subsidence + abyssal hills + sediment',
            'subsidence_model': config['subsidence']['model'],
            'sediment_mode': config['sediment']['mode'],
            'base_H': params['H'],
            'base_lambda_n': params['lambda_n'],
            'base_lambda_s': params['lambda_s'],
            'base_D': params['D'],
            'device': str(af_gpu.DEVICE),
            'random_seed': random_seed
        }
    )
    
    # Add CF-compliant coordinate attributes
    complete_grid.lon.attrs.update({
        'units': 'degrees_east',
        'standard_name': 'longitude',
        'axis': 'X'
    })
    complete_grid.lat.attrs.update({
        'units': 'degrees_north',
        'standard_name': 'latitude',
        'axis': 'Y'
    })
    
    # ========================================================================
    # TRIM TO ORIGINAL EXTENT (for global grids)
    # ========================================================================
    if is_global and original_lon_min is not None:
        if verbose:
            print("\nTrimming to original longitude extent...")
        
        # Remove duplicate longitudes from extension
        lon_vals = complete_grid.lon.values
        _, unique_indices = np.unique(lon_vals, return_index=True)
        unique_indices = np.sort(unique_indices)
        complete_grid = complete_grid.isel(lon=unique_indices)
        
        # Trim to original extent (exclusive of max to avoid duplicate at wrap)
        lon_max_exclusive = original_lon_max - lon_spacing_deg / 2
        complete_grid = complete_grid.sel(lon=slice(original_lon_min, lon_max_exclusive))
        
        if verbose:
            print(f"  Trimmed range: {float(complete_grid.lon.min()):.1f}° to {float(complete_grid.lon.max()):.1f}°")
    
    # ========================================================================
    # SAVE OUTPUT
    # ========================================================================
    output_nc = config['output']['netcdf']
    if verbose:
        print(f"\n{'='*70}")
        print("Saving Output")
        print('='*70)
        print(f"NetCDF: {output_nc}")
    
    complete_grid.to_netcdf(output_nc)
    
    # ========================================================================
    # GENERATE VISUALIZATION
    # ========================================================================
    output_fig = config['output'].get('figure')
    if output_fig:
        if verbose:
            print(f"Figure: {output_fig}")
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # Complete bathymetry
        ax1 = fig.add_subplot(gs[0, :])
        im1 = ax1.pcolormesh(complete_grid.lon, complete_grid.lat, complete_grid.data,
                            cmap='terrain', shading='auto')
        ax1.set_title(f'Complete Synthetic Bathymetry (GPU)\n'
                     f'Subsidence [{config["subsidence"]["model"]}] + '
                     f'Abyssal Hills + Sediment [{config["sediment"]["mode"]}]',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Longitude (°E)')
        ax1.set_ylabel('Latitude (°N)')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='Depth (m)', orientation='horizontal', pad=0.08)
        
        # Subsidence only - reload age for trimmed region
        ax2 = fig.add_subplot(gs[1, 0])
        # Reload age data for the trimmed output region
        age_for_plot = pygmt.grdsample(
            config['input']['age_file'],
            region=[float(complete_grid.lon.min()), float(complete_grid.lon.max()),
                   float(complete_grid.lat.min()), float(complete_grid.lat.max())],
            spacing=config['region']['spacing']
        )
        subsidence = af_gpu.to_numpy(
            af_gpu.calculate_thermal_subsidence_gpu(
                af_gpu.to_torch(age_for_plot.data.astype(np.float32)),
                model=config['subsidence']['model']
            )
        )
        im2 = ax2.pcolormesh(age_for_plot.lon, age_for_plot.lat, subsidence,
                            cmap='terrain', shading='auto')
        ax2.set_title(f'Thermal Subsidence Only\n({config["subsidence"]["model"]})')
        ax2.set_xlabel('Longitude (°E)')
        ax2.set_ylabel('Latitude (°N)')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='Depth (m)')
        
        # Abyssal hills component
        ax3 = fig.add_subplot(gs[1, 1])
        # Resample subsidence to match complete_grid if shapes differ
        if subsidence.shape != complete_grid.data.shape:
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(
                (age_for_plot.lat.values, age_for_plot.lon.values), 
                subsidence,
                bounds_error=False, fill_value=np.nan
            )
            lat_grid, lon_grid = np.meshgrid(complete_grid.lat.values, complete_grid.lon.values, indexing='ij')
            subsidence_resampled = interp((lat_grid, lon_grid))
        else:
            subsidence_resampled = subsidence
        
        hills = complete_grid.data - subsidence_resampled
        # Load sediment for trimmed region if needed
        if config['sediment']['mode'] in ['drape', 'fill']:
            if config['input'].get('sediment_file'):
                sed_for_plot = pygmt.grdsample(
                    config['input']['sediment_file'],
                    region=[float(complete_grid.lon.min()), float(complete_grid.lon.max()),
                           float(complete_grid.lat.min()), float(complete_grid.lat.max())],
                    spacing=config['region']['spacing']
                )
                sed_for_plot = sed_for_plot.where(np.isfinite(sed_for_plot), 1.)
                # Resample if needed
                if sed_for_plot.shape != complete_grid.data.shape:
                    interp_sed = RegularGridInterpolator(
                        (sed_for_plot.lat.values, sed_for_plot.lon.values), 
                        sed_for_plot.data,
                        bounds_error=False, fill_value=0
                    )
                    sed_resampled = interp_sed((lat_grid, lon_grid))
                else:
                    sed_resampled = sed_for_plot.data
                hills = hills - sed_resampled
            elif config['input'].get('constant_sediment'):
                hills = hills - config['input']['constant_sediment']
        
        vmax = np.nanpercentile(np.abs(hills[np.isfinite(hills)]), 99) if np.any(np.isfinite(hills)) else 500
        im3 = ax3.pcolormesh(complete_grid.lon, complete_grid.lat, hills,
                            cmap='seismic', shading='auto', vmin=-vmax, vmax=vmax)
        ax3.set_title('Abyssal Hills Component\n(Residual)')
        ax3.set_xlabel('Longitude (°E)')
        ax3.set_ylabel('Latitude (°N)')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3, label='Height (m)')
        
        plt.savefig(output_fig, dpi=config['output'].get('dpi', 300), bbox_inches='tight')
        plt.close(fig)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_end = time.time()
    
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        print(f"\nTotal processing time: {total_end - total_start:.1f}s")
        print(f"  GPU generation: {t_gpu_end - t_gpu_start:.1f}s")
        if config['sediment']['mode'] == 'fill':
            print(f"  Diffusive infill: {t_diff_end - t_diff_start:.1f}s")
        
        print(f"\nOutput grid:")
        print(f"  Shape: {complete_grid.shape}")
        print(f"  Depth range: {float(np.nanmin(bathymetry)):.0f} to {float(np.nanmax(bathymetry)):.0f} m")
        print(f"  Valid pixels: {np.sum(~np.isnan(bathymetry)):,}/{bathymetry.size:,}")
        
        # Performance metrics
        pixels_per_second = bathymetry.size / (t_gpu_end - t_gpu_start)
        print(f"\nPerformance:")
        print(f"  {pixels_per_second:,.0f} pixels/second")
        print(f"  {pixels_per_second * 3600 / 1e6:.1f} megapixels/hour")
        
        print(f"\n{'='*70}")
        print("COMPLETE")
        print('='*70)
        print(f"\nGenerated files:")
        print(f"  • {output_nc}")
        if output_fig:
            print(f"  • {output_fig}")
    
    return complete_grid


def main():
    """Main entry point for CLI."""
    print("="*70)
    print("AbFab GPU - Complete Bathymetry Generation")
    print("="*70)
    
    # Check GPU availability
    if not af_gpu.MPS_AVAILABLE:
        print("\nWARNING: Apple MPS (GPU) not available!")
        print("This will run on CPU, which is slower.")
        print("Ensure you have PyTorch 2.0+ with MPS support.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'config_default.yaml'
    
    # Load configuration
    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Modify output filenames to indicate GPU version
    if 'gpu' not in config['output']['netcdf'].lower():
        base, ext = os.path.splitext(config['output']['netcdf'])
        config['output']['netcdf'] = f"{base}_gpu{ext}"
    
    if config['output'].get('figure'):
        if 'gpu' not in config['output']['figure'].lower():
            base, ext = os.path.splitext(config['output']['figure'])
            config['output']['figure'] = f"{base}_gpu{ext}"
    
    # Display settings
    print(f"\nConfiguration: {config_file}")
    print(f"Region: {config['region']['lon_min']}° to {config['region']['lon_max']}°E, "
          f"{config['region']['lat_min']}° to {config['region']['lat_max']}°N")
    print(f"Spacing: {config['region']['spacing']}")
    print(f"Output: {config['output']['netcdf']}")
    
    # Run workflow
    try:
        complete_grid = run_gpu_workflow(config)
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
