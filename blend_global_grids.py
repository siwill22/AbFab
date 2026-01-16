#!/usr/bin/env python3
"""
Blend three bathymetry grids (Mercator + Arctic + Antarctic) into a single global grid.

Uses GMT's grdblend command-line tool for seamless blending with automatic feathering.
"""

import pygmt
import xarray as xr
import numpy as np
import os
import subprocess

# ============================================================================
# Configuration
# ============================================================================

# Input grids (in output/ directory)
MERCATOR_GRID = 'output/test_mercator_large_150Ma_gpu.nc'
ARCTIC_GRID = 'output/test_arctic_gpu.nc'
ANTARCTIC_GRID = 'output/test_antarctic_gpu.nc'

# Output
OUTPUT_GRID = 'output/global_blended_bathymetry.nc'

# Blending parameters
SPACING = '2m'  # 2 arcminutes
REGION = 'd'    # Global: -180/180/-90/90

# Verbose output
VERBOSE = True

# ============================================================================
# Main Script
# ============================================================================

def check_grid_info(grid_path):
    """Load and display grid information."""
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    grid = xr.open_dataset(grid_path)

    # Get dimension names
    dims = list(grid.dims.keys())

    # Check if geographic or projected
    if 'lat' in dims and 'lon' in dims:
        coord_type = 'geographic'
        x_range = (float(grid.lon.min()), float(grid.lon.max()))
        y_range = (float(grid.lat.min()), float(grid.lat.max()))
        x_label, y_label = 'lon', 'lat'
    elif 'x' in dims and 'y' in dims:
        coord_type = 'projected'
        x_range = (float(grid.x.min()), float(grid.x.max()))
        y_range = (float(grid.y.min()), float(grid.y.max()))
        x_label, y_label = 'x', 'y'
    else:
        coord_type = 'unknown'
        x_range = None
        y_range = None
        x_label, y_label = dims[0], dims[1]

    # Get data variable name
    data_vars = list(grid.data_vars.keys())
    var_name = data_vars[0] if data_vars else 'unknown'

    grid.close()

    return {
        'path': grid_path,
        'coord_type': coord_type,
        'dims': dims,
        'x_label': x_label,
        'y_label': y_label,
        'x_range': x_range,
        'y_range': y_range,
        'var_name': var_name
    }

def reproject_if_needed(grid_path, info):
    """Reproject grid to geographic coordinates if it's in projected coordinates."""
    if info['coord_type'] == 'geographic':
        if VERBOSE:
            print(f"  Already in geographic coordinates")
        return grid_path

    if VERBOSE:
        print(f"  Reprojecting from projected to geographic coordinates...")

    # Determine projection based on filename/path
    grid_name = os.path.basename(grid_path).lower()

    if 'arctic' in grid_name:
        projection = 'EPSG:3413'  # North Polar Stereographic
        region_geo = [-180, 180, 60, 90]  # Arctic coverage
    elif 'antarctic' in grid_name:
        projection = 'EPSG:3031'  # South Polar Stereographic
        region_geo = [-180, 180, -90, -60]  # Antarctic coverage
    else:
        raise ValueError(f"Cannot determine projection for {grid_name}")

    # Output path for reprojected grid
    output_path = grid_path.replace('.nc', '_geographic.nc')

    # Use grdproject to convert to geographic
    if VERBOSE:
        print(f"    Projection: {projection}")
        print(f"    Output: {output_path}")

    pygmt.grdproject(
        grid=grid_path,
        projection=f"{projection}",
        inverse=True,  # Project FROM projected TO geographic
        region=region_geo,
        spacing=SPACING,
        outgrid=output_path
    )

    if VERBOSE:
        print(f"  ✓ Reprojected to {output_path}")

    return output_path

def main():
    """Main blending workflow."""

    print("="*70)
    print("Global Grid Blending")
    print("="*70)

    # ========================================================================
    # Step 1: Check all input grids
    # ========================================================================
    print("\nStep 1: Checking input grids...")

    grids_info = {}
    for name, path in [
        ('Mercator', MERCATOR_GRID),
        ('Arctic', ARCTIC_GRID),
        ('Antarctic', ANTARCTIC_GRID)
    ]:
        print(f"\n{name} grid:")
        print(f"  Path: {path}")

        info = check_grid_info(path)
        grids_info[name] = info

        if VERBOSE:
            print(f"  Coordinate type: {info['coord_type']}")
            print(f"  Dimensions: {info['dims']}")
            print(f"  Data variable: {info['var_name']}")
            if info['x_range'] and info['y_range']:
                if info['coord_type'] == 'geographic':
                    print(f"  Longitude range: {info['x_range'][0]:.2f} to {info['x_range'][1]:.2f}°")
                    print(f"  Latitude range: {info['y_range'][0]:.2f} to {info['y_range'][1]:.2f}°")
                else:
                    print(f"  X range: {info['x_range'][0]/1e6:.2f} to {info['x_range'][1]/1e6:.2f} × 10⁶ m")
                    print(f"  Y range: {info['y_range'][0]/1e6:.2f} to {info['y_range'][1]/1e6:.2f} × 10⁶ m")

    # ========================================================================
    # Step 2: Reproject polar grids if needed
    # ========================================================================
    print("\n" + "="*70)
    print("Step 2: Reprojecting polar grids to geographic coordinates...")
    print("="*70)

    grid_paths = {}

    # Mercator should already be geographic (or we'll use as-is)
    print("\nMercator grid:")
    grid_paths['Mercator'] = reproject_if_needed(MERCATOR_GRID, grids_info['Mercator'])

    # Arctic needs reprojection if projected
    print("\nArctic grid:")
    grid_paths['Arctic'] = reproject_if_needed(ARCTIC_GRID, grids_info['Arctic'])

    # Antarctic needs reprojection if projected
    print("\nAntarctic grid:")
    grid_paths['Antarctic'] = reproject_if_needed(ANTARCTIC_GRID, grids_info['Antarctic'])

    # ========================================================================
    # Step 3: Blend grids
    # ========================================================================
    print("\n" + "="*70)
    print("Step 3: Blending grids...")
    print("="*70)

    print(f"\nTarget region: {REGION} (global)")
    print(f"Target spacing: {SPACING}")
    print(f"\nInput grids:")
    for name, path in grid_paths.items():
        print(f"  - {path}")

    print(f"\nBlending with GMT grdblend...")

    try:
        # Build GMT grdblend command
        # gmt grdblend grid1.nc grid2.nc grid3.nc -Rregion -Ispacing -Goutput.nc
        cmd = [
            'gmt', 'grdblend',
            grid_paths['Antarctic'],  # Southern grid
            grid_paths['Mercator'],   # Middle grid
            grid_paths['Arctic'],     # Northern grid
            f'-R{REGION}',            # Region
            f'-I{SPACING}',           # Spacing/increment
            f'-G{OUTPUT_GRID}',       # Output grid
            '-V'                      # Verbose output
        ]

        if VERBOSE:
            print(f"  Command: {' '.join(cmd)}")

        # Run GMT grdblend
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        if VERBOSE and result.stdout:
            print(f"  GMT output:\n{result.stdout}")

        if result.stderr:
            # GMT often writes info to stderr even on success
            print(f"  GMT messages:\n{result.stderr}")

        print(f"  ✓ Blending complete!")

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error during blending:")
        print(f"    Return code: {e.returncode}")
        if e.stdout:
            print(f"    stdout: {e.stdout}")
        if e.stderr:
            print(f"    stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"  ✗ Error: GMT not found in PATH")
        print(f"    Please ensure GMT is installed and accessible")
        print(f"    Try: conda install gmt  or  brew install gmt")
        raise

    # ========================================================================
    # Step 4: Verify output
    # ========================================================================
    print("\n" + "="*70)
    print("Step 4: Verifying output...")
    print("="*70)

    output_info = check_grid_info(OUTPUT_GRID)

    print(f"\nOutput grid: {OUTPUT_GRID}")
    print(f"  Coordinate type: {output_info['coord_type']}")
    print(f"  Data variable: {output_info['var_name']}")
    print(f"  Longitude range: {output_info['x_range'][0]:.2f} to {output_info['x_range'][1]:.2f}°")
    print(f"  Latitude range: {output_info['y_range'][0]:.2f} to {output_info['y_range'][1]:.2f}°")

    # Check file size
    file_size_mb = os.path.getsize(OUTPUT_GRID) / 1024 / 1024
    print(f"  File size: {file_size_mb:.1f} MB")

    # Load and check for data
    output_ds = xr.open_dataset(OUTPUT_GRID)
    var_name = output_info['var_name']
    data = output_ds[var_name].values

    n_valid = np.sum(~np.isnan(data))
    n_total = data.size
    coverage = 100.0 * n_valid / n_total

    print(f"  Valid pixels: {n_valid:,} / {n_total:,} ({coverage:.1f}%)")
    print(f"  Value range: {np.nanmin(data):.1f} to {np.nanmax(data):.1f} m")

    output_ds.close()

    # ========================================================================
    # Done!
    # ========================================================================
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nBlended global bathymetry saved to:")
    print(f"  {OUTPUT_GRID}")
    print()


if __name__ == '__main__':
    main()
