#!/usr/bin/env python
"""
Quick test to verify the config-based workflow works and generates PNG files.
"""

import yaml

# Create a minimal test config
test_config = {
    'input': {
        'age_file': '/Users/simon/Data/AgeGrids/2020/age.2020.1.GeeK2007.6m.nc',
        'sediment_file': None,
        'constant_sediment': None
    },
    'region': {
        'lon_min': 0,
        'lon_max': 20,
        'lat_min': -30,
        'lat_max': -20,
        'spacing': '10m'  # Very coarse for fast test
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
        'mode': 'none',
        'diffusion': 0.3
    },
    'optimization': {
        'enabled': True,
        'azimuth_bins': 18,
        'sediment_bins': 5,
        'spreading_rate_bins': 10
    },
    'parallel': {
        'num_cpus': 2,
        'chunk_size': 50,
        'chunk_pad': 10
    },
    'output': {
        'netcdf': 'test_workflow.nc',
        'figure': 'test_workflow.png',  # THIS SHOULD BE GENERATED
        'dpi': 150,
        'verbose': True
    },
    'advanced': {
        'random_seed': 42,
        'timeout': 300
    }
}

print("="*70)
print("Testing PNG Generation with Config Workflow")
print("="*70)
print("\nExpected output files:")
print("  • test_workflow.nc")
print("  • test_workflow.png  <-- This should be created!")
print("\n" + "="*70 + "\n")

import AbFab as af

try:
    result = af.run_complete_bathymetry_workflow(test_config)
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)

    # Check if files were created
    import os
    nc_exists = os.path.exists('test_workflow.nc')
    png_exists = os.path.exists('test_workflow.png')

    print(f"\nFile status:")
    print(f"  NetCDF: {'✓ EXISTS' if nc_exists else '✗ MISSING'}")
    print(f"  PNG:    {'✓ EXISTS' if png_exists else '✗ MISSING'}")

    if png_exists:
        png_size = os.path.getsize('test_workflow.png') / 1024  # KB
        print(f"  PNG size: {png_size:.1f} KB")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
