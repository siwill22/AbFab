#!/usr/bin/env python
"""
Complete Bathymetry Generation - Command Line Interface

This script provides a simple command-line interface to generate synthetic
ocean floor bathymetry. All parameters are configured via YAML files.

Usage:
    python generate_complete_bathymetry.py [config.yaml]

    If no config file is specified, uses config_default.yaml

Examples:
    # Use default configuration
    python generate_complete_bathymetry.py

    # Use custom configuration
    python generate_complete_bathymetry.py my_config.yaml

    # Quick regional test
    python generate_complete_bathymetry.py config_regional_test.yaml
"""

import sys
import os
import yaml
import AbFab as af


def load_config(config_file='config_default.yaml'):
    """
    Load configuration from YAML file with defaults.

    Parameters:
    -----------
    config_file : str
        Path to YAML configuration file

    Returns:
    --------
    dict
        Configuration dictionary
    """
    # Default configuration
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
            'netcdf': 'complete_bathymetry.nc',
            'figure': 'complete_bathymetry.png',
            'dpi': 300,
            'verbose': True
        },
        'advanced': {
            'random_seed': 42,
            'timeout': 600
        }
    }

    # Load user config if file exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)

        # Merge with defaults (recursive dict update)
        def merge_configs(default, user):
            """Recursively merge user config into default config."""
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
        else:
            print("Using default configuration")
        config = default_config

    return config


def main():
    """Main entry point for CLI."""
    print("="*70)
    print("AbFab - Complete Bathymetry Generation")
    print("="*70)

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

    # Display key settings
    print("\nKey Settings:")
    print(f"  Region: {config['region']['lon_min']}° to {config['region']['lon_max']}°E, "
          f"{config['region']['lat_min']}° to {config['region']['lat_max']}°N")
    print(f"  Spacing: {config['region']['spacing']}")
    print(f"  Sediment mode: {config['sediment']['mode']}")
    print(f"  Subsidence: {config['subsidence']['model']}")
    print(f"  CPUs: {config['parallel']['num_cpus']}")
    print(f"  Output: {config['output']['netcdf']}")

    # Run the workflow
    try:
        complete_grid = af.run_complete_bathymetry_workflow(config)
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nGenerated files:")
        print(f"  • {config['output']['netcdf']}")
        print(f"\nGrid statistics:")
        print(f"  Shape: {complete_grid.shape}")
        print(f"  Depth range: {float(complete_grid.min()):.0f} to {float(complete_grid.max()):.0f} m")
        print(f"  Valid pixels: {(~complete_grid.isnull()).sum().values}/{complete_grid.size}")

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
