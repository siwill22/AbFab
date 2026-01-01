# AbFab
Generate Synthetic Abyssal Hill Fabric

Code to generate maps of synthetic abyssal hill fabric for ocean basins following the method described by Goff and Arbic (2010), *Ocean Modelling*, doi:[10.1016/j.ocemod.2009.10.001](https://doi.org/10.1016/j.ocemod.2009.10.001).

## Overview

AbFab generates realistic synthetic bathymetry for abyssal hill fabric using a spatial filtering approach. The inputs are maps of seafloor age and sediment thickness.

## Features

- **Dual filter options**: Gaussian (fast, default) or von Kármán (theoretically correct)
- **Spatial variability**: Handles spatially-varying parameters (age, sediment, azimuth)
- **Sediment modification**: Automatically adjusts roughness based on sediment thickness
- **Verified orientation**: Produces linear ridges perpendicular to spreading direction
- **Parallel processing**: Chunk-based approach for large grids

## Installation

```bash
# Dependencies
pip install numpy scipy xarray
```

## Quick Start

```python
import numpy as np
import AbFab as af

# Generate random field
random_field = af.generate_random_field(seafloor_age.shape)

# Set parameters
params = {
    'H': 50,       # RMS height (m)
    'kn': 0.05,    # Perpendicular wavenumber (deg⁻¹)
    'ks': 0.2,     # Parallel wavenumber (deg⁻¹)
    'D': 2.2       # Fractal dimension
}

# Generate bathymetry
bathymetry = af.generate_bathymetry_spatial_filter(
    seafloor_age,
    sediment_thickness,
    params,
    random_field
)
```

## Usage Examples

### Example 1: Basic Usage (Default Gaussian Filter)

```python
import AbFab as af

params = {'H': 50, 'kn': 0.05, 'ks': 0.2, 'D': 2.2}
random_field = af.generate_random_field(age_grid.shape)

bathymetry = af.generate_bathymetry_spatial_filter(
    age_grid,
    sediment_grid,
    params,
    random_field
)
```

### Example 2: Using von Kármán Filter

```python
bathymetry = af.generate_bathymetry_spatial_filter(
    age_grid,
    sediment_grid,
    params,
    random_field,
    filter_type='von_karman'  # Theoretically correct filter
)
```

### Example 3: Large Grid Processing (Parallel Chunks)

See [tiling_test_updated.ipynb](tiling_test_updated.ipynb) for a complete example of parallel chunk-based processing for large grids.

## Key Functions

### `generate_bathymetry_spatial_filter(seafloor_age, sediment_thickness, params, random_field, filter_type='gaussian')`

Main function for generating synthetic bathymetry.

**Parameters:**
- `seafloor_age`: 2D array of seafloor ages (Myr)
- `sediment_thickness`: 2D array of sediment thickness (m)
- `params`: Dict with keys `H`, `kn`, `ks`, `D`
- `random_field`: 2D array of random noise (same shape as inputs)
- `filter_type`: 'gaussian' (default) or 'von_karman'

**Returns:**
- 2D array of synthetic bathymetry (m)

### `calculate_azimuth_from_age(seafloor_age)`

Calculate spreading direction from age gradient.

### `modify_by_sediment(H, kn, ks, sediment_thickness, D=None)`

Modify parameters based on sediment thickness (Goff & Arbic Equations 5-7).

### `generate_random_field(grid_size)`

Generate Gaussian random field for convolution.

## Filter Types

### Gaussian Filter (Default)
- Fast computation
- Smooth, clean results
- Backward compatible
- **Recommended for most applications**

### von Kármán Filter
- Theoretically correct autocorrelation using Bessel function K_ν
- Slightly rougher texture with heavier tails
- ~40% higher RMS output
- Use for research applications requiring theoretical rigor

## Parameter Guidelines

### Fixed Parameters (Recommended)
Based on empirical validation:
```python
params = {
    'H': 50,       # RMS height: 50-300m depending on spreading rate
    'kn': 0.05,    # Perpendicular: smaller = wider hills
    'ks': 0.2,     # Parallel: larger = narrower hills
    'D': 2.2       # Fractal dimension: typically 2.0-2.3
}
```

### Typical Ranges
- **H** (RMS height): 50-300 m (increases with spreading rate)
- **kn** (perpendicular): 0.01-0.1 deg⁻¹ for lat/lon grids
- **ks** (parallel): 0.1-0.5 deg⁻¹ for lat/lon grids
- **D** (fractal dimension): 2.0-2.3
- **λn** (perpendicular wavelength): 10-30 km (physical space)
- **λs** (parallel wavelength): 5-10 km (physical space)

## Notebooks

- **[tiling_test.ipynb](tiling_test.ipynb)** - Original working example
- **[tiling_test_updated.ipynb](tiling_test_updated.ipynb)** - Demonstrates new features (dual filters)
- **[test_azimuth.ipynb](test_azimuth.ipynb)** - Verification of ridge orientation

## Known Issues

### Spreading Rate Utilities (⚠️ Needs Calibration)

The `spreading_rate_to_params()` and `calculate_spreading_rate_from_age()` functions are implemented but **require calibration** before use. The current empirical relationships produce values ~1000× too large for lat/lon grids.

**Workaround**: Use fixed parameters validated for your application.

See [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) for details.

## Performance Tips

### Optimized Implementation (Default)

`generate_bathymetry_spatial_filter()` includes built-in optimization using a filter bank approach:
- **50× speedup** over naive pixel-by-pixel method
- **<4% error** with default settings (36 azimuth × 5 sediment bins)
- Automatically enabled with `optimize=True` (default)

**Tuning parameters**:
```python
bathymetry = af.generate_bathymetry_spatial_filter(
    seafloor_age, sediment_thickness, params, random_field,
    azimuth_bins=36,    # Higher = more accurate, slower
    sediment_bins=5,    # Higher = more accurate, slower
    optimize=True       # Set False for pixel-perfect original method
)
```

**Accuracy vs Speed**:
- 3 sediment bins: ~60× faster, 8% error
- 5 sediment bins: ~50× faster, 4% error (default)
- 10 sediment bins: ~25× faster, 2% error

### Large Grid Processing

For very large grids (>1000×1000):
1. Use chunk-based processing (see `tiling_test_updated.ipynb`)
2. Process chunks in parallel using `joblib.Parallel`
3. Use padding (≥20 pixels) to avoid edge artifacts
4. Gaussian filter is ~30% faster than von Kármán

## References

Goff, J. A., & Arbic, B. K. (2010). Global prediction of abyssal hill root‐mean‐square heights from small‐scale altimetric gravity variability. *Journal of Geophysical Research: Solid Earth*, 115(B12). https://doi.org/10.1016/j.ocemod.2009.10.001

## License

[Add your license here]

## Contributing

This code was developed with assistance from Claude Code. See [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) for implementation details and future development guidance.
