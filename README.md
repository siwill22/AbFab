# AbFab
Generate Synthetic Abyssal Hill Fabric

Code to generate maps of synthetic abyssal hill fabric for ocean basins following the method described by Goff and Arbic (2010), *Ocean Modelling*, doi:[10.1016/j.ocemod.2009.10.001](https://doi.org/10.1016/j.ocemod.2009.10.001).

## Quick Start

```python
import numpy as np
import AbFab as af

# Calculate grid spacing (km/pixel)
lon_spacing_deg = float(np.abs(age_da.lon.values[1] - age_da.lon.values[0]))
mean_lat = float(np.mean(age_da.lat.values))
grid_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(mean_lat))

# Set parameters (physical units: km)
params = {
    'H': 250.0,       # RMS height (m)
    'lambda_n': 6.5,  # Perpendicular wavelength (km) - smaller = wider hills
    'lambda_s': 16.0, # Parallel wavelength (km) - larger = narrower ridges
    'D': 2.2          # Fractal dimension
}

# Generate random field
np.random.seed(42)
random_field = af.generate_random_field(age_da.shape)

# Generate bathymetry
bathymetry = af.generate_bathymetry_spatial_filter(
    age_da.data,
    sediment_da.data,
    params,
    grid_spacing_km,
    random_field,
    filter_type='gaussian',  # or 'von_karman'
    optimize=True            # 50x speedup with filter bank
)
```

## Key Features

### Physical Units (km)
- **Resolution independent**: Same parameters work at any grid resolution
- **Physically meaningful**: "6.5 km wavelength" is interpretable
- **Literature compatible**: Direct comparison to published observations

### Spatially Varying Spreading Rate
```python
# Enable spatially varying parameters
bathymetry = af.generate_bathymetry_spatial_filter(
    age_da.data,
    sediment_da.data,
    params,  # Base parameters to scale
    grid_spacing_km,
    random_field,
    spreading_rate_bins=5,  # Enable spatial variation
    base_params=params      # Parameters to apply spreading rate scaling to
)
```

Automatically:
- Calculates spreading rate from age gradient
- Bins rates into 5 levels (default)
- Scales parameters: fast spreading → smaller wavelengths, smaller H
- Uses **trilinear bin interpolation** for smooth transitions

### Dual Filter Options
- **Gaussian** (default): Fast, clean results - recommended for most use
- **von Kármán**: Theoretically correct Bessel function - for research rigor

### Performance Optimization
- **Filter bank approach**: 50× speedup over pixel-by-pixel
- **<4% error** with default settings (36 azimuth × 5 sediment bins)
- **Parallel chunk processing**: See [tiling_test_updated.ipynb](tiling_test_updated.ipynb)

## Parameter Guidelines

### Typical Values
Based on Goff & Arbic (2010):

| Spreading Rate | H (m) | λ_n (km) | λ_s (km) |
|---------------|-------|----------|----------|
| Slow (10 mm/yr) | 85 | 18.8 | 7.5 |
| Medium (40 mm/yr) | 190 | 15.2 | 6.0 |
| Fast (100 mm/yr) | 400 | 8.0 | 3.0 |

**Note**: Faster spreading produces *smaller* wavelengths (smoother, smaller hills)

### Fixed Parameters (Recommended Starting Point)
```python
params = {
    'H': 250.0,       # RMS height: 50-400m
    'lambda_n': 6.5,  # Perpendicular: 5-20 km
    'lambda_s': 16.0, # Parallel: 3-10 km
    'D': 2.2          # Fractal: 2.0-2.3
}
```

### Auto-calculate from Spreading Rate
```python
# Option 1: Single median spreading rate
spreading_rate = af.calculate_spreading_rate_from_age(age_da.data, grid_spacing_km)
median_rate = np.nanmedian(spreading_rate)
params = af.spreading_rate_to_params(median_rate, base_params=params)

# Option 2: Spatially varying (recommended for heterogeneous regions)
bathymetry = af.generate_bathymetry_spatial_filter(
    ..., spreading_rate_bins=5, base_params=params
)
```

## Installation

```bash
# Core dependencies
pip install numpy scipy xarray

# For notebooks
pip install jupyter matplotlib pygmt joblib
```

## Examples

- **[tiling_test_updated.ipynb](tiling_test_updated.ipynb)** - Complete demonstration with 4 methods:
  1. Fixed parameters
  2. Median spreading rate parameters
  3. von Kármán filter
  4. Spatially varying spreading rate (NEW!)

## Key Functions

### `generate_bathymetry_spatial_filter()`
Main function for generating synthetic bathymetry.

```python
bathymetry = af.generate_bathymetry_spatial_filter(
    seafloor_age,        # 2D array (Myr)
    sediment_thickness,  # 2D array (m)
    params,              # Dict: H, lambda_n, lambda_s, D
    grid_spacing_km,     # Float: km/pixel
    random_field,        # 2D array: same shape as inputs
    filter_type='gaussian',      # 'gaussian' or 'von_karman'
    optimize=True,               # Use filter bank (50x faster)
    azimuth_bins=36,             # Azimuth discretization
    sediment_bins=5,             # Sediment discretization
    spreading_rate_bins=1,       # Set >1 for spatial variation
    base_params=None             # Base params for SR scaling
)
```

### `spreading_rate_to_params()`
Convert spreading rate to parameters using empirical relationships.

```python
params = af.spreading_rate_to_params(
    spreading_rate,  # mm/yr (half-spreading)
    base_params=None # Optional base params to scale
)
# Returns: {'H': float, 'lambda_n': float, 'lambda_s': float, 'D': float}
```

### `calculate_spreading_rate_from_age()`
Calculate spreading rate from age gradient.

```python
spreading_rate = af.calculate_spreading_rate_from_age(
    seafloor_age,     # 2D array (Myr)
    grid_spacing_km   # Float: km/pixel
)
# Returns: 2D array of half-spreading rates (mm/yr)
```

## Tests

Test scripts are in the `tests/` directory:
- `test_optimization.py` - Validates filter bank optimization
- `test_sediment_effect.py` - Validates sediment modification
- `test_spreading_rate_scaling.py` - Validates spreading rate parameter scaling
- `test_spreading_rate_variation.py` - Validates spatially varying spreading rate
- `test_bin_interpolation.py` - Validates smooth bin transitions

Run tests:
```bash
cd tests
python test_optimization.py
```

## References

Goff, J. A., & Arbic, B. K. (2010). Global prediction of abyssal hill root‐mean‐square heights from small‐scale altimetric gravity variability. *Journal of Geophysical Research: Solid Earth*, 115(B12). https://doi.org/10.1016/j.ocemod.2009.10.001

## Technical Details

For detailed implementation notes, development history, and guidance for future Claude Code sessions, see [CLAUDE.md](CLAUDE.md).
