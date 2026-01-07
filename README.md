# AbFab
Generate Synthetic Abyssal Hill Fabric

Code to generate maps of synthetic abyssal hill fabric for ocean basins following the method described by Goff and Arbic (2010), *Ocean Modelling*, doi:[10.1016/j.ocemod.2009.10.001](https://doi.org/10.1016/j.ocemod.2009.10.001).

## Quick Start

### Command Line (Recommended)

The easiest way to generate complete bathymetry is using the command-line interface with YAML configuration:

```bash
# Use default configuration (global grid, 5m resolution)
python generate_complete_bathymetry.py

# Use custom configuration
python generate_complete_bathymetry.py config_test.yaml

# Quick regional test (completes in ~1-2 minutes)
python generate_complete_bathymetry.py config_test.yaml
```

See [Configuration Files](#configuration-files) section below for details.

### Python API (Advanced)

For direct Python API usage:

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

## Configuration Files

AbFab uses YAML configuration files for easy parameter management. This approach allows you to:
- Keep your parameters organized and version-controlled
- Easily switch between different configurations
- Override only the parameters you need (others use defaults)
- Share configurations with collaborators

### Basic Usage

```bash
# Use default configuration
python generate_complete_bathymetry.py

# Use custom configuration
python generate_complete_bathymetry.py my_config.yaml
```

### Configuration Structure

See [config_default.yaml](config_default.yaml) for the complete configuration template. Key sections:

```yaml
# Input data files
input:
  age_file: '/path/to/age_grid.nc'
  sediment_file: '/path/to/sediment.nc'  # or null to disable
  constant_sediment: null  # or a value in meters for uniform sediment

# Region selection
region:
  lon_min: -180
  lon_max: 180
  lat_min: -70
  lat_max: 70
  spacing: '5m'  # '1m', '2m', '5m', '10m', etc.

# Abyssal hill parameters
abyssal_hills:
  H: 250.0          # RMS height (m)
  lambda_n: 3.0     # Perpendicular wavelength (km)
  lambda_s: 30.0    # Parallel wavelength (km)
  D: 2.2            # Fractal dimension

# Sediment treatment
sediment:
  mode: 'fill'      # 'none', 'drape', or 'fill'
  diffusion: 0.3    # For 'fill' mode (0-1)

# Optimization and parallel processing
optimization:
  enabled: true
  azimuth_bins: 36
  sediment_bins: 10
  spreading_rate_bins: 20

parallel:
  num_cpus: 8
  chunk_size: 100
```

### Example Configurations

#### Quick Test (No Sediment)
```yaml
# config_no_sediment.yaml
input:
  sediment_file: null

region:
  lon_min: 0
  lon_max: 50
  lat_min: -40
  lat_max: -20
  spacing: '5m'

sediment:
  mode: 'none'

output:
  netcdf: 'no_sediment_bathymetry.nc'
```

#### Regional Study with Uniform Sediment
```yaml
# config_uniform_sediment.yaml
input:
  constant_sediment: 100.0  # 100m everywhere

region:
  lon_min: 0
  lon_max: 50
  lat_min: -40
  lat_max: -20

sediment:
  mode: 'drape'
```

### Making Sediment Optional

To run without sediment:

**Option 1**: Set `sediment_file: null` in config
**Option 2**: Use `sediment_mode: 'none'`
**Option 3**: Both (recommended for clarity)

```yaml
input:
  sediment_file: null  # No sediment data

sediment:
  mode: 'none'  # Skip sediment processing
```

This generates bathymetry with only thermal subsidence and abyssal hills.

## Installation

```bash
# Core dependencies
pip install numpy scipy xarray pyyaml

# For complete bathymetry workflow
pip install pygmt joblib tqdm

# For notebooks and visualization
pip install jupyter matplotlib
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

## Processing Time Considerations

Understanding how different options affect processing time helps you optimize workflows for your specific needs.

### Major Factors (Ordered by Impact)

#### 1. Grid Resolution & Region Size ⚡ **DOMINANT FACTOR**
Processing time scales **quadratically** with grid size (number of pixels):

| Resolution | Global Grid Size | Region Example | Typical Time* |
|-----------|------------------|----------------|--------------|
| `30m` (30 arcmin) | 360 × 720 | Testing only | ~30 seconds |
| `10m` | 1,080 × 2,160 | Ocean basin | ~3 minutes |
| `5m` | 2,160 × 4,320 | Ocean basin | ~12 minutes |
| `2m` | 5,400 × 10,800 | High resolution | ~1.5 hours |
| `1m` | 10,800 × 21,600 | Very high res | ~6 hours |

*Approximate times for global grids on 8-core system with optimization enabled

**Regional grids** process proportionally faster:
- Southern Africa (50° × 20°): 5× faster than global
- Pacific Basin (120° × 60°): 50% of global time

#### 2. Sediment Mode ⚡ **MODERATE IMPACT**

| Mode | Speed | Description |
|------|-------|-------------|
| `'none'` | **Fastest** | No sediment processing |
| `'drape'` | **Fast** | Simple subtraction (~1% overhead) |
| `'fill'` | **Slower** | Adds 20-40% time for global diffusive infill |

**Diffusive infill (`'fill'` mode)**:
- Adds ~30% to total time for global grids
- Time depends on sediment thickness (more iterations for thicker sediment)
- Applied ONCE globally after chunk assembly (not per-chunk)

#### 3. Optimization Settings ⚡ **50× SPEEDUP**

| Setting | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| `optimize=False` | **50× slower** | Exact | Testing/validation only |
| `optimize=True` | **Baseline** | <4% error | Production (recommended) |

**Bin counts** (when `optimize=True`):
```python
# Fast (good for testing)
AZIMUTH_BINS = 18
SEDIMENT_BINS = 5
SPREADING_RATE_BINS = 10
# → ~10% faster than defaults

# Default (recommended balance)
AZIMUTH_BINS = 36
SEDIMENT_BINS = 10
SPREADING_RATE_BINS = 20
# → Baseline speed, <4% error

# High accuracy (minimal speed impact)
AZIMUTH_BINS = 72
SEDIMENT_BINS = 20
SPREADING_RATE_BINS = 40
# → ~15% slower, <1% error
```

**Impact**: Bin counts have **minimal impact** on speed (±20% max). Use higher values for better accuracy with negligible time cost.

#### 4. Parallel Processing (NUM_CPUS)

| CPUs | Speedup | Efficiency |
|------|---------|------------|
| 1 | 1× | 100% |
| 2 | 1.9× | 95% |
| 4 | 3.6× | 90% |
| 8 | 6.5× | 81% |
| 16 | 10× | 63% |

**Recommendations**:
- Use 75-100% of available cores
- Diminishing returns beyond 8-12 cores due to overhead
- Optimal chunk size: 50-150 pixels

#### 5. Chunk Size (CHUNKSIZE)

| Chunk Size | Memory | Speed | Notes |
|-----------|--------|-------|-------|
| 50 | Low | Slower | More overhead, better parallelism |
| 100 | Medium | **Optimal** | Recommended default |
| 200 | High | Slower | Less parallelism, larger chunks |

**Rule of thumb**: `CHUNKSIZE = 100` works well for most cases. Adjust if:
- **Low memory**: Use 50-75
- **Very large grids**: Use 100-150
- **Few CPUs**: Use larger chunks (150-200)

#### 6. Filter Type (Minor Impact)

| Filter | Speed | Use Case |
|--------|-------|----------|
| `'gaussian'` | **Baseline** | Production (recommended) |
| `'von_karman'` | ~10% slower | Research/publications |

### Example Configurations

#### Fast Testing (1-2 minutes)
```python
XMIN, XMAX = 0, 50
YMIN, YMAX = -40, -20
SPACING = '5m'
SEDIMENT_MODE = 'drape'
NUM_CPUS = 4
USE_OPTIMIZATION = True
AZIMUTH_BINS = 18
SEDIMENT_BINS = 5
```

#### Production Regional (5-15 minutes)
```python
XMIN, XMAX = 0, 50
YMIN, YMAX = -40, -20
SPACING = '2m'
SEDIMENT_MODE = 'fill'
NUM_CPUS = 8
USE_OPTIMIZATION = True
AZIMUTH_BINS = 36
SEDIMENT_BINS = 10
```

#### High-Resolution Global (4-8 hours)
```python
XMIN, XMAX = -180, 180
YMIN, YMAX = -70, 70
SPACING = '1m'
SEDIMENT_MODE = 'fill'
NUM_CPUS = 12
USE_OPTIMIZATION = True
AZIMUTH_BINS = 72
SEDIMENT_BINS = 20
CHUNKSIZE = 150
```

### Progress Monitoring

The code includes progress indicators (via `tqdm`) for:
- **Chunk processing**: Shows chunks/second and ETA
- **Sediment resampling**: Shows rows processed (when using `'fill'` mode)
- **Diffusive infill**: Shows iteration progress

Example output:
```
Processing chunks: 100%|██████| 272/272 [05:23<00:00, 0.84chunk/s]
Resampling sediment: 100%|██| 1680/1680 [00:45<00:00, 37.3row/s]
Diffusion iteration 3/5 (sigma=2.45 pixels)
```

### Quick Time Estimates

**Formula**: `Time (minutes) ≈ (Grid_Height × Grid_Width / 10^6) × Complexity_Factor`

Where `Complexity_Factor`:
- `optimize=True, drape`: **1.0** (baseline)
- `optimize=True, fill`: **1.3**
- `optimize=False`: **50.0** (not recommended)

**Example**: 5m global grid (2160 × 4320) with `fill` mode:
- Pixels: 9,331,200
- Time ≈ (9.33 × 1.3) = **~12 minutes** (8 CPUs)

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
