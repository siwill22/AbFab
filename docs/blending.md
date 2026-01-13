# Multi-Grid Blending

When generating global bathymetry, you may want to combine **multiple grids** with different projections to minimize distortion:

- **Global Mercator** (±70° latitude) - equatorial and mid-latitude regions
- **Arctic Polar Stereographic** (>70°N) - Arctic Ocean
- **Antarctic Polar Stereographic** (<-70°S) - Southern Ocean

## The Problem: Bin Inconsistency

AbFab uses a **filter bank optimization** that discretizes continuous parameters (sediment thickness, spreading rate, azimuth) into bins:

- **36 azimuth bins** (every 10°)
- **10 sediment bins** (e.g., 0, 100, 200, ... 900 m)
- **20 spreading rate bins** (e.g., 0, 10, 20, ... 190 mm/yr)

When generating three separate grids independently:

| Grid | Sediment Range | Spreading Rate Range |
|------|----------------|---------------------|
| Global Mercator | 0-1000 m | 0-200 mm/yr |
| Arctic | 0-500 m | 5-50 mm/yr |
| Antarctic | 0-300 m | 10-80 mm/yr |

**Result:** Different bin boundaries → different filters → **visible discontinuities** at blend boundaries!

## The Solution: Bin Consistency System

Ensure **all grids use identical bin ranges** by either:

1. **Manual range specification** (simple, explicit)
2. **Save/load workflow** (automatic, recommended)

### Option 1: Manual Range Specification

Specify the same ranges in all three configuration files:

```yaml
# config_global_mercator.yaml
optimization:
  sediment_range: [0, 1000]        # [min, max] in meters
  spreading_rate_range: [0, 250]   # [min, max] in mm/yr

# config_arctic.yaml
optimization:
  sediment_range: [0, 1000]        # SAME ranges
  spreading_rate_range: [0, 250]   # SAME ranges

# config_antarctic.yaml
optimization:
  sediment_range: [0, 1000]        # SAME ranges
  spreading_rate_range: [0, 250]   # SAME ranges
```

**Pros:**
- Simple and explicit
- Full control over ranges

**Cons:**
- Must know appropriate ranges ahead of time
- Manual coordination across config files

### Option 2: Save/Load Workflow (Recommended)

Generate the **global grid first**, which auto-detects ranges from data and saves them. Then load those ranges for polar grids:

```yaml
# config_global_mercator.yaml
optimization:
  save_bin_config: 'output/bin_config.yaml'  # Save detected ranges

# config_arctic.yaml
optimization:
  load_bin_config: 'output/bin_config.yaml'  # Load saved ranges

# config_antarctic.yaml
optimization:
  load_bin_config: 'output/bin_config.yaml'  # Load saved ranges
```

**Pros:**
- Automatic (no need to know ranges)
- Guaranteed consistency
- Ranges optimal for global data

**Cons:**
- Must generate global grid first
- Creates dependency between runs

## Complete Workflow

### Step 1: Generate Global Mercator Grid

Create `config_global_mercator.yaml`:

```yaml
projection:
  enabled: true
  type: 'mercator'
  lat_limits: [-70, 70]
  projected_spacing: '5k'
  output_projected: true

region:
  lon_min: -180
  lon_max: 180

input:
  age_file: '/path/to/age_grid.nc'
  sediment_file: '/path/to/sediment.nc'

abyssal_hills:
  H: 250.0
  lambda_n: 3.0
  lambda_s: 30.0
  D: 2.2

optimization:
  enabled: true
  azimuth_bins: 36
  sediment_bins: 10
  spreading_rate_bins: 20
  save_bin_config: 'output/bin_config.yaml'  # SAVE bins

output:
  netcdf: 'output/mercator_bathymetry.nc'
```

**Run:**
```bash
python generate_complete_bathymetry_gpu.py config_global_mercator.yaml
```

**Output:**
- `output/mercator_bathymetry.nc` - Bathymetry grid
- `output/bin_config.yaml` - Bin configuration (NEW!)

### Step 2: Check Saved Bin Configuration

The saved `output/bin_config.yaml` contains:

```yaml
sediment:
  min: 0.0
  max: 982.5
  bins: 10
  levels: [0.0, 109.2, 218.3, 327.5, 436.7, 545.8, 655.0, 764.2, 873.3, 982.5]

spreading_rate:
  min: 0.1
  max: 197.3
  bins: 20
  levels: [0.1, 10.5, 20.9, 31.3, ..., 197.3]

azimuth:
  bins: 36
  levels: [-3.14159, -3.05433, ..., 3.05433]
```

These ranges were auto-detected from the global Mercator data.

### Step 3: Generate Arctic Grid (Load Bins)

Create `config_arctic.yaml`:

```yaml
projection:
  enabled: true
  type: 'polar_stereo'
  pole: 'north'
  lat_limit: 71  # Data from 71°N to 90°N
  lat_standard: 70
  projected_spacing: '5k'
  output_projected: true

region:
  lon_min: -180
  lon_max: 180

input:
  age_file: '/path/to/age_grid.nc'
  sediment_file: '/path/to/sediment.nc'

abyssal_hills:
  H: 250.0
  lambda_n: 3.0
  lambda_s: 30.0
  D: 2.2

optimization:
  enabled: true
  azimuth_bins: 36
  sediment_bins: 10
  spreading_rate_bins: 20
  load_bin_config: 'output/bin_config.yaml'  # LOAD bins from global

output:
  netcdf: 'output/arctic_bathymetry.nc'
```

**Run:**
```bash
python generate_complete_bathymetry_gpu.py config_arctic.yaml
```

### Step 4: Generate Antarctic Grid (Load Bins)

Create `config_antarctic.yaml`:

```yaml
projection:
  enabled: true
  type: 'polar_stereo'
  pole: 'south'
  lat_limit: -60  # Data from -60°S to -90°S
  lat_standard: -71
  projected_spacing: '5k'
  output_projected: true

region:
  lon_min: -180
  lon_max: 180

input:
  age_file: '/path/to/age_grid.nc'
  sediment_file: '/path/to/sediment.nc'

abyssal_hills:
  H: 250.0
  lambda_n: 3.0
  lambda_s: 30.0
  D: 2.2

optimization:
  enabled: true
  azimuth_bins: 36
  sediment_bins: 10
  spreading_rate_bins: 20
  load_bin_config: 'output/bin_config.yaml'  # LOAD bins from global

output:
  netcdf: 'output/antarctic_bathymetry.nc'
```

**Run:**
```bash
python generate_complete_bathymetry_gpu.py config_antarctic.yaml
```

### Step 5: Blend Grids

Now you have three grids with **identical bin ranges**, ready for seamless blending:

- `output/mercator_bathymetry.nc` (70°S to 70°N)
- `output/arctic_bathymetry.nc` (71°N to 90°N)
- `output/antarctic_bathymetry.nc` (60°S to 90°S)

**Using PyGMT to blend:**

```python
import pygmt

# Load grids
mercator = pygmt.load_dataarray('output/mercator_bathymetry.nc')
arctic = pygmt.load_dataarray('output/arctic_bathymetry.nc')
antarctic = pygmt.load_dataarray('output/antarctic_bathymetry.nc')

# Project polar grids back to geographic if needed
if mercator.dims == ('lat', 'lon') and arctic.dims == ('y', 'x'):
    arctic_geo = pygmt.grdproject(
        arctic,
        projection='EPSG:3413',  # Input CRS
        inverse=True,  # Project to geographic
        region=[-180, 180, 71, 90]
    )
    antarctic_geo = pygmt.grdproject(
        antarctic,
        projection='EPSG:3031',
        inverse=True,
        region=[-180, 180, -90, -60]
    )
else:
    arctic_geo = arctic
    antarctic_geo = antarctic

# Blend with feathering at boundaries (70°-71°)
# Use grdblend or manual feathering
blended = pygmt.grdblend(
    [antarctic_geo, mercator, arctic_geo],
    region=[-180, 180, -90, 90],
    spacing='5m'
)

# Save
blended.to_netcdf('output/global_blended_bathymetry.nc')
```

## Why Bin Consistency Matters

### Without Bin Consistency

```
Arctic bins:    [0, 50, 100, 150, ..., 500 m]
Mercator bins:  [0, 100, 200, 300, ..., 1000 m]
                 ↑ Different boundaries!
```

**Result:** At 70°N boundary, same sediment value (e.g., 200m) maps to:
- Arctic: bin 4
- Mercator: bin 2

Different bins → different filters → **discontinuity**

### With Bin Consistency

```
Arctic bins:    [0, 100, 200, ..., 1000 m]
Mercator bins:  [0, 100, 200, ..., 1000 m]
                 ✓ Identical boundaries!
```

**Result:** Same sediment value maps to same bin in both grids → **seamless transition**

## Best Practices

### 1. Always Use Bin Consistency for Multi-Grid Workflows

If generating >1 grid that will be blended, **always** use bin consistency system.

### 2. Generate Largest Grid First

The largest grid (usually global Mercator) should be generated first with `save_bin_config`, since:
- It captures the widest range of values
- Ensures polar grids don't exceed saved ranges
- More representative of global conditions

### 3. Use Same Parameters Across All Grids

Not just bin ranges - also use identical:
- `abyssal_hills` parameters (H, λ_n, λ_s, D)
- `optimization` bin counts
- `sediment.mode` and `sediment.diffusion`
- `random_seed` (if you want reproducible patterns)

### 4. Check Bin Config After Global Run

After generating global grid, inspect `bin_config.yaml`:
- Are ranges reasonable?
- Do they cover expected values in polar regions?
- If not, adjust with manual ranges

### 5. Keep All Config Files Together

Organize configurations:
```
config_files/
  ├── blending_workflow/
  │   ├── 1_global_mercator.yaml
  │   ├── 2_arctic.yaml
  │   └── 3_antarctic.yaml
  └── output/
      └── bin_config.yaml  # Shared by all
```

## Parameter Priorities

The bin range determination follows this priority:

1. **Loaded config** (highest priority)
   - If `load_bin_config` specified and file exists
   - Uses ranges from saved YAML file

2. **Manual ranges**
   - If `sediment_range` or `spreading_rate_range` specified
   - Uses explicitly provided values

3. **Auto-detect** (fallback)
   - Computes min/max from input data
   - Used when neither of above specified

## Troubleshooting

### Issue: "Loaded bin config but still see discontinuities"

**Check:**
1. Are all three grids using `load_bin_config`?
2. Did global run complete successfully and create `bin_config.yaml`?
3. Are the `abyssal_hills` parameters identical across configs?

**Verify:**
```bash
# Check that bin config exists
cat output/bin_config.yaml

# Check that ranges are used
grep "load_bin_config" config_*.yaml
```

### Issue: "Sediment/spreading rate in polar grid exceeds saved range"

**Cause:** Polar regions have values outside global range

**Solutions:**
- **Option A:** Use manual ranges that cover all regions:
  ```yaml
  sediment_range: [0, 1500]  # Expanded to cover polar values
  ```

- **Option B:** Generate polar grid first (if it has wider range), then load for global

### Issue: "Want different bin counts for different grids"

**Not recommended!** Different bin counts = different interpolation = potential discontinuities.

If you **must** use different counts:
- Keep `save_bin_config` / `load_bin_config` for ranges
- Accept some minor interpolation differences
- Test carefully at boundaries

## Advanced: Three-Step Blending

For perfect transitions, use a **three-region approach** with overlap:

1. **Global Mercator:** 70°S to 70°N (baseline)
2. **Arctic transition:** 65°N to 75°N (overlap with Mercator)
3. **Antarctic transition:** 75°S to 65°S (overlap with Mercator)

Then blend with feathering in the 5° overlap zones. This provides:
- Smooth gradients across projection boundaries
- Flexibility to adjust blend weights
- More forgiving of small bin differences

## See Also

- [Projection System](projections.md) - Details on Mercator and polar stereographic projections
- [Configuration Files](../config_files/config_default.yaml) - Complete configuration reference
- [PyGMT grdblend](https://www.pygmt.org/latest/api/generated/pygmt.grdblend.html) - Grid blending documentation
