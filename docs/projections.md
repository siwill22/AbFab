# Projection System

AbFab supports generating bathymetry in **projected coordinates** for regions where geographic (lat/lon) coordinates are suboptimal. This is particularly useful for:

- **High-latitude regions** (polar areas where lat/lon grids are distorted)
- **Equatorial regions** (Mercator projection for uniform sampling)
- **Multi-grid blending** (combining global + polar grids)

## Supported Projections

### 1. Mercator (EPSG:3395)

**Best for:** Equatorial to mid-latitude regions (typically ±70°)

**Characteristics:**
- Uniform grid spacing in projected coordinates
- Conformal (preserves angles and shapes locally)
- Increasing distortion toward poles
- Standard parallel at equator (true scale at 0°)

**Configuration:**
```yaml
projection:
  enabled: true
  type: 'mercator'
  lat_limits: [-70, 70]  # Latitude range for data inclusion
  projected_spacing: '5k'  # Grid spacing in projected space (meters)
  output_projected: true   # Save projected grid
  output_geographic: false  # Optional: inverse project to lat/lon
```

**Use cases:**
- Global ocean bathymetry (avoiding poles)
- Equatorial/mid-latitude regional studies
- Blending with polar stereographic grids

### 2. Polar Stereographic (EPSG:3413 North / EPSG:3031 South)

**Best for:** High-latitude regions (typically >70° or <-70°)

**Characteristics:**
- Minimal distortion at poles
- Conformal projection
- True scale at standard parallel (default = latitude limit)
- Uniform grid spacing in projected coordinates

**Configuration:**
```yaml
projection:
  enabled: true
  type: 'polar_stereo'
  pole: 'north'  # or 'south'
  lat_limit: 71  # Data boundary (latitude cutoff)
  lat_standard: 70  # Optional: standard parallel (true scale)
  projected_spacing: '5k'  # Grid spacing in projected space (meters)
  output_projected: true
  output_geographic: false
```

**Parameters:**
- `pole`: `'north'` (Arctic, EPSG:3413) or `'south'` (Antarctic, EPSG:3031)
- `lat_limit`: Latitude boundary for data inclusion (e.g., 71° means data from 71°-90°N)
- `lat_standard`: Standard parallel where scale is true (defaults to `lat_limit` if not specified)

**Use cases:**
- Arctic Ocean bathymetry
- Antarctic bathymetry
- Polar-focused regional studies
- Blending with Mercator grids

## Key Differences from Geographic Mode

### Grid Spacing

**Geographic mode:**
- Spacing in degrees (e.g., `'5m'` = 5 arcminutes)
- Variable km/pixel (depends on latitude)
- Requires spherical Earth corrections

**Projected mode:**
- Spacing in meters (e.g., `'5k'` = 5 kilometers)
- Uniform km/pixel across entire grid
- NO spherical corrections needed (Cartesian math)

### Coordinate Systems

**Geographic:**
- Coordinates: longitude (°E), latitude (°N)
- Dimension names: `lon`, `lat`
- Example: `(45.5°E, -20.3°N)`

**Projected:**
- Coordinates: easting (m), northing (m)
- Dimension names: `x`, `y`
- Example: `(5067000 m, -2254000 m)`

## Configuration Examples

### Example 1: Global Mercator Grid

Generate bathymetry from 70°S to 70°N in Mercator projection:

```yaml
# config_global_mercator.yaml
projection:
  enabled: true
  type: 'mercator'
  lat_limits: [-70, 70]
  projected_spacing: '5k'  # 5 km uniform spacing
  output_projected: true
  output_geographic: true  # Also save lat/lon version

region:
  lon_min: -180
  lon_max: 180
  # lat_min/lat_max not used in projected mode (set by lat_limits)

input:
  age_file: '/path/to/age_grid.nc'
  sediment_file: '/path/to/sediment.nc'

output:
  netcdf: 'output/mercator_bathymetry.nc'
```

**Output files:**
- `mercator_bathymetry.nc` - Projected coordinates (x, y in meters)
- `mercator_bathymetry_geographic.nc` - Inverse projected to lat/lon (if `output_geographic: true`)

### Example 2: Arctic Polar Stereographic

Generate bathymetry for Arctic Ocean (>71°N):

```yaml
# config_arctic.yaml
projection:
  enabled: true
  type: 'polar_stereo'
  pole: 'north'
  lat_limit: 71  # Only process data north of 71°N
  lat_standard: 70  # True scale at 70°N
  projected_spacing: '5k'
  output_projected: true

region:
  lon_min: -180
  lon_max: 180
  # Latitude determined by lat_limit and pole

input:
  age_file: '/path/to/age_grid.nc'
  sediment_file: '/path/to/sediment.nc'

output:
  netcdf: 'output/arctic_bathymetry.nc'
```

### Example 3: Antarctic Polar Stereographic

Generate bathymetry for Southern Ocean (<-60°S):

```yaml
# config_antarctic.yaml
projection:
  enabled: true
  type: 'polar_stereo'
  pole: 'south'
  lat_limit: -60  # Only process data south of -60°S
  lat_standard: -71  # True scale at 71°S (standard for Antarctic)
  projected_spacing: '5k'
  output_projected: true

region:
  lon_min: -180
  lon_max: 180

input:
  age_file: '/path/to/age_grid.nc'
  sediment_file: '/path/to/sediment.nc'

output:
  netcdf: 'output/antarctic_bathymetry.nc'
```

## Inverse Projection

You can output **both** projected and geographic grids:

```yaml
projection:
  output_projected: true   # Save x/y grid
  output_geographic: true  # Also save lat/lon grid via inverse projection
```

**Output:**
- `output_name.nc` - Projected coordinates
- `output_name_geographic.nc` - Inverse projected to geographic

**Note:** Inverse projection uses PyGMT's `grdproject` with bicubic interpolation. Small numerical differences (<1m) may occur at grid boundaries.

## When to Use Projections

### Use Geographic Mode When:
- Working at mid-latitudes (20°-60°)
- Small regional domains
- No blending with other grids needed
- Standard lat/lon output preferred

### Use Mercator Projection When:
- Global or near-global domains
- Avoiding polar regions (>70° or <-70°)
- Blending with polar grids
- Uniform sampling in projected space desired

### Use Polar Stereographic When:
- High-latitude regions (>70° or <-70°)
- Arctic or Antarctic focused studies
- Blending with Mercator grids
- Minimal distortion at poles critical

## Technical Details

### Projection Workflow

1. **Input data** loaded in geographic coordinates (lat/lon)
2. **Reproject** age and sediment grids to projected coordinates using PyGMT
3. **Calculate** azimuth and spreading rate in projected space (Cartesian math)
4. **Generate** bathymetry on projected grid
5. **Optional:** Inverse project to geographic coordinates

### No Spherical Corrections

In projected mode, **all calculations use Cartesian math**:
- No `cos(lat)` corrections on longitude gradients
- Uniform grid spacing throughout
- Simpler, faster gradient calculations
- Identical `grid_spacing_km` everywhere

### Coordinate Reference Systems (CRS)

**Mercator (EPSG:3395):**
- Also known as "World Mercator"
- Central meridian: 0°E
- Standard parallel: 0°N (equator)
- Units: meters
- Authority: EPSG

**North Polar Stereographic (EPSG:3413):**
- Projection center: North Pole (90°N)
- Central meridian: varies by convention
- Standard parallel: 70°N (typical)
- Units: meters
- Used by: NSIDC, sea ice datasets

**South Polar Stereographic (EPSG:3031):**
- Projection center: South Pole (90°S)
- Central meridian: 0°E
- Standard parallel: 71°S (typical)
- Units: meters
- Used by: Antarctic datasets, BEDMAP2

### Grid Spacing Specification

**Projected mode uses meters:**
- `'5k'` = 5 km = 5000 m
- `'10k'` = 10 km = 10000 m
- `'500'` = 500 m
- Can also use float: `5000.0` (in meters)

**Geographic mode uses degrees:**
- `'5m'` = 5 arcminutes = 0.0833°
- `'1m'` = 1 arcminute = 0.0167°
- Can also use float: `0.0833` (in degrees)

## Performance Considerations

**Projected mode is typically faster because:**
1. No spherical Earth corrections needed
2. Uniform grid spacing (simpler interpolation)
3. Cartesian distance calculations (no haversine)

**BUT:**
- Adds overhead for projection/inverse projection steps
- Overall similar performance for large grids
- Negligible difference (<5%) for most use cases

## Troubleshooting

### Issue: "x_inc does not divide 180" warning

**Cause:** PyGMT warning when grid spacing doesn't evenly divide global extent

**Solution:** This is harmless. PyGMT adjusts boundary conditions automatically.

### Issue: Inverse projection produces NaN values

**Cause:** Grid extends outside valid projection range

**Solution:**
- For Mercator: Ensure `lat_limits` are within ±85°
- For polar: Ensure data respects hemisphere boundaries

### Issue: Discontinuity at projection boundaries

**Cause:** Different bin ranges in Mercator vs polar grids (see [blending.md](blending.md))

**Solution:** Use bin consistency system for multi-grid workflows

## See Also

- [Multi-Grid Blending](blending.md) - Combining Mercator + polar grids
- [Configuration Files](../config_files/config_default.yaml) - Full configuration reference
- [PyGMT Documentation](https://www.pygmt.org/) - Projection details
