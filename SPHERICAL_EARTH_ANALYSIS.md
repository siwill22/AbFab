# Spherical Earth Distortion - Analysis and Improvement Plan

## Current Implementation Issues

### Problem 1: Gradients Assume Cartesian Coordinates

**Current Code** ([AbFab.py:162-165](AbFab.py#L162-L165)):
```python
def calculate_azimuth_from_age(seafloor_age):
    grad_y, grad_x = np.gradient(seafloor_age)
    azimuth = np.arctan2(grad_y, grad_x)
    return azimuth
```

**Issues:**
1. `np.gradient()` treats lon/lat as Cartesian (uniform spacing in meters)
2. At high latitudes, longitude spacing contracts: `dx_km = dx_deg × 111.32 × cos(lat)`
3. This causes **azimuth errors** that increase with latitude
4. Spreading rate calculation also affected (uses same gradient)

**Example Error:**
- At equator (0°): 1° lon = 111.32 km
- At 60°N: 1° lon = 55.66 km (50% shorter!)
- Gradient calculated without correction → azimuth rotated incorrectly

### Problem 2: Grid Spacing Varies with Latitude

**Current Code** ([run_tiling_comparison.py](run_tiling_comparison.py)):
```python
# Uses SINGLE grid spacing for entire domain
grid_spacing_km = lon_spacing_deg * 111.32 * cos(mean_lat)
```

**Issues:**
1. Uses `mean_lat` - good approximation for small regions
2. For large lat ranges (e.g., -50° to 0°), spacing varies 50%+
3. Filters have fixed size in pixels → variable physical size
4. Wavelength parameters become latitude-dependent

**Example:**
- Region: 30°E-100°E, -50°N to 0°N
- At -50°: cos(-50°) = 0.643 → grid_spacing = 2.16 km
- At 0°: cos(0°) = 1.0 → grid_spacing = 3.36 km
- **55% variation** across domain!

### Problem 3: Filter Parameters Don't Account for Latitude

**Current Code** ([AbFab.py:392-448](AbFab.py#L392-L448)):
```python
def generate_spatial_filter(H, lambda_n, lambda_s, azimuth, grid_spacing_km, ...):
    # Convert wavelength to wavenumber
    kn = grid_spacing_km / lambda_n  # Uses single grid_spacing_km
    ks = grid_spacing_km / lambda_s

    # Generate filter in pixel space
    # ... but pixels represent different physical distances at different latitudes
```

**Issues:**
1. Filter generated in pixel space with uniform physical interpretation
2. But pixels ≠ uniform physical size across latitudes
3. A 25-pixel filter at equator ≠ 25-pixel filter at 60°N
4. This creates **latitude-dependent wavelengths**

## Proposed Solutions

### Approach: Latitude-Aware Processing

We want to keep the code simple while accounting for major spherical effects. The key is to make grid_spacing_km **latitude-dependent**.

### Solution 1: Latitude-Corrected Gradients ⭐ PRIORITY

**Implementation:**
```python
def calculate_azimuth_from_age(seafloor_age, lat_coords):
    """
    Calculate azimuth accounting for spherical distortion.

    Parameters
    ----------
    seafloor_age : 2D array
        Ages in Myr
    lat_coords : 1D array
        Latitude coordinates in degrees

    Returns
    -------
    azimuth : 2D array
        Spreading direction in radians (corrected for latitude)
    """
    # Calculate gradients in index space
    grad_y, grad_x = np.gradient(seafloor_age)

    # Correct for latitude distortion
    # At each latitude, longitude spacing contracts by cos(lat)
    # So d_age/d_lon in physical space = (d_age/d_lon_index) / cos(lat)

    # Broadcast latitude to 2D
    lat_2d = np.broadcast_to(lat_coords[:, np.newaxis], seafloor_age.shape)
    cos_lat = np.cos(np.radians(lat_2d))

    # Correct x-gradient (longitude direction)
    grad_x_corrected = grad_x / cos_lat

    # Calculate azimuth with corrected gradients
    # Note: arctan2(y, x) where y is N-S, x is E-W
    azimuth = np.arctan2(grad_y, grad_x_corrected)

    return azimuth
```

**Impact:**
- ✓ Correct azimuth at all latitudes
- ✓ Ridges properly oriented regardless of latitude
- ✓ Simple modification to existing code
- ✓ No performance penalty

**Trade-offs:**
- Requires passing latitude coordinates
- Assumes regular lat/lon grid (reasonable for global grids)

### Solution 2: Latitude-Dependent Grid Spacing

**Option A: Per-Pixel Grid Spacing (Most Accurate)**
```python
def generate_bathymetry_spatial_filter(..., lat_coords=None):
    # Calculate grid spacing at each latitude
    lat_2d = np.broadcast_to(lat_coords[:, np.newaxis], (ny, nx))
    lon_spacing_deg = ...  # from data
    grid_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(lat_2d))

    # Now grid_spacing_km is 2D array
    # Use in filter generation and binning
```

**Pros:**
- Most accurate
- Handles any latitude range

**Cons:**
- `grid_spacing_km` becomes 2D array
- Requires significant code changes
- Filter generation more complex

**Option B: Per-Chunk Grid Spacing (Practical)**
```python
def process_bathymetry_chunk(coord, ..., lat_coords):
    # Calculate grid spacing at chunk's mean latitude
    chunk_lat_mean = np.mean(lat_coords[coord[0]:coord[0]+chunksize])
    grid_spacing_km = lon_spacing_deg * 111.32 * cos(radians(chunk_lat_mean))

    # Use chunk-specific grid spacing
    synthetic = af.generate_bathymetry_spatial_filter(
        ..., grid_spacing_km, ...
    )
```

**Pros:**
- ✓ Simple to implement
- ✓ Minimal code changes
- ✓ Good approximation (chunks are ~100 pixels ~300 km)
- ✓ Each chunk gets appropriate filter size

**Cons:**
- Still an approximation within each chunk
- Small discontinuities possible at chunk boundaries (but much smaller than without correction)

**Option C: Latitude-Binned Grid Spacing (Simplest)**
```python
# Bin latitudes into bands (e.g., 10° bands)
lat_bins = [-90, -60, -30, 0, 30, 60, 90]
grid_spacings = [lon_spacing * 111.32 * cos(radians(lat))
                 for lat in [-75, -45, -15, 15, 45, 75]]

# Assign each chunk to a latitude bin
for coord in coords:
    chunk_lat = mean(lat_coords[coord[0]:coord[0]+chunksize])
    bin_idx = digitize(chunk_lat, lat_bins)
    grid_spacing_km = grid_spacings[bin_idx]
```

**Pros:**
- Very simple
- Pre-calculated values
- Fast

**Cons:**
- Step changes at bin boundaries
- Less accurate than per-chunk

### Solution 3: Spherical Spreading Rate Calculation

**Current Issue:**
```python
# From AbFab.py:193-196
grad_y, grad_x = np.gradient(seafloor_age)
age_gradient_magnitude = sqrt(grad_x**2 + grad_y**2)
spreading_rate = grid_spacing_km / age_gradient_magnitude
```

This assumes `grad_x` and `grad_y` have same physical units!

**Corrected Version:**
```python
def calculate_spreading_rate_from_age(seafloor_age, lon_spacing_deg, lat_coords):
    """
    Calculate spreading rate with spherical correction.
    """
    # Gradients in index space
    grad_y, grad_x = np.gradient(seafloor_age)

    # Physical spacing
    lat_2d = np.broadcast_to(lat_coords[:, np.newaxis], seafloor_age.shape)
    dx_km = lon_spacing_deg * 111.32 * np.cos(np.radians(lat_2d))
    dy_km = lat_spacing_deg * 111.32  # Constant

    # Gradient magnitude in km/Myr
    # Note: need to convert index gradients to physical gradients
    physical_grad_x = grad_x * dx_km  # km of age change per km distance
    physical_grad_y = grad_y * dy_km

    # Actually wait - gradients are in Myr per index
    # We need: (Myr/km) so we can invert to km/Myr
    age_gradient_x_physical = grad_x / dx_km  # Myr per km
    age_gradient_y_physical = grad_y / dy_km  # Myr per km

    age_gradient_magnitude = np.sqrt(age_gradient_x_physical**2 +
                                     age_gradient_y_physical**2)

    # Spreading rate = 1 / age_gradient (km/Myr = mm/yr)
    spreading_rate = 1.0 / age_gradient_magnitude

    return spreading_rate
```

## Recommended Implementation Plan

### Phase 1: Critical Fixes (Immediate) ⭐

**1. Fix Azimuth Calculation**
- Add `lat_coords` parameter to `calculate_azimuth_from_age()`
- Apply `cos(lat)` correction to longitude gradient
- Update all callers

**Impact:** Correct ridge orientations at all latitudes

**2. Fix Spreading Rate Calculation**
- Add spherical correction to `calculate_spreading_rate_from_age()`
- Account for varying longitude spacing

**Impact:** Correct spreading rates at all latitudes

### Phase 2: Grid Spacing Improvements (Important)

**3. Per-Chunk Grid Spacing** (Option B above)
- Calculate `grid_spacing_km` at each chunk's mean latitude
- Pass to `generate_bathymetry_spatial_filter()`

**Impact:** Correct filter sizes/wavelengths for each chunk

### Phase 3: API Updates (Moderate Effort)

**4. Update Function Signatures**
```python
# OLD
def generate_bathymetry_spatial_filter(
    seafloor_age, sediment, params, grid_spacing_km, ...
)

# NEW
def generate_bathymetry_spatial_filter(
    seafloor_age, sediment, params,
    lon_spacing_deg, lat_coords,  # Replace grid_spacing_km
    ...
)
```

Calculate `grid_spacing_km` internally based on latitude.

### Phase 4: Advanced (Optional)

**5. True 2D Grid Spacing**
- Make `grid_spacing_km` a 2D array
- Update filter generation to handle varying spacing
- Most accurate but most complex

**6. Mercator Projection Option**
- Offer to reproject data to Mercator (equal-area)
- Process in Mercator space (uniform spacing)
- Reproject back to lat/lon
- More complex but handles extreme latitudes better

## Accuracy Assessment

### Current Errors (No Correction)

**Azimuth Error:**
- At ±30°: ~13% error in longitude gradient → ~7° azimuth error
- At ±60°: ~50% error in longitude gradient → ~27° azimuth error
- At ±80°: ~83% error → ridges almost perpendicular!

**Wavelength Error:**
- At ±60°: Filters too large by 50%
- Hills appear smoother than they should
- Spreading rate errors of 50%+

### With Phase 1-2 Corrections

**Azimuth Error:**
- < 1° at all latitudes (limited by grid resolution)

**Wavelength Error:**
- < 5% per chunk (chunks span ~2° latitude)
- Acceptable for most applications

**Spreading Rate Error:**
- < 3% at all latitudes

## Backward Compatibility

### Breaking Changes
- `calculate_azimuth_from_age()` needs `lat_coords`
- `calculate_spreading_rate_from_age()` needs `lon_spacing_deg` and `lat_coords`
- `generate_bathymetry_spatial_filter()` may need latitude info

### Migration Path
1. Add new parameters with defaults (None)
2. If None, fall back to old behavior + warning
3. Deprecate old behavior in next version
4. Remove old behavior in major version

### Example
```python
def calculate_azimuth_from_age(seafloor_age, lat_coords=None):
    if lat_coords is None:
        warnings.warn("Spherical correction disabled. Pass lat_coords for accuracy.",
                     DeprecationWarning)
        # Old behavior
        grad_y, grad_x = np.gradient(seafloor_age)
        return np.arctan2(grad_y, grad_x)
    else:
        # New corrected behavior
        return calculate_azimuth_corrected(seafloor_age, lat_coords)
```

## Testing Strategy

1. **Synthetic Test**: Create synthetic age grid with known azimuth
   - Test at 0°, 30°, 60° latitude
   - Verify calculated azimuth matches input

2. **Real Data Test**: Use Pacific or Atlantic data
   - Compare uncorrected vs corrected
   - Verify ridges align with magnetic anomalies

3. **Spreading Rate Validation**:
   - Compare to published spreading rate maps
   - Check global consistency

## Summary

**Minimum Viable Fix** (Phase 1):
- Latitude-corrected gradients
- 1-2 hours implementation
- Huge accuracy improvement

**Recommended Full Fix** (Phases 1-2):
- Above + per-chunk grid spacing
- 4-6 hours implementation
- Handles all realistic use cases well

**Maximum Accuracy** (Phases 1-4):
- 2D grid spacing throughout
- 2-3 days implementation
- Marginal improvement over Phase 2 for most cases
