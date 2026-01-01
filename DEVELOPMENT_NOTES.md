# AbFab Development Notes

This document provides implementation details, known issues, and guidance for future development. It consolidates information for future Claude Code sessions or other developers working on this codebase.

---

## Recent Changes (January 2026)

### Summary
Comprehensive improvements to AbFab.py focusing on the spatial filter method (production code), NOT the FFT method.

### What Was Added

#### 1. Dual Filter Support
- **File**: `AbFab.py`, lines 212-450
- **Functions Modified**:
  - `generate_spatial_filter()` - Added `filter_type` parameter
  - `generate_bathymetry_spatial_filter()` - Passes through `filter_type`

**Filter Options:**
- `'gaussian'` (default): Fast, smooth, backward compatible
- `'von_karman'`: Uses modified Bessel function K_ν for theoretically correct von Kármán autocorrelation

**Testing**: Verified both produce correct N-S oriented ridges for E-W spreading. von Kármán produces ~40% higher RMS due to heavier tails.

#### 4. Performance Optimization (NEW)
- **File**: `AbFab.py`, lines 493-615
- **Function**: `generate_bathymetry_spatial_filter()` - Optimized with filter bank approach

**Implementation**:
```python
# Pre-compute filters at discrete (azimuth, sediment) combinations
for az in azimuth_angles:  # e.g., 36 angles
    for sed in sediment_levels:  # e.g., 5 sediment bins
        filter = generate_spatial_filter(H_mod, kn_mod, ks_mod, az)
        convolved_results[az, sed] = convolve(random_field, filter)

# Extract results using nearest-neighbor lookup
bathymetry = convolved_results[azimuth_bin_idx, sediment_bin_idx]
```

**Performance**:
- **50-53× speedup** over original pixel-by-pixel method (100×100 grid)
- **3.88% relative error** with default settings (36 azimuth × 5 sediment bins)
- **0.9995 correlation** with original implementation
- Scales from 10,000 convolutions → 180 convolutions (36×5)

**Key Insight**:
Original attempt only binned by azimuth and applied sediment modification as post-processing. This failed because sediment affects filter SHAPE (kn, ks) not just amplitude (H). Solution: bin by BOTH azimuth and sediment.

**Tunable Parameters**:
- `azimuth_bins=36` (default) - More bins = more accurate, slower
- `sediment_bins=5` (default) - More bins = more accurate, slower
- `optimize=True` (default) - Set False for pixel-perfect original method

**Trade-offs**:
- 3 sediment bins: ~60× faster, 8% error
- 5 sediment bins: ~50× faster, 4% error (default)
- 10 sediment bins: ~25× faster, 2% error
- 20 sediment bins: ~15× faster, <1% error

**Testing**: See `test_optimization.py` and `test_optimization_visual.ipynb`

#### 2. Azimuth Verification
- **Function**: `calculate_azimuth_from_age()` (lines 132-159)
- **Status**: ✓ Verified correct
- **Test**: `test_azimuth.ipynb`, `test_azimuth_orientation.png`
- **Result**: Produces linear ridges perpendicular to spreading direction (parallel to ridge)

#### 3. Utility Functions (⚠️ NEEDS CALIBRATION)

**`calculate_spreading_rate_from_age(seafloor_age, grid_spacing_km)`**
- **File**: Lines 162-204
- **Purpose**: Estimate half-spreading rate from age gradient
- **Formula**: `rate = grid_spacing / age_gradient` (km/Myr = mm/yr numerically)
- **Status**: ✓ Works correctly (tested 40 mm/yr → 40.0 mm/yr recovery)

**`spreading_rate_to_params(spreading_rate, output_units='deg')`**
- **File**: Lines 207-294
- **Purpose**: Convert spreading rate to abyssal hill parameters
- **Status**: ⚠️ **NEEDS CALIBRATION** - produces values ~1000× too large
- **Issue**: See "Known Issues" section below

### What Was NOT Changed

- ❌ FFT method (`generate_synthetic_bathymetry_fft`) - Not used in production
- ❌ `von_karman_spectrum()` - Only used by FFT method
- ❌ Random field generation - Working correctly
- ❌ Tiling/chunking - User's implementation is optimal

---

## Known Issues

### Issue 1: `spreading_rate_to_params()` Calibration Problem

**Severity**: Medium (workaround exists)

**Problem**:
Function produces kn/ks values ~1000× too large for lat/lon grids, resulting in overly smooth output.

**Example**:
```python
rate = 21.5  # mm/yr
params = spreading_rate_to_params(rate, output_units='deg')
# Returns: kn=48.3, ks=114.8 deg⁻¹
# Expected: kn~0.05, ks~0.2 deg⁻¹ (based on working fixed values)
```

**Root Cause**:
The filter parameters kn/ks are **dimensionless scaling factors** on normalized coordinates [-1, 1], NOT physical wavenumbers (k = 2π/λ). The function incorrectly assumes they are wavenumbers and applies unit conversions.

**From the filter code**:
```python
x = np.linspace(-1, 1, filter_size)  # Normalized coordinates
filter_exp = -(x_rot**2 / (2 * kn**2) + y_rot**2 / (2 * ks**2))
```

Small kn (e.g., 0.05) → wide filter → realistic abyssal hills
Large kn (e.g., 48) → narrow filter → overly smooth (almost delta function)

**Workaround**:
Use fixed parameters that have been empirically validated:
```python
params = {'H': 50, 'kn': 0.05, 'ks': 0.2, 'D': 2.2}
```

**Future Fix Required**:
1. Empirically determine relationship between physical wavelength and filter scaling factor
2. Generate bathymetry with known kn/ks values
3. Measure actual wavelengths via FFT/autocorrelation
4. Calibrate formula: `kn_filter = f(λ_physical)`

**Documentation**:
- Warning added to function docstring (lines 244-253)
- Details in `CALIBRATION_NEEDED.md` (can be deleted after fix)

---

## Architecture Notes

### Spatial Filter Method (Production)

**Function**: `generate_bathymetry_spatial_filter()` (lines 493-615)

**Current Implementation** (✓ Optimized):
The function now uses a filter bank approach by default (`optimize=True`), providing 50× speedup. The original pixel-by-pixel method is preserved as `generate_bathymetry_spatial_filter_original()` (lines 453-489) for reference and testing.

**Original Implementation** (Still available via `optimize=False`):
```python
for i in range(ny):
    for j in range(nx):
        # Generate NEW filter for EVERY pixel
        spatial_filter = generate_spatial_filter(...)
        # Convolve ENTIRE random field for EVERY pixel
        filtered_value = oaconvolve(random_field, spatial_filter, mode='same')[i, j]
```

**Why Original Was Slow**:
- Generates ny × nx filters (e.g., 100×100 = 10,000 filters!)
- Performs ny × nx full convolutions
- Each convolution processes entire field but only uses one output pixel

**Optimization Approach**:
Pre-compute filters at discrete (azimuth, sediment) bins and use nearest-neighbor lookup. See "Performance Optimization" section above for details.

**User's Chunk-Based Approach (Still Recommended for Large Grids)**:
```python
def process_bathymetry_chunk(coord, age_dataarray, sed_dataarray,
                             rand_dataarray, chunksize, chunkpad,
                             params, filter_type='gaussian'):
    # Extract chunk with padding
    chunk_age = age_dataarray[y0:y1, x0:x1]
    # ... generate bathymetry for chunk
    # Trim padding before returning
    return result[pad:-pad, pad:-pad]

# Parallel processing
results = Parallel(n_jobs=4)(
    delayed(process_bathymetry_chunk)(...) for coord in coords
)
```

**Performance**: ~20-50 seconds for 360×842 grid with 4 CPUs, 50×50 chunks, 20-pixel padding

**Future Optimization** (Low Priority):
Could vectorize the inner loop by:
1. Pre-computing filter bank for discrete azimuth/sediment values
2. Using FFT-based convolution once
3. Interpolating results

But current approach works well enough for typical use cases.

---

## File Organization

### Core Implementation
- **AbFab.py** - Main module with all functions

### Notebooks
- **tiling_test.ipynb** - Original working example (keep as reference)
- **tiling_test_updated.ipynb** - Demonstrates new features (dual filters)
- **test_azimuth.ipynb** - Azimuth verification test

### Documentation
- **README.md** - User-facing documentation
- **DEVELOPMENT_NOTES.md** - This file (for developers/future Claude sessions)

### Test Files (Can be removed after verification)
- `test_azimuth.py` - Standalone azimuth test
- `test_filter_types.py` - Filter comparison test
- `test_utility_functions.py` - Spreading rate utilities test
- `compare_filters.py` - Detailed filter comparison
- `debug_vonkarman_filter.py` - von Kármán filter debugging

### Images (Generated by tests - can keep or remove)
- Various `.png` files showing test results

---

## Parameter Reference

### Units and Conventions

**For lat/lon grids** (current implementation):
- Coordinates are in **degrees**
- kn, ks are **dimensionless scaling factors** (not true wavenumbers!)
- H is in **meters**
- Ages in **Myr** (million years)
- Sediment thickness in **meters**
- Spreading rates in **mm/yr**

**Physical space** (for reference):
- Typical abyssal hill wavelengths: 10-30 km (perpendicular), 5-10 km (parallel)
- 1 degree ≈ 111 km at equator

### Working Parameters

These values have been empirically validated for lat/lon grids:

```python
params = {
    'H': 50,       # RMS height (m): 50-300 typical range
    'kn': 0.05,    # Perpendicular scaling: 0.01-0.1 typical
    'ks': 0.2,     # Parallel scaling: 0.1-0.5 typical
    'D': 2.2       # Fractal dimension: 2.0-2.3 typical
}
```

### Filter Scaling Interpretation

Given kn=0.05, ks=0.2 on normalized coords [-1, 1]:
- Filter extent dominated by kn (smaller value → wider spread)
- Anisotropy ratio: ks/kn = 4 (4× narrower parallel to ridge)
- This produces realistic abyssal hill fabric

**Physical wavelength** (approximate, needs calibration):
- kn=0.05 → λn ≈ 10-20 km (perpendicular)
- ks=0.2 → λs ≈ 5-10 km (parallel)

---

## Testing Checklist

When modifying AbFab.py, verify:

1. **Orientation**: Run `test_azimuth.ipynb`
   - Check that ridges are perpendicular to spreading direction
   - Look for clear N-S ridges in E-W spreading example

2. **Filter comparison**: Run `test_filter_types.py` or notebook cell
   - Gaussian and von Kármán should both show linear ridges
   - von Kármán should have ~1.4× higher RMS
   - Both should have same orientation

3. **Backward compatibility**: Run original `tiling_test.ipynb`
   - Should produce same results as before changes
   - Default behavior unchanged

4. **Parameter units**: Verify degrees vs km handling
   - Lat/lon grids require degree-based parameters
   - Check kn/ks values are reasonable (0.01-0.5 range)

---

## Future Development Tasks

### High Priority
1. **Calibrate `spreading_rate_to_params()`**
   - Empirically determine kn_filter = f(wavelength) relationship
   - Update formulas to produce correct scaling factors
   - Add comprehensive tests

### Medium Priority
2. **Optimize `generate_bathymetry_spatial_filter()`**
   - Vectorize filter generation
   - Use FFT-based convolution
   - Pre-compute filter banks

3. **Add proper unit handling**
   - Support both lat/lon (degrees) and projected (km) grids
   - Automatic unit detection and conversion
   - Clear documentation of which units apply where

### Low Priority
4. **Additional features**
   - Save/load filter banks for reuse
   - GPU acceleration for large grids
   - Alternative sediment modification models

---

## Tips for Future Claude Code Sessions

### Context to Provide
- This codebase uses **spatial filter method**, NOT FFT method
- Parameters kn/ks are **scaling factors**, not physical wavenumbers
- Users work with **lat/lon grids** (degrees), not projected grids (km)
- Current focus: practical utility over theoretical perfection

### What Works Well
- Dual filter implementation (Gaussian + von Kármán)
- Azimuth calculation
- Chunk-based parallel processing (user's notebook approach)
- Sediment modification

### What Needs Work
- `spreading_rate_to_params()` calibration (see Issue 1 above)
- Performance optimization (low priority - chunking works)
- Unit conversions between degrees and km

### Files to Read First
1. `README.md` - User perspective
2. This file - Development context
3. `AbFab.py` lines 453-513 - Core spatial filter function
4. `tiling_test.ipynb` - Real-world usage example

---

## Change Log

### 2026-01-01: Comprehensive Update (Claude Code Session)
- ✓ Added dual filter support (Gaussian + von Kármán)
- ✓ Verified azimuth calculation
- ✓ Added utility functions (needs calibration)
- ✓ Created updated notebook with examples
- ✓ Comprehensive testing and documentation
- ⚠️ Identified calibration issue with spreading_rate_to_params()

### Pre-2026: Original Implementation
- Original spatial filter method (Gaussian only)
- Sediment modification
- Basic chunking approach

---

**Last Updated**: 2026-01-01
**Status**: Functional with known calibration issue documented
**Recommended for Production**: Yes (using fixed parameters)
