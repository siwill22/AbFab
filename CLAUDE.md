# AbFab - Technical Documentation for Claude Code

This document contains comprehensive technical information about the AbFab project for future Claude Code sessions.

## Project Overview

**AbFab** generates synthetic abyssal hill bathymetry using spatial filtering based on Goff & Arbic (2010). The code has evolved through several major redesigns to improve usability and physical accuracy.

## Current Implementation (January 2025)

### Core Algorithm
1. Calculate azimuth from seafloor age gradient (spreading direction)
2. Optionally calculate spreading rate from age gradient
3. Generate anisotropic spatial filters (Gaussian or von Kármán)
4. Modify filter parameters based on sediment thickness
5. Convolve random field with filters
6. Use filter bank + binning for 50× speedup

### Key Design Decisions

**Physical Units (December 2024)**
- **Why**: Resolution independence, physical interpretability, literature comparison
- **Parameters**: `lambda_n`, `lambda_s` (km) instead of `kn`, `ks` (pixel⁻¹)
- **Breaking change**: All functions now require `grid_spacing_km` parameter
- **Conversion**: `k = grid_spacing_km / lambda` (wavenumber from wavelength)

**Trilinear Bin Interpolation (January 2025)**
- **Problem**: Nearest-neighbor bin lookup created visible discontinuities at bin transitions
- **Solution**: Trilinear interpolation between 8 neighboring bins in 3D filter bank
- **Location**: [AbFab.py:713-784](AbFab.py#L713-L784)
- **Result**: 0.00% discontinuities (validated by `test_bin_interpolation.py`)
- **Key insight**: Blending was unnecessary - the root cause was bin quantization, not chunking

### Major Components

#### 1. Filter Generation (`generate_spatial_filter()`)
**Location**: Lines 104-183 in AbFab.py

Creates anisotropic 2D filters:
- **Gaussian**: Fast, clean, backward compatible (default)
- **von Kármán**: Theoretically correct with Bessel function K_ν

**Parameters**:
- `H`: RMS height (m)
- `lambda_n`: Perpendicular wavelength (km) - controls hill width
- `lambda_s`: Parallel wavelength (km) - controls ridge narrowness
- `azimuth`: Spreading direction (radians)
- `grid_spacing_km`: km/pixel for physical units
- `filter_type`: 'gaussian' or 'von_karman'

**Key equations**:
```python
# Convert wavelength to wavenumber
kn = grid_spacing_km / lambda_n
ks = grid_spacing_km / lambda_s

# Gaussian filter
filt = H² * exp(-0.5 * ((x_rot*kn)² + (y_rot*ks)²)) / (2π/(kn*ks))

# von Kármán filter
filt = H² * (2^(ν-1) * Γ(ν))^-1 * (kr)^ν * K_ν(kr) / (π/(kn*ks))
```

#### 2. Sediment Modification (`modify_by_sediment()`)
**Location**: Lines 185-213 in AbFab.py

Implements Goff & Arbic (2010) Equations 5-7:
- Sediment reduces roughness (increases H, decreases λ)
- Empirical relationships from observations

**Equations**:
```python
# Wavenumber increases with sediment
k_sed = k₀ * (1 + 1.3 * S/H₀)

# Convert back to wavelength
lambda_sed = grid_spacing_km / k_sed

# Height increases
H_sed = H₀ + 0.7 * S
```

#### 3. Spreading Rate Utilities

**`calculate_spreading_rate_from_age()`** - Lines 242-267
- Calculates half-spreading rate from age gradient
- Uses finite differences on age grid
- Returns mm/yr

**`spreading_rate_to_params()`** - Lines 215-338
- Empirical relationships from Goff & Arbic (2010)
- **Two modes**:
  1. `base_params=None`: Returns absolute parameters
  2. `base_params=dict`: Applies multiplicative/additive scaling

**Scaling relationships** (when `base_params` provided):
```python
# Scaling factors
f_H = 1.3 - 0.01 * u  # Height decreases with spreading rate
f_lambda = 0.3 + 0.014 * u  # Wavelength increases slightly
delta_D = -0.003 * (u - u_ref)  # Fractal dimension changes

# Apply to base parameters
H = H_base * f_H
lambda_n = lambda_n_base * f_lambda
lambda_s = lambda_s_base * f_lambda
D = D_base + delta_D
```

#### 4. Optimized Filter Bank (`generate_bathymetry_spatial_filter()`)
**Location**: Lines 557-784 in AbFab.py

**Strategy**:
1. Bin continuous variables (azimuth, sediment, spreading_rate) into discrete levels
2. Pre-compute filters at discrete combinations
3. Convolve random field with each filter once
4. **Use trilinear interpolation** to select/blend between bins for each pixel

**Performance**:
- Reduces ~10,000 convolutions → ~900 convolutions (36×5×5 bins)
- 50× speedup with <4% error
- Automatically enabled with `optimize=True` (default)

**Bin Interpolation** (Lines 716-784):
```python
# Calculate continuous bin positions
azimuth_bin_pos = azimuth_normalized / azimuth_bin_width
sediment_bin_pos = (sediment - sediment_min) / sediment_bin_width
sr_bin_pos = (spreading_rate - sr_min) / sr_bin_width

# Get integer indices and fractional parts
az_idx0 = floor(azimuth_bin_pos)
az_frac = azimuth_bin_pos - az_idx0

# Trilinear interpolation across 8 corners
c000 = convolved_stack[az_idx0, sed_idx0, sr_idx0, ...]
c001 = convolved_stack[az_idx0, sed_idx0, sr_idx1, ...]
# ... (8 corners total)

# Interpolate in 3D
bathymetry = trilinear_interp(c000, ..., c111, az_frac, sed_frac, sr_frac)
```

This eliminates visible boundaries at bin transitions without any post-processing blending.

## Spatially Varying Spreading Rate

**Enabled with**: `spreading_rate_bins > 1` and `base_params` provided

**How it works**:
1. Calculate spreading rate at each pixel from age gradient
2. Determine min/max spreading rate in domain
3. Create `spreading_rate_bins` discrete levels (e.g., 5 bins: 10, 20, 30, 40, 50 mm/yr)
4. For each (azimuth, sediment, spreading_rate) combination:
   - Apply spreading rate scaling to `base_params`
   - Apply sediment modification
   - Generate filter
   - Convolve with random field
5. For each pixel, use trilinear interpolation to blend between nearest bins

**Result**: Continuous spatial variation in parameters reflecting local spreading conditions.

## Chunk Processing for Large Grids

**See**: [tiling_test_updated.ipynb](tiling_test_updated.ipynb)

**Strategy**:
1. Process grid in chunks (e.g., 100×100 pixels)
2. Add padding (e.g., 20 pixels) to handle convolution edge effects
3. Use **global random field** (not per-chunk) for continuity
4. Trim padding after processing
5. Assemble chunks edge-to-edge

**Key insight**: Padding handles edge effects, bin interpolation handles smoothness. No blending needed!

**Implementation**:
```python
def process_bathymetry_chunk(coord, age_da, sed_da, rand_da, chunksize, chunkpad, ...):
    # Extract chunk WITH padding from global arrays
    chunk_age = age_da[y:y+chunksize+chunkpad, x:x+chunksize+chunkpad]
    chunk_sed = sed_da[y:y+chunksize+chunkpad, x:x+chunksize+chunkpad]
    chunk_random = rand_da[y:y+chunksize+chunkpad, x:x+chunksize+chunkpad]  # Global!

    # Generate bathymetry
    synthetic = af.generate_bathymetry_spatial_filter(...)

    # Trim padding
    return synthetic[chunkpad/2:-chunkpad/2, chunkpad/2:-chunkpad/2]

# Parallel processing
results = Parallel(n_jobs=4)(
    delayed(process_bathymetry_chunk)(coord, ...) for coord in coords
)

# Simple concatenation (no blending!)
for chunk, coord in zip(results, coords):
    output[coord[0]:coord[0]+chunksize, coord[1]:coord[1]+chunksize] = chunk
```

## Development History

### Phase 1: Original Implementation
- Pixel-based wavenumbers (`kn`, `ks`)
- Resolution-dependent (parameters must change with grid spacing)
- Slow pixel-by-pixel processing

### Phase 2: Physical Units Redesign (December 2024)
- **Motivation**: Resolution independence, physical interpretability
- Changed to wavelengths in km (`lambda_n`, `lambda_s`)
- Added required `grid_spacing_km` parameter
- **Breaking change**: Updated all function signatures

### Phase 3: Spreading Rate Integration (December 2024)
- Added `calculate_spreading_rate_from_age()`
- Enhanced `spreading_rate_to_params()` with parameter scaling
- Integrated into main generation function with binning

### Phase 4: Optimization (December 2024)
- Implemented filter bank approach
- 50× speedup with binning
- Maintained <4% error vs pixel-by-pixel

### Phase 5: Spatially Varying Spreading Rate (January 2025)
- Extended filter bank to 3D: azimuth × sediment × spreading_rate
- Added `spreading_rate_bins` and `base_params` parameters
- Enabled continuous spatial variation in parameters

### Phase 6: Bin Interpolation Fix (January 2025)
- **Problem discovered**: Nearest-neighbor bin lookup created visible discontinuities
- **Initial wrong approach**: Attempted feather blending (failed because chunks don't overlap)
- **Root cause identified**: Bin quantization, not chunking
- **Solution**: Trilinear interpolation between bins
- **Result**: 0% discontinuities, no blending needed

### Phase 7: Complete Bathymetry & Diffusive Sediment Fix (January 2025)
- **New feature**: Added `generate_complete_bathymetry()` combining subsidence + hills + sediment
- **Problem discovered**: 1092m discontinuities at chunk boundaries with `sediment_mode='fill'`
- **Root cause**: `gaussian_filter(..., mode='nearest')` in `apply_diffusive_sediment_infill()` applied per-chunk
- **Solution**: Refactored workflow to apply diffusive infill GLOBALLY after chunk assembly
- **Key insight**: ANY smoothing operation with edge-dependent modes MUST be global, not per-chunk
- **Result**: Complete elimination of chunk boundary artifacts

## Common Pitfalls & Solutions

### 1. "I see visible boundaries/discontinuities at chunk edges"
**⚠️ CRITICAL GOTCHA - January 2025**

**Root Cause**: Smoothing operations with edge-dependent modes applied PER-CHUNK

**The Problem**:
When using `scipy.ndimage.gaussian_filter()` or similar smoothing with `mode='nearest'`, `mode='reflect'`, etc., each chunk gets different edge padding. This creates discontinuities when chunks are assembled.

**Specific case identified**: `apply_diffusive_sediment_infill()` in [AbFab.py:244](AbFab.py#L244)
```python
smoothed_basement = gaussian_filter(smoothed_basement, sigma=mean_sigma, mode='nearest')
```

This caused **1092m discontinuities** at chunk boundaries when `sediment_mode='fill'` was used!

**THE FIX**: Apply smoothing operations GLOBALLY after chunk assembly, not per-chunk.

**Implementation** ([generate_complete_bathymetry.py:377-414](generate_complete_bathymetry.py#L377-L414)):
```python
# WRONG: Apply diffusive infill per chunk
for chunk in chunks:
    chunk_bathy = generate_complete_bathymetry(..., sediment_mode='fill')  # ❌

# CORRECT: Simple drape per chunk, then global diffusive infill
for chunk in chunks:
    chunk_bathy = generate_complete_bathymetry(..., sediment_mode='drape')  # ✅

# After assembly:
if SEDIMENT_MODE == 'fill':
    complete_grid = apply_diffusive_sediment_infill(complete_grid, ...)  # ✅
```

**General Rule**: ANY operation using `scipy.ndimage` with edge modes (`gaussian_filter`, `convolve`, etc.) MUST be applied globally after assembly, not per-chunk.

**Other diagnoses**:
- Check if boundaries align with chunk edges (100px apart) → smoothing/padding issue
- Check if boundaries are irregular/not aligned with chunks → bin transitions (fixed with trilinear interpolation)

### 2. "Results look too smooth/rough"
**Check**:
- `lambda_n` and `lambda_s` values (km)
- Spreading rate relationships (faster = smaller wavelengths = smoother)
- Filter type (von Kármán is ~40% rougher than Gaussian)

**Adjust**:
- Decrease wavelengths for rougher texture
- Increase `H` for more vertical relief
- Try different filter type

### 3. "Performance is slow"
**Solutions**:
- Ensure `optimize=True` (default)
- Reduce `azimuth_bins` (36 → 18) for 2× speedup, minimal error increase
- Use Gaussian filter (30% faster than von Kármán)
- Use chunk-based parallel processing for large grids

### 4. "Spreading rate parameters look wrong"
**Common issue**: Using `spreading_rate_to_params()` without `base_params`

**Solutions**:
- **Option 1**: Provide `base_params` for scaling approach (recommended)
- **Option 2**: Use returned params but verify they're reasonable for your region
- **Option 3**: Use fixed parameters and skip spreading rate entirely

## File Structure

```
AbFab/
├── AbFab.py                    # Main module (784 lines)
├── README.md                   # User documentation
├── CLAUDE.md                   # This file - AI assistant guide
├── tiling_test_updated.ipynb  # Demonstration notebook
├── tests/                      # Test scripts
│   ├── test_optimization.py
│   ├── test_sediment_effect.py
│   ├── test_spreading_rate_scaling.py
│   ├── test_spreading_rate_variation.py
│   └── test_bin_interpolation.py
└── [utility scripts]
```

## Testing

**Run tests**:
```bash
cd tests
python test_bin_interpolation.py  # Validates smooth transitions (0% discontinuities)
python test_optimization.py       # Validates filter bank speedup
```

## Key Code Locations

**Core Functions**:
- `generate_spatial_filter()`: Lines 104-183
- `modify_by_sediment()`: Lines 185-213
- `spreading_rate_to_params()`: Lines 215-338
- `calculate_spreading_rate_from_age()`: Lines 242-267
- `generate_bathymetry_spatial_filter()`: Lines 557-784

**Trilinear Interpolation**: Lines 716-784 (the fix for bin discontinuities)

## Future Development Considerations

### Potential Enhancements
1. **Adaptive binning**: Automatically adjust bin count based on domain variation
2. **GPU acceleration**: Port convolution to GPU for massive grids
3. **Alternative spreading rate relationships**: Region-specific calibrations
4. **Machine learning**: Train on observed bathymetry for better parameter prediction

### Code Maintenance
- **Do not remove** trilinear interpolation - it's essential for smooth results
- **Preserve** `optimize=True` as default - speedup is critical for large grids
- **Keep** both filter types - Gaussian for speed, von Kármán for theory
- **Maintain** physical units - resolution independence is key feature

### Breaking Changes to Avoid
- Changing `grid_spacing_km` units (must stay km/pixel)
- Removing `base_params` from `spreading_rate_to_params()` (needed for scaling mode)
- Changing bin interpolation back to nearest-neighbor (creates discontinuities)

## References & Theory

**Primary Reference**:
Goff, J. A., & Arbic, B. K. (2010). Global prediction of abyssal hill root-mean-square heights from small-scale altimetric gravity variability. *Journal of Geophysical Research: Solid Earth*, 115(B12).

**Key Equations from Paper**:
- Equation 1: Abyssal hill RMS height vs spreading rate
- Equations 5-7: Sediment modification of parameters
- Appendix: Von Kármán autocorrelation function

**Physical Basis**:
- Abyssal hills form at mid-ocean ridges from tectonic extension
- Ridges are perpendicular to spreading direction
- Faster spreading → smoother, smaller hills (less vertical relief time)
- Sediment infills topography → reduces roughness

## Session Continuity Notes

**If starting a new Claude Code session**:
1. Read this document first for complete context
2. Check [generate_complete_bathymetry.py](generate_complete_bathymetry.py) for current workflow
3. Remember: **No blending needed** - trilinear interpolation handles smoothness
4. **CRITICAL**: Smoothing operations MUST be global, not per-chunk

**Critical Understanding - Chunk Boundary Issues**:
There were TWO separate causes of chunk boundaries, both now solved:

1. **Bin transitions** (solved Phase 6):
   - Cause: Nearest-neighbor bin lookup
   - Fix: Trilinear interpolation [AbFab.py:716-784](AbFab.py#L716-L784)

2. **Diffusive sediment infill** (solved Phase 7):
   - Cause: `gaussian_filter(..., mode='nearest')` applied per-chunk
   - Fix: Apply diffusive infill globally after assembly [generate_complete_bathymetry.py:377-414](generate_complete_bathymetry.py#L377-L414)

**If someone reports chunk boundaries**:
1. Check if using `sediment_mode='fill'` → ensure global application
2. Check for ANY `scipy.ndimage` operations in chunk processing → move to post-assembly
3. Verify trilinear interpolation is enabled (it's automatic with `optimize=True`)
4. Don't suggest blending - it doesn't work and isn't needed

**Last Major Update**: January 2025 (diffusive sediment fix, complete bathymetry workflow)
