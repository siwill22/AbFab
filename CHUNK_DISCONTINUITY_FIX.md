# Chunk Discontinuity Fix - January 2025

## Summary

Fixed **1092 meter discontinuities** at chunk boundaries in complete bathymetry generation when using diffusive sediment infill (`sediment_mode='fill'`).

## Problem

When generating complete bathymetry with diffusive sediment infill, massive discontinuities (up to 1092m) appeared at chunk boundaries, particularly visible in the Southern Africa test region at longitude ~34°E.

## Root Cause

The `apply_diffusive_sediment_infill()` function in [AbFab.py:244](AbFab.py#L244) uses:

```python
smoothed_basement = gaussian_filter(smoothed_basement, sigma=mean_sigma, mode='nearest')
```

When this smoothing operation is applied **per-chunk** before assembly:
- Each chunk sees different boundary values
- `mode='nearest'` pads edges differently for each chunk
- Chunks produce different smoothed results at boundaries
- Assembly creates visible discontinuities

## The Fix

**Refactored workflow** in [generate_complete_bathymetry.py](generate_complete_bathymetry.py):

### Before (WRONG):
```python
# Each chunk applies diffusive infill independently
for chunk in chunks:
    complete_bathy = generate_complete_bathymetry(
        ...,
        sediment_mode='fill'  # ❌ Gaussian smoothing per chunk
    )
```

### After (CORRECT):
```python
# Step 1: Each chunk uses simple drape only
for chunk in chunks:
    basement_bathy = generate_complete_bathymetry(
        ...,
        sediment_mode='drape'  # ✅ No smoothing per chunk
    )

# Step 2: Assemble all chunks
complete_grid = assemble_chunks(chunks)

# Step 3: Apply diffusive infill globally
if SEDIMENT_MODE == 'fill':
    complete_grid = apply_diffusive_sediment_infill(
        complete_grid, ...  # ✅ Smoothing on complete grid
    )
```

## Implementation Details

### Changes to [generate_complete_bathymetry.py](generate_complete_bathymetry.py)

**Line 125**: Force simple drape mode per chunk
```python
sediment_mode='drape',  # Use simple drape per chunk, diffusion applied globally later
```

**Lines 377-414**: Apply diffusive infill globally after assembly
```python
if SEDIMENT_MODE == 'fill':
    print("\nApplying global diffusive sediment infill...")

    # Get sediment thickness matching complete grid
    sed_trimmed_data = resample_sediment_to_grid(complete_grid)

    # Remove drape to get basement
    basement_grid = complete_grid.data - sed_trimmed_data

    # Apply diffusive infill globally
    final_grid = af.apply_diffusive_sediment_infill(
        basement_grid, sed_trimmed_data,
        grid_spacing_global, SEDIMENT_DIFFUSION
    )

    complete_grid.data[:] = final_grid
```

## Verification

**Before fix**:
- 1092m discontinuity at 34.33°E (chunk boundary)
- Visible steps in cross-section profile
- Clear artifacts in 2D bathymetry map

**After fix**:
- NO discontinuities at any chunk boundaries (0°, 8.3°, 16.7°, 25.0°, 33.3°, 41.6°E)
- Smooth profiles across entire domain
- Natural seafloor appearance

**Verification script**: [check_discontinuities.py](check_discontinuities.py)

## General Rule

⚠️ **CRITICAL**: ANY operation using `scipy.ndimage` with edge-dependent modes (`gaussian_filter`, `convolve`, `uniform_filter`, etc.) MUST be applied **globally after chunk assembly**, not per-chunk.

**Edge-dependent modes include**:
- `mode='nearest'`
- `mode='reflect'`
- `mode='mirror'`
- `mode='wrap'`
- `mode='constant'`

These modes create different edge padding for each chunk, leading to discontinuities.

## Operations Safe Per-Chunk

Operations that ARE safe to apply per-chunk:
- Thermal subsidence (point-wise, no neighbor dependencies)
- Abyssal hill generation (uses padding to handle edges)
- Simple sediment drape (point-wise subtraction)
- Parameter calculations (spreading rate, azimuth - when using global fields)

## Related Issues

This fix is separate from the earlier **bin interpolation fix** (Phase 6):
- **Bin interpolation**: Fixed discontinuities from nearest-neighbor bin lookup → trilinear interpolation
- **Diffusive sediment**: Fixed discontinuities from per-chunk smoothing → global application

Both were independently necessary for fully smooth results.

## Testing

Test the fix:
```bash
cd /Users/simon/GIT/AbFab
conda activate pygmt17

# Generate Southern Africa test case
python generate_complete_bathymetry.py

# Check for discontinuities
python check_discontinuities.py
```

Expected result: No spikes at chunk boundaries in the discontinuity analysis.

## References

- Issue discovered: January 2025
- Root cause identified: Comparison with 'unwind' branch (abyssal hills only, no complete bathymetry)
- Fix implemented: Lines 125, 377-414 in [generate_complete_bathymetry.py](generate_complete_bathymetry.py)
- Documented in: [CLAUDE.md](CLAUDE.md) Phase 7
