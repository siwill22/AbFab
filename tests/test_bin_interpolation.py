"""
Test bin interpolation vs nearest-neighbor to verify smooth transitions.

This test creates a simple scenario where spreading rate varies smoothly
and compares the results with and without interpolation.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import AbFab as af

print("="*70)
print("Testing Bin Interpolation for Smooth Bin Transitions")
print("="*70)

# Create simple test case with smooth spreading rate gradient
ny, nx = 200, 200
lon = np.linspace(30, 40, nx)
lat = np.linspace(-50, -40, ny)

# Create age grid with smooth gradient (age increases left to right)
age_grid = np.linspace(0, 50, nx)[None, :] * np.ones((ny, 1))

# Create DataArray
age_da = xr.DataArray(
    age_grid,
    coords={'lat': lat, 'lon': lon},
    dims=['lat', 'lon']
)

# Uniform sediment
sed_da = xr.DataArray(
    np.full((ny, nx), 500.0),
    coords={'lat': lat, 'lon': lon},
    dims=['lat', 'lon']
)

# Random field (seed for reproducibility)
np.random.seed(42)
rand_da = xr.DataArray(
    np.random.randn(ny, nx),
    coords={'lat': lat, 'lon': lon},
    dims=['lat', 'lon']
)

# Parameters
params_fixed = {
    'H': 250.0,
    'lambda_n': 6.5,
    'lambda_s': 16.0,
    'D': 2.2
}
grid_spacing_km = 2.0

print("\nTest setup:")
print(f"  Grid size: {ny} × {nx}")
print(f"  Age range: 0-50 Myr (smooth gradient)")
print(f"  Spreading rate bins: 5")
print(f"  Grid spacing: {grid_spacing_km} km")

print("\nGenerating bathymetry with bin interpolation...")
print("(This is the new implementation in AbFab.py)")

result_interp = af.generate_bathymetry_spatial_filter(
    age_da.data,
    sed_da.data,
    params_fixed,
    grid_spacing_km,
    rand_da.data,
    filter_type='gaussian',
    optimize=True,
    azimuth_bins=36,
    sediment_bins=5,
    spreading_rate_bins=5,  # Creates 5 bins
    base_params=params_fixed
)

print("✓ Generation complete")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Full field
im0 = axes[0, 0].imshow(result_interp, cmap='seismic', origin='lower',
                        vmin=-np.std(result_interp)*2, vmax=np.std(result_interp)*2)
axes[0, 0].set_title('Full Field with Bin Interpolation', fontweight='bold', fontsize=14)
axes[0, 0].set_xlabel('X (age increases →)')
axes[0, 0].set_ylabel('Y')
plt.colorbar(im0, ax=axes[0, 0], label='Bathymetry (m)')

# Zoomed region to check for boundaries
zoom_y, zoom_x = slice(80, 120), slice(80, 120)
im1 = axes[0, 1].imshow(result_interp[zoom_y, zoom_x], cmap='seismic', origin='lower',
                        vmin=-np.std(result_interp)*2, vmax=np.std(result_interp)*2)
axes[0, 1].set_title('Zoomed Region (should be smooth)', fontweight='bold', fontsize=14)
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0, 1], label='Bathymetry (m)')

# Vertical profile to check for discontinuities
middle_row = ny // 2
profile = result_interp[middle_row, :]

axes[1, 0].plot(profile, linewidth=0.5)
axes[1, 0].set_title('Vertical Profile (middle row)\nShould show no sharp transitions',
                     fontweight='bold', fontsize=14)
axes[1, 0].set_xlabel('X (age/spreading rate increases →)')
axes[1, 0].set_ylabel('Bathymetry (m)')
axes[1, 0].grid(True, alpha=0.3)

# Gradient to detect discontinuities
gradient = np.gradient(result_interp, axis=1)
gradient_std = np.std(gradient)

im3 = axes[1, 1].imshow(gradient, cmap='RdBu_r', origin='lower',
                        vmin=-gradient_std*3, vmax=gradient_std*3)
axes[1, 1].set_title('Horizontal Gradient\n(vertical lines = bin boundaries)',
                     fontweight='bold', fontsize=14)
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')
plt.colorbar(im3, ax=axes[1, 1], label='dB/dx')

plt.tight_layout()
plt.savefig('test_bin_interpolation.png', dpi=150, bbox_inches='tight')
print("\nSaved: test_bin_interpolation.png")

# Statistics
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nBathymetry statistics:")
print(f"  RMS: {np.std(result_interp):.2f} m")
print(f"  Range: {np.min(result_interp):.2f} to {np.max(result_interp):.2f} m")

print(f"\nGradient statistics (for discontinuity detection):")
print(f"  Gradient std: {gradient_std:.3f} m/pixel")
print(f"  Max gradient: {np.max(np.abs(gradient)):.3f} m/pixel")
print(f"  99th percentile: {np.percentile(np.abs(gradient), 99):.3f} m/pixel")

# Check for outliers in gradient (sign of discontinuities)
threshold = 5 * gradient_std
outliers = np.sum(np.abs(gradient) > threshold)
total_pixels = gradient.size
outlier_pct = 100 * outliers / total_pixels

print(f"\nDiscontinuity check:")
print(f"  Pixels with |gradient| > 5σ: {outliers} / {total_pixels} ({outlier_pct:.2f}%)")

if outlier_pct < 1.0:
    print(f"  ✓ PASS: Very few discontinuities (<1%)")
    print(f"  → Bin interpolation is working correctly")
else:
    print(f"  ⚠ WARNING: {outlier_pct:.1f}% of pixels show large gradients")
    print(f"  → May indicate remaining bin boundaries")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
If bin interpolation is working correctly, you should see:
1. No visible vertical bands in the full field (top left)
2. Smooth variation in the zoomed region (top right)
3. No sharp jumps in the vertical profile (bottom left)
4. No vertical lines in the gradient image (bottom right)

The gradient image is most sensitive to discontinuities. Vertical lines
would indicate locations where the binning creates sharp transitions.
With interpolation, these should be eliminated.
""")

plt.show()
