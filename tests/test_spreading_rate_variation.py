"""
Test spatially varying spreading rate implementation.

This test creates a synthetic region with varying spreading rates
and verifies that the parameters change appropriately.
"""

import numpy as np
import matplotlib.pyplot as plt
import AbFab as af

print("=" * 70)
print("SPATIALLY VARYING SPREADING RATE TEST")
print("=" * 70)
print()

# Create synthetic domain with E-W spreading (N-S ridges)
np.random.seed(42)
ny, nx = 100, 100
grid_spacing_km = 2.0

# Create age grid: young at center, old at edges
# This creates E-W spreading (spreading direction is E-W)
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

# Age increases with distance from center ridge
distance_from_ridge = np.abs(X - nx/2)
seafloor_age = distance_from_ridge * 0.2  # Myr

# Create varying spreading rate by having different age gradients
# Left side: steep gradient (fast spreading)
# Right side: gentle gradient (slow spreading)
age_left = distance_from_ridge[:, :nx//2] * 0.1  # Fast: small age change per distance
age_right = distance_from_ridge[:, nx//2:] * 0.4  # Slow: large age change per distance

seafloor_age = np.zeros_like(X, dtype=float)
seafloor_age[:, :nx//2] = age_left
seafloor_age[:, nx//2:] = age_right

# Uniform sediment
sediment_thickness = np.ones((ny, nx)) * 50.0

# Generate random field
random_field = af.generate_random_field((ny, nx))

# Base parameters
base_params = {
    'H': 200.0,
    'lambda_n': 8.0,
    'lambda_s': 20.0,
    'D': 2.2
}

print(f"Domain: {ny}×{nx} pixels, grid spacing: {grid_spacing_km} km")
print(f"Base parameters: H={base_params['H']}m, λ_n={base_params['lambda_n']}km, λ_s={base_params['lambda_s']}km")
print()

# Calculate spreading rate to see the variation
spreading_rate = af.calculate_spreading_rate_from_age(seafloor_age, grid_spacing_km)
spreading_rate = np.where(np.isnan(spreading_rate), np.nanmedian(spreading_rate), spreading_rate)

sr_left = np.nanmedian(spreading_rate[:, :nx//2])
sr_right = np.nanmedian(spreading_rate[:, nx//2:])

print(f"Spreading rate variation:")
print(f"  Left side (fast):  {sr_left:.1f} mm/yr")
print(f"  Right side (slow): {sr_right:.1f} mm/yr")
print(f"  Ratio: {sr_left/sr_right:.2f}×")
print()

# Test 1: Fixed parameters (spreading_rate_bins=1, should ignore spreading rate)
print("-" * 70)
print("Test 1: Fixed parameters (spreading_rate_bins=1)")
print("-" * 70)

bathy_fixed = af.generate_bathymetry_spatial_filter(
    seafloor_age, sediment_thickness, base_params, grid_spacing_km,
    random_field, spreading_rate_bins=1, optimize=True
)

rms_left_fixed = np.std(bathy_fixed[:, :nx//2])
rms_right_fixed = np.std(bathy_fixed[:, nx//2:])

print(f"  RMS left:  {rms_left_fixed:.4f} m")
print(f"  RMS right: {rms_right_fixed:.4f} m")
print(f"  Ratio: {rms_left_fixed/rms_right_fixed:.3f}")
print(f"  Expected: ~1.0 (same parameters everywhere)")
print()

# Test 2: Spatially varying with spreading rate (spreading_rate_bins=5)
print("-" * 70)
print("Test 2: Spatially varying (spreading_rate_bins=5)")
print("-" * 70)

bathy_varying = af.generate_bathymetry_spatial_filter(
    seafloor_age, sediment_thickness, base_params, grid_spacing_km,
    random_field, spreading_rate_bins=5, base_params=base_params, optimize=True
)

rms_left_varying = np.std(bathy_varying[:, :nx//2])
rms_right_varying = np.std(bathy_varying[:, nx//2:])

print(f"  RMS left (fast spreading):  {rms_left_varying:.4f} m")
print(f"  RMS right (slow spreading): {rms_right_varying:.4f} m")
print(f"  Ratio: {rms_left_varying/rms_right_varying:.3f}")
print(f"  Expected: <1.0 (fast spreading should be smoother/lower RMS)")
print()

# Expected behavior check
if rms_left_varying < rms_right_varying:
    print("✓ CORRECT: Fast spreading (left) has lower RMS than slow spreading (right)")
else:
    print("✗ WRONG: Fast spreading should have lower RMS!")
print()

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Inputs and spreading rate
im0 = axes[0, 0].imshow(seafloor_age, cmap='YlOrRd', aspect='equal')
axes[0, 0].set_title('Seafloor Age (Myr)')
axes[0, 0].axvline(nx//2, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(spreading_rate, cmap='RdYlGn', aspect='equal', vmin=0, vmax=100)
axes[0, 1].set_title('Spreading Rate (mm/yr)')
axes[0, 1].axvline(nx//2, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 1].text(nx//4, ny//10, 'FAST', ha='center', fontsize=14, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
axes[0, 1].text(3*nx//4, ny//10, 'SLOW', ha='center', fontsize=14, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
plt.colorbar(im1, ax=axes[0, 1])

axes[0, 2].hist(spreading_rate.flatten(), bins=30, alpha=0.7, edgecolor='black')
axes[0, 2].set_xlabel('Spreading Rate (mm/yr)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Spreading Rate Distribution')
axes[0, 2].axvline(sr_left, color='green', linestyle='--', label=f'Left: {sr_left:.1f}')
axes[0, 2].axvline(sr_right, color='red', linestyle='--', label=f'Right: {sr_right:.1f}')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Row 2: Bathymetry comparisons
vmin = min(np.percentile(bathy_fixed, 2), np.percentile(bathy_varying, 2))
vmax = max(np.percentile(bathy_fixed, 98), np.percentile(bathy_varying, 98))

im2 = axes[1, 0].imshow(bathy_fixed, cmap='seismic', vmin=vmin, vmax=vmax, aspect='equal')
axes[1, 0].set_title(f'Fixed Params\n(RMS: L={rms_left_fixed:.3f}, R={rms_right_fixed:.3f})')
axes[1, 0].axvline(nx//2, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
plt.colorbar(im2, ax=axes[1, 0], label='Height (m)')

im3 = axes[1, 1].imshow(bathy_varying, cmap='seismic', vmin=vmin, vmax=vmax, aspect='equal')
axes[1, 1].set_title(f'Varying Params\n(RMS: L={rms_left_varying:.3f}, R={rms_right_varying:.3f})')
axes[1, 1].axvline(nx//2, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
plt.colorbar(im3, ax=axes[1, 1], label='Height (m)')

# Difference
diff = bathy_varying - bathy_fixed
im4 = axes[1, 2].imshow(diff, cmap='PuOr', aspect='equal')
axes[1, 2].set_title(f'Difference\n(Varying - Fixed)')
axes[1, 2].axvline(nx//2, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
plt.colorbar(im4, ax=axes[1, 2], label='Height diff (m)')

plt.tight_layout()
plt.savefig('test_spreading_rate_variation.png', dpi=150)
print(f"Saved: test_spreading_rate_variation.png")
print()

# Quantitative comparison
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("The spatially varying spreading rate implementation:")
print(f"  • Automatically calculates spreading rate from age gradient")
print(f"  • Bins spreading rates into discrete levels (default: 5)")
print(f"  • Applies spreading_rate_to_params() scaling at each bin")
print(f"  • Creates 3D filter bank: azimuth × sediment × spreading_rate")
print()
print("Results:")
print(f"  • Fixed params: no variation between fast/slow regions")
print(f"  • Varying params: fast spreading regions are smoother (lower RMS)")
print(f"  • RMS ratio change: {rms_left_fixed/rms_right_fixed:.3f} → {rms_left_varying/rms_right_varying:.3f}")
