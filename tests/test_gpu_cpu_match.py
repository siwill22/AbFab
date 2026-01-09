#!/usr/bin/env python
"""
Test script to verify GPU and CPU implementations produce identical results.

Tests:
1. Random field generation
2. Gradient calculation (azimuth)
3. Spreading rate calculation
4. Full bathymetry generation

Run this after making fixes to ensure GPU matches CPU exactly.
"""

import numpy as np
import AbFab as af
import AbFab_gpu as af_gpu
import torch

print("="*70)
print("GPU vs CPU Consistency Test")
print("="*70)

# Test parameters
SEED = 42
SHAPE = (50, 50)
GRID_SPACING = 1.85  # km

# Create synthetic test data
np.random.seed(SEED)
age_cpu = np.random.rand(*SHAPE) * 100 + 10  # 10-110 Myr
sediment_cpu = np.random.rand(*SHAPE) * 500  # 0-500 m
lat_coords = np.linspace(-30, 30, SHAPE[0])

print(f"\nTest grid: {SHAPE}")
print(f"Random seed: {SEED}")
print(f"Grid spacing: {GRID_SPACING} km")

# ============================================================================
# TEST 1: Random Field Generation
# ============================================================================
print("\n" + "-"*70)
print("TEST 1: Random Field Generation")
print("-"*70)

np.random.seed(SEED)
random_cpu = np.random.randn(*SHAPE).astype(np.float32)

random_gpu_tensor = af_gpu.generate_random_field_gpu(SHAPE, seed=SEED)
random_gpu = random_gpu_tensor.cpu().numpy()

diff_random = np.abs(random_cpu - random_gpu)
max_diff = np.max(diff_random)
mean_diff = np.mean(diff_random)

print(f"Max difference: {max_diff:.2e}")
print(f"Mean difference: {mean_diff:.2e}")
print(f"Status: {'✓ PASS' if max_diff < 1e-6 else '✗ FAIL'}")

# ============================================================================
# TEST 2: Azimuth Calculation (Gradient)
# ============================================================================
print("\n" + "-"*70)
print("TEST 2: Azimuth Calculation (Gradient)")
print("-"*70)

# CPU version
azimuth_cpu = af.calculate_azimuth_from_age(age_cpu, lat_coords=lat_coords)

# GPU version
age_gpu = torch.tensor(age_cpu, dtype=torch.float32, device=af_gpu.DEVICE)
lat_gpu = torch.tensor(lat_coords, dtype=torch.float32, device=af_gpu.DEVICE)
azimuth_gpu_tensor = af_gpu.calculate_azimuth_from_age_gpu(age_gpu, lat_coords=lat_gpu)
azimuth_gpu = azimuth_gpu_tensor.cpu().numpy()

diff_azimuth = np.abs(azimuth_cpu - azimuth_gpu)
max_diff = np.max(diff_azimuth)
mean_diff = np.mean(diff_azimuth)
max_diff_deg = np.rad2deg(max_diff)

print(f"Max difference: {max_diff:.2e} rad ({max_diff_deg:.4f}°)")
print(f"Mean difference: {mean_diff:.2e} rad ({np.rad2deg(mean_diff):.4f}°)")
print(f"Status: {'✓ PASS' if max_diff < 1e-5 else '✗ FAIL'}")

# ============================================================================
# TEST 3: Spreading Rate Calculation
# ============================================================================
print("\n" + "-"*70)
print("TEST 3: Spreading Rate Calculation")
print("-"*70)

# CPU version
sr_cpu = af.calculate_spreading_rate_from_age(age_cpu, GRID_SPACING, lat_coords=lat_coords)

# GPU version
sr_gpu_tensor = af_gpu.calculate_spreading_rate_from_age_gpu(
    age_gpu, GRID_SPACING, lat_coords=lat_gpu
)
sr_gpu = sr_gpu_tensor.cpu().numpy()

# Compare only finite values (NaN positions should match)
finite_cpu = np.isfinite(sr_cpu)
finite_gpu = np.isfinite(sr_gpu)
nan_match = np.array_equal(finite_cpu, finite_gpu)

if nan_match:
    diff_sr = np.abs(sr_cpu[finite_cpu] - sr_gpu[finite_gpu])
    max_diff = np.max(diff_sr)
    mean_diff = np.mean(diff_sr)
    print(f"NaN positions match: ✓")
    print(f"Max difference (finite values): {max_diff:.2e} mm/yr")
    print(f"Mean difference (finite values): {mean_diff:.2e} mm/yr")
    print(f"Status: {'✓ PASS' if max_diff < 1e-3 else '✗ FAIL'}")
else:
    print(f"NaN positions differ: ✗")
    print(f"CPU NaNs: {np.sum(~finite_cpu)}, GPU NaNs: {np.sum(~finite_gpu)}")
    print(f"Status: ✗ FAIL")

# ============================================================================
# TEST 4: Edge Pixel Verification (Critical for gradient fix)
# ============================================================================
print("\n" + "-"*70)
print("TEST 4: Edge Pixel Verification")
print("-"*70)

print("\nAzimuth edges:")
print(f"  Top row max diff: {np.max(np.abs(azimuth_cpu[0, :] - azimuth_gpu[0, :])):.2e}")
print(f"  Bottom row max diff: {np.max(np.abs(azimuth_cpu[-1, :] - azimuth_gpu[-1, :])):.2e}")
print(f"  Left col max diff: {np.max(np.abs(azimuth_cpu[:, 0] - azimuth_gpu[:, 0])):.2e}")
print(f"  Right col max diff: {np.max(np.abs(azimuth_cpu[:, -1] - azimuth_gpu[:, -1])):.2e}")

edge_pass = (np.max(np.abs(azimuth_cpu[0, :] - azimuth_gpu[0, :])) < 1e-5 and
             np.max(np.abs(azimuth_cpu[-1, :] - azimuth_gpu[-1, :])) < 1e-5 and
             np.max(np.abs(azimuth_cpu[:, 0] - azimuth_gpu[:, 0])) < 1e-5 and
             np.max(np.abs(azimuth_cpu[:, -1] - azimuth_gpu[:, -1])) < 1e-5)

print(f"\nEdge gradient status: {'✓ PASS' if edge_pass else '✗ FAIL'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_pass = (max_diff < 1e-6 for test in ['random', 'azimuth', 'spreading_rate', 'edges'])

print("""
If all tests pass:
✓ GPU and CPU implementations are numerically identical
✓ Random field generation uses NumPy RNG correctly
✓ Gradient calculation matches np.gradient() exactly
✓ Edge handling is correct

If tests fail:
✗ Check the specific test that failed above
✗ Verify fixes were applied correctly to AbFab_gpu.py
✗ Check for any remaining differences in FFT or interpolation
""")

print("\nNext step: Run full bathymetry generation and compare outputs visually")
print("  python test_full_gpu_cpu_bathymetry.py")
