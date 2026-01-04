"""
Test the new spreading rate scaling functionality that applies scaling
to user-defined base parameters rather than replacing them.
"""

import numpy as np
import AbFab as af

print("=" * 70)
print("SPREADING RATE SCALING TEST")
print("=" * 70)
print()

# Test 1: Default base parameters
print("Test 1: Default base parameters (no base_params argument)")
print("-" * 70)

spreading_rates = [10, 50, 100]

print(f"{'u (mm/yr)':<12} {'H (m)':<12} {'λ_n (km)':<12} {'λ_s (km)':<12} {'D':<8}")
print("-" * 70)

for u in spreading_rates:
    params = af.spreading_rate_to_params(u)
    print(f"{u:<12} {params['H']:<12.1f} {params['lambda_n']:<12.2f} "
          f"{params['lambda_s']:<12.2f} {params['D']:<8.3f}")

print()
print("Note: At u=50 mm/yr, values should be close to defaults")
print("      (H≈200m, λ_n≈6.5km, λ_s≈16km)")
print()

# Test 2: Custom base parameters (small features)
print("=" * 70)
print("Test 2: Custom base parameters - SMALL features")
print("-" * 70)

base_small = {
    'H': 150.0,      # Smaller amplitude
    'lambda_n': 3.0,  # Narrower width
    'lambda_s': 8.0,  # Shorter length
    'D': 2.3
}

print(f"Base parameters: H={base_small['H']}m, λ_n={base_small['lambda_n']}km, "
      f"λ_s={base_small['lambda_s']}km, D={base_small['D']}")
print()
print(f"{'u (mm/yr)':<12} {'H (m)':<12} {'λ_n (km)':<12} {'λ_s (km)':<12} {'D':<8}")
print("-" * 70)

for u in spreading_rates:
    params = af.spreading_rate_to_params(u, base_params=base_small)
    print(f"{u:<12} {params['H']:<12.1f} {params['lambda_n']:<12.2f} "
          f"{params['lambda_s']:<12.2f} {params['D']:<8.3f}")

print()
print("Note: Features stay SMALL but still show spreading rate trends")
print("      Fast spreading (100 mm/yr) has larger λ and smaller H than slow")
print()

# Test 3: Custom base parameters (large features)
print("=" * 70)
print("Test 3: Custom base parameters - LARGE features")
print("-" * 70)

base_large = {
    'H': 300.0,       # Larger amplitude
    'lambda_n': 12.0,  # Wider width
    'lambda_s': 30.0,  # Longer length
    'D': 2.1
}

print(f"Base parameters: H={base_large['H']}m, λ_n={base_large['lambda_n']}km, "
      f"λ_s={base_large['lambda_s']}km, D={base_large['D']}")
print()
print(f"{'u (mm/yr)':<12} {'H (m)':<12} {'λ_n (km)':<12} {'λ_s (km)':<12} {'D':<8}")
print("-" * 70)

for u in spreading_rates:
    params = af.spreading_rate_to_params(u, base_params=base_large)
    print(f"{u:<12} {params['H']:<12.1f} {params['lambda_n']:<12.2f} "
          f"{params['lambda_s']:<12.2f} {params['D']:<8.3f}")

print()
print("Note: Features stay LARGE but still show spreading rate trends")
print("      Fast spreading (100 mm/yr) has larger λ and smaller H than slow")
print()

# Test 4: Verify scaling factors are consistent
print("=" * 70)
print("Test 4: Verify scaling factors applied correctly")
print("-" * 70)

# Test at u=50 mm/yr (reference rate where f_lambda ≈ 1.0)
u_ref = 50.0
params_ref_small = af.spreading_rate_to_params(u_ref, base_params=base_small)
params_ref_large = af.spreading_rate_to_params(u_ref, base_params=base_large)

print(f"At reference rate u={u_ref} mm/yr:")
print(f"  Small base: λ_n = {params_ref_small['lambda_n']:.2f} km (expect ≈ {base_small['lambda_n']:.1f})")
print(f"  Large base: λ_n = {params_ref_large['lambda_n']:.2f} km (expect ≈ {base_large['lambda_n']:.1f})")
print()

# Check ratio is preserved
ratio_base = base_large['lambda_n'] / base_small['lambda_n']
ratio_derived = params_ref_large['lambda_n'] / params_ref_small['lambda_n']
print(f"Ratio (large/small):")
print(f"  Base parameters: {ratio_base:.2f}")
print(f"  At u={u_ref}: {ratio_derived:.2f}")
print(f"  Match: {'✓ YES' if abs(ratio_derived - ratio_base) < 0.1 else '✗ NO'}")
print()

# Test 5: Verify trends are still correct
print("=" * 70)
print("Test 5: Verify spreading rate trends (fast vs slow)")
print("-" * 70)

params_slow = af.spreading_rate_to_params(10, base_params=base_small)
params_fast = af.spreading_rate_to_params(100, base_params=base_small)

print(f"Using small base parameters:")
print(f"  Slow (10 mm/yr):  H={params_slow['H']:.1f}m, λ_n={params_slow['lambda_n']:.2f}km")
print(f"  Fast (100 mm/yr): H={params_fast['H']:.1f}m, λ_n={params_fast['lambda_n']:.2f}km")
print()

H_trend = params_fast['H'] < params_slow['H']
lambda_trend = params_fast['lambda_n'] > params_slow['lambda_n']

print(f"Expected trends:")
print(f"  Fast H < Slow H:      {H_trend} {'✓' if H_trend else '✗'}")
print(f"  Fast λ > Slow λ:      {lambda_trend} {'✓' if lambda_trend else '✗'}")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("The new spreading_rate_to_params() function now:")
print("  1. Accepts optional base_params to control baseline characteristics")
print("  2. Applies spreading rate scaling factors to these base values")
print("  3. Preserves relative differences in base parameters")
print("  4. Maintains correct physical trends (fast → smoother)")
print()
print("Usage:")
print("  # Default base parameters:")
print("  params = spreading_rate_to_params(spreading_rate)")
print()
print("  # Custom base parameters:")
print("  base = {'H': 200, 'lambda_n': 5, 'lambda_s': 15, 'D': 2.2}")
print("  params = spreading_rate_to_params(spreading_rate, base_params=base)")
