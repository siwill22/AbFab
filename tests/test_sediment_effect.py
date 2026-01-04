"""
Test sediment modification effect.
"""

import numpy as np
import AbFab as af

# Test with different H values and same sediment
sediment = 100.0

print("Sediment modification test (sediment = 100m):")
print("=" * 60)

for spreading_rate in [10, 40, 100]:
    params = af.spreading_rate_to_params(spreading_rate)
    H = params['H']
    lambda_n = params['lambda_n']
    lambda_s = params['lambda_s']

    H_sed, ln_sed, ls_sed = af.modify_by_sediment(H, lambda_n, lambda_s, sediment)

    print(f"\n{spreading_rate} mm/yr:")
    print(f"  Original: H={H:.1f}m, λ_n={lambda_n:.2f}km, λ_s={lambda_s:.2f}km")
    print(f"  Modified: H={H_sed:.1f}m, λ_n={ln_sed:.2f}km, λ_s={ls_sed:.2f}km")
    print(f"  Reduction: H={H-H_sed:.1f}m, λ_n={lambda_n-ln_sed:.2f}km ({(lambda_n-ln_sed)/lambda_n*100:.1f}%)")

print()
print("Expected: Sediment should reduce H and λ uniformly across spreading rates")
print("The percentage reduction in λ should be similar for all spreading rates")
