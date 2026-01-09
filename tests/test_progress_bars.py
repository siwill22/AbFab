#!/usr/bin/env python
"""
Quick test to verify tqdm progress bars are working in generate_complete_bathymetry.py
"""

import subprocess
import sys

# Create a test configuration with very small region for fast testing
test_config = """
# Minimal test configuration - small region
XMIN, XMAX = 0, 20
YMIN, YMAX = -30, -20
SPACING = '5m'  # Coarse for speed
CHUNKSIZE = 50
NUM_CPUS = 2
SEDIMENT_MODE = 'fill'  # Test diffusive infill with progress
"""

print("Testing progress bar implementation...")
print("Running generate_complete_bathymetry.py with small test region")
print("Expected progress bars:")
print("  1. 'Processing chunks' - for parallel chunk processing")
print("  2. 'Resampling sediment' - when applying diffusive infill")
print("  3. 'Diffusion iteration' messages - during global diffusive infill")
print("\n" + "="*70 + "\n")

# Note: This will run the actual script with current configuration
# If you want to test with custom config, modify generate_complete_bathymetry.py parameters first

result = subprocess.run(
    ["conda", "run", "-n", "pygmt17", "python", "generate_complete_bathymetry.py"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("Progress bars should have been visible above.")
else:
    print("\n" + "="*70)
    print("✗ Test failed with return code:", result.returncode)
    sys.exit(1)
