#!/usr/bin/env python
"""
AbFab GPU vs CPU Benchmark

Compares performance of GPU-accelerated vs CPU bathymetry generation
across different grid sizes.

Usage:
    python benchmark_gpu_vs_cpu.py [--full]
    
    --full: Run extended benchmark with larger grids (requires more time)

Requirements:
    pip install torch>=2.0 matplotlib
"""

import numpy as np
import time
import sys

# Check for GPU availability first
try:
    import torch
    MPS_AVAILABLE = torch.backends.mps.is_available()
    if MPS_AVAILABLE:
        print(f"✓ Apple MPS (GPU) available")
        print(f"  PyTorch version: {torch.__version__}")
    else:
        print("✗ MPS not available, GPU tests will use CPU fallback")
except ImportError:
    print("✗ PyTorch not installed")
    MPS_AVAILABLE = False

import AbFab as af
import AbFab_gpu as af_gpu


def generate_test_data(shape, seed=42):
    """Generate test data for benchmarking."""
    np.random.seed(seed)
    ny, nx = shape
    
    # Create realistic-ish test data
    x = np.linspace(0, 100, nx)
    y = np.linspace(-40, -20, ny)
    xx, yy = np.meshgrid(x, y)
    
    # Age increases away from a synthetic ridge
    age = 10 + np.abs(xx - 50) * 1.5 + np.random.randn(ny, nx) * 2
    age = np.clip(age, 1, 150).astype(np.float32)
    
    # Sediment thickness
    sediment = 50 + age * 2 + np.random.randn(ny, nx) * 20
    sediment = np.clip(sediment, 1, 500).astype(np.float32)
    
    # Random field
    random_field = np.random.randn(ny, nx).astype(np.float32)
    
    # Latitude coordinates
    lat_coords = y.astype(np.float32)
    
    return age, sediment, random_field, lat_coords


def benchmark_cpu(age, sediment, random_field, params, grid_spacing_km,
                  azimuth_bins=36, sediment_bins=10, spreading_rate_bins=20,
                  warmup=True):
    """Benchmark CPU implementation."""
    if warmup:
        # Warmup run
        _ = af.generate_bathymetry_spatial_filter(
            age[:100, :100] if age.shape[0] > 100 else age,
            sediment[:100, :100] if sediment.shape[0] > 100 else sediment,
            params,
            grid_spacing_km,
            random_field[:100, :100] if random_field.shape[0] > 100 else random_field,
            optimize=True,
            azimuth_bins=18,
            sediment_bins=5,
            spreading_rate_bins=5
        )
    
    t_start = time.time()
    result = af.generate_bathymetry_spatial_filter(
        age, sediment, params, grid_spacing_km, random_field,
        optimize=True,
        azimuth_bins=azimuth_bins,
        sediment_bins=sediment_bins,
        spreading_rate_bins=spreading_rate_bins
    )
    t_end = time.time()
    
    return result, t_end - t_start


def benchmark_gpu(age, sediment, random_field, lat_coords, params, grid_spacing_km,
                  azimuth_bins=36, sediment_bins=10, spreading_rate_bins=20,
                  tile_size=200, warmup=True):
    """Benchmark GPU implementation."""
    if warmup:
        # Warmup run (important for MPS)
        _ = af_gpu.generate_bathymetry_gpu(
            age[:100, :100] if age.shape[0] > 100 else age,
            sediment[:100, :100] if sediment.shape[0] > 100 else sediment,
            params,
            grid_spacing_km,
            random_field[:100, :100] if random_field.shape[0] > 100 else random_field,
            lat_coords[:100] if len(lat_coords) > 100 else lat_coords,
            azimuth_bins=18,
            sediment_bins=5,
            spreading_rate_bins=5,
            tile_size=100,
            verbose=False
        )
    
    # Synchronize before timing
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    t_start = time.time()
    result = af_gpu.generate_bathymetry_gpu(
        age, sediment, params, grid_spacing_km, random_field, lat_coords,
        azimuth_bins=azimuth_bins,
        sediment_bins=sediment_bins,
        spreading_rate_bins=spreading_rate_bins,
        tile_size=tile_size,
        verbose=False
    )
    
    # Synchronize after computation
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    t_end = time.time()
    
    return result, t_end - t_start


def run_benchmark(sizes, params, grid_spacing_km=5.0, n_runs=3):
    """
    Run benchmark across multiple grid sizes.
    
    Returns dict with results.
    """
    results = {
        'sizes': sizes,
        'cpu_times': [],
        'gpu_times': [],
        'speedups': [],
        'pixels': [],
        'errors': []
    }
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Grid size: {size}×{size} = {size*size:,} pixels")
        print('='*60)
        
        # Generate test data
        age, sediment, random_field, lat_coords = generate_test_data((size, size))
        
        cpu_times = []
        gpu_times = []
        
        for run in range(n_runs):
            print(f"\nRun {run+1}/{n_runs}:")
            
            # CPU benchmark
            warmup = (run == 0)
            _, cpu_time = benchmark_cpu(
                age, sediment, random_field, params, grid_spacing_km,
                warmup=warmup
            )
            cpu_times.append(cpu_time)
            print(f"  CPU: {cpu_time:.2f}s")
            
            # GPU benchmark - use tile_size proportional to grid size
            tile_sz = min(size, 300)  # Cap at 300 for memory
            _, gpu_time = benchmark_gpu(
                age, sediment, random_field, lat_coords, params, grid_spacing_km,
                tile_size=tile_sz,
                warmup=warmup
            )
            gpu_times.append(gpu_time)
            print(f"  GPU: {gpu_time:.2f}s")
        
        # Calculate statistics
        cpu_mean = np.mean(cpu_times)
        gpu_mean = np.mean(gpu_times)
        speedup = cpu_mean / gpu_mean
        
        # Verify results match (using fresh data to ensure same random seed effect)
        age, sediment, random_field, lat_coords = generate_test_data((size, size))
        cpu_result, _ = benchmark_cpu(age, sediment, random_field, params, grid_spacing_km, warmup=False)
        tile_sz = min(size, 300)
        gpu_result, _ = benchmark_gpu(age, sediment, random_field, lat_coords, params, grid_spacing_km, tile_size=tile_sz, warmup=False)
        
        # Calculate error
        valid_mask = ~np.isnan(cpu_result) & ~np.isnan(gpu_result)
        if np.any(valid_mask):
            mae = np.mean(np.abs(cpu_result[valid_mask] - gpu_result[valid_mask]))
            rmse = np.sqrt(np.mean((cpu_result[valid_mask] - gpu_result[valid_mask])**2))
        else:
            mae, rmse = np.nan, np.nan
        
        results['cpu_times'].append(cpu_mean)
        results['gpu_times'].append(gpu_mean)
        results['speedups'].append(speedup)
        results['pixels'].append(size * size)
        results['errors'].append({'mae': mae, 'rmse': rmse})
        
        print(f"\nResults for {size}×{size}:")
        print(f"  CPU mean: {cpu_mean:.2f}s")
        print(f"  GPU mean: {gpu_mean:.2f}s")
        print(f"  Speedup: {speedup:.1f}×")
        print(f"  Error (MAE): {mae:.4f}m")
    
    return results


def plot_results(results, output_file='benchmark_results.png'):
    """Plot benchmark results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sizes = results['sizes']
    pixels = [s*s for s in sizes]
    
    # Plot 1: Execution time
    ax = axes[0]
    ax.plot(sizes, results['cpu_times'], 'bo-', label='CPU', linewidth=2, markersize=8)
    ax.plot(sizes, results['gpu_times'], 'ro-', label='GPU', linewidth=2, markersize=8)
    ax.set_xlabel('Grid Size (pixels per side)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Speedup
    ax = axes[1]
    ax.bar(range(len(sizes)), results['speedups'], color='green', alpha=0.7)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel('Grid Size (pixels per side)', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('GPU Speedup Factor', fontsize=14, fontweight='bold')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup labels on bars
    for i, v in enumerate(results['speedups']):
        ax.text(i, v + 0.1, f'{v:.1f}×', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Throughput
    ax = axes[2]
    cpu_throughput = [p/t for p, t in zip(pixels, results['cpu_times'])]
    gpu_throughput = [p/t for p, t in zip(pixels, results['gpu_times'])]
    
    ax.plot(sizes, [t/1000 for t in cpu_throughput], 'bo-', label='CPU', linewidth=2, markersize=8)
    ax.plot(sizes, [t/1000 for t in gpu_throughput], 'ro-', label='GPU', linewidth=2, markersize=8)
    ax.set_xlabel('Grid Size (pixels per side)', fontsize=12)
    ax.set_ylabel('Throughput (kilo-pixels/second)', fontsize=12)
    ax.set_title('Processing Throughput', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")
    plt.close()


def main():
    print("="*70)
    print("AbFab GPU vs CPU Benchmark")
    print("="*70)
    
    # Check for --full flag
    full_benchmark = '--full' in sys.argv
    
    # Parameters
    params = {
        'H': 250.0,
        'lambda_n': 3.0,
        'lambda_s': 30.0,
        'D': 2.2
    }
    
    grid_spacing_km = 5.0
    
    if full_benchmark:
        print("\nRunning FULL benchmark (this may take 10-20 minutes)...")
        sizes = [100, 200, 300, 500, 750, 1000, 1500]
        n_runs = 3
    else:
        print("\nRunning QUICK benchmark...")
        print("(Use --full for extended benchmark with larger grids)")
        sizes = [100, 200, 300, 500]
        n_runs = 2
    
    print(f"\nGrid sizes to test: {sizes}")
    print(f"Runs per size: {n_runs}")
    print(f"Parameters: H={params['H']}m, λ_n={params['lambda_n']}km, λ_s={params['lambda_s']}km")
    
    # Run benchmark
    results = run_benchmark(sizes, params, grid_spacing_km, n_runs)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n{:>10} {:>12} {:>12} {:>10} {:>12}".format(
        'Size', 'CPU (s)', 'GPU (s)', 'Speedup', 'Throughput'))
    print("-"*60)
    
    for i, size in enumerate(results['sizes']):
        pixels = size * size
        gpu_throughput = pixels / results['gpu_times'][i] / 1000
        print("{:>10} {:>12.2f} {:>12.2f} {:>10.1f}× {:>10.0f} kpx/s".format(
            f"{size}×{size}",
            results['cpu_times'][i],
            results['gpu_times'][i],
            results['speedups'][i],
            gpu_throughput
        ))
    
    # Average speedup
    avg_speedup = np.mean(results['speedups'])
    print("-"*60)
    print(f"Average speedup: {avg_speedup:.1f}×")
    
    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
    
    # Memory estimate for large grids
    print("\n" + "="*70)
    print("MEMORY ESTIMATES FOR LARGE GRIDS")
    print("="*70)
    
    # Memory per filter bank: n_az × n_sed × n_sr × filter_size² × 4 bytes
    # Plus convolved results: n_filters × ny × nx × 4 bytes
    n_filters = 36 * 10 * 20
    filter_size = 25
    filter_bank_mb = n_filters * filter_size * filter_size * 4 / 1e6
    
    print(f"\nFilter bank memory: {filter_bank_mb:.1f} MB ({n_filters} filters)")
    
    large_sizes = [2000, 3000, 4000, 5000]
    print(f"\nConvolved results memory estimate:")
    for s in large_sizes:
        conv_mb = n_filters * s * s * 4 / 1e6
        total_mb = filter_bank_mb + conv_mb
        print(f"  {s}×{s}: {conv_mb/1000:.1f} GB convolved → {total_mb/1000:.1f} GB total")
    
    print("\nWith 96GB unified memory, grids up to ~4000×4000 should fit entirely in GPU memory")
    print("For larger grids, consider chunked processing")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
