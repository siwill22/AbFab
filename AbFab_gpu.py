"""
AbFab GPU-Accelerated Version

GPU-accelerated synthetic abyssal hill bathymetry generation using PyTorch MPS
backend for Apple Silicon (M2/M3/M4 Macs).

Key optimizations:
1. Batched FFT-based convolution (all filters at once)
2. GPU-resident trilinear interpolation
3. Unified memory utilization (no CPU-GPU transfers for large grids)

Requirements:
    pip install torch  # PyTorch with MPS support (2.0+)

Usage:
    python generate_complete_bathymetry_gpu.py [config.yaml]

Author: AbFab project
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import time

# Check for MPS availability
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    DEVICE = torch.device("mps")
    print(f"GPU: Apple Silicon MPS backend available")
else:
    DEVICE = torch.device("cpu")
    print(f"WARNING: MPS not available, falling back to CPU")


# ============================================================================
# GPU UTILITY FUNCTIONS
# ============================================================================

def to_torch(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor on GPU."""
    return torch.tensor(arr, dtype=dtype, device=DEVICE)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.cpu().numpy()


# ============================================================================
# GPU-ACCELERATED CORE FUNCTIONS
# ============================================================================

def generate_random_field_gpu(shape: Tuple[int, int], seed: Optional[int] = None) -> torch.Tensor:
    """
    Generate random field using NumPy for consistency with CPU version.

    CRITICAL: This must use NumPy's RNG, not PyTorch's, to ensure identical
    results between CPU and GPU versions. NumPy and PyTorch RNGs produce
    different sequences even with the same seed!

    Parameters
    ----------
    shape : tuple
        (ny, nx) shape of random field
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    torch.Tensor
        Random field on GPU device
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate on CPU using NumPy (matches CPU version exactly)
    random_field_cpu = np.random.randn(shape[0], shape[1]).astype(np.float32)

    # Transfer to GPU
    return torch.tensor(random_field_cpu, dtype=torch.float32, device=DEVICE)


def numpy_gradient_gpu(f: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Replicate NumPy's gradient() behavior exactly on GPU.

    NumPy uses:
    - Forward difference at first point: (f[1] - f[0])
    - Central differences for interior: (f[i+1] - f[i-1]) / 2
    - Backward difference at last point: (f[-1] - f[-2])

    This ensures GPU and CPU gradients are identical.

    Parameters
    ----------
    f : torch.Tensor
        Input tensor
    axis : int
        Axis along which to compute gradient (0=rows, 1=columns)

    Returns
    -------
    torch.Tensor
        Gradient along specified axis
    """
    if axis == 0:  # Gradient along rows (y-direction)
        grad = torch.zeros_like(f)
        # First row: forward difference
        grad[0, :] = f[1, :] - f[0, :]
        # Interior: central differences
        if f.shape[0] > 2:
            grad[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0
        # Last row: backward difference
        grad[-1, :] = f[-1, :] - f[-2, :]
        return grad

    elif axis == 1:  # Gradient along columns (x-direction)
        grad = torch.zeros_like(f)
        # First column: forward difference
        grad[:, 0] = f[:, 1] - f[:, 0]
        # Interior: central differences
        if f.shape[1] > 2:
            grad[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0
        # Last column: backward difference
        grad[:, -1] = f[:, -1] - f[:, -2]
        return grad

    else:
        raise ValueError(f"Invalid axis: {axis}")


def calculate_azimuth_from_age_gpu(seafloor_age: torch.Tensor,
                                    lat_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate azimuth from seafloor age gradient on GPU.

    Uses numpy.gradient()-compatible gradient calculation to ensure
    identical results with CPU version.

    Parameters
    ----------
    seafloor_age : torch.Tensor
        2D tensor of seafloor ages (Myr)
    lat_coords : torch.Tensor, optional
        1D tensor of latitude coordinates for spherical correction

    Returns
    -------
    torch.Tensor
        Azimuth in radians (spreading direction)
    """
    # Compute gradients using NumPy-compatible method
    grad_y = numpy_gradient_gpu(seafloor_age, axis=0)  # d/dy (lat direction)
    grad_x = numpy_gradient_gpu(seafloor_age, axis=1)  # d/dx (lon direction)

    # Apply spherical correction if latitude provided
    if lat_coords is not None:
        lat_2d = lat_coords.unsqueeze(1).expand_as(grad_x)
        cos_lat = torch.cos(lat_2d * np.pi / 180.0)
        cos_lat = torch.clamp(cos_lat, min=1e-10)
        grad_x = grad_x / cos_lat

    # Azimuth from gradient direction
    azimuth = torch.atan2(grad_y, grad_x)

    return azimuth


def calculate_spreading_rate_from_age_gpu(seafloor_age: torch.Tensor,
                                           grid_spacing_km: float,
                                           lat_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate half-spreading rate from age gradient on GPU.

    Uses numpy.gradient()-compatible gradient calculation to ensure
    identical results with CPU version.

    Returns spreading rate in mm/yr.
    """
    # Compute gradients using NumPy-compatible method
    grad_y = numpy_gradient_gpu(seafloor_age, axis=0)  # d/dy (Myr per grid cell)
    grad_x = numpy_gradient_gpu(seafloor_age, axis=1)  # d/dx (Myr per grid cell)

    # Apply spherical correction
    if lat_coords is not None:
        lat_2d = lat_coords.unsqueeze(1).expand_as(grad_x)
        cos_lat = torch.cos(lat_2d * np.pi / 180.0)
        cos_lat = torch.clamp(cos_lat, min=1e-10)

        dx_km = grid_spacing_km * cos_lat
        dy_km = grid_spacing_km

        grad_x_per_km = grad_x / dx_km
        grad_y_per_km = grad_y / dy_km

        age_gradient_magnitude = torch.sqrt(grad_x_per_km**2 + grad_y_per_km**2)
    else:
        age_gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2) / grid_spacing_km

    # Avoid division by zero
    age_gradient_magnitude = torch.where(
        age_gradient_magnitude > 1e-10,
        age_gradient_magnitude,
        torch.tensor(float('nan'), device=DEVICE)
    )

    # Spreading rate: km/Myr = mm/yr
    spreading_rate = 1.0 / age_gradient_magnitude

    return spreading_rate


def spreading_rate_to_params_gpu(spreading_rate: torch.Tensor,
                                  base_params: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """
    Apply spreading rate scaling to parameters on GPU.
    
    Returns tensors of spatially-varying parameters.
    """
    H_base = base_params['H']
    lambda_n_base = base_params['lambda_n']
    lambda_s_base = base_params['lambda_s']
    D_base = base_params['D']
    
    u_ref = 50.0
    
    # Scaling factors (vectorized on GPU)
    f_H = torch.clamp(1.3 - 0.01 * spreading_rate, 0.2, 1.5)
    f_lambda = torch.clamp(0.3 + 0.014 * spreading_rate, 0.3, 2.0)
    delta_D = torch.clamp(-0.003 * (spreading_rate - u_ref), -0.3, 0.3)
    
    return {
        'H': torch.clamp(H_base * f_H, 10, 500),
        'lambda_n': torch.clamp(lambda_n_base * f_lambda, 0.5, 30),
        'lambda_s': torch.clamp(lambda_s_base * f_lambda, 1.0, 60),
        'D': torch.clamp(D_base + delta_D, 2.0, 2.4)
    }


def modify_by_sediment_gpu(H: torch.Tensor, lambda_n: torch.Tensor, 
                           lambda_s: torch.Tensor, sediment_thickness: torch.Tensor,
                           D: Optional[torch.Tensor] = None) -> Tuple:
    """
    Modify parameters by sediment thickness on GPU.
    """
    H0 = torch.clamp(H, min=0.01)
    H_sed = torch.clamp(H0 - sediment_thickness / 2, min=0.01)
    
    H_ref = 200.0
    sediment_ratio = sediment_thickness / H_ref
    lambda_n_sed = lambda_n * (1.0 + 1.3 * sediment_ratio)
    lambda_s_sed = lambda_s * (1.0 + 1.3 * sediment_ratio)
    
    if D is not None:
        D_sed = torch.where(
            sediment_ratio <= 0.1,
            2.2 - 1.5 * sediment_ratio,
            torch.tensor(2.05, device=DEVICE)
        )
        D_sed = torch.clamp(D_sed, 2.0, 2.3)
        return H_sed, lambda_n_sed, lambda_s_sed, D_sed
    
    return H_sed, lambda_n_sed, lambda_s_sed


def generate_filter_bank_gpu(azimuth_levels: torch.Tensor,
                              sediment_levels: torch.Tensor,
                              spreading_rate_levels: torch.Tensor,
                              base_params: Dict[str, float],
                              grid_spacing_km: float,
                              filter_size: int = 25) -> torch.Tensor:
    """
    Generate complete filter bank on GPU.
    
    Returns tensor of shape (n_az, n_sed, n_sr, filter_size, filter_size)
    """
    n_az = len(azimuth_levels)
    n_sed = len(sediment_levels)
    n_sr = len(spreading_rate_levels)
    
    # Pre-allocate filter bank
    filters = torch.zeros((n_az, n_sed, n_sr, filter_size, filter_size), 
                          dtype=torch.float32, device=DEVICE)
    
    # Create filter coordinate grid (once)
    x = torch.linspace(-1, 1, filter_size, device=DEVICE)
    y = torch.linspace(-1, 1, filter_size, device=DEVICE)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    # Generate all filters
    for i_az, az in enumerate(azimuth_levels):
        # Pre-compute rotation
        cos_az = torch.cos(az)
        sin_az = torch.sin(az)
        x_rot = xx * cos_az + yy * sin_az
        y_rot = -xx * sin_az + yy * cos_az
        
        for i_sed, sed in enumerate(sediment_levels):
            for i_sr, sr in enumerate(spreading_rate_levels):
                # Get spreading-rate-scaled parameters
                params_sr = spreading_rate_to_params_gpu(
                    torch.tensor([sr], device=DEVICE),
                    base_params
                )
                H_sr = params_sr['H'][0]
                lambda_n_sr = params_sr['lambda_n'][0]
                lambda_s_sr = params_sr['lambda_s'][0]
                
                # Apply sediment modification
                H_mod, lambda_n_mod, lambda_s_mod = modify_by_sediment_gpu(
                    H_sr.unsqueeze(0), 
                    lambda_n_sr.unsqueeze(0),
                    lambda_s_sr.unsqueeze(0),
                    torch.tensor([sed], device=DEVICE)
                )
                H_mod = H_mod[0]
                lambda_n_mod = lambda_n_mod[0]
                lambda_s_mod = lambda_s_mod[0]
                
                # Convert wavelengths to filter coordinates
                lambda_n_pixels = lambda_n_mod / grid_spacing_km
                lambda_s_pixels = lambda_s_mod / grid_spacing_km
                lambda_n_scaled = lambda_n_pixels / (filter_size / 2.0)
                lambda_s_scaled = lambda_s_pixels / (filter_size / 2.0)
                
                # Gaussian filter
                filter_exp = -((x_rot / lambda_n_scaled)**2 + (y_rot / lambda_s_scaled)**2) / 2
                filt = H_mod * torch.exp(filter_exp)
                
                # Normalize
                filt_sum = filt.sum()
                if filt_sum > 0:
                    filt = filt / filt_sum
                else:
                    filt = torch.zeros_like(filt)
                    filt[filter_size//2, filter_size//2] = 1.0
                
                filters[i_az, i_sed, i_sr] = filt
    
    return filters


def fft_convolve_batch_gpu(signal: torch.Tensor, 
                           filters: torch.Tensor,
                           batch_size: int = 50) -> torch.Tensor:
    """
    Perform batched FFT convolution on GPU with memory management.
    
    Parameters
    ----------
    signal : torch.Tensor
        2D input signal (ny, nx)
    filters : torch.Tensor
        Filter bank (n_filters, fh, fw) or (n_az, n_sed, n_sr, fh, fw)
    batch_size : int
        Number of filters to process at once (lower = less memory)
        
    Returns
    -------
    torch.Tensor
        Convolved outputs with same spatial shape as signal (stored on CPU to save GPU memory)
    """
    original_shape = filters.shape
    ny, nx = signal.shape
    
    # Flatten filter dimensions for batch processing
    if filters.dim() == 5:
        n_az, n_sed, n_sr, fh, fw = filters.shape
        filters_flat = filters.reshape(-1, fh, fw)
    else:
        filters_flat = filters
        n_az, n_sed, n_sr = None, None, None
    
    n_filters = filters_flat.shape[0]
    fh, fw = filters_flat.shape[1], filters_flat.shape[2]
    
    # Pad signal and filters to same size for FFT
    pad_h = fh // 2
    pad_w = fw // 2
    
    # Pad signal (replicate edges)
    signal_padded = F.pad(signal.unsqueeze(0).unsqueeze(0), 
                          (pad_w, pad_w, pad_h, pad_h), 
                          mode='replicate').squeeze()
    
    # FFT size
    fft_h = signal_padded.shape[0]
    fft_w = signal_padded.shape[1]
    
    # FFT of signal (compute once, keep on GPU)
    signal_fft = torch.fft.rfft2(signal_padded, s=(fft_h, fft_w))
    
    # Pre-allocate output on CPU to save GPU memory
    # This is the key change - store results on CPU, not GPU
    all_convolved_cpu = torch.zeros((n_filters, ny, nx), dtype=torch.float32, device='cpu')
    
    # Process filters in small batches
    for i in range(0, n_filters, batch_size):
        batch_filters = filters_flat[i:i+batch_size]
        actual_batch = batch_filters.shape[0]
        
        # Pad filters to FFT size
        filters_padded = torch.zeros((actual_batch, fft_h, fft_w), 
                                      dtype=torch.float32, device=DEVICE)
        filters_padded[:, :fh, :fw] = batch_filters
        
        # Circular shift to center the filter
        filters_padded = torch.roll(filters_padded, shifts=(-fh//2, -fw//2), dims=(1, 2))
        
        # FFT of filters
        filters_fft = torch.fft.rfft2(filters_padded)
        
        # Multiply in frequency domain (broadcast signal FFT)
        product = signal_fft.unsqueeze(0) * filters_fft
        
        # Inverse FFT
        convolved = torch.fft.irfft2(product, s=(fft_h, fft_w))
        
        # Crop to original size and move to CPU immediately
        convolved = convolved[:, pad_h:pad_h+ny, pad_w:pad_w+nx]
        all_convolved_cpu[i:i+actual_batch] = convolved.cpu()
        
        # Free GPU memory
        del filters_padded, filters_fft, product, convolved
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
    
    # Clean up
    del signal_fft, signal_padded
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    
    # Move back to GPU for interpolation (or keep on CPU if memory tight)
    # Reshape to original filter dimensions
    if n_az is not None:
        all_convolved_cpu = all_convolved_cpu.reshape(n_az, n_sed, n_sr, ny, nx)
    
    # Move to GPU for interpolation
    return all_convolved_cpu.to(DEVICE)


def trilinear_interpolation_gpu(convolved_stack: torch.Tensor,
                                 azimuth: torch.Tensor,
                                 sediment: torch.Tensor,
                                 spreading_rate: torch.Tensor,
                                 azimuth_bins: int,
                                 sediment_levels: torch.Tensor,
                                 spreading_rate_levels: torch.Tensor) -> torch.Tensor:
    """
    Perform trilinear interpolation on GPU to select from convolved results.
    
    Parameters
    ----------
    convolved_stack : torch.Tensor
        Shape (n_az, n_sed, n_sr, ny, nx)
    azimuth : torch.Tensor
        Azimuth at each pixel (ny, nx), radians
    sediment : torch.Tensor
        Sediment at each pixel (ny, nx), meters
    spreading_rate : torch.Tensor
        Spreading rate at each pixel (ny, nx), mm/yr
    azimuth_bins : int
        Number of azimuth bins
    sediment_levels : torch.Tensor
        Sediment bin levels
    spreading_rate_levels : torch.Tensor
        Spreading rate bin levels
        
    Returns
    -------
    torch.Tensor
        Interpolated bathymetry (ny, nx)
    """
    ny, nx = azimuth.shape
    n_sed = len(sediment_levels)
    n_sr = len(spreading_rate_levels)
    
    # Normalize azimuth to [0, 2π)
    azimuth_normalized = torch.remainder(azimuth + np.pi, 2 * np.pi)
    
    # Calculate continuous bin positions
    azimuth_bin_width = 2 * np.pi / azimuth_bins
    azimuth_bin_pos = azimuth_normalized / azimuth_bin_width
    
    sed_min = sediment_levels[0]
    sed_max = sediment_levels[-1]
    sed_range = sed_max - sed_min
    
    if n_sed == 1 or sed_range < 1e-6:
        sediment_bin_pos = torch.zeros_like(sediment)
    else:
        sed_bin_width = sed_range / (n_sed - 1)
        sediment_bin_pos = (sediment - sed_min) / sed_bin_width
    
    sr_min = spreading_rate_levels[0]
    sr_max = spreading_rate_levels[-1]
    sr_range = sr_max - sr_min
    
    if n_sr == 1 or sr_range < 1e-6:
        sr_bin_pos = torch.zeros_like(spreading_rate)
    else:
        sr_bin_width = sr_range / (n_sr - 1)
        sr_bin_pos = (spreading_rate - sr_min) / sr_bin_width
    
    # Handle NaN values
    azimuth_bin_pos = torch.nan_to_num(azimuth_bin_pos, nan=0.0)
    sediment_bin_pos = torch.nan_to_num(sediment_bin_pos, nan=0.0)
    sr_bin_pos = torch.nan_to_num(sr_bin_pos, nan=0.0)
    
    # Get integer indices and fractional parts
    az_idx0 = torch.floor(azimuth_bin_pos).long()
    az_idx1 = az_idx0 + 1
    az_frac = azimuth_bin_pos - az_idx0.float()
    
    # Handle azimuth wraparound
    az_idx0 = torch.clamp(az_idx0, 0, azimuth_bins - 1)
    az_idx1 = torch.remainder(az_idx1, azimuth_bins)
    
    sed_idx0 = torch.floor(sediment_bin_pos).long()
    sed_idx1 = torch.clamp(sed_idx0 + 1, max=n_sed - 1)
    sed_frac = sediment_bin_pos - sed_idx0.float()
    sed_idx0 = torch.clamp(sed_idx0, 0, n_sed - 1)
    
    sr_idx0 = torch.floor(sr_bin_pos).long()
    sr_idx1 = torch.clamp(sr_idx0 + 1, max=n_sr - 1)
    sr_frac = sr_bin_pos - sr_idx0.float()
    sr_idx0 = torch.clamp(sr_idx0, 0, n_sr - 1)
    
    # Create index grids
    i_indices = torch.arange(ny, device=DEVICE).unsqueeze(1).expand(ny, nx)
    j_indices = torch.arange(nx, device=DEVICE).unsqueeze(0).expand(ny, nx)
    
    # Get values at 8 corners (trilinear interpolation)
    c000 = convolved_stack[az_idx0, sed_idx0, sr_idx0, i_indices, j_indices]
    c001 = convolved_stack[az_idx0, sed_idx0, sr_idx1, i_indices, j_indices]
    c010 = convolved_stack[az_idx0, sed_idx1, sr_idx0, i_indices, j_indices]
    c011 = convolved_stack[az_idx0, sed_idx1, sr_idx1, i_indices, j_indices]
    c100 = convolved_stack[az_idx1, sed_idx0, sr_idx0, i_indices, j_indices]
    c101 = convolved_stack[az_idx1, sed_idx0, sr_idx1, i_indices, j_indices]
    c110 = convolved_stack[az_idx1, sed_idx1, sr_idx0, i_indices, j_indices]
    c111 = convolved_stack[az_idx1, sed_idx1, sr_idx1, i_indices, j_indices]
    
    # Interpolate along spreading rate axis
    c00 = c000 * (1 - sr_frac) + c001 * sr_frac
    c01 = c010 * (1 - sr_frac) + c011 * sr_frac
    c10 = c100 * (1 - sr_frac) + c101 * sr_frac
    c11 = c110 * (1 - sr_frac) + c111 * sr_frac
    
    # Interpolate along sediment axis
    c0 = c00 * (1 - sed_frac) + c01 * sed_frac
    c1 = c10 * (1 - sed_frac) + c11 * sed_frac
    
    # Interpolate along azimuth axis
    bathymetry = c0 * (1 - az_frac) + c1 * az_frac
    
    return bathymetry


def generate_bathymetry_gpu(seafloor_age: np.ndarray,
                            sediment_thickness: np.ndarray,
                            params: Dict[str, float],
                            grid_spacing_km: float,
                            random_field: Optional[np.ndarray] = None,
                            lat_coords: Optional[np.ndarray] = None,
                            azimuth_bins: int = 36,
                            sediment_bins: int = 10,
                            spreading_rate_bins: int = 20,
                            filter_size: int = 25,
                            tile_size: int = 500,
                            verbose: bool = True) -> np.ndarray:
    """
    Generate synthetic abyssal hill bathymetry using GPU acceleration.
    
    This is the main GPU-accelerated function that replaces
    generate_bathymetry_spatial_filter() from AbFab.py.
    
    Uses internal spatial tiling to manage GPU memory for large grids.
    
    Parameters
    ----------
    seafloor_age : np.ndarray
        Seafloor age in Myr (2D array)
    sediment_thickness : np.ndarray
        Sediment thickness in meters (2D array)
    params : dict
        Base abyssal hill parameters: {'H', 'lambda_n', 'lambda_s', 'D'}
    grid_spacing_km : float
        Grid spacing in km/pixel
    random_field : np.ndarray, optional
        Pre-generated random field (if None, generates on GPU)
    lat_coords : np.ndarray, optional
        Latitude coordinates for spherical correction
    azimuth_bins : int
        Number of azimuth bins (default: 36)
    sediment_bins : int
        Number of sediment bins (default: 10)
    spreading_rate_bins : int
        Number of spreading rate bins (default: 20)
    filter_size : int
        Size of convolution filter (default: 25)
    tile_size : int
        Size of internal tiles for memory management (default: 500)
    verbose : bool
        Print progress information
        
    Returns
    -------
    np.ndarray
        Synthetic bathymetry in meters
    """
    ny, nx = seafloor_age.shape
    
    if verbose:
        print(f"  GPU bathymetry generation: {ny}×{nx} grid")
        n_filters = azimuth_bins * sediment_bins * spreading_rate_bins
        print(f"  Filter bank: {azimuth_bins}×{sediment_bins}×{spreading_rate_bins} = {n_filters} filters")
    
    t_start = time.time()
    
    # Determine global ranges for consistent binning
    # Do this on CPU to avoid transferring full arrays to GPU yet
    if lat_coords is not None:
        # Calculate spreading rate on CPU first for global stats
        import AbFab as af
        spreading_rate_cpu = af.calculate_spreading_rate_from_age(
            seafloor_age, grid_spacing_km, lat_coords
        )
    else:
        import AbFab as af
        spreading_rate_cpu = af.calculate_spreading_rate_from_age(
            seafloor_age, grid_spacing_km
        )
    
    sr_valid = spreading_rate_cpu[~np.isnan(spreading_rate_cpu)]
    if len(sr_valid) > 0:
        sr_median = np.median(sr_valid)
        sr_min = np.min(sr_valid)
        sr_max = np.max(sr_valid)
    else:
        sr_median, sr_min, sr_max = 50.0, 10.0, 100.0
    
    spreading_rate_cpu = np.where(np.isnan(spreading_rate_cpu), sr_median, spreading_rate_cpu)
    
    sed_min = float(np.nanmin(sediment_thickness))
    sed_max = float(np.nanmax(sediment_thickness))
    
    if verbose:
        print(f"  Spreading rate range: {sr_min:.1f} - {sr_max:.1f} mm/yr")
        print(f"  Sediment range: {sed_min:.1f} - {sed_max:.1f} m")
    
    # Create bin levels
    azimuth_levels = torch.linspace(-np.pi, np.pi, azimuth_bins, device=DEVICE)
    
    if sed_max - sed_min < 1e-6:
        sediment_levels = torch.tensor([sed_min], device=DEVICE)
        sediment_bins = 1
    else:
        sediment_levels = torch.linspace(sed_min, sed_max, sediment_bins, device=DEVICE)
    
    if sr_max - sr_min < 1e-6 or spreading_rate_bins == 1:
        spreading_rate_levels = torch.tensor([(sr_min + sr_max) / 2], device=DEVICE)
        spreading_rate_bins = 1
    else:
        spreading_rate_levels = torch.linspace(sr_min, sr_max, spreading_rate_bins, device=DEVICE)
    
    # Generate filter bank on GPU (this stays constant)
    t_fb_start = time.time()
    filter_bank = generate_filter_bank_gpu(
        azimuth_levels, sediment_levels, spreading_rate_levels,
        params, grid_spacing_km, filter_size
    )
    t_fb_end = time.time()
    if verbose:
        print(f"  Filter bank generation: {t_fb_end - t_fb_start:.2f}s")
    
    # Calculate azimuth on CPU (will tile later)
    if lat_coords is not None:
        import AbFab as af
        azimuth_cpu = af.calculate_azimuth_from_age(seafloor_age, lat_coords)
    else:
        import AbFab as af
        azimuth_cpu = af.calculate_azimuth_from_age(seafloor_age)
    
    # Prepare random field
    if random_field is None:
        random_field = np.random.randn(ny, nx).astype(np.float32)
    
    # Output array (on CPU)
    result = np.zeros((ny, nx), dtype=np.float32)
    
    # Process in spatial tiles
    pad = filter_size // 2 + 5  # Padding for convolution edges
    
    n_tiles_y = (ny + tile_size - 1) // tile_size
    n_tiles_x = (nx + tile_size - 1) // tile_size
    total_tiles = n_tiles_y * n_tiles_x
    
    if verbose:
        print(f"  Processing {total_tiles} tiles ({tile_size}×{tile_size} each)...")
    
    t_conv_start = time.time()
    tile_count = 0
    
    for ty in range(0, ny, tile_size):
        for tx in range(0, nx, tile_size):
            tile_count += 1
            
            # Tile boundaries with padding
            y0 = max(0, ty - pad)
            y1 = min(ny, ty + tile_size + pad)
            x0 = max(0, tx - pad)
            x1 = min(nx, tx + tile_size + pad)
            
            # Extract tile data
            tile_random = random_field[y0:y1, x0:x1]
            tile_azimuth = azimuth_cpu[y0:y1, x0:x1]
            tile_sediment = sediment_thickness[y0:y1, x0:x1]
            tile_spreading = spreading_rate_cpu[y0:y1, x0:x1]
            
            tile_ny, tile_nx = tile_random.shape
            
            # Transfer tile to GPU
            rand_gpu = to_torch(tile_random.astype(np.float32))
            az_gpu = to_torch(tile_azimuth.astype(np.float32))
            sed_gpu = to_torch(tile_sediment.astype(np.float32))
            sr_gpu = to_torch(tile_spreading.astype(np.float32))
            
            # Convolve random field with all filters for this tile
            convolved_stack = fft_convolve_batch_gpu(rand_gpu, filter_bank, batch_size=100)
            
            # Scale by target RMS height
            theoretical_rms = 1.0 / np.sqrt(12.0)
            for i_sed in range(len(sediment_levels)):
                for i_sr in range(len(spreading_rate_levels)):
                    params_sr = spreading_rate_to_params_gpu(
                        spreading_rate_levels[i_sr:i_sr+1], params
                    )
                    H_sr = params_sr['H'][0]
                    H_mod, _, _ = modify_by_sediment_gpu(
                        H_sr.unsqueeze(0),
                        params_sr['lambda_n'],
                        params_sr['lambda_s'],
                        sediment_levels[i_sed:i_sed+1]
                    )
                    scale_factor = H_mod[0] / theoretical_rms
                    convolved_stack[:, i_sed, i_sr] *= scale_factor
            
            # Trilinear interpolation for this tile
            tile_bathy = trilinear_interpolation_gpu(
                convolved_stack, az_gpu, sed_gpu, sr_gpu,
                azimuth_bins, sediment_levels, spreading_rate_levels
            )
            
            # Copy result back, accounting for padding
            out_y0 = ty - y0  # Offset in tile coords
            out_y1 = out_y0 + min(tile_size, ny - ty)
            out_x0 = tx - x0
            out_x1 = out_x0 + min(tile_size, nx - tx)
            
            result[ty:ty+out_y1-out_y0, tx:tx+out_x1-out_x0] = \
                to_numpy(tile_bathy[out_y0:out_y1, out_x0:out_x1])
            
            # Free GPU memory
            del rand_gpu, az_gpu, sed_gpu, sr_gpu, convolved_stack, tile_bathy
            if DEVICE.type == 'mps':
                torch.mps.empty_cache()
            
            if verbose and tile_count % 10 == 0:
                print(f"    Tile {tile_count}/{total_tiles}")
    
    t_conv_end = time.time()
    if verbose:
        print(f"  Convolution + interpolation: {t_conv_end - t_conv_start:.2f}s")
    
    t_end = time.time()
    if verbose:
        print(f"  Total GPU time: {t_end - t_start:.2f}s")
    
    return result


# ============================================================================
# THERMAL SUBSIDENCE (GPU VERSION)
# ============================================================================

def calculate_thermal_subsidence_gpu(seafloor_age: torch.Tensor,
                                      model: str = 'GDH1') -> torch.Tensor:
    """
    Calculate basement depth from seafloor age on GPU.
    """
    age_valid = torch.where(torch.isnan(seafloor_age), 
                            torch.zeros_like(seafloor_age), 
                            seafloor_age)
    age_valid = torch.clamp(age_valid, min=0)
    
    if model.upper() == 'GDH1':
        depth = torch.where(
            age_valid <= 20.0,
            2600.0 + 365.0 * torch.sqrt(age_valid),
            5651.0 - 2473.0 * torch.exp(-0.0278 * age_valid)
        )
    elif model.lower() == 'half_space':
        depth = 2600.0 + 365.0 * torch.sqrt(age_valid)
    elif model.lower() == 'plate':
        plate_timescale = 62.8
        depth = 2600.0 + 365.0 * torch.sqrt(age_valid) * (
            1.0 - torch.exp(-age_valid / plate_timescale)
        )
    else:
        raise ValueError(f"Unknown subsidence model: {model}")
    
    depth = torch.where(torch.isnan(seafloor_age), 
                        torch.tensor(float('nan'), device=DEVICE), 
                        depth)
    
    return -depth


def generate_complete_bathymetry_gpu(seafloor_age: np.ndarray,
                                      sediment_thickness: np.ndarray,
                                      params: Dict[str, float],
                                      grid_spacing_km: float,
                                      random_field: Optional[np.ndarray] = None,
                                      lat_coords: Optional[np.ndarray] = None,
                                      subsidence_model: str = 'GDH1',
                                      sediment_mode: str = 'drape',
                                      azimuth_bins: int = 36,
                                      sediment_bins: int = 10,
                                      spreading_rate_bins: int = 20,
                                      tile_size: int = 500,
                                      verbose: bool = True) -> np.ndarray:
    """
    Generate complete bathymetry (subsidence + hills + sediment) on GPU.
    
    Parameters
    ----------
    seafloor_age : np.ndarray
        Seafloor age in Myr
    sediment_thickness : np.ndarray
        Sediment thickness in meters
    params : dict
        Base abyssal hill parameters
    grid_spacing_km : float
        Grid spacing in km/pixel
    random_field : np.ndarray, optional
        Pre-generated random field
    lat_coords : np.ndarray, optional
        Latitude coordinates for spherical correction
    subsidence_model : str
        Thermal subsidence model ('GDH1', 'half_space', 'plate')
    sediment_mode : str
        Sediment treatment ('none', 'drape')
        Note: 'fill' mode requires global diffusion, done separately
    azimuth_bins, sediment_bins, spreading_rate_bins : int
        Bin counts for filter bank
    tile_size : int
        Internal tile size for GPU memory management (default: 500)
    verbose : bool
        Print progress
        
    Returns
    -------
    np.ndarray
        Complete bathymetry in meters (negative below sea level)
    """
    if verbose:
        print("Generating complete bathymetry on GPU...")
    
    t_start = time.time()
    
    # Transfer to GPU
    age_gpu = to_torch(seafloor_age)
    sed_gpu = to_torch(sediment_thickness)
    
    # Calculate thermal subsidence on GPU
    if verbose:
        print(f"  Calculating thermal subsidence ({subsidence_model})...")
    basement_regional = calculate_thermal_subsidence_gpu(age_gpu, model=subsidence_model)
    
    # Generate abyssal hills on GPU
    if verbose:
        print("  Generating abyssal hills...")
    abyssal_hills = generate_bathymetry_gpu(
        seafloor_age, sediment_thickness, params, grid_spacing_km,
        random_field, lat_coords,
        azimuth_bins, sediment_bins, spreading_rate_bins,
        tile_size=tile_size,
        verbose=verbose
    )
    
    # Combine on GPU
    abyssal_hills_gpu = to_torch(abyssal_hills)
    basement_topo = basement_regional + abyssal_hills_gpu
    
    # Apply sediment
    if sediment_mode == 'none':
        final_bathymetry = basement_topo
    elif sediment_mode == 'drape':
        final_bathymetry = basement_topo + sed_gpu
    else:
        # For 'fill' mode, return drape and apply diffusion globally after
        final_bathymetry = basement_topo + sed_gpu
        if verbose:
            print("  Note: 'fill' mode diffusion should be applied globally after assembly")
    
    result = to_numpy(final_bathymetry)
    
    t_end = time.time()
    if verbose:
        print(f"  Complete bathymetry total: {t_end - t_start:.2f}s")
    
    return result


# ============================================================================
# HIGH-LEVEL WORKFLOW FUNCTION
# ============================================================================

def run_gpu_bathymetry_workflow(config: Dict) -> np.ndarray:
    """
    Run complete bathymetry workflow with GPU acceleration.
    
    This is designed to be a drop-in replacement for 
    af.run_complete_bathymetry_workflow() but using GPU.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (same format as CPU version)
        
    Returns
    -------
    xarray.DataArray
        Complete bathymetry grid
    """
    import xarray as xr
    import pygmt
    
    verbose = config['output'].get('verbose', True)
    
    if verbose:
        print("="*70)
        print("GPU-Accelerated Bathymetry Generation")
        print(f"Device: {DEVICE}")
        print("="*70)
    
    # Load input data
    if verbose:
        print("\nLoading input data...")
    
    age_file = config['input']['age_file']
    sediment_file = config['input'].get('sediment_file')
    constant_sediment = config['input'].get('constant_sediment')
    spacing = config['region']['spacing']
    
    age_da = pygmt.grdsample(age_file, region='g', spacing=spacing)
    
    # Handle sediment
    if sediment_file is not None:
        sed_da = pygmt.grdsample(sediment_file, region='g', spacing=spacing)
        sed_da = sed_da.where(np.isfinite(sed_da), 1.)
        sed_da = sed_da.where(sed_da < 1000., 1000.)
    elif constant_sediment is not None:
        sed_da = age_da.copy()
        sed_da.data = np.full_like(age_da.data, constant_sediment)
        sed_da = sed_da.where(~np.isnan(age_da.data), np.nan)
    else:
        sed_da = age_da.copy()
        sed_da.data = np.zeros_like(age_da.data)
    
    # Select region
    lon_min = config['region']['lon_min']
    lon_max = config['region']['lon_max']
    lat_min = config['region']['lat_min']
    lat_max = config['region']['lat_max']
    
    age_da = age_da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    sed_da = sed_da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    
    if verbose:
        print(f"  Region: {lon_min}° to {lon_max}°E, {lat_min}° to {lat_max}°N")
        print(f"  Grid shape: {age_da.shape}")
    
    # Calculate grid spacing
    lon_spacing_deg = float(np.abs(age_da.lon.values[1] - age_da.lon.values[0]))
    mean_lat = float(np.mean(age_da.lat.values))
    grid_spacing_km = lon_spacing_deg * 111.32 * np.cos(np.radians(mean_lat))
    
    if verbose:
        print(f"  Grid spacing: {grid_spacing_km:.3f} km/pixel")
    
    # Generate random field
    random_seed = config['advanced'].get('random_seed')
    if random_seed is not None:
        np.random.seed(random_seed)
    random_field = np.random.randn(*age_da.shape)
    
    # Get parameters
    params = config['abyssal_hills']
    
    # Generate complete bathymetry on GPU
    # Get tile_size from config (default 500 for memory safety)
    tile_size = config.get('gpu', {}).get('tile_size', 500)
    
    bathymetry = generate_complete_bathymetry_gpu(
        age_da.data,
        sed_da.data,
        params,
        grid_spacing_km,
        random_field,
        lat_coords=age_da.lat.values,
        subsidence_model=config['subsidence']['model'],
        sediment_mode=config['sediment']['mode'],
        azimuth_bins=config['optimization']['azimuth_bins'],
        sediment_bins=config['optimization']['sediment_bins'],
        spreading_rate_bins=config['optimization']['spreading_rate_bins'],
        tile_size=tile_size,
        verbose=verbose
    )
    
    # Create output DataArray
    complete_grid = xr.DataArray(
        bathymetry,
        coords={'lat': age_da.lat, 'lon': age_da.lon},
        dims=['lat', 'lon'],
        name='bathymetry',
        attrs={
            'long_name': 'Complete synthetic bathymetry (GPU)',
            'units': 'm',
            'subsidence_model': config['subsidence']['model'],
            'sediment_mode': config['sediment']['mode'],
            'device': str(DEVICE)
        }
    )
    
    # Save output
    output_nc = config['output']['netcdf']
    if verbose:
        print(f"\nSaving to: {output_nc}")
    complete_grid.to_netcdf(output_nc)
    
    if verbose:
        print(f"\nGrid statistics:")
        print(f"  Shape: {complete_grid.shape}")
        print(f"  Depth range: {float(np.nanmin(bathymetry)):.0f} to {float(np.nanmax(bathymetry)):.0f} m")
        print(f"  Valid pixels: {np.sum(~np.isnan(bathymetry))}/{bathymetry.size}")
    
    return complete_grid


# ============================================================================
# BENCHMARKING UTILITIES
# ============================================================================

def benchmark_gpu_vs_cpu(shape: Tuple[int, int] = (500, 500),
                         params: Optional[Dict] = None,
                         n_runs: int = 3) -> Dict:
    """
    Benchmark GPU vs CPU performance.
    
    Parameters
    ----------
    shape : tuple
        Grid shape to test
    params : dict, optional
        Abyssal hill parameters
    n_runs : int
        Number of runs for averaging
        
    Returns
    -------
    dict
        Benchmark results
    """
    import AbFab as af
    
    if params is None:
        params = {
            'H': 250.0,
            'lambda_n': 3.0,
            'lambda_s': 30.0,
            'D': 2.2
        }
    
    ny, nx = shape
    grid_spacing_km = 5.0
    
    # Generate test data
    np.random.seed(42)
    age = np.random.uniform(10, 100, shape).astype(np.float32)
    sediment = np.random.uniform(0, 500, shape).astype(np.float32)
    random_field = np.random.randn(*shape).astype(np.float32)
    
    results = {
        'shape': shape,
        'n_filters': 36 * 10 * 20,
        'gpu_times': [],
        'cpu_times': [],
    }
    
    print(f"\nBenchmarking {shape[0]}×{shape[1]} grid...")
    print(f"Filter bank: 36×10×20 = 7200 filters")
    
    # GPU benchmark
    print("\nGPU runs:")
    for i in range(n_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.time()
        
        _ = generate_bathymetry_gpu(
            age, sediment, params, grid_spacing_km, random_field,
            azimuth_bins=36, sediment_bins=10, spreading_rate_bins=20,
            verbose=False
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_end = time.time()
        results['gpu_times'].append(t_end - t_start)
        print(f"  Run {i+1}: {t_end - t_start:.2f}s")
    
    # CPU benchmark
    print("\nCPU runs:")
    for i in range(n_runs):
        t_start = time.time()
        
        _ = af.generate_bathymetry_spatial_filter(
            age, sediment, params, grid_spacing_km, random_field,
            optimize=True,
            azimuth_bins=36, sediment_bins=10, spreading_rate_bins=20
        )
        
        t_end = time.time()
        results['cpu_times'].append(t_end - t_start)
        print(f"  Run {i+1}: {t_end - t_start:.2f}s")
    
    # Summary
    gpu_mean = np.mean(results['gpu_times'])
    cpu_mean = np.mean(results['cpu_times'])
    speedup = cpu_mean / gpu_mean
    
    results['gpu_mean'] = gpu_mean
    results['cpu_mean'] = cpu_mean
    results['speedup'] = speedup
    
    print(f"\nResults:")
    print(f"  GPU mean: {gpu_mean:.2f}s")
    print(f"  CPU mean: {cpu_mean:.2f}s")
    print(f"  Speedup: {speedup:.1f}×")
    
    return results


if __name__ == "__main__":
    # Run benchmark if called directly
    print("AbFab GPU Module")
    print(f"MPS Available: {MPS_AVAILABLE}")
    print(f"Device: {DEVICE}")
    
    if len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == 'benchmark':
        benchmark_gpu_vs_cpu()
