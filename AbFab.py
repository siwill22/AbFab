import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import convolve
from scipy.signal import oaconvolve, fftconvolve
from scipy.special import kv  # Modified Bessel function for von Kármán filter
import xarray as xr

# Optional tqdm for progress bars
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable



#### FFT Section

# Von Kármán spectrum function with azimuthal dependence
def von_karman_spectrum(kx, ky, H, kn, ks, D, azimuth, use_full_equation=False):
    """
    Create the von Kármán spectrum based on the abyssal hill parameters.

    Implements two forms of the von Kármán spectrum:
    - Simplified (default): P_h(k) = H² / [u²(k) + 1]^((D+1)/2)
    - Full (Eq 1): P_h(k) = 4π·ν·H²·(kn×ks) / [u²(k) + 1]^(ν+1)
      where ν = 3 - D

    Parameters:
    -----------
    kx, ky : ndarray
        Wavenumber arrays (2D grids)
    H : float
        RMS height (meters)
    kn, ks : float
        Characteristic wavenumbers (km⁻¹) in ridge-normal and ridge-parallel directions
    D : float
        Fractal dimension (typically 2.0-2.3)
    azimuth : float or ndarray
        Ridge orientation angle in radians (azimuth of abyssal hill fabric)
    use_full_equation : bool, optional
        If True, use full Equation 1 from Goff & Arbic (2010)
        If False (default), use simplified form

    Returns:
    --------
    spectrum : ndarray
        Von Kármán power spectral density

    References:
    -----------
    Goff & Arbic (2010) Ocean Modelling, Equation 1
    """
    # Rotate the wavenumbers by the azimuth angle
    kx_rot = kx * np.cos(azimuth) + ky * np.sin(azimuth)
    ky_rot = -kx * np.sin(azimuth) + ky * np.cos(azimuth)

    # Dimensionless wavenumbers scaled by the characteristic widths
    kx_dim = kx_rot / kn
    ky_dim = ky_rot / ks

    # Dimensionless wavenumber squared: u²(k)
    k_squared = kx_dim**2 + ky_dim**2

    if use_full_equation:
        # Full Equation 1 from Goff & Arbic (2010)
        # P_h(k) = 4π·ν·H²·|Q|^(1/2) / [u²(k) + 1]^(ν+1)
        # where ν = m (Hurst number) = 3 - D
        # and |Q| = k_n² × k_s², so |Q|^(1/2) = k_n × k_s
        nu = 3 - D  # Hurst number
        Q_det_sqrt = kn * ks  # √(k_n² × k_s²) = k_n × k_s
        spectrum = 4 * np.pi * nu * (H**2) * Q_det_sqrt / (1 + k_squared)**(nu + 1)
    else:
        # Simplified form (original working version)
        # P_h(k) = H² / [u²(k) + 1]^((D+1)/2)
        spectrum = (H**2) * (1 + k_squared)**(-(D+1)/2)

    return spectrum

# Random phase for the noise field
def generate_random_phase(nx, ny):
    """ Generate a random field with normally distributed random phases. """
    return np.exp(2j * np.pi * np.random.rand(nx, ny))

# Sediment modification for abyssal hill parameters
def modify_by_sediment(H, kn, ks, sediment_thickness, D=None):
    """
    Modify abyssal hill parameters based on sediment thickness.
    Sediment drape reduces the rms height (H) and increases the width (kn, ks).

    Implements Goff & Arbic (2010) Equations 5-7:
    - Equation 5: H(S) = H₀ - S/2
    - Equations 6-7: k(S/H₀) = k₀(1 + 1.3·S/H₀)

    Optionally implements Equation 8 for fractal dimension if D is provided:
    - Equation 8: D(S/H₀) = 2.2 - 1.5·(S/H₀) for S/H₀ ≤ 0.1, else 2.05

    Parameters:
    -----------
    H, kn, ks : float
        Original parameters
    sediment_thickness : float
        Sediment thickness (m)
    D : float, optional
        Fractal dimension. If provided, also returns modified D. If None, only returns (H, kn, ks)

    Returns:
    --------
    (H_sed, kn_sed, ks_sed) if D is None
    (H_sed, kn_sed, ks_sed, D_sed) if D is provided
    """
    H0 = max(H, 0.01)  # Avoid division by zero
    H_sed = np.maximum(H0 - sediment_thickness / 2, 0.01)  # Equation 5
    kn_sed = kn + 1.3 * kn * (sediment_thickness / H0)  # Equation 6 (fixed: use H0 not H_sed)
    ks_sed = ks + 1.3 * ks * (sediment_thickness / H0)  # Equation 7 (fixed: use H0 not H_sed)

    # Equation 8: Fractal dimension modification (optional)
    if D is not None:
        ratio = sediment_thickness / H0
        if ratio <= 0.1:
            D_sed = 2.2 - 1.5 * ratio
        else:
            D_sed = 2.05
        D_sed = np.clip(D_sed, 2.0, 2.3)
        return H_sed, kn_sed, ks_sed, D_sed

    return H_sed, kn_sed, ks_sed

# Azimuth calculation from seafloor age gradient
def calculate_azimuth_from_age(seafloor_age):
    """
    Calculate the azimuth from the gradient of seafloor age.

    The age gradient points in the spreading direction (away from ridge).
    This function returns the spreading direction azimuth, which is used
    to orient the anisotropic filter in the spatial convolution method.

    Note: This is the spreading direction, NOT the ridge-perpendicular direction.
    The filter orientation handles the perpendicular relationship internally.

    Parameters:
    -----------
    seafloor_age : 2D array
        Seafloor age in Myr

    Returns:
    --------
    azimuth : 2D array
        Spreading direction azimuth in radians (measured clockwise from north)
    """
    # Compute the gradients in x and y direction
    grad_y, grad_x = np.gradient(seafloor_age)
    
    # Compute azimuth as the angle of the gradient vector (radians)
    azimuth = np.arctan2(grad_y, grad_x)

    return azimuth


def calculate_spreading_rate_from_age(seafloor_age, grid_spacing_km=1.0):
    """
    Estimate half-spreading rate from seafloor age gradient.

    Parameters:
    -----------
    seafloor_age : 2D array
        Seafloor age in Myr
    grid_spacing_km : float, optional
        Grid spacing in km (default: 1.0 km). Needed to convert gradient to physical units.

    Returns:
    --------
    spreading_rate : 2D array
        Half-spreading rate in mm/yr

    Notes:
    ------
    Half-spreading rate u = distance / (2 × time)
    From age gradient: u = (grid_spacing / age_gradient) / 2
    where age_gradient is in Myr per grid cell.
    """
    # Compute the gradients in x and y direction (Myr per grid cell)
    grad_y, grad_x = np.gradient(seafloor_age)

    # Magnitude of age gradient (Myr per grid cell)
    age_gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Avoid division by zero
    age_gradient_magnitude = np.where(age_gradient_magnitude > 1e-10, age_gradient_magnitude, np.nan)

    # Half-spreading rate: distance = rate × time
    # So: rate = distance / time
    # distance per grid cell = grid_spacing_km
    # time per grid cell = age_gradient_magnitude (Myr per grid cell)
    # rate = grid_spacing_km / age_gradient_magnitude (km/Myr)
    #
    # Convert km/Myr to mm/yr:
    # 1 km/Myr = 1000 m / 1e6 yr = 1e-3 m/yr = 1 mm/yr
    # So km/Myr and mm/yr are numerically equivalent!
    spreading_rate = grid_spacing_km / age_gradient_magnitude  # km/Myr = mm/yr

    return spreading_rate


def spreading_rate_to_params(spreading_rate, output_units='deg'):
    """
    Convert spreading rate to abyssal hill parameters using empirical relationships.

    Based on Figure 1 from Goff & Arbic (2010), which shows relationships between
    half-spreading rate and abyssal hill parameters from observational data.

    Parameters:
    -----------
    spreading_rate : float or array
        Half-spreading rate in mm/yr
    output_units : str, optional
        Units for wavenumbers: 'km' for km⁻¹, 'deg' for degrees⁻¹ (default: 'deg')
        Use 'deg' when working with lat/lon grids, 'km' for projected grids

    Returns:
    --------
    params : dict
        Dictionary containing:
        - 'H': RMS height in meters
        - 'kn': characteristic wavenumber perpendicular to ridge (km⁻¹ or deg⁻¹)
        - 'ks': characteristic wavenumber parallel to ridge (km⁻¹ or deg⁻¹)
        - 'D': fractal dimension

    Notes:
    ------
    Empirical relationships from Goff & Arbic (2010) Figure 1:
    - H increases with spreading rate: ~100m at 10 mm/yr to ~300m at 80 mm/yr
    - λn (perpendicular wavelength) increases with spreading rate
    - λs (parallel wavelength) relatively constant
    - D relatively constant around 2.2

    These are approximate fits to the observed data.

    When output_units='deg', wavenumbers are converted assuming 1 degree ≈ 111 km.
    This is appropriate for lat/lon grids where coordinates are in degrees.

    **WARNING - NEEDS CALIBRATION**:
    The current empirical formulas produce kn/ks values that are ~1000× too large
    for typical usage with lat/lon grids. The spatial filter uses kn/ks as dimensionless
    scaling factors on normalized coordinates [-1,1], NOT as physical wavenumbers.

    This function needs calibration to match your specific filter implementation.
    For now, it's recommended to use fixed parameters that have been empirically
    validated for your application (e.g., kn=0.05, ks=0.2 for degree-based grids).

    TODO: Calibrate empirical relationships to produce appropriate scaling factors.
    """
    # Convert spreading rate to array for vectorized operations
    u = np.atleast_1d(spreading_rate)

    # Empirical relationships (approximate fits to Figure 1 data)
    # RMS height H (meters): increases roughly linearly with spreading rate
    # At u=10 mm/yr: H~100m, at u=80 mm/yr: H~300m
    H = 50.0 + 3.5 * u  # H in meters
    H = np.clip(H, 50, 350)  # Reasonable bounds

    # Characteristic wavelength perpendicular to ridge λn (km)
    # Increases with spreading rate: ~10 km at slow spreading, ~30 km at fast
    lambda_n = 8.0 + 0.3 * u  # λn in km
    lambda_n = np.clip(lambda_n, 8, 35)

    # Characteristic wavelength parallel to ridge λs (km)
    # Relatively constant around 5-8 km
    lambda_s = 5.0 + 0.05 * u  # λs in km
    lambda_s = np.clip(lambda_s, 4, 10)

    # Convert wavelengths to wavenumbers: k = 2π/λ
    kn = 2.0 * np.pi / lambda_n  # km⁻¹
    ks = 2.0 * np.pi / lambda_s  # km⁻¹

    # Convert to degrees if requested
    if output_units == 'deg':
        # 1 degree ≈ 111 km at equator (this is approximate)
        # λ (deg) = λ (km) / (111 km/deg)
        # k (deg⁻¹) = k (km⁻¹) × (111 km/deg)
        km_per_degree = 111.0
        kn = kn * km_per_degree
        ks = ks * km_per_degree

    # Fractal dimension D: relatively constant around 2.2
    # Slightly lower at faster spreading rates
    D = 2.25 - 0.003 * u
    D = np.clip(D, 2.0, 2.3)

    # Return as dictionary (or scalar values if input was scalar)
    if np.isscalar(spreading_rate):
        return {
            'H': float(H),
            'kn': float(kn),
            'ks': float(ks),
            'D': float(D)
        }
    else:
        return {
            'H': H,
            'kn': kn,
            'ks': ks,
            'D': D
        }


# Generate synthetic bathymetry with variable azimuth and sediment modification
def generate_synthetic_bathymetry_fft(grid_size, seafloor_age, sediment_thickness, params, use_full_equation=False):
    """
    Generate synthetic bathymetry using a von Kármán model with azimuthal orientation
    and sediment thickness modification.

    grid_size: tuple (nx, ny) specifying the size of the grid
    seafloor_age: 2D array of seafloor ages (used to calculate azimuth)
    sediment_thickness: 2D array of sediment thicknesses
    params: Dictionary containing the base abyssal hill parameters for each grid
            e.g., {'H': H, 'kn': kn, 'ks': ks, 'D': D}
    use_full_equation: bool, optional - use full Equation 1 (default: False, uses simplified)
    """
    nx, ny = grid_size

    # Calculate azimuth from seafloor age gradient
    azimuth = calculate_azimuth_from_age(seafloor_age)

    # Create wavenumber arrays (kx, ky)
    kx = np.fft.fftfreq(nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)

    # Modify parameters based on sediment thickness
    H_sed, kn_sed, ks_sed = modify_by_sediment(params['H'], params['kn'], params['ks'], sediment_thickness)

    # Generate the von Kármán spectrum with azimuth
    spectrum = von_karman_spectrum(kx, ky, H_sed, kn_sed, ks_sed, params['D'], azimuth, use_full_equation)
    
    # Generate random phase field
    random_phase = generate_random_phase(nx, ny)
    
    # Multiply the spectrum by random phase
    bathymetry_fft = np.sqrt(spectrum) * random_phase
    
    # Perform inverse FFT to get the synthetic bathymetry in space domain
    synthetic_bathymetry = np.real(ifft2(bathymetry_fft))
    
    return synthetic_bathymetry




#### Convolution Filter section

# Generate a random noise field
def generate_random_field(grid_size):
    """ Generate a random field with normally distributed random values. """
    return np.random.randn(*grid_size)

# Generate a spatial filter based on local parameters
def generate_spatial_filter(H, kn, ks, azimuth, filter_size=25, filter_type='gaussian'):
    """
    Generate a spatial filter based on local parameters H, kn, ks, and azimuth.

    Parameters:
    -----------
    H : float
        RMS height (meters)
    kn, ks : float
        Characteristic wavenumbers (km⁻¹) in ridge-normal and ridge-parallel directions
    azimuth : float
        Ridge orientation in radians
    filter_size : int, optional
        Size of the filter kernel (default: 25)
    filter_type : str, optional
        Type of filter to use:
        - 'gaussian' (default): Gaussian filter - faster, simpler
        - 'von_karman': von Kármán autocorrelation using Bessel function - theoretically correct

    Returns:
    --------
    spatial_filter : ndarray
        Normalized spatial filter (sum = 1)

    Notes:
    ------
    The Gaussian filter is simpler and faster but the von Kármán filter is theoretically
    correct for fractal terrain following the von Kármán autocorrelation function.
    In practice, both produce similar results.
    """
    # Create a 2D grid for the filter
    # Use consistent domain for both filter types
    x = np.linspace(-1, 1, filter_size)
    y = np.linspace(-1, 1, filter_size)

    xx, yy = np.meshgrid(x, y)

    # Rotate the coordinates based on the azimuth
    x_rot = xx * np.cos(azimuth) + yy * np.sin(azimuth)
    y_rot = -xx * np.sin(azimuth) + yy * np.cos(azimuth)

    if filter_type == 'von_karman':
        # Von Kármán autocorrelation: R(r) ∝ (kr)^ν × K_ν(kr)
        # where K_ν is the modified Bessel function of the second kind
        # and ν is the Hurst number (ν = 3 - D, but we don't have D here)
        # For typical abyssal hills, ν ≈ 0.8 (D ≈ 2.2)

        # Use typical ν = 0.8 for abyssal hills (corresponds to D = 2.2)
        nu = 0.8

        # Calculate ANISOTROPIC distance - same scaling as Gaussian
        # Divide by kn (small) → narrow in x_rot direction (perpendicular to ridge)
        # Divide by ks (large) → wide in y_rot direction (parallel to ridge)
        r_aniso = np.sqrt((x_rot / kn)**2 + (y_rot / ks)**2)

        # Avoid r=0 singularity
        r_aniso = np.where(r_aniso < 1e-10, 1e-10, r_aniso)

        # Von Kármán autocorrelation: (r)^ν × K_ν(r)
        # Scale the argument to get reasonable filter extent
        # Use a scaling factor to match the Gaussian filter spatial extent
        r_scaled = r_aniso * 2.0  # Adjust this factor to match Gaussian width

        spatial_filter = (r_scaled ** nu) * kv(nu, r_scaled)

        # Handle any infinities or NaNs
        spatial_filter = np.where(np.isfinite(spatial_filter), spatial_filter, 0)

        # Scale by H (relative to max, since Bessel function amplitude depends on ν)
        if np.max(spatial_filter) > 0:
            spatial_filter = H * spatial_filter / np.max(spatial_filter)
        else:
            spatial_filter = np.zeros_like(spatial_filter)

    else:  # filter_type == 'gaussian' (default)
        # Apply anisotropic Gaussian scaling based on kn and ks
        filter_exp = -(x_rot**2 / (2 * kn**2) + y_rot**2 / (2 * ks**2))

        # Generate a Gaussian filter and scale it by the RMS height (H)
        spatial_filter = H * np.exp(filter_exp)

    # Normalize the filter to ensure proper convolution (sum = 1)
    if np.sum(spatial_filter) > 0:
        spatial_filter /= np.sum(spatial_filter)
    else:
        # Fallback: if filter is all zeros, create a delta function
        spatial_filter = np.zeros_like(spatial_filter)
        center = filter_size // 2
        spatial_filter[center, center] = 1.0

    return spatial_filter

# Generate synthetic bathymetry using a spatial filter (ORIGINAL - SLOW)
def generate_bathymetry_spatial_filter_original(seafloor_age, sediment_thickness, params, random_field=None, filter_type='gaussian'):
    """
    Original pixel-by-pixel implementation (SLOW but simple).

    Kept for reference and backward compatibility testing.
    Use generate_bathymetry_spatial_filter() instead for better performance.
    """
    ny, nx = seafloor_age.shape

    # Calculate azimuth from seafloor age gradient
    azimuth = calculate_azimuth_from_age(seafloor_age)

    # Initialize the output bathymetry array
    bathymetry = np.zeros((ny,nx))

    # Loop over the grid to apply a spatial filter at each point
    for i in range(ny):
        for j in range(nx):
            # Extract local parameters
            H_local = params['H']
            kn_local = params['kn']
            ks_local = params['ks']
            azimuth_local = azimuth[i, j]

            # Modify the parameters based on sediment thickness
            H_local, kn_local, ks_local = modify_by_sediment(H_local, kn_local, ks_local, sediment_thickness[i, j])

            # Generate the local filter
            spatial_filter = generate_spatial_filter(H_local, kn_local, ks_local, azimuth_local, filter_type=filter_type)

            # Apply the filter to the random noise field at location (i, j)
            filtered_value = oaconvolve(random_field, spatial_filter, mode='same')[i, j]

            # Store the filtered value in the bathymetry map
            bathymetry[i, j] = filtered_value

    return bathymetry


# Generate synthetic bathymetry using a spatial filter (OPTIMIZED)
def generate_bathymetry_spatial_filter(seafloor_age, sediment_thickness, params, random_field=None,
                                       filter_type='gaussian', azimuth_bins=36, sediment_bins=5, optimize=True):
    """
    Generate synthetic bathymetry using a spatially varying filter based on von Kármán model.

    This optimized version pre-computes filters at discrete azimuth angles and sediment levels,
    reducing computation from O(ny×nx) full convolutions to O(azimuth_bins × sediment_bins) convolutions.

    Parameters:
    -----------
    seafloor_age : 2D array
        Seafloor ages in Myr (used to calculate azimuth)
    sediment_thickness : 2D array
        Sediment thicknesses in meters
    params : dict
        Dictionary containing base abyssal hill parameters
        e.g., {'H': H, 'kn': kn, 'ks': ks, 'D': D}
    random_field : 2D array, optional
        Pre-generated random field (if None, must be provided externally)
    filter_type : str, optional
        Type of spatial filter to use:
        - 'gaussian' (default): Gaussian filter - faster, simpler
        - 'von_karman': von Kármán autocorrelation using Bessel function
    azimuth_bins : int, optional
        Number of azimuth angles to pre-compute (default: 36, every 10°)
        Higher = more accurate but slower. Typical range: 18-72.
    sediment_bins : int, optional
        Number of sediment thickness bins to pre-compute (default: 5)
        Higher = more accurate but slower. Typical range: 3-10.
    optimize : bool, optional
        If True, use optimized filter bank approach (default)
        If False, use original pixel-by-pixel method (very slow!)

    Returns:
    --------
    bathymetry : 2D array
        Synthetic bathymetry in meters

    Notes:
    ------
    Optimization approach:
    1. Bin sediment thickness into discrete levels
    2. Pre-compute filters at discrete azimuth × sediment combinations
    3. Convolve random field with each filter once
    4. For each pixel, select result from nearest (azimuth, sediment) bin

    This reduces computation from ny×nx convolutions to ~36×5=180 convolutions,
    giving typical speedup of 10-50× for 100×100 grids while maintaining accuracy.
    """
    if not optimize:
        # Fall back to original slow implementation
        return generate_bathymetry_spatial_filter_original(
            seafloor_age, sediment_thickness, params, random_field, filter_type
        )

    ny, nx = seafloor_age.shape

    # Calculate azimuth from seafloor age gradient
    azimuth = calculate_azimuth_from_age(seafloor_age)

    # Bin sediment thickness
    sediment_min = np.min(sediment_thickness)
    sediment_max = np.max(sediment_thickness)
    sediment_range = sediment_max - sediment_min

    if sediment_range < 1e-6:
        # Uniform sediment - only need one bin
        sediment_levels = np.array([sediment_min])
        sediment_bins = 1
    else:
        sediment_levels = np.linspace(sediment_min, sediment_max, sediment_bins)

    # Pre-compute filters at discrete (azimuth, sediment) combinations
    azimuth_angles = np.linspace(-np.pi, np.pi, azimuth_bins, endpoint=False)

    H_base = params['H']
    kn_base = params['kn']
    ks_base = params['ks']

    # Generate filter bank: shape will be (azimuth_bins, sediment_bins, filter_size, filter_size)
    # But we only store convolved results: (azimuth_bins, sediment_bins, ny, nx)
    convolved_results = []

    for az in azimuth_angles:
        convolved_results_for_azimuth = []
        for sed in sediment_levels:
            # Modify parameters based on sediment thickness
            H_mod, kn_mod, ks_mod = modify_by_sediment(H_base, kn_base, ks_base, sed)

            # Generate filter at this (azimuth, sediment) combination
            filt = generate_spatial_filter(H_mod, kn_mod, ks_mod, az, filter_type=filter_type)

            # Convolve with random field (this is the expensive operation)
            convolved = oaconvolve(random_field, filt, mode='same')
            convolved_results_for_azimuth.append(convolved)

        convolved_results.append(convolved_results_for_azimuth)

    # Stack convolved results for efficient indexing
    convolved_stack = np.array(convolved_results)  # Shape: (azimuth_bins, sediment_bins, ny, nx)

    # For each pixel, find nearest (azimuth, sediment) bin
    # Normalize azimuth to [0, 2π) for easier binning
    azimuth_normalized = np.mod(azimuth + np.pi, 2*np.pi)  # [0, 2π)

    # Find nearest azimuth bin for each pixel
    azimuth_bin_width = 2 * np.pi / azimuth_bins
    azimuth_bin_idx = (azimuth_normalized / azimuth_bin_width).astype(int)
    azimuth_bin_idx = np.clip(azimuth_bin_idx, 0, azimuth_bins - 1)

    # Find nearest sediment bin for each pixel
    if sediment_bins == 1:
        sediment_bin_idx = np.zeros_like(sediment_thickness, dtype=int)
    else:
        sediment_bin_width = sediment_range / (sediment_bins - 1)
        sediment_bin_idx = ((sediment_thickness - sediment_min) / sediment_bin_width).astype(int)
        sediment_bin_idx = np.clip(sediment_bin_idx, 0, sediment_bins - 1)

    # Extract values using nearest-neighbor lookup (vectorized - no loops!)
    i_indices, j_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    bathymetry = convolved_stack[azimuth_bin_idx, sediment_bin_idx, i_indices, j_indices]

    return bathymetry



def generate_bathymetry_tiled(seafloor_age, sediment_thickness, params,
                              random_field, chunk_size=50, chunk_pad=20,
                              show_progress=True):
    """
    Generate synthetic bathymetry using tiled processing with progress bar.

    This is a wrapper around generate_bathymetry_spatial_filter() that processes
    the domain in overlapping tiles for better performance and progress tracking.

    Parameters:
    -----------
    seafloor_age : xarray.DataArray or ndarray
        Seafloor age grid (Myr)
    sediment_thickness : xarray.DataArray or ndarray
        Sediment thickness grid (m)
    params : dict
        Abyssal hill parameters {'H': float, 'kn': float, 'ks': float, 'D': float}
    random_field : xarray.DataArray or ndarray
        Pre-generated random field (same shape as inputs)
    chunk_size : int
        Size of processing chunks (default 50)
    chunk_pad : int
        Padding around chunks to avoid edge effects (default 20)
    show_progress : bool
        Show tqdm progress bar (default True)

    Returns:
    --------
    result : xarray.DataArray
        Synthetic bathymetry with same coordinates as input
    """
    # Ensure chunk_pad is even
    chunk_pad = int(2 * np.round(chunk_pad / 2))

    # Get full shape
    full_ny, full_nx = seafloor_age.shape

    # Generate chunk coordinates
    coords = []
    for y in range(0, full_ny - 1, chunk_size):
        for x in range(0, full_nx - 1, chunk_size):
            coords.append((y, x))

    # Process chunks
    results = []
    iterator = tqdm(coords, desc="Processing chunks") if (HAS_TQDM and show_progress) else coords

    for coord in iterator:
        y0, x0 = coord
        y1 = min(y0 + chunk_size + chunk_pad, full_ny)
        x1 = min(x0 + chunk_size + chunk_pad, full_nx)

        # Extract chunk
        chunk_age = seafloor_age[y0:y1, x0:x1]
        chunk_sed = sediment_thickness[y0:y1, x0:x1]
        chunk_random = random_field[y0:y1, x0:x1]

        # Skip empty chunks
        if hasattr(chunk_age, 'data'):
            if np.all(np.isnan(chunk_age.data)):
                continue
            chunk_age_data = chunk_age.data
            chunk_sed_data = chunk_sed.data
            chunk_random_data = chunk_random.data
        else:
            if np.all(np.isnan(chunk_age)):
                continue
            chunk_age_data = chunk_age
            chunk_sed_data = chunk_sed
            chunk_random_data = chunk_random

        # Generate bathymetry for chunk
        synthetic_chunk = generate_bathymetry_spatial_filter(
            chunk_age_data,
            chunk_sed_data,
            params,
            chunk_random_data
        )

        # Trim padding
        pad_half = chunk_pad // 2
        y_trim_start = pad_half if y0 > 0 else 0
        y_trim_end = synthetic_chunk.shape[0] - pad_half if y1 < full_ny else synthetic_chunk.shape[0]
        x_trim_start = pad_half if x0 > 0 else 0
        x_trim_end = synthetic_chunk.shape[1] - pad_half if x1 < full_nx else synthetic_chunk.shape[1]

        trimmed_chunk = synthetic_chunk[y_trim_start:y_trim_end, x_trim_start:x_trim_end]

        # Create DataArray if input was DataArray
        if hasattr(seafloor_age, 'lon'):
            result_chunk = xr.DataArray(
                trimmed_chunk,
                coords={
                    'lat': chunk_age.lat[y_trim_start:y_trim_end],
                    'lon': chunk_age.lon[x_trim_start:x_trim_end]
                },
                name='z'
            )
            results.append(result_chunk)
        else:
            results.append(trimmed_chunk)

    return results


def extend_longitude_range(da):
    """
    Extend a DataArray's longitude range from -180:180 to -190:190
    
    Parameters:
    -----------
    da : xarray.DataArray
        Input DataArray with longitude coordinates from -180 to 180
    
    Returns:
    --------
    xarray.DataArray
        Extended DataArray with longitude range -190 to 190
    """
    # Identify the longitude dimension
    lon_dim = [dim for dim in da.dims if 'lon' in dim.lower()][0]
    
    # Get original longitude coordinates
    original_lons = da[lon_dim].values
    
    # Create new longitude coordinates
    new_lons = np.concatenate([
        original_lons - 360,  # Add negative extension
        original_lons,         # Keep original data
        original_lons + 360    # Add positive extension
    ])
    
    # Create new data array with extended longitude
    extended_data = np.concatenate([
        da.sel({lon_dim: slice(original_lons.min(), original_lons.max())}),
        da,
        da.sel({lon_dim: slice(original_lons.min(), original_lons.max())})
    ], axis=da.dims.index(lon_dim))
    
    # Create new DataArray
    extended_da = xr.DataArray(
        extended_data, 
        coords={**da.coords, lon_dim: new_lons},
        dims=da.dims,
        attrs=da.attrs
    )
    
    return extended_da