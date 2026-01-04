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
def modify_by_sediment(H, lambda_n, lambda_s, sediment_thickness, D=None):
    """
    Modify abyssal hill parameters based on sediment thickness.
    Sediment drape reduces the RMS height (H) and increases wavelength (spatial smoothing).

    Based on physical expectation:
    - Equation 5 (from Goff & Arbic 2010): H(S) = H₀ - S/2 (reduces amplitude)
    - Wavelength modification (INVERTED from spectral form): λ(S) = λ₀ × (1 + 1.3·S/H₀)
      → More sediment → larger wavelengths → smoother appearance in map view

    Optionally implements Equation 8 for fractal dimension if D is provided:
    - Equation 8: D(S/H₀) = 2.2 - 1.5·(S/H₀) for S/H₀ ≤ 0.1, else 2.05

    Parameters:
    -----------
    H : float
        Original RMS height (m)
    lambda_n, lambda_s : float
        Original characteristic wavelengths (km)
    sediment_thickness : float
        Sediment thickness (m)
    D : float, optional
        Fractal dimension. If provided, also returns modified D. If None, only returns (H, lambda_n, lambda_s)

    Returns:
    --------
    (H_sed, lambda_n_sed, lambda_s_sed) if D is None
    (H_sed, lambda_n_sed, lambda_s_sed, D_sed) if D is provided
    """
    H0 = max(H, 0.01)  # Avoid division by zero
    H_sed = np.maximum(H0 - sediment_thickness / 2, 0.01)  # Equation 5

    # Sediment smooths seafloor → INCREASES wavelength (buries small features)
    # For spatial smoothness: more sediment → larger wavelengths → smoother appearance
    # Use reference H=200m to avoid spreading rate dependence in sediment effect
    H_ref = 200.0  # Reference RMS height for sediment normalization
    sediment_ratio = sediment_thickness / H_ref
    lambda_n_sed = lambda_n * (1.0 + 1.3 * sediment_ratio)  # INVERTED: sediment increases wavelength
    lambda_s_sed = lambda_s * (1.0 + 1.3 * sediment_ratio)  # INVERTED: sediment increases wavelength

    # Equation 8: Fractal dimension modification (optional)
    if D is not None:
        if sediment_ratio <= 0.1:
            D_sed = 2.2 - 1.5 * sediment_ratio
        else:
            D_sed = 2.05
        D_sed = np.clip(D_sed, 2.0, 2.3)
        return H_sed, lambda_n_sed, lambda_s_sed, D_sed

    return H_sed, lambda_n_sed, lambda_s_sed

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


def spreading_rate_to_params(spreading_rate, base_params=None):
    """
    Apply spreading rate scaling to base parameters using empirical relationships.

    This function allows you to control baseline abyssal hill characteristics while
    still applying physically-motivated spreading rate trends.

    Parameters:
    -----------
    spreading_rate : float or array
        Half-spreading rate in mm/yr
    base_params : dict, optional
        Base parameters to scale. If None, uses default values.
        Dictionary containing:
        - 'H': Base RMS height in meters (default: 250m)
        - 'lambda_n': Base wavelength perpendicular to ridge in km (default: 6.5km)
        - 'lambda_s': Base wavelength parallel to ridge in km (default: 16km)
        - 'D': Base fractal dimension (default: 2.2)

    Returns:
    --------
    params : dict
        Dictionary containing spreading-rate-modified parameters:
        - 'H': RMS height in meters
        - 'lambda_n': characteristic wavelength perpendicular to ridge (km)
        - 'lambda_s': characteristic wavelength parallel to ridge (km)
        - 'D': fractal dimension

    Notes:
    ------
    **Spreading Rate Scaling:**
    Based on empirical relationships from Goff & Arbic (2010) Figure 1:

    1. **H (amplitude) scaling:**
       - DECREASES with spreading rate
       - Faster spreading → hotter, weaker lithosphere → lower amplitude
       - Scale factor: f_H = 1.3 - 0.01 * u
       - At 10 mm/yr: f_H ≈ 1.2 (20% higher)
       - At 50 mm/yr: f_H ≈ 0.8 (20% lower)
       - At 100 mm/yr: f_H ≈ 0.3 (70% lower)

    2. **Wavelength (lambda_n, lambda_s) scaling:**
       - INCREASES with spreading rate
       - Faster spreading → more frequent resurfacing → smoother, wider features
       - Scale factor: f_λ = 0.3 + 0.014 * u
       - At 10 mm/yr: f_λ ≈ 0.44 (56% smaller)
       - At 50 mm/yr: f_λ ≈ 1.0 (unchanged)
       - At 100 mm/yr: f_λ ≈ 1.7 (70% larger)

    3. **D (fractal dimension) scaling:**
       - Slightly DECREASES with spreading rate
       - Relatively small effect
       - Additive shift: δD = -0.003 * (u - 50)

    **Example:**
    Base params: H=200m, λ_n=5km, λ_s=15km, D=2.2

    At 10 mm/yr (slow):  H=240m, λ_n=2.2km, λ_s=6.6km, D=2.32
    At 50 mm/yr (medium): H=160m, λ_n=5.0km, λ_s=15.0km, D=2.2
    At 100 mm/yr (fast): H=60m, λ_n=8.5km, λ_s=25.5km, D=2.05
    """
    # Set default base parameters if not provided
    if base_params is None:
        base_params = {
            'H': 250.0,      # meters
            'lambda_n': 6.5,  # km (width, normal to ridge)
            'lambda_s': 16.0, # km (length, parallel to ridge)
            'D': 2.2
        }

    # Extract base values
    H_base = base_params['H']
    lambda_n_base = base_params['lambda_n']
    lambda_s_base = base_params['lambda_s']
    D_base = base_params['D']

    # Convert spreading rate to array for vectorized operations
    u = np.atleast_1d(spreading_rate)

    # Scaling factors based on Goff & Arbic (2010) empirical relationships
    # Reference spreading rate: 50 mm/yr (where scale factors ≈ 1.0)
    u_ref = 50.0

    # H scale factor: DECREASES with spreading rate
    # At u=10: ~1.2, at u=50: ~0.8, at u=100: ~0.3
    f_H = 1.3 - 0.01 * u
    f_H = np.clip(f_H, 0.2, 1.5)

    # Wavelength scale factor: INCREASES with spreading rate
    # At u=10: ~0.44, at u=50: ~1.0, at u=100: ~1.7
    f_lambda = 0.3 + 0.014 * u
    f_lambda = np.clip(f_lambda, 0.3, 2.0)

    # D additive shift: slight decrease with spreading rate
    delta_D = -0.003 * (u - u_ref)
    delta_D = np.clip(delta_D, -0.3, 0.3)

    # Apply scaling
    H = H_base * f_H
    lambda_n = lambda_n_base * f_lambda
    lambda_s = lambda_s_base * f_lambda
    D = D_base + delta_D

    # Apply reasonable bounds
    H = np.clip(H, 10, 500)
    lambda_n = np.clip(lambda_n, 0.5, 30)
    lambda_s = np.clip(lambda_s, 1.0, 60)
    D = np.clip(D, 2.0, 2.4)

    # Return as dictionary (or scalar values if input was scalar)
    if np.isscalar(spreading_rate):
        return {
            'H': float(H),
            'lambda_n': float(lambda_n),
            'lambda_s': float(lambda_s),
            'D': float(D)
        }
    else:
        return {
            'H': H,
            'lambda_n': lambda_n,
            'lambda_s': lambda_s,
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
def generate_spatial_filter(H, lambda_n, lambda_s, azimuth, grid_spacing_km, filter_size=25, filter_type='gaussian'):
    """
    Generate a spatial filter based on local parameters H, lambda_n, lambda_s, and azimuth.

    Parameters:
    -----------
    H : float
        RMS height (meters)
    lambda_n, lambda_s : float
        Characteristic wavelengths (km):
        - lambda_n: WIDTH (normal to ridge, SMALLER ~3-8 km)
        - lambda_s: LENGTH (parallel to ridge, LARGER ~10-20 km)
        These are PHYSICAL wavelengths representing the dominant scales of abyssal hills.
        lambda_s > lambda_n creates elongated ridges parallel to paleo-ridge axis.
    azimuth : float
        Ridge orientation in radians
    grid_spacing_km : float
        Grid spacing in kilometers (e.g., 1.85 km for 1 arcmin at equator)
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
    The lambda_n/lambda_s parameters are now in PHYSICAL units (km), making them:
    - Directly interpretable in terms of real-world abyssal hill characteristics
    - Independent of grid resolution (same physical wavelength works at any resolution)
    - Comparable to observations in the literature

    The wavelengths are converted to wavenumbers in pixel⁻¹ units using the grid spacing:
        kn = grid_spacing_km / lambda_n
        ks = grid_spacing_km / lambda_s

    The Gaussian filter is simpler and faster but the von Kármán filter is theoretically
    correct for fractal terrain following the von Kármán autocorrelation function.
    In practice, both produce similar results.
    """
    # Convert physical wavelengths (km) to pixels
    # These are the characteristic length scales for the filter
    lambda_n_pixels = lambda_n / grid_spacing_km  # pixels
    lambda_s_pixels = lambda_s / grid_spacing_km  # pixels

    # Create a 2D grid for the filter
    # Use consistent domain for both filter types
    # The filter coordinates span filter_size pixels but are normalized to [-1, 1]
    # So we need to scale wavelengths: 1 unit in filter coords = filter_size/2 pixels
    x = np.linspace(-1, 1, filter_size)
    y = np.linspace(-1, 1, filter_size)

    # Scale wavelengths to filter coordinate system
    # lambda in pixels → lambda in filter coords = lambda / (filter_size/2)
    lambda_n_scaled = lambda_n_pixels / (filter_size / 2.0)
    lambda_s_scaled = lambda_s_pixels / (filter_size / 2.0)

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

        # Calculate ANISOTROPIC distance using scaled wavelengths
        # lambda_n is WIDTH (small) → narrow in that direction
        # lambda_s is LENGTH (large) → wide in that direction
        # y_rot aligns with ridge-parallel (elongation) → use lambda_s_scaled
        # x_rot aligns with ridge-perpendicular (width) → use lambda_n_scaled
        r_aniso = np.sqrt((x_rot / lambda_n_scaled)**2 + (y_rot / lambda_s_scaled)**2)

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
        # Apply anisotropic Gaussian scaling using scaled wavelengths
        # Larger wavelength → wider filter → smoother texture (INTUITIVE!)
        # y_rot → ridge-parallel (elongation) → use lambda_s_scaled (large)
        # x_rot → ridge-perpendicular (width) → use lambda_n_scaled (small)
        filter_exp = -((x_rot / lambda_n_scaled)**2 + (y_rot / lambda_s_scaled)**2) / 2

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
def generate_bathymetry_spatial_filter_original(seafloor_age, sediment_thickness, params, grid_spacing_km, random_field=None, filter_type='gaussian'):
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
            lambda_n_local = params['lambda_n']
            lambda_s_local = params['lambda_s']
            azimuth_local = azimuth[i, j]

            # Modify the parameters based on sediment thickness
            H_local, lambda_n_local, lambda_s_local = modify_by_sediment(H_local, lambda_n_local, lambda_s_local, sediment_thickness[i, j])

            # Generate the local filter
            spatial_filter = generate_spatial_filter(H_local, lambda_n_local, lambda_s_local, azimuth_local, grid_spacing_km, filter_type=filter_type)

            # Apply the filter to the random noise field at location (i, j)
            filtered_value = oaconvolve(random_field, spatial_filter, mode='same')[i, j]

            # Store the filtered value in the bathymetry map
            bathymetry[i, j] = filtered_value

    return bathymetry


# Generate synthetic bathymetry using a spatial filter (OPTIMIZED)
def generate_bathymetry_spatial_filter(seafloor_age, sediment_thickness, params, grid_spacing_km, random_field=None,
                                       filter_type='gaussian', azimuth_bins=36, sediment_bins=5,
                                       spreading_rate_bins=5, base_params=None, optimize=True):
    """
    Generate synthetic bathymetry using a spatially varying filter based on von Kármán model.

    This optimized version pre-computes filters at discrete azimuth angles, sediment levels,
    and spreading rates, reducing computation from O(ny×nx) full convolutions to
    O(azimuth_bins × sediment_bins × spreading_rate_bins) convolutions.

    Parameters:
    -----------
    seafloor_age : 2D array
        Seafloor ages in Myr (used to calculate azimuth and spreading rate)
    sediment_thickness : 2D array
        Sediment thicknesses in meters
    params : dict
        Dictionary containing base abyssal hill parameters
        e.g., {'H': H, 'lambda_n': lambda_n, 'lambda_s': lambda_s, 'D': D}
        where H is in meters and lambda_n, lambda_s are in km

        If base_params is provided, these params are IGNORED and base_params is used instead.
        If base_params is None, these params are used as the base for spreading rate scaling.
    grid_spacing_km : float
        Grid spacing in kilometers (e.g., 1.85 km for 1 arcmin at equator)
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
    spreading_rate_bins : int, optional
        Number of spreading rate bins to pre-compute (default: 5)
        Higher = more accurate but slower. Typical range: 3-10.
        Set to 1 to disable spreading rate variation (use single params).
    base_params : dict, optional
        Base parameters to use for spreading rate scaling. If None, uses params.
        Dictionary with keys: 'H', 'lambda_n', 'lambda_s', 'D'
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
    1. Calculate spreading rate from seafloor age gradient
    2. Bin spreading rate, sediment thickness, and azimuth into discrete levels
    3. Pre-compute filters at discrete (azimuth, sediment, spreading_rate) combinations
    4. Convolve random field with each filter once
    5. For each pixel, select result from nearest (azimuth, sediment, spreading_rate) bin

    This reduces computation from ny×nx convolutions to ~36×5×5=900 convolutions,
    giving typical speedup of 5-20× for 100×100 grids while maintaining accuracy.

    Spreading rate variation:
    - If spreading_rate_bins > 1, parameters vary spatially with spreading rate
    - Each bin gets scaled parameters via spreading_rate_to_params(rate, base_params)
    - This allows smooth spatial transitions in abyssal hill characteristics
    """
    if not optimize:
        # Fall back to original slow implementation
        return generate_bathymetry_spatial_filter_original(
            seafloor_age, sediment_thickness, params, grid_spacing_km, random_field, filter_type
        )

    ny, nx = seafloor_age.shape

    # Determine base parameters for spreading rate scaling
    if base_params is None:
        base_params = params

    # Calculate azimuth from seafloor age gradient
    azimuth = calculate_azimuth_from_age(seafloor_age)

    # Calculate spreading rate from seafloor age gradient
    spreading_rate = calculate_spreading_rate_from_age(seafloor_age, grid_spacing_km)

    # Handle NaN values in spreading rate (e.g., from uniform age)
    spreading_rate_valid = spreading_rate[~np.isnan(spreading_rate)]
    if len(spreading_rate_valid) == 0:
        # No valid spreading rate - use base params everywhere
        spreading_rate = np.full_like(seafloor_age, 50.0)  # Default to 50 mm/yr
    else:
        # Fill NaN values with median
        spreading_rate = np.where(np.isnan(spreading_rate),
                                  np.nanmedian(spreading_rate),
                                  spreading_rate)

    # Bin spreading rate
    sr_min = np.min(spreading_rate)
    sr_max = np.max(spreading_rate)
    sr_range = sr_max - sr_min

    if sr_range < 1e-6 or spreading_rate_bins == 1:
        # Uniform spreading rate or disabled - only need one bin
        spreading_rate_levels = np.array([np.mean(spreading_rate)])
        spreading_rate_bins = 1
    else:
        spreading_rate_levels = np.linspace(sr_min, sr_max, spreading_rate_bins)

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

    # Pre-compute filters at discrete (azimuth, sediment, spreading_rate) combinations
    azimuth_angles = np.linspace(-np.pi, np.pi, azimuth_bins, endpoint=False)

    # Generate 3D filter bank: (azimuth_bins, sediment_bins, spreading_rate_bins)
    # We store convolved results: (azimuth_bins, sediment_bins, spreading_rate_bins, ny, nx)
    convolved_results = []

    for az in azimuth_angles:
        convolved_results_for_azimuth = []
        for sed in sediment_levels:
            convolved_results_for_sediment = []
            for sr in spreading_rate_levels:
                # Get parameters for this spreading rate
                params_sr = spreading_rate_to_params(sr, base_params=base_params)

                # Extract spreading-rate-scaled parameters
                H_sr = params_sr['H']
                lambda_n_sr = params_sr['lambda_n']
                lambda_s_sr = params_sr['lambda_s']

                # Further modify parameters based on sediment thickness
                H_mod, lambda_n_mod, lambda_s_mod = modify_by_sediment(H_sr, lambda_n_sr, lambda_s_sr, sed)

                # Generate filter at this (azimuth, sediment, spreading_rate) combination
                filt = generate_spatial_filter(H_mod, lambda_n_mod, lambda_s_mod, az, grid_spacing_km, filter_type=filter_type)

                # Convolve with random field (this is the expensive operation)
                convolved = oaconvolve(random_field, filt, mode='same')
                convolved_results_for_sediment.append(convolved)

            convolved_results_for_azimuth.append(convolved_results_for_sediment)
        convolved_results.append(convolved_results_for_azimuth)

    # Stack convolved results for efficient indexing
    convolved_stack = np.array(convolved_results)  # Shape: (azimuth_bins, sediment_bins, spreading_rate_bins, ny, nx)

    # For each pixel, interpolate between bins (trilinear interpolation)
    # This eliminates visible boundaries at bin transitions

    # Normalize azimuth to [0, 2π) for easier binning
    azimuth_normalized = np.mod(azimuth + np.pi, 2*np.pi)  # [0, 2π)

    # Calculate continuous bin positions (not rounded)
    azimuth_bin_width = 2 * np.pi / azimuth_bins
    azimuth_bin_pos = azimuth_normalized / azimuth_bin_width

    if sediment_bins == 1:
        sediment_bin_pos = np.zeros_like(sediment_thickness)
    else:
        sediment_bin_width = sediment_range / (sediment_bins - 1)
        sediment_bin_pos = (sediment_thickness - sediment_min) / sediment_bin_width

    if spreading_rate_bins == 1:
        sr_bin_pos = np.zeros_like(spreading_rate)
    else:
        sr_bin_width = sr_range / (spreading_rate_bins - 1)
        sr_bin_pos = (spreading_rate - sr_min) / sr_bin_width

    # Get integer bin indices and fractional parts for interpolation
    az_idx0 = np.floor(azimuth_bin_pos).astype(int)
    az_idx1 = az_idx0 + 1
    az_frac = azimuth_bin_pos - az_idx0

    # Handle azimuth wraparound (circular)
    az_idx0 = np.clip(az_idx0, 0, azimuth_bins - 1)
    az_idx1 = np.mod(az_idx1, azimuth_bins)

    sed_idx0 = np.floor(sediment_bin_pos).astype(int)
    sed_idx1 = np.minimum(sed_idx0 + 1, sediment_bins - 1)
    sed_frac = sediment_bin_pos - sed_idx0
    sed_idx0 = np.clip(sed_idx0, 0, sediment_bins - 1)

    sr_idx0 = np.floor(sr_bin_pos).astype(int)
    sr_idx1 = np.minimum(sr_idx0 + 1, spreading_rate_bins - 1)
    sr_frac = sr_bin_pos - sr_idx0
    sr_idx0 = np.clip(sr_idx0, 0, spreading_rate_bins - 1)

    # Trilinear interpolation: interpolate across 8 corners of bin cube
    # This is computationally efficient and eliminates bin discontinuities
    i_indices, j_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # Get values at 8 corners
    c000 = convolved_stack[az_idx0, sed_idx0, sr_idx0, i_indices, j_indices]
    c001 = convolved_stack[az_idx0, sed_idx0, sr_idx1, i_indices, j_indices]
    c010 = convolved_stack[az_idx0, sed_idx1, sr_idx0, i_indices, j_indices]
    c011 = convolved_stack[az_idx0, sed_idx1, sr_idx1, i_indices, j_indices]
    c100 = convolved_stack[az_idx1, sed_idx0, sr_idx0, i_indices, j_indices]
    c101 = convolved_stack[az_idx1, sed_idx0, sr_idx1, i_indices, j_indices]
    c110 = convolved_stack[az_idx1, sed_idx1, sr_idx0, i_indices, j_indices]
    c111 = convolved_stack[az_idx1, sed_idx1, sr_idx1, i_indices, j_indices]

    # Interpolate along spreading rate (z) axis
    c00 = c000 * (1 - sr_frac) + c001 * sr_frac
    c01 = c010 * (1 - sr_frac) + c011 * sr_frac
    c10 = c100 * (1 - sr_frac) + c101 * sr_frac
    c11 = c110 * (1 - sr_frac) + c111 * sr_frac

    # Interpolate along sediment (y) axis
    c0 = c00 * (1 - sed_frac) + c01 * sed_frac
    c1 = c10 * (1 - sed_frac) + c11 * sed_frac

    # Interpolate along azimuth (x) axis
    bathymetry = c0 * (1 - az_frac) + c1 * az_frac

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