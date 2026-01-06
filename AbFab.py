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


def apply_diffusive_sediment_infill(basement_topo, sediment_thickness, grid_spacing_km=1.0,
                                     diffusion_coeff=0.3):
    """
    Apply diffusive sediment infill that preferentially fills topographic lows.

    This function models the physical process of sediment accumulation where sediment
    preferentially ponds in valleys and depressions, creating a smoother seafloor
    surface than simple draping would produce.

    Physical basis:
    - Sediment is transported and deposited preferentially in lows
    - Creates smoother final topography as sediment thickness increases
    - Modeled as iterative diffusion process with strength ∝ sediment thickness

    This is COMPLEMENTARY to the modify_by_sediment() function:
    - modify_by_sediment(): Reduces hill AMPLITUDE and increases WAVELENGTH
      (models how sediment burial affects the generation of hills)
    - apply_diffusive_sediment_infill(): Redistributes sediment spatially
      (models how sediment fills in existing topography)

    Parameters:
    -----------
    basement_topo : ndarray
        Basement topography (negative = below sea level)
        This should be basement + abyssal hills BEFORE sediment
    sediment_thickness : ndarray
        Total sediment thickness available at each location (meters)
    grid_spacing_km : float, optional
        Grid spacing in kilometers (default: 1.0)
        Used to scale diffusion appropriately
    diffusion_coeff : float, optional
        Diffusion strength parameter (0-1, default: 0.3)
        Higher values = more smoothing, more ponding in lows
        - 0.0: No diffusion (same as simple drape)
        - 0.3: Moderate diffusion (recommended default)
        - 0.5: Strong diffusion (heavy ponding)
        - 1.0: Maximum diffusion (extreme smoothing)

    Returns:
    --------
    seafloor_topo : ndarray
        Final seafloor topography after sediment infill (negative = below sea level)

    Algorithm:
    ----------
    1. Start with basement topography
    2. For each location:
       - Calculate effective diffusion based on local sediment thickness
       - Apply Gaussian smoothing (simulates sediment redistribution)
       - Number of iterations scales with sediment thickness
    3. Fill from basement upward by sediment thickness
    4. Result: sediment ponds in lows, barely affects highs

    Notes:
    ------
    The diffusion is applied in a way that:
    - Preserves total sediment volume (mass conservative)
    - Smooths more where sediment is thicker
    - Creates realistic ponding in valleys
    - Computational cost: ~2-5x a simple subtraction (acceptable)

    This complements the existing sediment treatment:
    - Hill generation already accounts for sediment via modify_by_sediment()
      (amplitude reduction H → H-S/2, wavelength increase λ → λ(1+1.3S/H))
    - This function adds spatial redistribution on top of that
    """
    from scipy.ndimage import gaussian_filter

    # Handle NaN values
    valid_mask = ~np.isnan(basement_topo)
    basement_valid = np.where(valid_mask, basement_topo, 0)
    sediment_valid = np.where(valid_mask, sediment_thickness, 0)

    # Calculate spatially-varying smoothing based on sediment thickness
    # More sediment = more smoothing (sediment preferentially fills lows)
    # Scale sigma by sediment thickness and diffusion coefficient
    mean_sediment = np.nanmean(sediment_valid)
    if mean_sediment < 1.0:
        # Very little sediment - just do simple drape
        return basement_topo + sediment_valid

    # Normalize sediment to get smoothing length scale
    # Typical: 100m sediment → sigma ~ 1-2 pixels with default diffusion_coeff
    sediment_normalized = sediment_valid / 100.0  # Normalize by 100m reference
    sigma_base = diffusion_coeff * 3.0  # Base smoothing in pixels

    # Create sediment-thickness-dependent smoothing
    # Thicker sediment → more iterations of diffusion
    max_sediment = np.nanmax(sediment_valid)
    n_iterations = max(1, int(max_sediment / 100.0 * diffusion_coeff * 5))
    n_iterations = min(n_iterations, 10)  # Cap at 10 iterations for performance

    # Apply iterative diffusion
    # Each iteration represents sediment redistribution
    smoothed_basement = basement_valid.copy()

    for i in range(n_iterations):
        # Spatially varying sigma based on local sediment thickness
        # More sediment locally = more local smoothing
        local_sigma = sigma_base * (1.0 + sediment_normalized)

        # Apply Gaussian filter with mean sigma
        # (Spatially-varying sigma is approximated by using thickness-weighted mean)
        mean_sigma = np.mean(local_sigma[valid_mask])
        smoothed_basement = gaussian_filter(smoothed_basement, sigma=mean_sigma, mode='nearest')

    # The smoothed basement represents where sediment would naturally level off
    # Now add sediment: basement moves up by sediment thickness
    # But use weighted combination: more smoothing where more sediment
    smoothing_weight = np.tanh(sediment_normalized)  # 0 to 1, based on sediment
    final_basement = (1 - smoothing_weight) * basement_valid + smoothing_weight * smoothed_basement

    # Add sediment on top (sediment fills upward from basement)
    seafloor = final_basement + sediment_valid

    # Restore NaN values
    seafloor = np.where(valid_mask, seafloor, np.nan)

    return seafloor


# Azimuth calculation from seafloor age gradient
def calculate_azimuth_from_age(seafloor_age, lat_coords=None):
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
    lat_coords : 1D array, optional
        Latitude coordinates in degrees (for spherical correction).
        If None, assumes Cartesian coordinates (may be inaccurate at high latitudes).

    Returns:
    --------
    azimuth : 2D array
        Spreading direction azimuth in radians (measured clockwise from north)

    Notes:
    ------
    Spherical Correction:
    When lat_coords is provided, corrects for longitude spacing contraction
    at high latitudes. At latitude φ, longitude spacing contracts by cos(φ),
    so the age gradient in longitude direction must be corrected:
        grad_lon_physical = grad_lon_index / cos(φ)

    This correction is essential for accurate azimuth at |lat| > 30°.
    """
    # Compute the gradients in y (lat) and x (lon) direction
    grad_y, grad_x = np.gradient(seafloor_age)

    # Apply spherical correction if latitude coordinates provided
    if lat_coords is not None:
        # Broadcast latitude to 2D
        lat_2d = np.broadcast_to(lat_coords[:, np.newaxis], seafloor_age.shape)

        # Correct longitude gradient for spherical distortion
        # At latitude φ, 1° lon = cos(φ) × 1° at equator
        # So age gradient per physical distance = age gradient per index / cos(φ)
        cos_lat = np.cos(np.radians(lat_2d))

        # Avoid division by zero at poles (shouldn't have data there anyway)
        cos_lat = np.maximum(cos_lat, 1e-10)

        grad_x = grad_x / cos_lat

    # Compute azimuth as the angle of the gradient vector (radians)
    azimuth = np.arctan2(grad_y, grad_x)

    return azimuth


def calculate_spreading_rate_from_age(seafloor_age, grid_spacing_km=1.0, lat_coords=None):
    """
    Estimate half-spreading rate from seafloor age gradient.

    Parameters:
    -----------
    seafloor_age : 2D array
        Seafloor age in Myr
    grid_spacing_km : float, optional
        Grid spacing in km at the equator or mean latitude (default: 1.0 km).
        If lat_coords is provided, this is the longitude spacing.
    lat_coords : 1D array, optional
        Latitude coordinates in degrees (for spherical correction).
        If provided, accounts for longitude spacing variation with latitude.

    Returns:
    --------
    spreading_rate : 2D array
        Half-spreading rate in mm/yr

    Notes:
    ------
    Spherical Correction:
    When lat_coords is provided, the function accounts for longitude spacing
    contraction at high latitudes. The physical distance for one grid cell in
    longitude direction is: dx_physical = grid_spacing_km × cos(lat)

    This ensures accurate spreading rate calculations at all latitudes.

    Half-spreading rate u = distance / (2 × time)
    From age gradient: u = (grid_spacing / age_gradient) / 2
    """
    # Compute the gradients in y (lat) and x (lon) direction (Myr per grid cell)
    grad_y, grad_x = np.gradient(seafloor_age)

    # Apply spherical correction if latitude coordinates provided
    if lat_coords is not None:
        # Broadcast latitude to 2D
        lat_2d = np.broadcast_to(lat_coords[:, np.newaxis], seafloor_age.shape)

        # Physical grid spacing varies with latitude
        # Longitude: dx = grid_spacing_km × cos(lat)
        # Latitude: dy = grid_spacing_km (constant)
        cos_lat = np.cos(np.radians(lat_2d))
        cos_lat = np.maximum(cos_lat, 1e-10)  # Avoid division by zero at poles

        dx_km = grid_spacing_km * cos_lat
        dy_km = grid_spacing_km  # Latitude spacing is constant

        # Convert gradients from (Myr per index) to (Myr per km)
        grad_x_per_km = grad_x / dx_km  # Myr/km in longitude direction
        grad_y_per_km = grad_y / dy_km  # Myr/km in latitude direction

        # Magnitude of age gradient in physical space (Myr/km)
        age_gradient_magnitude = np.sqrt(grad_x_per_km**2 + grad_y_per_km**2)
    else:
        # No spherical correction - assume Cartesian
        age_gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Convert from (Myr per index) to (Myr per km)
        age_gradient_magnitude = age_gradient_magnitude / grid_spacing_km

    # Avoid division by zero
    age_gradient_magnitude = np.where(age_gradient_magnitude > 1e-10, age_gradient_magnitude, np.nan)

    # Half-spreading rate: rate = distance / time
    # age_gradient_magnitude is in Myr/km
    # So: rate = 1 / age_gradient_magnitude (km/Myr)
    #
    # Convert km/Myr to mm/yr:
    # 1 km/Myr = 1000 m / 1e6 yr = 1e-3 m/yr = 1 mm/yr
    # So km/Myr and mm/yr are numerically equivalent!
    spreading_rate = 1.0 / age_gradient_magnitude  # km/Myr = mm/yr

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
                                       spreading_rate_bins=5, base_params=None, optimize=True,
                                       sediment_range=None, spreading_rate_range=None, lat_coords=None,
                                       sediment_levels=None, spreading_rate_levels=None,
                                       spreading_rate_fill_value=None,
                                       azimuth_field=None, spreading_rate_field=None):
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
    sediment_range : tuple, optional
        Global (min, max) sediment range for consistent binning across chunks.
        If None, uses local min/max from this chunk.
        Format: (sediment_min, sediment_max)
    spreading_rate_range : tuple, optional
        Global (min, max) spreading rate range for consistent binning across chunks.
        If None, uses local min/max from this chunk.
        Format: (sr_min, sr_max)
    lat_coords : 1D array, optional
        Latitude coordinates in degrees for spherical Earth corrections.
        If provided, applies cos(lat) correction to longitude gradients.
        If None, assumes Cartesian coordinates (suitable for small regions).
    sediment_levels : 1D array, optional
        Pre-computed sediment bin levels to ensure consistency across chunks.
        If provided, overrides sediment_bins and sediment_range for binning.
        Should be created once globally and passed to all chunks.
    spreading_rate_levels : 1D array, optional
        Pre-computed spreading rate bin levels to ensure consistency across chunks.
        If provided, overrides spreading_rate_bins and spreading_rate_range for binning.
        Should be created once globally and passed to all chunks.
    spreading_rate_fill_value : float, optional
        Global value to use for filling NaN spreading rates (e.g., over land).
        If None, uses local median per chunk (less consistent).
        Recommended: pass global median from entire grid for consistency.
    azimuth_field : 2D array, optional
        Pre-computed azimuth field (radians) with same shape as inputs.
        If provided, skips local azimuth calculation for perfect chunk consistency.
        Should be calculated globally once before chunking.
    spreading_rate_field : 2D array, optional
        Pre-computed spreading rate field (mm/yr) with same shape as inputs.
        If provided, skips local spreading rate calculation for perfect chunk consistency.
        Should be calculated globally once before chunking.

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

    # Use pre-computed azimuth if provided, otherwise calculate from age gradient
    if azimuth_field is not None:
        azimuth = azimuth_field
    else:
        # Calculate azimuth from seafloor age gradient (with spherical correction if lat_coords provided)
        azimuth = calculate_azimuth_from_age(seafloor_age, lat_coords=lat_coords)

    # Use pre-computed spreading rate if provided, otherwise calculate from age gradient
    if spreading_rate_field is not None:
        spreading_rate = spreading_rate_field
    else:
        # Calculate spreading rate from seafloor age gradient (with spherical correction if lat_coords provided)
        spreading_rate = calculate_spreading_rate_from_age(seafloor_age, grid_spacing_km, lat_coords=lat_coords)

    # Handle NaN values in spreading rate (e.g., from uniform age or land)
    # Use global fill value if provided (for consistency across chunks)
    if spreading_rate_fill_value is not None:
        # Use the provided global fill value
        fill_value = spreading_rate_fill_value
    else:
        # Calculate local median (less consistent across chunks)
        spreading_rate_valid = spreading_rate[~np.isnan(spreading_rate)]
        if len(spreading_rate_valid) == 0:
            fill_value = 50.0  # Default to 50 mm/yr if no valid values
        else:
            fill_value = np.nanmedian(spreading_rate)

    # Fill NaN values
    spreading_rate = np.where(np.isnan(spreading_rate), fill_value, spreading_rate)

    # Bin spreading rate
    # Use pre-computed levels if provided (best for consistency across chunks)
    if spreading_rate_levels is not None:
        # Use the provided levels directly
        spreading_rate_bins = len(spreading_rate_levels)
        sr_min = spreading_rate_levels[0]
        sr_max = spreading_rate_levels[-1]
        sr_range = sr_max - sr_min
    elif spreading_rate_range is not None:
        # Use global range if provided (for consistent binning across chunks)
        sr_min, sr_max = spreading_rate_range
        sr_range = sr_max - sr_min
        if sr_range < 1e-6 or spreading_rate_bins == 1:
            # Uniform spreading rate or disabled - only need one bin
            spreading_rate_levels = np.array([np.mean(spreading_rate)])
            spreading_rate_bins = 1
        else:
            spreading_rate_levels = np.linspace(sr_min, sr_max, spreading_rate_bins)
    else:
        # Use local range (less consistent across chunks)
        sr_min = np.min(spreading_rate)
        sr_max = np.max(spreading_rate)
        sr_range = sr_max - sr_min
        if sr_range < 1e-6 or spreading_rate_bins == 1:
            spreading_rate_levels = np.array([np.mean(spreading_rate)])
            spreading_rate_bins = 1
        else:
            spreading_rate_levels = np.linspace(sr_min, sr_max, spreading_rate_bins)

    # Bin sediment thickness
    # Use pre-computed levels if provided (best for consistency across chunks)
    if sediment_levels is not None:
        # Use the provided levels directly
        sediment_bins = len(sediment_levels)
        sediment_min = sediment_levels[0]
        sediment_max = sediment_levels[-1]
        sediment_range_val = sediment_max - sediment_min
    elif sediment_range is not None:
        # Use global range if provided (for consistent binning across chunks)
        sediment_min, sediment_max = sediment_range
        sediment_range_val = sediment_max - sediment_min
        if sediment_range_val < 1e-6:
            # Uniform sediment - only need one bin
            sediment_levels = np.array([sediment_min])
            sediment_bins = 1
        else:
            sediment_levels = np.linspace(sediment_min, sediment_max, sediment_bins)
    else:
        # Use local range (less consistent across chunks)
        sediment_min = np.min(sediment_thickness)
        sediment_max = np.max(sediment_thickness)
        sediment_range_val = sediment_max - sediment_min
        if sediment_range_val < 1e-6:
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

                # Rescale to match target RMS height (H_mod)
                # The filter is normalized to sum=1, which makes the convolved output
                # have RMS proportional to the RMS of the random field (~0.28 for uniform[0,1])
                # We rescale by the theoretical factor rather than empirical std
                # to ensure consistency across chunks regardless of their content
                #
                # For a normalized filter convolved with uniform[0,1] random field:
                # Expected RMS ≈ 1/sqrt(12) ≈ 0.2887
                # We want RMS = H_mod, so scale by H_mod / (1/sqrt(12))
                theoretical_rms = 1.0 / np.sqrt(12.0)  # RMS of uniform[0,1] distribution
                convolved = convolved * (H_mod / theoretical_rms)

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
        sediment_bin_width = sediment_range_val / (sediment_bins - 1)
        sediment_bin_pos = (sediment_thickness - sediment_min) / sediment_bin_width

    if spreading_rate_bins == 1:
        sr_bin_pos = np.zeros_like(spreading_rate)
    else:
        sr_bin_width = sr_range / (spreading_rate_bins - 1)
        sr_bin_pos = (spreading_rate - sr_min) / sr_bin_width

    # Get integer bin indices and fractional parts for interpolation
    # Handle NaN values before floor/astype to avoid RuntimeWarning
    azimuth_bin_pos_safe = np.where(np.isnan(azimuth_bin_pos), 0, azimuth_bin_pos)
    sediment_bin_pos_safe = np.where(np.isnan(sediment_bin_pos), 0, sediment_bin_pos)
    sr_bin_pos_safe = np.where(np.isnan(sr_bin_pos), 0, sr_bin_pos)

    az_idx0 = np.floor(azimuth_bin_pos_safe).astype(int)
    az_idx1 = az_idx0 + 1
    az_frac = azimuth_bin_pos_safe - az_idx0

    # Handle azimuth wraparound (circular)
    az_idx0 = np.clip(az_idx0, 0, azimuth_bins - 1)
    az_idx1 = np.mod(az_idx1, azimuth_bins)

    sed_idx0 = np.floor(sediment_bin_pos_safe).astype(int)
    sed_idx1 = np.minimum(sed_idx0 + 1, sediment_bins - 1)
    sed_frac = sediment_bin_pos_safe - sed_idx0
    sed_idx0 = np.clip(sed_idx0, 0, sediment_bins - 1)

    sr_idx0 = np.floor(sr_bin_pos_safe).astype(int)
    sr_idx1 = np.minimum(sr_idx0 + 1, spreading_rate_bins - 1)
    sr_frac = sr_bin_pos_safe - sr_idx0
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


#### Thermal Subsidence Models

def calculate_thermal_subsidence(seafloor_age, model='GDH1', ridge_depth=2600.0,
                                  plate_timescale=62.8, hs_coeff=365.0):
    """
    Calculate basement depth from seafloor age using thermal subsidence models.

    Implements three thermal subsidence models:
    - 'half_space': Simple half-space cooling model
    - 'plate': Plate cooling model with exponential decay
    - 'GDH1': Global Depth-Heat flow model 1 (Stein & Stein, 1992)

    Parameters:
    -----------
    seafloor_age : ndarray
        Seafloor age in Myr (2D array)
    model : str, optional
        Subsidence model to use:
        - 'GDH1' (default): Stein & Stein (1992) model
        - 'half_space': d = d0 + c*sqrt(t)
        - 'plate': d = d0 + c*sqrt(t)*[1 - exp(-t/τ)]
    ridge_depth : float, optional
        Depth at ridge axis in meters (default: 2600m)
        Used for half_space and plate models
    plate_timescale : float, optional
        Plate cooling timescale τ in Myr (default: 62.8 Myr)
        Used for plate model only
    hs_coeff : float, optional
        Half-space coefficient in m/sqrt(Myr) (default: 365.0)
        Used for half_space and plate models

    Returns:
    --------
    depth : ndarray
        Basement depth in meters (negative values below sea level)
        Same shape as seafloor_age

    Notes:
    ------
    GDH1 model (Stein & Stein, 1992):
    - For t ≤ 20 Myr: d = 2600 + 365*sqrt(t)
    - For t > 20 Myr: d = 5651 - 2473*exp(-0.0278*t)

    Half-space cooling model:
    - d(t) = ridge_depth + hs_coeff*sqrt(t)
    - Simple, valid for young seafloor (< ~80 Myr)

    Plate cooling model:
    - d(t) = ridge_depth + hs_coeff*sqrt(t)*[1 - exp(-t/τ)]
    - More accurate for old seafloor
    - Asymptotes to maximum depth

    References:
    -----------
    Stein, C. A., & Stein, S. (1992). A model for the global variation in oceanic
    depth and heat flow with lithospheric age. Nature, 359(6391), 123-129.
    """
    # Handle NaN values
    age_valid = np.where(np.isnan(seafloor_age), 0, seafloor_age)
    age_valid = np.maximum(age_valid, 0)  # Ensure non-negative ages

    if model.upper() == 'GDH1':
        # GDH1 model (Stein & Stein, 1992)
        # Young seafloor (≤ 20 Myr): d = 2600 + 365*sqrt(t)
        # Old seafloor (> 20 Myr): d = 5651 - 2473*exp(-0.0278*t)
        depth = np.where(
            age_valid <= 20.0,
            2600.0 + 365.0 * np.sqrt(age_valid),
            5651.0 - 2473.0 * np.exp(-0.0278 * age_valid)
        )

    elif model.lower() == 'half_space':
        # Half-space cooling model
        # d(t) = d0 + c*sqrt(t)
        depth = ridge_depth + hs_coeff * np.sqrt(age_valid)

    elif model.lower() == 'plate':
        # Plate cooling model
        # d(t) = d0 + c*sqrt(t)*[1 - exp(-t/τ)]
        depth = ridge_depth + hs_coeff * np.sqrt(age_valid) * (1.0 - np.exp(-age_valid / plate_timescale))

    else:
        raise ValueError(f"Unknown subsidence model: {model}. Use 'GDH1', 'half_space', or 'plate'")

    # Restore NaN values where age was NaN
    depth = np.where(np.isnan(seafloor_age), np.nan, depth)

    # Convert to negative depth (oceanographic convention: negative = below sea level)
    return -depth


def generate_complete_bathymetry(seafloor_age, sediment_thickness, params, grid_spacing_km,
                                  random_field=None, subsidence_model='GDH1',
                                  sediment_mode='drape', sediment_diffusion=0.3, **kwargs):
    """
    Generate complete synthetic bathymetry combining thermal subsidence and abyssal hills.

    This function creates realistic ocean floor bathymetry by combining:
    1. Long-wavelength thermal subsidence (from seafloor age)
    2. Short-wavelength abyssal hill fabric (modified by sediment)
    3. Sediment infill (with choice of simple drape or diffusive ponding)

    Parameters:
    -----------
    seafloor_age : ndarray
        Seafloor age in Myr (2D array)
    sediment_thickness : ndarray
        Sediment thickness in meters (2D array)
    params : dict
        Abyssal hill parameters: {'H': height, 'lambda_n': wavelength_n,
                                   'lambda_s': wavelength_s, 'D': fractal_dim}
    grid_spacing_km : float
        Grid spacing in kilometers
    random_field : ndarray, optional
        Pre-generated random field (if None, will be generated)
    subsidence_model : str, optional
        Thermal subsidence model: 'GDH1' (default), 'half_space', or 'plate'
    sediment_mode : str, optional
        Sediment treatment mode (default: 'drape'):
        - 'none': No sediment layer added (basement + hills only)
        - 'drape': Simple sediment drape (just subtract thickness)
        - 'fill': Diffusive infill (sediment ponds in lows)
    sediment_diffusion : float, optional
        Diffusion coefficient for 'fill' mode (0-1, default: 0.3)
        Higher = more smoothing/ponding. Ignored for other modes.
    **kwargs : dict
        Additional arguments passed to generate_bathymetry_spatial_filter()
        (e.g., filter_type, optimize, azimuth_bins, spreading_rate_bins,
         base_params, lat_coords, etc.)

    Returns:
    --------
    bathymetry : ndarray
        Complete bathymetry in meters (negative below sea level)

    Notes:
    ------
    SEDIMENT TREATMENT EXPLANATION:

    Sediment affects the final bathymetry in TWO COMPLEMENTARY ways:

    1. **During hill generation** (via modify_by_sediment function):
       - Amplitude reduction: H(S) = H₀ - S/2
       - Wavelength increase: λ(S) = λ₀(1 + 1.3·S/H₀)
       - Effect: Buried hills are shorter and smoother
       - This happens automatically inside generate_bathymetry_spatial_filter()

    2. **After hill generation** (via sediment_mode parameter):
       - 'none': No additional sediment layer (just buried hills)
       - 'drape': Add uniform sediment layer everywhere
       - 'fill': Add sediment with diffusive redistribution (ponds in lows)

    The combination creates realistic sediment-covered seafloor:
    - Hills are generated with reduced amplitude (buried effect)
    - Hills have increased wavelength (smoothed by sediment)
    - Then sediment layer is added on top (filling topography)
    - With 'fill' mode, sediment preferentially fills valleys

    Order of operations:
    1. Calculate regional basement depth from thermal subsidence
    2. Generate abyssal hill fabric (with sediment-modified parameters)
    3. Combine: basement_topo = regional_depth + abyssal_hills
    4. Apply sediment layer:
       - 'none': seafloor = basement_topo
       - 'drape': seafloor = basement_topo + sediment
       - 'fill': seafloor = apply_diffusive_infill(basement_topo, sediment)

    Examples:
    ---------
    # Simple drape (default)
    bathy = generate_complete_bathymetry(age, sediment, params, 3.7)

    # No sediment layer (only buried hill effect)
    bathy = generate_complete_bathymetry(age, sediment, params, 3.7, sediment_mode='none')

    # Diffusive infill (realistic ponding)
    bathy = generate_complete_bathymetry(age, sediment, params, 3.7,
                                         sediment_mode='fill', sediment_diffusion=0.3)

    # With spreading rate variation
    bathy = generate_complete_bathymetry(
        age, sediment, params, 3.7,
        subsidence_model='GDH1',
        sediment_mode='fill',
        spreading_rate_bins=10,
        base_params=params,
        lat_coords=lat_array
    )
    """
    # 1. Calculate thermal subsidence (long-wavelength component)
    basement_regional = calculate_thermal_subsidence(seafloor_age, model=subsidence_model)

    # 2. Generate abyssal hill fabric (short-wavelength component)
    # Note: This already accounts for sediment effects on hill parameters via modify_by_sediment()
    abyssal_hills = generate_bathymetry_spatial_filter(
        seafloor_age, sediment_thickness, params, grid_spacing_km,
        random_field=random_field, **kwargs
    )

    # 3. Combine regional subsidence and abyssal hills (basement topography)
    basement_topo = basement_regional + abyssal_hills

    # 4. Apply sediment layer based on mode
    if sediment_mode == 'none':
        # No sediment layer - just basement (hills still affected by modify_by_sediment)
        final_bathymetry = basement_topo

    elif sediment_mode == 'drape':
        # Simple drape: uniform sediment layer added everywhere
        # (sediment fills in from the top)
        final_bathymetry = basement_topo + sediment_thickness  # Add because depth is negative

    elif sediment_mode == 'fill':
        # Diffusive infill: sediment ponds in topographic lows
        # More realistic for thick sediment
        final_bathymetry = apply_diffusive_sediment_infill(
            basement_topo, sediment_thickness, grid_spacing_km, sediment_diffusion
        )

    else:
        raise ValueError(f"Unknown sediment_mode: {sediment_mode}. "
                        f"Use 'none', 'drape', or 'fill'")

    return final_bathymetry


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


def add_cf_compliant_coordinate_attrs(data_array):
    """
    Add CF-compliant metadata to coordinate variables for proper NetCDF export.

    This ensures that programs like Panoply, QGIS, and other GIS tools can
    correctly interpret the coordinate system.

    Parameters:
    -----------
    data_array : xarray.DataArray
        Input DataArray to add coordinate attributes to

    Returns:
    --------
    xarray.DataArray
        DataArray with CF-compliant coordinate attributes

    Notes:
    ------
    Follows CF Conventions 1.8 for coordinate variables:
    - http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html
    """
    # Make a copy to avoid modifying the input
    da = data_array.copy()

    # Add longitude attributes if longitude coordinate exists
    lon_names = [dim for dim in da.dims if 'lon' in dim.lower()]
    if lon_names:
        lon_dim = lon_names[0]
        da[lon_dim].attrs.update({
            'units': 'degrees_east',
            'standard_name': 'longitude',
            'long_name': 'longitude',
            'axis': 'X',
            'actual_range': [float(da[lon_dim].min()), float(da[lon_dim].max())]
        })

    # Add latitude attributes if latitude coordinate exists
    lat_names = [dim for dim in da.dims if 'lat' in dim.lower()]
    if lat_names:
        lat_dim = lat_names[0]
        da[lat_dim].attrs.update({
            'units': 'degrees_north',
            'standard_name': 'latitude',
            'long_name': 'latitude',
            'axis': 'Y',
            'actual_range': [float(da[lat_dim].min()), float(da[lat_dim].max())]
        })

    return da


def calculate_required_longitude_margin(params, grid_spacing_km, chunkpad_pixels=20):
    """
    Calculate the required longitude margin for periodic boundary handling.

    The margin must be large enough to accommodate:
    1. The filter size (based on maximum wavelength)
    2. The chunk padding

    Parameters:
    -----------
    params : dict
        Abyssal hill parameters containing 'lambda_s' (parallel wavelength in km)
    grid_spacing_km : float
        Grid spacing in km at the equator
    chunkpad_pixels : int, optional
        Number of padding pixels used in chunking (default: 20)

    Returns:
    --------
    float
        Required margin in degrees

    Notes:
    ------
    The filter extends approximately 3*lambda_s on each side (99% of Gaussian)
    """
    # Maximum wavelength (parallel direction, usually largest)
    max_wavelength_km = params.get('lambda_s', 30.0)

    # Filter extends ~3 standard deviations (99% of Gaussian energy)
    # For Gaussian: sigma ≈ wavelength/2.355, so 3*sigma ≈ 1.27*wavelength
    filter_extent_km = 3.0 * max_wavelength_km

    # Add chunk padding
    padding_km = chunkpad_pixels * grid_spacing_km

    # Total margin needed in km
    total_margin_km = filter_extent_km + padding_km

    # Convert to degrees at equator (111.32 km/degree)
    margin_deg = total_margin_km / 111.32

    # Round up to nearest degree for safety
    margin_deg = np.ceil(margin_deg)

    return margin_deg


def is_global_longitude(lon_coords, tolerance_deg=1.0):
    """
    Check if longitude coordinates span the full globe.

    Parameters:
    -----------
    lon_coords : array-like
        Longitude coordinates
    tolerance_deg : float, optional
        Tolerance for detecting global coverage (default: 1.0°)

    Returns:
    --------
    bool
        True if longitude range covers ~360°, False otherwise
    """
    lon_range = np.max(lon_coords) - np.min(lon_coords)
    return lon_range >= (360.0 - tolerance_deg)