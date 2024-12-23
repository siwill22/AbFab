import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import convolve
from scipy.signal import oaconvolve
import xarray as xr



#### FFT Section

# Von Kármán spectrum function with azimuthal dependence
def von_karman_spectrum(kx, ky, H, kn, ks, D, azimuth):
    """
    Create the von Kármán spectrum based on the abyssal hill parameters.
    kx, ky: Wavenumber arrays
    H: rms height
    kn, ks: Characteristic widths (normal and strike directions)
    D: Fractal dimension
    azimuth: Ridge orientation angle in radians (azimuth of abyssal hill fabric)
    """
    # Rotate the wavenumbers by the azimuth angle
    kx_rot = kx * np.cos(azimuth) + ky * np.sin(azimuth)
    ky_rot = -kx * np.sin(azimuth) + ky * np.cos(azimuth)
    
    # Dimensionless wavenumbers scaled by the characteristic widths
    kx_dim = kx_rot / kn
    ky_dim = ky_rot / ks
    
    # Von Kármán power spectral density
    k_squared = kx_dim**2 + ky_dim**2
    spectrum = (H**2) * (1 + k_squared)**(-(D+1)/2)
    return spectrum

# Random phase for the noise field
def generate_random_phase(nx, ny):
    """ Generate a random field with normally distributed random phases. """
    return np.exp(2j * np.pi * np.random.rand(nx, ny))

# Sediment modification for abyssal hill parameters
def modify_by_sediment(H, kn, ks, sediment_thickness):
    """
    Modify abyssal hill parameters based on sediment thickness.
    Sediment drape reduces the rms height (H) and increases the width (kn, ks).
    """
    H_sed = np.maximum(H - sediment_thickness / 2, 0.01)  # Modify H based on sediment thickness
    kn_sed = kn + 1.3 * kn * (sediment_thickness / H_sed)  # Modify kn based on sediment
    ks_sed = ks + 1.3 * ks * (sediment_thickness / H_sed)  # Modify ks similarly
    return H_sed, kn_sed, ks_sed

# Azimuth calculation from seafloor age gradient
def calculate_azimuth_from_age(seafloor_age):
    """
    Calculate the azimuth from the gradient of seafloor age.
    Azimuth is perpendicular to the gradient vector.
    """
    # Compute the gradients in x and y direction
    grad_y, grad_x = np.gradient(seafloor_age)
    
    # Compute azimuth as the angle of the gradient vector (radians)
    azimuth = np.arctan2(grad_y, grad_x)
    
    return azimuth

# Generate synthetic bathymetry with variable azimuth and sediment modification
def generate_synthetic_bathymetry_fft(grid_size, seafloor_age, sediment_thickness, params):
    """
    Generate synthetic bathymetry using a von Kármán model with azimuthal orientation
    and sediment thickness modification.
    
    grid_size: tuple (nx, ny) specifying the size of the grid
    seafloor_age: 2D array of seafloor ages (used to calculate azimuth)
    sediment_thickness: 2D array of sediment thicknesses
    params: Dictionary containing the base abyssal hill parameters for each grid
            e.g., {'H': H, 'kn': kn, 'ks': ks, 'D': D}
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
    spectrum = von_karman_spectrum(kx, ky, H_sed, kn_sed, ks_sed, params['D'], azimuth)
    
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
def generate_spatial_filter(H, kn, ks, azimuth, filter_size=25):
    """
    Generate a spatial filter (Gaussian-like) based on local parameters H, kn, ks, and azimuth.
    The filter size controls how large the filter window is.
    
    H: RMS height
    kn, ks: Characteristic widths (normal and parallel to ridge)
    azimuth: Ridge orientation in radians
    """
    # Create a 2D grid for the filter
    x = np.linspace(-1, 1, filter_size)
    y = np.linspace(-1, 1, filter_size)
    xx, yy = np.meshgrid(x, y)
    
    # Rotate the coordinates based on the azimuth
    x_rot = xx * np.cos(azimuth) + yy * np.sin(azimuth)
    y_rot = -xx * np.sin(azimuth) + yy * np.cos(azimuth)
    
    # Apply anisotropic scaling based on kn and ks
    filter_exp = -(x_rot**2 / (2 * kn**2) + y_rot**2 / (2 * ks**2))
    
    # Generate a Gaussian-like filter and scale it by the RMS height (H)
    spatial_filter = H * np.exp(filter_exp)
    
    # Normalize the filter to ensure proper convolution
    spatial_filter /= np.sum(spatial_filter)
    
    return spatial_filter

# Generate synthetic bathymetry using a spatial filter
def generate_bathymetry_spatial_filter(seafloor_age, sediment_thickness, params, random_field=None):
    """
    Generate synthetic bathymetry using a spatially varying filter based on von Kármán model.
    
    grid_size: tuple (nx, ny) specifying the size of the grid
    seafloor_age: 2D array of seafloor ages (used to calculate azimuth)
    sediment_thickness: 2D array of sediment thicknesses
    params: Dictionary containing base abyssal hill parameters
            e.g., {'H': H, 'kn': kn, 'ks': ks, 'D': D}
    """
    ny, nx = seafloor_age.shape
        
    # Generate random noise field
    #if random_field is None:
    #    random_field = generate_random_field((ny,nx))
    
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
            #print(H_local, kn_local, ks_local, sediment_thickness[i, j])
            H_local, kn_local, ks_local = modify_by_sediment(H_local, kn_local, ks_local, sediment_thickness[i, j])
            #print(H_local, kn_local, ks_local)
            
            # Generate the local filter
            spatial_filter = generate_spatial_filter(H_local, kn_local, ks_local, azimuth_local)

            # Apply the filter to the random noise field at location (i, j)
            # Convolve the filter with the random field (centered at the current point)
            #filtered_value = convolve(random_field, spatial_filter, mode='constant', cval=0.0)[i, j]
            #print(random_field.shape, spatial_filter.shape, i, j)
            filtered_value = oaconvolve(random_field, spatial_filter, mode='same')[i, j]
            #print(filtered_value)
            #print(random_field * spatial_filter)
            #break

            # Store the filtered value in the bathymetry map
            bathymetry[i, j] = filtered_value

    return bathymetry



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