# GPU vs CPU Implementation Comparison

## Code Organization

### Files Structure
- **CPU Version**: `AbFab.py` (2400+ lines) + `generate_complete_bathymetry.py`
- **GPU Version**: `AbFab_gpu.py` (1096 lines) + `generate_complete_bathymetry_gpu.py`

### Key Differences in Implementation

## 1. **Gradient Calculation (Azimuth)**

### CPU Version (`AbFab.py:266-316`)
```python
# Uses numpy.gradient() - which implements:
# - Central differences for interior: (f[i+1] - f[i-1]) / 2
# - Forward/backward differences at edges
grad_y, grad_x = np.gradient(seafloor_age)
```

### GPU Version (`AbFab_gpu.py:62-100`)
```python
# Manual implementation with F.pad() + manual differences:
padded = F.pad(seafloor_age.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
grad_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0  # Central differences
grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
```

**⚠️ POTENTIAL ISSUE #1: Edge Handling**
- CPU: `np.gradient()` uses forward/backward differences at edges
- GPU: Manual padding with `mode='replicate'` then central differences everywhere
- **Impact**: Edge pixels (first and last row/column) will differ

---

## 2. **FFT Convolution**

### CPU Version (`AbFab.py:932-1010`)
```python
# Uses scipy.signal.fftconvolve()
from scipy.signal import fftconvolve
...
convolved = fftconvolve(signal, filter, mode='same')
```
- **Padding**: Automatic, determined by scipy
- **FFT backend**: FFTPACK (Fortran-based)
- **Centering**: Handled by scipy internally

### GPU Version (`AbFab_gpu.py:279-376`)
```python
# Manual FFT implementation using torch.fft.rfft2()
signal_fft = torch.fft.rfft2(signal_padded, s=(fft_h, fft_w))
filters_fft = torch.fft.rfft2(filters_padded)
product = signal_fft * filters_fft
convolved = torch.fft.irfft2(product, s=(fft_h, fft_w))
```
- **Padding**: Manual with `F.pad(..., mode='replicate')`
- **FFT backend**: PyTorch (different library)
- **Centering**: Manual `torch.roll(filters_padded, shifts=(-fh//2, -fw//2), dims=(1, 2))`

**⚠️ POTENTIAL ISSUE #2: FFT Implementation Differences**
- Different FFT libraries may have slightly different numerical precision
- Different padding strategies (scipy automatic vs. manual replicate)
- Different filter centering methods

---

## 3. **Trilinear Interpolation**

### CPU Version (`AbFab.py:975-1058`)
```python
# Uses scipy.interpolate.interpn() with bounds_error=False, fill_value=None
from scipy.interpolate import interpn
...
result = interpn(points, values, xi, method='linear',
                 bounds_error=False, fill_value=None)
```
- **Out-of-bounds**: Extrapolates using nearest boundary value
- **Implementation**: SciPy's C-based interpolation

### GPU Version (`AbFab_gpu.py:378-492`)
```python
# Manual trilinear interpolation on GPU
# Find surrounding bin indices and weights
# Perform 8-corner weighted sum manually
```
- **Out-of-bounds**: Manual clamping with `torch.clamp()`
- **Implementation**: Pure PyTorch tensor operations

**⚠️ POTENTIAL ISSUE #3: Extrapolation Behavior**
- CPU: SciPy's `fill_value=None` extrapolates
- GPU: Manual clamping may behave differently at boundaries

---

## 4. **Random Field Generation**

### CPU Version (`AbFab.py:652-678`)
```python
np.random.seed(seed)
return np.random.randn(ny, nx)
```

### GPU Version (`AbFab_gpu.py:55-59`)
```python
torch.manual_seed(seed)
return torch.randn(shape, dtype=torch.float32, device=DEVICE)
```

**⚠️ POTENTIAL ISSUE #4: Different RNG Implementations**
- NumPy and PyTorch use different random number generators
- **Even with same seed, outputs will differ!**
- This is a **MAJOR** source of discrepancy

---

## 5. **Spherical Correction**

### Both Versions
Apply `grad_x = grad_x / cos(lat)` correction.

**Minor potential difference**:
- CPU: Uses `np.maximum(cos_lat, 1e-10)` for clamping
- GPU: Uses `torch.clamp(cos_lat, min=1e-10)` for clamping
- Should be identical in practice

---

## Summary of Likely Discrepancy Sources

### **CRITICAL (Will cause differences)**:
1. ✗ **Random field generation** - Different RNGs in numpy vs torch
2. ✗ **Edge gradient calculation** - Different handling at array boundaries
3. ✗ **FFT convolution** - Different libraries and padding strategies

### **MODERATE (May cause small differences)**:
4. ? **Trilinear interpolation** - Different extrapolation behavior
5. ? **Numerical precision** - GPU float32 vs CPU float64 operations

### **MINOR (Unlikely to cause visible differences)**:
6. ✓ **Spherical correction** - Should be identical
7. ✓ **Parameter modifications** - Same mathematical operations

---

## Recommendations for Ensuring Identical Outputs

### 1. **Random Field**
**MUST** generate random field on CPU using numpy, then transfer to GPU:
```python
# In GPU code:
random_field_cpu = np.random.randn(ny, nx)  # Use numpy!
random_field_gpu = to_torch(random_field_cpu)
```

### 2. **Gradient Calculation**
**MUST** match numpy.gradient() behavior exactly:
```python
# Replace manual GPU gradient with numpy-compatible version
# Use forward/backward differences at edges, not replicate padding
```

### 3. **FFT Convolution**
**TEST** if scipy and torch FFT give identical results. If not:
- Use scipy on CPU for reference
- Or adjust torch FFT parameters to match scipy

### 4. **Trilinear Interpolation**
**VERIFY** out-of-bounds behavior matches between scipy.interpn and manual GPU version.

---

## Testing Strategy

1. **Generate identical random field** (numpy on both CPU and GPU versions)
2. **Test each function independently**:
   - Gradient calculation: Compare `calculate_azimuth_from_age()` outputs
   - FFT convolution: Compare single-filter convolution
   - Trilinear interpolation: Test with known boundary cases
3. **Use small test cases** (10x10 grid) to isolate differences
4. **Check numerical precision** (float32 vs float64)
