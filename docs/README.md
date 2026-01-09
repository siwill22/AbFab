# AbFab - Detailed Technical Documentation

This folder contains detailed technical documentation for specific aspects of the AbFab project. For general usage, see the main [README.md](../README.md).

## Documentation Files

### GPU Implementation
- **[GPU_FIX_SUMMARY.md](GPU_FIX_SUMMARY.md)** - Summary of fixes applied to make GPU version match CPU version exactly
  - Random field generation using NumPy
  - Gradient calculation matching `np.gradient()` behavior
  - Test results showing < 1e-6 precision

- **[GPU_CPU_COMPARISON.md](GPU_CPU_COMPARISON.md)** - Detailed comparison of GPU vs CPU implementations
  - Line-by-line analysis of differences
  - Root cause identification for numerical discrepancies
  - Original analysis before fixes were applied

### Bug Fixes & Solutions
- **[CHUNK_DISCONTINUITY_FIX.md](CHUNK_DISCONTINUITY_FIX.md)** - Documentation of the 1092m chunk boundary discontinuity fix
  - Problem: Diffusive sediment infill applied per-chunk
  - Solution: Apply globally after chunk assembly
  - Critical insight about smoothing operations with edge modes

### Technical Analysis
- **[SPHERICAL_EARTH_ANALYSIS.md](SPHERICAL_EARTH_ANALYSIS.md)** - Analysis of spherical Earth corrections
  - Latitude-dependent grid spacing
  - Gradient calculations on sphere
  - When corrections are necessary

## Quick Reference

### For Users
Start with [../README.md](../README.md) - contains all practical usage information including:
- Installation instructions
- Quick start examples
- Configuration file format
- GPU acceleration guide
- Processing time estimates

### For Developers / AI Assistants
See [../CLAUDE.md](../CLAUDE.md) - contains:
- Complete development history
- Implementation details
- Common pitfalls and solutions
- Code structure and key functions
- Testing procedures

### For Debugging
If you encounter specific issues:
1. **Chunk boundaries visible**: See [CHUNK_DISCONTINUITY_FIX.md](CHUNK_DISCONTINUITY_FIX.md)
2. **GPU/CPU differences**: See [GPU_FIX_SUMMARY.md](GPU_FIX_SUMMARY.md) and [GPU_CPU_COMPARISON.md](GPU_CPU_COMPARISON.md)
3. **High latitude artifacts**: See [SPHERICAL_EARTH_ANALYSIS.md](SPHERICAL_EARTH_ANALYSIS.md)

## Document Status

All documents in this folder are **reference material** - the essential information has been incorporated into README.md and CLAUDE.md. These files are kept for:
- Historical reference
- Detailed technical explanations
- Debugging complex issues
- Understanding design decisions
