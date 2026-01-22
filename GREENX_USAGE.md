# GreenX Analytic Continuation - Usage Guide

## Overview

GreenX library integration for multi-precision Pade analytic continuation is now available in LibRPA.

**Location**: `/home/elhacedor/LibRPA-develop/`
**Branch**: `qsgw`
**Build Date**: 2026-01-23

## Test Results Summary

**Important Finding**: Higher precision (64→128→256 bit) does **NOT** reduce error amplification in Pade approximation.

- 64-bit (standard): 3.03x amplification
- 128-bit (GreenX): 3.06x amplification
- 256-bit (GreenX): 3.26x amplification **(worse!)**

**Conclusion**: Error amplification is algorithm ill-conditioning, not floating-point precision.
**Details**: See `/home/elhacedor/long-term-project/pade_precision_effect_test/PROJECT_SUMMARY.md`

## Available Methods

### 1. Standard Pade (Default)
- **Method**: `pade`
- **Precision**: 64-bit (double)
- **Recommendation**: ✅ Use this for production (most stable)

### 2. GreenX Multi-Precision (Testing/Research)
- **Methods**:
  - `greenx64` - 64-bit (for comparison)
  - `greenx128` - 128-bit (high precision)
  - `greenx256` - 256-bit (very high precision)
- **Recommendation**: ⚠️ Only for testing/research, not for production

## How to Use

### Basic Usage (Standard Pade)
```bash
# In librpa.in
task = qsgw
analycont_method = pade
```

### Testing GreenX Precision
```bash
# In librpa.in
task = qsgw
analycont_method = greenx128    # or greenx64, greenx256
```

### Enabling Debug Output (Testing)
Uncomment lines in `driver/task_qsgw.cpp`:
- Line ~720-728: Controlled perturbation
- Line ~833-860: Debug output for sigc_mn and Pade results

```bash
# After uncommenting, recompile
cd /home/elhacedor/LibRPA-develop/build
make -j8
```

## Code Modifications

### Files Modified
1. **src/analycont.h** - GreenX class declaration
2. **src/analycont.cpp** - GreenX implementation
3. **driver/task_qsgw.cpp** - AC method selection (Line ~753-805)
4. **src/params.h/cpp** - `analycont_method` and `perturbation_magnitude` parameters

### Test Code Location
- **Perturbation**: Line ~720-728 (commented out)
- **Debug Output**: Line ~833-860 (commented out)
- **Method Selection**: Line ~753-805 (active)

## Compilation

### Requirements
```bash
cmake -DUSE_LIBRI=ON -DUSE_GREENX_API=ON ..
make -j8
```

### Executable
`/home/elhacedor/LibRPA-develop/build/chi0_main.exe` (20 MB)

## Testing Your Implementation

To verify GreenX is working:
```bash
# Run with GreenX128
echo "analycont_method = greenx128" >> librpa.in
mpirun -np 1 ./chi0_main.exe

# Check output for successful completion
grep "libRPA finished successfully" *.out
```

## Performance Notes

- **64-bit (standard)**: Fastest, most stable
- **128-bit**: ~2-3x slower than 64-bit
- **256-bit**: ~5-10x slower than 64-bit

## Troubleshooting

### GreenX Not Found
```bash
# Check CMake configuration
cd build
cmake -DUSE_GREENX_API=ON ..
make clean
make -j8
```

### Compilation Errors
- Ensure GreenX submodule is initialized:
  ```bash
  cd /home/elhacedor/LibRPA-develop
  git submodule update --init --recursive thirdparty/greenX
  ```

### Runtime Errors
- Check `analycont_method` spelling (lowercase)
- Ensure input file format is correct

## Future Work

Since higher precision doesn't help, consider:
1. Regularized Pade (add L2 regularization)
2. Conformal mapping (optimize frequency grid)
3. Shrinkage methods (dampen Pade coefficients)
4. Other analytic continuation methods (Max Entropy, etc.)

## References

- **Test Results**: `/home/elhacedor/long-term-project/pade_precision_effect_test/`
- **Project Summary**: `PROJECT_SUMMARY.md`
- **Full Report**: `PRECISION_EFFECT_FINAL_REPORT.md`
- **Best Practices**: `/home/elhacedor/long-term-project/CLAUDE.md`

---

**Last Updated**: 2026-01-23
**Status**: ✅ GreenX integrated and tested
**Recommendation**: Use standard Pade (64-bit) for production
