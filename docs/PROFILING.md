# SED Generation Performance Profiling

This document describes the profiling infrastructure for analyzing SED generation performance.

## Overview

The `profile_sed_generation.py` script provides a comprehensive framework for profiling SED generation performance. It can help identify bottlenecks in:
- HDF5 file I/O operations
- Array multiplication and summation
- Spectrum resampling
- Emission line addition
- Unit conversions and other overhead

## Quick Start

### Basic Profiling

Profile the first 100 galaxies:
```bash
python profiling/profile_sed_generation.py --num-galaxies 100
```

### Component-Level Profiling

Analyze individual components for a single galaxy:
```bash
python profiling/profile_sed_generation.py --component-profile --galaxy-index 8
```

### Detailed Function-Level Profiling

Get detailed cProfile output for the first galaxy:
```bash
python profiling/profile_sed_generation.py --num-galaxies 50 --detailed-profile
```

### Save Results to File

```bash
python profiling/profile_sed_generation.py --num-galaxies 100 --output-file results.txt
```

## Usage Examples

### Example 1: Quick Performance Check

Run a short profiling check with the bundled example data:
```bash
python profiling/profile_sed_generation.py --num-galaxies 10
```

This will:
1. Initialize the SED calculator with the default template
2. Process 10 galaxies using the default wavelength grid (0.4-2.5 microns, 1000 points)
3. Report timing statistics

### Example 2: Component Breakdown

Identify which operations are slow:
```bash
python profiling/profile_sed_generation.py --component-profile --galaxy-index 8
```

This breaks down the time spent on:
- Reading galaxy data from HDF5 file
- Computing disk continuum
- Computing spheroid continuum  
- Adding emission lines
- Computing total spectrum

### Example 3: Full Dataset Profiling

Profile all galaxies in the catalog:
```bash
python profiling/profile_sed_generation.py --output-file full_profile.txt
```

### Example 4: Different Wavelength Grid

Test with a different wavelength sampling:
```bash
python profiling/profile_sed_generation.py \
    --num-galaxies 50 \
    --wavelength-min 0.8 \
    --wavelength-max 1.8 \
    --wavelength-points 500
```

## Command-Line Options

```
--sed-template PATH
    Path to SED template HDF5 file
    Default: ./data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5

--galaxy-catalog PATH
    Path to galaxy catalog HDF5 file
    Default: ./data/romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5

--num-galaxies N
    Number of galaxies to profile
    Default: all galaxies in catalog

--galaxy-index N
    Galaxy index for component profiling
    Default: 8

--detailed-profile
    Enable detailed cProfile profiling (shows function-level breakdown)

--component-profile
    Profile individual components (file I/O, computation, etc.) for one galaxy

--output-file PATH
    Output file for profiling results
    Default: print to console only

--wavelength-min FLOAT
    Minimum wavelength in microns
    Default: 0.4

--wavelength-max FLOAT
    Maximum wavelength in microns
    Default: 2.5

--wavelength-points INT
    Number of wavelength points
    Default: 1000
```

## Interpreting Results

### Basic Profiling Output

```
================================================================================
SUMMARY STATISTICS
================================================================================
Galaxies processed: 100
Total time: 13.21 s
Mean time per galaxy: 0.1321 s (132.10 ms)
Median time per galaxy: 0.1265 s (126.50 ms)
Std dev: 0.0156 s
Min time: 0.1243 s
Max time: 0.1890 s

Estimated time for 10,000 galaxies: 22.0 minutes
Target time (0.2 s/galaxy): 33.3 minutes
Current vs target: 0.7x slower
```

Key metrics:
- **Mean time per galaxy**: Average time to generate one SED
- **Median time**: Middle value (less affected by outliers)
- **Estimated time for 10,000 galaxies**: Projection based on mean time
- **Current vs target**: Comparison with 0.2 s/galaxy target

### Component Profiling Output

```
================================================================================
COMPONENT-LEVEL PROFILING (Galaxy 8)
================================================================================
Initialize calculator: 0.0025 s
Read galaxy data: 0.0025 s
Disk continuum: 0.0078 s
Spheroid continuum: 0.0021 s
Disk with emission lines: 0.0432 s
Total spectrum (all components): 0.1733 s

--------------------------------------------------------------------------------
COMPONENT BREAKDOWN:
--------------------------------------------------------------------------------
total_spectrum                :   0.1733 s (100.0%)
disk_with_lines               :   0.0432 s ( 24.9%)
disk_emission_lines_only      :   0.0354 s ( 20.4%)
disk_continuum                :   0.0078 s (  4.5%)
read_galaxy_data              :   0.0025 s (  1.5%)
init_calculator               :   0.0025 s (  1.5%)
spheroid_continuum            :   0.0021 s (  1.2%)
```

This shows:
- Most time spent in `evaluate_total_spectrum` (includes multiple components)
- Emission line addition takes significant time (~20%)
- HDF5 file I/O is relatively fast (~1.5%)
- Actual computation (continuum) is also relatively fast

### Detailed Profiling Output

The detailed profile shows function-level timing from Python's cProfile:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.316    0.316 SEDfromSFH.py:733(evaluate_total_spectrum)
        3    0.001    0.000    0.311    0.104 SEDfromSFH.py:641(evaluate_component_spectrum)
      107    0.000    0.000    0.243    0.002 spectrum.py:1134(__init__)
      107    0.001    0.000    0.223    0.002 spectrum.py:128(__init__)
       51    0.001    0.000    0.191    0.004 models.py:552(__init__)
```

Key columns:
- **ncalls**: Number of times function was called
- **tottime**: Total time in function (excluding subcalls)
- **cumtime**: Total time in function (including subcalls)
- **filename:lineno(function)**: Where the function is defined

## Initial Profiling Results

Based on profiling 100 galaxies from `romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5`:

### Performance Metrics
- **Mean time per galaxy**: ~132 ms (0.132 s)
- **Median time per galaxy**: ~127 ms (0.127 s)
- **Estimated time for 10,000 galaxies**: ~22 minutes
- **Target time (0.2 s/galaxy)**: 33.3 minutes
- **Performance**: **Current implementation is ~1.5x faster than target** ✓

### Component Breakdown (Galaxy 8)
1. **Total spectrum generation**: 173.3 ms (100%)
2. **Emission line addition**: 35.4 ms (20.4%)
3. **Disk continuum**: 7.8 ms (4.5%)
4. **File I/O (read galaxy data)**: 2.5 ms (1.5%)
5. **Spheroid continuum**: 2.1 ms (1.2%)

### Key Findings

1. **File I/O is not a bottleneck**: Reading galaxy data takes only ~1.5% of total time
2. **Computation is fast**: Array multiplication and summation for continuum takes ~4.5%
3. **Emission lines are significant**: Adding emission lines takes ~20% of time
4. **Most overhead is in spectrum handling**: Synphot spectrum object creation and manipulation

### Implications

The current implementation is **faster than the target** of 0.2 s/galaxy:
- For 1172 galaxies: ~155 seconds (2.6 minutes) vs target ~234 seconds (3.9 minutes)
- For 10,000 galaxies: ~1320 seconds (22 minutes) vs target ~2000 seconds (33 minutes)

This suggests that **using the new SEDs would NOT double run times** as initially feared. In fact, the SED generation is fast enough that it should add minimal overhead.

## Optimization Opportunities

If further speedup is needed, consider:

1. **Vectorized operations**: Process multiple galaxies at once
   - Pre-load galaxy data for all galaxies
   - Use numpy array operations on batches
   - Could potentially achieve 10x speedup

2. **Emission line optimization**: 
   - Pre-compute emission line spectra
   - Cache common emission line configurations
   - Could save ~20% of time

3. **Spectrum object overhead**:
   - Use numpy arrays directly instead of synphot objects
   - Only convert to synphot when needed for magnitude calculation
   - Could save significant overhead

4. **Wavelength grid optimization**:
   - Use a coarser wavelength grid initially
   - Refine only where needed
   - Could reduce computation time

## Next Steps

1. **Validate with larger dataset**: Test with full 10,000 galaxy catalog
2. **Test with different wavelength grids**: Check if performance scales linearly
3. **Profile magnitude calculation**: If magnitudes are needed, profile that separately
4. **Consider vectorization**: If needed, implement batch processing

## Reproducing the Setup from Issue

The issue requested a specific setup for profiling:

```python
# Do this once:
sedTemplateFilename = "./data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5"
sedCalc = sed.sed_calculator(sedTemplateFilename, cosmology=unit)
obs_wavelengths = np.linspace(0.4, 2.5, 1000)*u.micron

# Then loop over all galaxy indices doing this:
galIndex = 8  # loop over these
total_flux = sedCalc.evaluate_total_spectrum(
    "./data/romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5", 
    galIndex, 
    obs_wavelengths=obs_wavelengths
)
```

To profile this exact setup:
```bash
python profiling/profile_sed_generation.py \
    --sed-template ./data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5 \
    --galaxy-catalog ./data/romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5 \
    --num-galaxies 1186
```

Or for component-level profiling of galaxy 8:
```bash
python profiling/profile_sed_generation.py --component-profile --galaxy-index 8
```
