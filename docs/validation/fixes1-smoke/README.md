# Calibration1-Fixes1 diffuse-only smoke test

Recorded 2026-11-10 against PR #51. Inputs are read-only; no catalog data was
altered. Negative spheroid metals are flagged and retained as diagnostics,
because only disk metals determine diffuse dust. Small negative disk metals
warn and contribute zero dust within the configurable default tolerance
`max(1 Msun, 1e-4 * positive disk gas-plus-stellar mass)`; larger negatives
raise. This is the `d1_25` file, not all 400 release files.

Set `MOCK_RELEASE` to the local directory
`GP-MCMC-Calibration1-Fixes1_allUNIT_16sqDegMock_XMM-LSS` and `BENSON2018_ATLAS`
to the canonical Benson R_V=4 atlas. From the repository root:

```bash
PYTHONPATH=. python examples/example_diffuse_galaxy.py "$BENSON2018_ATLAS" \
    --catalog "$MOCK_RELEASE/galaxyCatalogs/romanUNIT-d1_25_16sqDeg.hdf5" \
    --template "$MOCK_RELEASE/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5" \
    --sample-size 256 --audit-metal-columns \
    --output-dir /tmp/fixes1-diffuse-smoke
```

The JSON report includes SHA-256 fingerprints for both input files, canonical
atlas identity, unchanged model parameters, every rejected sample row, and a
full scan of the two gas-metal columns. Audit masses are physical Msun after
applying catalog unit metadata, not raw numbers assuming a modern solar-mass
constant. Missing or nonfinite extrema are represented as JSON null.

Sampling is `unique(linspace(0, N-1, min(N, 256), dtype=int))`. It is deterministic
and unweighted, and does not match galaxy identities to the old catalog sample.
The plotted example is row 0. Timing covers only sample evaluation, excluding
template/atlas initialization, hashing, the full-column audit and plotting;
peak RSS is measured before hashing/audit/plotting. Timing and RSS vary by run.

The initial attempt exposed two compatibility requirements: modern structured
unit metadata and modern emission-line metadata. PR #51 now supports the former
and incorporates the latter from `main`. The report was then regenerated after
separating the spheroid diagnostic from disk-metal validation and introducing
bounded numerical disk residuals: all 256 sample rows succeed, including 53
flagged negative spheroid masses. No sample row needs disk sanitation; its
behavior is covered separately by regression tests. The negative disk entry
outside this sample still exceeds the provisional tolerance.

See [the main documentation](../../GALAXY_DIFFUSE.md#updated-test-calibration1-fixes1-2026-11-10)
for the results and their interpretation. The report and plot are generated
artifacts; regenerate them with the command above rather than editing values.
