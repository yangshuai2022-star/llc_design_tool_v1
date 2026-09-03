# V8.1 Release Validation Artifacts

`V8_1_baseline/` was generated from the release-candidate source with:

```bash
python -m llc_design model-compare \
  --vbus 400 --load 1.0 \
  --max-harmonic 7 --hb-samples 512 --td-samples 512 \
  --output <validation-output>
```

The files are a compact, reproducible numerical reference for the default
400 V / 53 V / 3 kW full-bridge LLC design. Waveform PNG/CSV files are not
committed here; the same command regenerates them offline.
