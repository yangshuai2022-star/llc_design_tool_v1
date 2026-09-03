# Apply V8.2 patch to V8.1

From the root of the V8.1 repository:

```bash
git apply --check V8_2_FROM_V8_1.patch
git apply V8_2_FROM_V8_1.patch
```

Then run:

```bash
python -m compileall -q .
pytest -q llc_design/tests/test_v8_sr_dcm_multiphase.py
pytest -q llc_design/tests/test_v8_multifidelity.py
```

Useful smoke commands:

```bash
python -m llc_design v8-sr --load 1.0
python -m llc_design v8-dcm --load 0.1 --samples 512
python -m llc_design interleaved --phases 2
python -m llc_design interleaved --phases 3
python -m llc_design v8-magnetics
```

The two interleaved phase definitions are hard-coded by design:

- 2 phase: `0 / 90 deg`
- 3 phase: `0 / 120 / 240 deg`

There is no custom phase-offset control in V8.2.
