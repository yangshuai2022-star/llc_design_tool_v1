# Documentation assets

`power-design-toolkit-overview.png` is a **real Qt launcher capture** produced by the repository code, not a mock-up.

Regenerate it from the current source tree with:

```bash
python scripts/capture_readme_screenshot.py docs/assets/power-design-toolkit-overview.png --language en
```

The `build-release` CI also generates the same launcher capture as an artifact. If the committed image and current UI diverge materially, regenerate and review the real capture rather than editing the screenshot by hand.

The image is documentation evidence of the application surface only. It is not evidence that numerical models or hardware have been validated; those claims follow `docs/ENGINEERING_VALIDATION.md` and the relevant Reference Design evidence matrix.
