# Power Design Toolkit Web Deployment

The web edition uses one Python service:

```text
Browser
  -> GET /                    static HTML/CSS/JS
  -> POST /api/llc/analyze    JSON parameters
       -> FastAPI
       -> original llc_design Python engineering kernels
       -> JSON results
  <- Plotly/browser rendering
```

The browser does **not** reproduce the LLC equations.  All engineering calculation remains in the repository's Python model.

## Local run

```bash
python -m pip install -e ".[web]"
uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`.

## GitHub Codespaces (runs on GitHub-hosted compute)

1. Open the repository on GitHub.
2. **Code -> Codespaces -> Create codespace on main**.
3. The included `.devcontainer/devcontainer.json` installs the web dependencies and forwards port 8000.
4. Run:

   ```bash
   uvicorn webapp.app:app --host 0.0.0.0 --port 8000
   ```

5. Open the forwarded `Power Design Web` port.  Set the port visibility to Public only if you intentionally want to share that temporary Codespace URL.

Codespaces is suitable for engineering evaluation but is not a permanent public web host; the codespace suspends when idle.

## Persistent deployment from GitHub

GitHub Pages cannot execute a persistent Python/FastAPI process.  For a stable public URL, connect this repository to a container host.  `Dockerfile` is provider-neutral and `render.yaml` is included for a one-click Render deployment.

On Render:

1. Create **New -> Blueprint**.
2. Select `yangshuai2022-star/llc_design_tool_v1`.
3. Render reads `render.yaml`, builds the Docker image and probes `/api/health`.
4. Every push to the selected branch can auto-deploy.

The same Docker image can run on Railway, Fly.io, Azure Container Apps, AWS App Runner, Cloud Run, or a private Linux server.

## API

- `GET /api/health` - liveness check
- `GET /api/llc/defaults` - browser defaults and available device/topology selections
- `POST /api/llc/analyze` - full LLC system analysis
- `GET /docs` - generated OpenAPI/Swagger UI

Example request:

```json
{
  "spec": {
    "vbus_nom_v": 400,
    "vbus_min_normal_v": 360,
    "vbus_max_v": 420,
    "vbus_hold_end_v": 300,
    "vout_v": 53,
    "pout_w": 3000,
    "resonant_frequency_hz": 100000,
    "minimum_frequency_hz": 60000,
    "maximum_frequency_hz": 180000,
    "ln_ratio": 5.0,
    "q_full_load": 0.35,
    "primary_turns": 30,
    "secondary_turns": 4
  }
}
```

## Security / resource boundary

The public endpoint accepts a bounded whitelist of electrical and tank parameters.  Internal solver-size controls are deliberately not exposed, preventing anonymous users from requesting pathological sample counts or arbitrary Python execution.

For an Internet-facing production instance, the next hardening layer should add authentication/rate limiting and per-request execution metrics before exposing heavier optimization/sweep endpoints.
