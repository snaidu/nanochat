# Development Environment

- Dockerfile contains a docker image definition with the dependencies needed for this code.
- docker-compose.yml defines two services `dev` and `jupyter`. `dev` container should be used for development/testing.
- The container will usually be running as nanochat-dev, if not bring it up to run commands.
- If docker related files are updated the docker commands should run on the host, not in the container.

## Container Structure

- The repo is mounted at `/workspace` inside the container
- Python venv is built into the image at `/opt/venv` (via `UV_PROJECT_ENVIRONMENT`)
- Dependencies: PyTorch, JAX, Flax, Optax (CPU-only by default)

## Running Commands

Run commands inside the development container:
```bash
docker exec nanochat-dev-1 <command>
```

If the container is not running, bring it up with:
```bash
docker compose up -d dev
```

## Rebuilding the Image

If `pyproject.toml` or dependencies change, rebuild the image:
```bash
docker compose build
docker compose up -d dev
```

## GPU Support

For GPU training (requires NVIDIA hardware), install with:
```bash
uv sync --extra gpu
```
