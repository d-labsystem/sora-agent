# sora's code sandbox: where run_code executes model-written Python.
# Build: uv run main.py sandbox build     (tag sora-sandbox:1)
#
# The image is only half the safety; the other half is how tools/sandbox.py
# runs it (no network, read-only root, workspace-only mount, limits).
# Versions are pinned so a run today and a run next month compute the same.
FROM python:3.12.7-slim-bookworm

RUN pip install --no-cache-dir \
        numpy==2.1.3 \
        pandas==2.2.3 \
        openpyxl==3.1.5 \
        matplotlib==3.9.2 \
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin sandbox

# matplotlib writes a font cache: point it at /tmp, the only writable place
# besides the workspace. Agg draws to files; there is no screen.
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

USER sandbox
WORKDIR /workspace
