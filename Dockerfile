# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

ARG PYTHON_IMAGE=python:3.13.18-slim-bookworm@sha256:281d39ff02e3b389401586134ff9e6572954687822a58055ec07a60ded4e0b7f

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/supplyguard

COPY pyproject.toml README.md ./
COPY src ./src

RUN /opt/supplyguard/bin/python -m pip install .


FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="SupplyGuard" \
      org.opencontainers.image.description="Local RAG API for OSV security advisories" \
      org.opencontainers.image.source="https://github.com/Goodfink/SupplyGuard"

ENV HF_HOME=/home/supplyguard/.cache/huggingface \
    PATH=/opt/supplyguard/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 supplyguard \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin supplyguard \
    && mkdir -p /app/data/indexes /app/data/processed /home/supplyguard/.cache/huggingface \
    && chown -R supplyguard:supplyguard /app /home/supplyguard

WORKDIR /app

COPY --from=builder --chown=10001:10001 /opt/supplyguard /opt/supplyguard
COPY --chown=10001:10001 data/processed/advisories.jsonl /app/data/processed/advisories.jsonl

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()"]

STOPSIGNAL SIGTERM

ENTRYPOINT ["supplyguard"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
