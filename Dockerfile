# syntax=docker/dockerfile:1
# Scoring service image: a built uplift-lab wheel with the serve extra plus one baked-in,
# pickle-free model directory (manifest.json + LightGBM text models). See docs/deployment-guide.md.
#
#   uplift-criteo-report                       # exports reports/criteo-uplift/model
#   docker build -t uplift-serve .
#   docker run --rm -p 8080:8080 uplift-serve
#
# MODEL_DIR picks another exported directory inside the build context. EXTRA_PACKAGES adds
# runtime packages, e.g. "mlflow-skinny==3.17.2" to resolve UPLIFT_LAB_MODEL_URI from a registry
# instead of the baked model.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --disable-pip-version-check --no-deps --wheel-dir /dist .

FROM python:${PYTHON_VERSION}-slim
ARG MODEL_DIR=reports/criteo-uplift/model
ARG EXTRA_PACKAGES=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    PORT=8080 \
    UPLIFT_LAB_MODEL_DIR=/app/model

# OMP_NUM_THREADS=1: one OpenMP thread per scoring call, since requests already run in parallel
# on the thread pool. LightGBM's Linux wheel links against the system OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /dist /tmp/dist
RUN pip install "$(ls /tmp/dist/uplift_lab-*.whl)[serve]" ${EXTRA_PACKAGES} \
    && rm -rf /tmp/dist \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

# Root-owned and read-only for the service user: the model cannot be modified at runtime.
COPY ${MODEL_DIR} /app/model
WORKDIR /app
USER app
EXPOSE 8080

# Cloud Run sets PORT; exec keeps uvicorn as PID 1 so it receives SIGTERM directly.
CMD ["sh", "-c", "exec uvicorn uplift_lab.serving.app:create_app --factory --host 0.0.0.0 --port ${PORT} --no-access-log"]
