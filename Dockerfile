# Reproducible GATFuse runtime.
#   docker build -t gatfuse:1.0.0 .
#   docker run --rm -v "$PWD:/data" gatfuse:1.0.0 detect -b /data/sample.bam ...
FROM python:3.11-slim

# pysam builds against these; keeping them out of the final image would require
# a multi-stage build, which is not worth the complexity for a research tool.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small; mount a CUDA base image instead when
# GPU inference is needed.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /opt/gatfuse
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir .

WORKDIR /data
ENTRYPOINT ["gatfuse"]
CMD ["--help"]
