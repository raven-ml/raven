# syntax=docker/dockerfile:1

ARG VARIANT="ubuntu-24.04"
FROM mcr.microsoft.com/devcontainers/base:${VARIANT}

ARG TARGETARCH
ARG OPAM_VERSION=2.4.1
ARG OCAML_VERSION=5.3.0

ENV DEBIAN_FRONTEND=noninteractive \
    OPAMYES=1 \
    OPAMCONFIRMLEVEL=unsafe-yes \
    OCAML_VERSION=${OCAML_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    m4 \
    bubblewrap \
    unzip \
    rsync \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    liblapacke-dev \
    libev-dev \
    libcurl4-gnutls-dev \
    libcairo2-dev \
    libsdl2-dev \
    libgmp-dev \
    libffi-dev \
    libssl-dev \
    libzstd-dev \
    zlib1g-dev \
    llvm-19-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH:-amd64}" in \
      amd64) OPAM_ARCH="x86_64" ;; \
      arm64) OPAM_ARCH="arm64" ;; \
      *) echo "Unsupported TARGETARCH: ${TARGETARCH}" && exit 1 ;; \
    esac \
    && curl -fsSL "https://github.com/ocaml/opam/releases/download/${OPAM_VERSION}/opam-${OPAM_VERSION}-${OPAM_ARCH}-linux" -o /usr/local/bin/opam \
    && chmod +x /usr/local/bin/opam
