# Use Ubuntu 24.04 as base image
FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV OPAMYES=1
ENV OPAMCONFIRMLEVEL=unsafe-yes

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    libev-dev \
    libcurl4-gnutls-dev \
    libcairo2-dev \
    pkg-config \
    libsdl2-dev \
    libgmp-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install OPAM binary directly
RUN curl -L https://github.com/ocaml/opam/releases/download/2.4.1/opam-2.4.1-arm64-linux -o /usr/local/bin/opam && \
    chmod +x /usr/local/bin/opam

# Initialize OPAM with OCaml 5.3.0
RUN opam init --disable-sandboxing --compiler=5.3.0

# Set working directory
WORKDIR /app

# Copy only opam files first
COPY *.opam ./

RUN apt-get update && apt-get install -y \
    libzstd-dev \
    llvm-19-dev

# Install dependencies
RUN eval $(opam env) && \
    opam install --deps-only --with-test --with-doc .

# Copy the rest of the project
COPY . .

# Build the project (dune will handle what can be built)
RUN eval $(opam env) && \
    opam install .

# Default command
CMD ["bash", "-c", "eval $(opam env) && bash"]
