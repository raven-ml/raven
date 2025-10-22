#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WORKSPACE_ROOT}"

echo ">>> Setting up opam for Raven at ${WORKSPACE_ROOT}"

OCAML_VERSION="${OCAML_VERSION:-5.3.0}"
OPAM_SWITCH_NAME="${OPAM_SWITCH_NAME:-raven}"
OPAM_HOME="${HOME}/.opam"

export OPAMYES="${OPAMYES:-1}"
export OPAMCONFIRMLEVEL="${OPAMCONFIRMLEVEL:-unsafe-yes}"

if ! command -v opam >/dev/null 2>&1; then
  echo "opam is not available on PATH. Check the devcontainer build." >&2
  exit 1
fi

if [ ! -f "${OPAM_HOME}/config" ]; then
  echo "Initializing opam root at ${OPAM_HOME}..."
  opam init --disable-sandboxing -y
else
  echo "Reusing opam root at ${OPAM_HOME}."
fi

echo "Updating opam repositories..."
opam update

if ! opam switch list --short | grep -Fx "${OPAM_SWITCH_NAME}" >/dev/null 2>&1; then
  echo "Creating opam switch '${OPAM_SWITCH_NAME}' with OCaml ${OCAML_VERSION}..."
  opam switch create "${OPAM_SWITCH_NAME}" "ocaml-base-compiler.${OCAML_VERSION}"
else
  echo "Reusing existing opam switch '${OPAM_SWITCH_NAME}'."
fi

eval "$(opam env --switch="${OPAM_SWITCH_NAME}" --set-switch)"

echo "Installing Raven dependencies..."
opam install . --deps-only --with-test

echo "Installing developer tooling..."
DEVTOOLS=(
  ocaml-lsp-server
  ocamlformat
  utop
  odoc
  odoc-driver
  soupault
  lambdasoup
  landmarks
  landmarks-ppx
)

for pkg in "${DEVTOOLS[@]}"; do
  if ! opam install "${pkg}"; then
    echo "Warning: failed to install ${pkg}. Some tooling may be unavailable." >&2
  fi
done

ENV_SNIPPET="eval \"\$(opam env --switch=${OPAM_SWITCH_NAME} --set-switch)\""

if ! grep -Fq "opam env --switch=${OPAM_SWITCH_NAME}" "${HOME}/.bashrc"; then
  {
    echo ""
    echo "# Load Raven opam switch automatically"
    echo "${ENV_SNIPPET}"
  } >> "${HOME}/.bashrc"
fi

echo ">>> Raven devcontainer setup complete."
