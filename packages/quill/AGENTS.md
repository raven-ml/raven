# quill

A REPL and notebooks for OCaml. A Markdown file is a notebook, run from a
terminal UI, a web frontend or a batch evaluator.

- The web frontend's bundle, `lib/server/frontend/dist/`, is committed so
  building quill needs no node toolchain. It changes only through
  `dune build @assets --auto-promote`, never by hand.
- A cell's `Invalid_argument` or `Failure` prints its message as raised.
