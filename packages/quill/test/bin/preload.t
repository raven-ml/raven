quill run loads each raven package that is installed. Packages placed first
on the search path stand in for raven's: a talon whose archive does not load,
with the subpackages that require it, and a hugin that requires a package
that is not installed.

  $ mkdir -p fake/talon fake/hugin
  $ cat > fake/talon/META <<'META'
  > archive(byte) = "talon.cma"
  > package "csv" (
  >   requires = "talon"
  > )
  > package "parquet" (
  >   requires = "talon"
  > )
  > META
  $ echo "this is not a bytecode archive" > fake/talon/talon.cma
  $ echo 'requires = "absent"' > fake/hugin/META
  $ cat > nb.md <<'NB'
  > <!-- quill:cell id="c_ran" -->
  > ```ocaml {.run}
  > let () = print_string "ran"
  > ```
  > NB

Each package that cannot load is reported once, on standard error, and the
cells still run. The packages that require talon add nothing.

  $ OCAMLPATH="$PWD/fake:$OCAMLPATH" quill run nb.md 2>&1 | sed "s|$PWD|.|"
  [quill] cannot load hugin: absent is not installed
  [quill] failed to load talon: File ./fake/talon/talon.cma is not a bytecode object file.
  <!-- quill:cell id="c_ran" -->
  ```ocaml {.run}
  let () = print_string "ran"
  ```
  <!-- quill:output -->
  <!-- out:stdout -->
  ran
  <!-- /quill:output -->

A package that is not installed is skipped without a word. Here the search
path holds only a talon whose META says it exists only beside a file that is
missing, so findlib finds no talon and no other raven package.

  $ mkdir -p alone/talon
  $ echo 'exists_if = "nope.cma"' > alone/talon/META
  $ OCAMLPATH="$PWD/alone" quill run nb.md 2>&1
  <!-- quill:cell id="c_ran" -->
  ```ocaml {.run}
  let () = print_string "ran"
  ```
  <!-- quill:output -->
  <!-- out:stdout -->
  ran
  <!-- /quill:output -->
