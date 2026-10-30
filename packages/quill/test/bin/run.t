quill run runs the code blocks marked {.run}, in their language. A block
without the mark is text and does not run. A cell in a language that no
kernel runs gets an error, and the cells after it still run.

  $ cat > nb.md <<'NB'
  > # Setup
  > 
  > Install with:
  > 
  > ```bash
  > opam install quill
  > ```
  > 
  > ```ocaml
  > let shown = 0
  > ```
  > 
  > <!-- quill:cell id="c_sh" -->
  > ```bash {.run}
  > ls
  > ```
  > 
  > <!-- quill:cell id="c_ml" -->
  > ```ocaml {.run}
  > let () = print_string "ran"
  > ```
  > NB
  $ quill run nb.md
  # Setup
  
  Install with:
  
  ```bash
  opam install quill
  ```
  
  ```ocaml
  let shown = 0
  ```
  
  
  <!-- quill:cell id="c_sh" -->
  ```bash {.run}
  ls
  ```
  <!-- quill:output -->
  <!-- out:error -->
  no kernel runs bash code: this notebook's kernel runs ocaml
  <!-- /quill:output -->
  
  <!-- quill:cell id="c_ml" -->
  ```ocaml {.run}
  let () = print_string "ran"
  ```
  <!-- quill:output -->
  <!-- out:stdout -->
  ran
  <!-- /quill:output -->

A block that asks to run in another shape is an error, which names its line
and the shape of a cell.

  $ cat > bad.md <<'NB'
  > # Title
  > 
  > ```{.ocaml .run}
  > let x = 1
  > ```
  > NB
  $ quill run bad.md
  Error: $TESTCASE_ROOT/bad.md: line 3: the info string "{.ocaml .run}" marks a block to run; write a language then {.run}, as in "ocaml {.run}"
  [1]
