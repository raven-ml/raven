A notebook older than the {.run} mark does not load. quill run names the
line and prints the command that migrates the notebook.

  $ cat > old.md <<'NB'
  > # Old
  > 
  > <!-- quill:cell id="c_a" -->
  > ```ocaml
  > let a = 1
  > ```
  > NB
  $ quill run old.md
  Error: $TESTCASE_ROOT/old.md: line 4: a quill:cell comment stands above a block without {.run}, so this notebook predates the run mark. Migrate it with sed -E -i.bak 's/^( {0,3})(`{3,}|~{3,})[[:blank:]]*ocaml[[:blank:]]*$/\1\2ocaml {.run}/' notebook.md, and delete the comment above a block that should stay text.
  [1]

The command marks the block, which keeps its id.

  $ sed -E -i.bak 's/^( {0,3})(`{3,}|~{3,})[[:blank:]]*ocaml[[:blank:]]*$/\1\2ocaml {.run}/' old.md
  $ quill run old.md
  # Old
  
  
  <!-- quill:cell id="c_a" -->
  ```ocaml {.run}
  let a = 1
  ```
  <!-- quill:output -->
  <!-- out:stdout -->
  val a : int = 1
  <!-- /quill:output -->

It marks an ocaml fence of any length, backticks or tildes, indented up to
three spaces, with blanks around the language. It also rewrites such a line
inside another code block.

  $ cat > nb.md <<'NB'
  > # T
  > 
  > ```ocaml
  > let a = 1
  > ```
  > 
  > ````ocaml
  > let s = {|
  > ```
  > |}
  > ````
  > 
  > ~~~ocaml
  > let b = 2
  > ~~~
  > 
  > ```ocaml 	
  > let c = 3
  > ```
  > 
  >   ```ocaml
  > let d = 4
  >   ```
  > 
  > ``` ocaml
  > let e = 5
  > ```
  > 
  > ````markdown
  > ```ocaml
  > shown
  > ```
  > ````
  > NB
  $ sed -E -i.bak 's/^( {0,3})(`{3,}|~{3,})[[:blank:]]*ocaml[[:blank:]]*$/\1\2ocaml {.run}/' nb.md
  $ cat nb.md
  # T
  
  ```ocaml {.run}
  let a = 1
  ```
  
  ````ocaml {.run}
  let s = {|
  ```
  |}
  ````
  
  ~~~ocaml {.run}
  let b = 2
  ~~~
  
  ```ocaml {.run}
  let c = 3
  ```
  
    ```ocaml {.run}
  let d = 4
    ```
  
  ```ocaml {.run}
  let e = 5
  ```
  
  ````markdown
  ```ocaml {.run}
  shown
  ```
  ````
