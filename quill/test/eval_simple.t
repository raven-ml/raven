Test evaluating simple OCaml code from stdin:

  $ printf '```ocaml\n1 + 1\n```' | quill eval 2>/dev/null
  ```
  1 + 1
  ```
  <!-- quill=output_start -->
  - : int = 2
  <!-- quill=output_end -->

Test evaluating from a file:

  $ cat > test.md << 'EOF'
  > # Test File
  > 
  > ```ocaml
  > Printf.printf "Hello from file!\n"
  > ```
  > EOF
  $ quill eval test.md 2>/dev/null
  # Test File
  
  ```
  Printf.printf "Hello from file!\n"
  ```
  <!-- quill=output_start -->
  - : unit = ()
  Hello from file!
  <!-- quill=output_end -->
  $ rm test.md

Test empty input:

  $ echo "" | quill eval 2>/dev/null

Test markdown without code blocks:

  $ printf "# Just a heading\n\nSome plain text." | quill eval 2>/dev/null
  # Just a heading
  
  Some plain text.
