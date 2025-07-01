Test that Hugin figure output is not wrapped in code blocks:

  $ cat > test_hugin.md << 'EOF'
  > # Test Hugin Output
  > 
  > ## Regular OCaml output
  > 
  > ```ocaml verbose
  > let x = 42
  > ```
  > 
  > ## Simulated Hugin figure output
  > 
  > ```ocaml verbose
  > (* Simulate what Hugin pretty printer would output *)
  > let () = print_string "val fig : Hugin.Figure.t = <abstr>\n"
  > let () = print_string "![figure](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==)"
  > ```
  > 
  > ## Mixed output (code and markdown)
  > 
  > ```ocaml verbose
  > let y = 100
  > let () = print_string "![another figure](data:image/png;base64,test)"
  > ```
  > EOF

  $ quill eval test_hugin.md 2>/dev/null | diff -u --label test_hugin.md --label - test_hugin.md -
  --- test_hugin.md
  +++ -
  @@ -5,6 +5,11 @@
   ```ocaml verbose
   let x = 42
   ```
  +<!-- quill=output_start -->
  +```
  +val x : int = 42
  +```
  +<!-- quill=output_end -->
   
   ## Simulated Hugin figure output
   
  @@ -13,6 +18,12 @@
   let () = print_string "val fig : Hugin.Figure.t = <abstr>\n"
   let () = print_string "![figure](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==)"
   ```
  +<!-- quill=output_start -->
  +```
  +val fig : Hugin.Figure.t = <abstr>
  +```
  +![figure](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==)
  +<!-- quill=output_end -->
   
   ## Mixed output (code and markdown)
   
  @@ -20,3 +31,9 @@
   let y = 100
   let () = print_string "![another figure](data:image/png;base64,test)"
   ```
  +<!-- quill=output_start -->
  +```
  +val y : int = 100
  +```
  +![another figure](data:image/png;base64,test)
  +<!-- quill=output_end -->
  [1]
