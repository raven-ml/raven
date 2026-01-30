Test pretty printers for Nx and Rune

  $ cat > test_printers.md << 'EOF'
  > # Test Pretty Printers
  > 
  > ## Nx Arrays
  > 
  > ```ocaml verbose
  > open Nx
  > let x = linspace float32 0. 10. 5
  > ```
  > 
  > ## Rune Tensors
  > 
  > ```ocaml verbose
  > open Rune
  > let t = zeros c float32 [|2; 3|]
  > ```
  > EOF

  $ quill eval test_printers.md 2>/dev/null | diff -u --label test_printers.md --label - test_printers.md -
  --- test_printers.md
  +++ -
  @@ -6,6 +6,11 @@
   open Nx
   let x = linspace float32 0. 10. 5
   ```
  +<!-- quill=output_start -->
  +```
  +val x : (float, Nx.float32_elt) Nx.t = [0, 2.5, 5, 7.5, 10]
  +```
  +<!-- quill=output_end -->
   
   ## Rune Tensors
   
  @@ -13,3 +18,9 @@
   open Rune
   let t = zeros c float32 [|2; 3|]
   ```
  +<!-- quill=output_start -->
  +```
  +val t : (float, Rune.float32_elt, [ `c ]) Rune.t = [[0, 0, 0],
  +                                                    [0, 0, 0]]
  +```
  +<!-- quill=output_end -->
  [1]
