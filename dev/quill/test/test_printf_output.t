Test that Printf.printf output is captured:

  $ cat > test_printf.md << 'EOF'
  > # Test Printf Output
  > 
  > ## Simple Printf
  > 
  > ```ocaml
  > let () = Printf.printf "Hello, world!\n"
  > let () = Printf.printf "Number: %d\n" 42
  > ```
  > 
  > ## Printf in let binding
  > 
  > ```ocaml
  > let () = 
  >   Printf.printf "Raw MNIST data loaded (unnormalized):\n";
  >   Printf.printf "Train samples: %d\n" 60000;
  >   Printf.printf "Test samples: %d\n" 10000
  > ```
  > EOF

  $ quill eval test_printf.md 2>/dev/null | diff -u --label test_printf.md --label - test_printf.md -
  --- test_printf.md
  +++ -
  @@ -6,6 +6,12 @@
   let () = Printf.printf "Hello, world!\n"
   let () = Printf.printf "Number: %d\n" 42
   ```
  +<!-- quill=output_start -->
  +```
  +Hello, world!
  +Number: 42
  +```
  +<!-- quill=output_end -->
   
   ## Printf in let binding
   
  @@ -15,3 +21,10 @@
     Printf.printf "Train samples: %d\n" 60000;
     Printf.printf "Test samples: %d\n" 10000
   ```
  +<!-- quill=output_start -->
  +```
  +Raw MNIST data loaded (unnormalized):
  +Train samples: 60000
  +Test samples: 10000
  +```
  +<!-- quill=output_end -->
  [1]
