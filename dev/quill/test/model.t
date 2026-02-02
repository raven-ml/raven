Test basic markdown parsing and evaluation:

  $ cat > basic.md << 'EOF'
  > # Hello World
  > 
  > This is a paragraph with **bold** and *italic* text.
  > 
  > ```ocaml
  > let x = 42;;
  > Printf.printf "The answer is %d\n" x;;
  > ```
  > EOF
  $ quill eval basic.md 2>/dev/null
  # Hello World
  
  This is a paragraph with **bold** and *italic* text.
  
  ```ocaml
  let x = 42;;
  Printf.printf "The answer is %d\n" x;;
  ```
  <!-- quill=output_start -->
  ```
  - : unit = ()
  The answer is 42
  ```
  <!-- quill=output_end -->
  $ rm basic.md

Test block quotes:

  $ cat > quote.md << 'EOF'
  > > This is a quote
  > > with multiple lines
  > >
  > > And multiple paragraphs
  > EOF
  $ quill eval quote.md 2>/dev/null | sed 's/^/  /'
    > This is a quote
    > with multiple lines
    > 
    > And multiple paragraphs
  $ rm quote.md

Test lists:

  $ cat > lists.md << 'EOF'
  > - Apples
  > - Bananas
  > - Oranges
  > 
  > 1. First
  > 2. Second
  > 3. Third
  > EOF
  $ quill eval lists.md 2>/dev/null
  - Apples
  - Bananas
  - Oranges
  
  1. First
  2. Second
  3. Third
  $ rm lists.md

Test code evaluation in lists:

  $ cat > list-code.md << 'EOF'
  > 1. Define a function:
  >    ```ocaml
  >    let square x = x * x
  >    ```
  > 
  > 2. Use it:
  >    ```ocaml
  >    square 5
  >    ```
  > EOF
  $ quill eval list-code.md 2>/dev/null
  1. Define a function:
     ```ocaml
     let square x = x * x
     ```
     <!-- quill=output_start -->
     ```
     val square : int -> int = <fun>
     ```
     <!-- quill=output_end -->
     
  2. Use it:
     ```ocaml
     square 5
     ```
     <!-- quill=output_start -->
     ```
     - : int = 25
     ```
     <!-- quill=output_end -->
  $ rm list-code.md

Test thematic breaks:

  $ cat > breaks.md << 'EOF'
  > First section
  > 
  > ---
  > 
  > Second section
  > EOF
  $ quill eval breaks.md 2>/dev/null
  First section
  
  ---
  
  Second section
  $ rm breaks.md

Test inline code and formatting:

  $ echo 'Use `List.map` with **bold** and *italic*.' | quill eval 2>/dev/null
  Use `List.map` with **bold** and *italic*.

Test HTML blocks:

  $ cat > html.md << 'EOF'
  > <div>
  >   <p>HTML content</p>
  > </div>
  > 
  > Regular text
  > EOF
  $ quill eval html.md 2>/dev/null
  <div>
    <p>HTML content</p>
  </div>
  
  Regular text
  $ rm html.md

Test nested block quotes with code:

  $ cat > nested.md << 'EOF'
  > > Nested quote with code:
  > > ```ocaml
  > > List.length [1; 2; 3];;
  > > ```
  > EOF
  $ quill eval nested.md 2>/dev/null | sed 's/^/  /'
    > Nested quote with code:
    > ```ocaml
    > List.length [1; 2; 3];;
    > ```
    > <!-- quill=output_start -->
    > ```
    > - : int = 3
    > ```
    > <!-- quill=output_end -->
  $ rm nested.md

Test complete example:

  $ cat > example.md << 'EOF'
  > # Factorial Example
  > 
  > The factorial function:
  > 
  > ```ocaml
  > let rec factorial n =
  >   if n <= 1 then 1
  >   else n * factorial (n - 1)
  > ```
  > 
  > Testing it:
  > 
  > ```ocaml
  > List.map factorial [1; 2; 3; 4; 5]
  > ```
  > 
  > > **Note**: This is a recursive implementation.
  > 
  > ---
  > 
  > More examples:
  > 
  > - `factorial 0` returns `1`
  > - `factorial 5` returns `120`
  > EOF
  $ quill eval example.md 2>/dev/null
  # Factorial Example
  
  The factorial function:
  
  ```ocaml
  let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
  ```
  <!-- quill=output_start -->
  ```
  val factorial : int -> int = <fun>
  ```
  <!-- quill=output_end -->
  
  Testing it:
  
  ```ocaml
  List.map factorial [1; 2; 3; 4; 5]
  ```
  <!-- quill=output_start -->
  ```
  - : int list = [1; 2; 6; 24; 120]
  ```
  <!-- quill=output_end -->
  
  > **Note**: This is a recursive implementation.
  
  ---
  
  More examples:
  
  - `factorial 0` returns `1`
  - `factorial 5` returns `120`
  $ rm example.md
