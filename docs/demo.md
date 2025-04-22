# Hello World

This is a simple example of a markdown file that can be used with Quill.

Support for **bold**, *italic*, and `inline code` is included.

```ocaml
open Ndarray

let () = 
  let arr = Ndarray.eye Ndarray.float32 4 in
  Ndarray.print arr
```
```output
[[1, 0, 0, 0],
 [0, 1, 0, 0],
 [0, 0, 1, 0],
 [0, 0, 0, 1]]

```
