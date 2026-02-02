# Hello World

This is a simple example of a markdown file that can be used with Quill.

Support for **bold**, *italic*, and `inline code` is included.

```ocaml
Rune.eye Rune.float32 4
```
<!-- quill=output_start -->
```
[[1, 0, 0, 0],
 [0, 1, 0, 0],
 [0, 0, 1, 0],
 [0, 0, 0, 1]]
```
<!-- quill=output_end -->

```ocaml
(* This is a code block *)
let () = 
  print_endline "Hello, world!"
```
<!-- quill=output_start -->
```
Hello, world!
```
<!-- quill=output_end -->


```ocaml
open Hugin
open Nx


let fig =
  let x = linspace float32 0. (2. *. Float.pi) 100 in
  let y1 = map Float.sin x in
  let y2 = map Float.cos x in
  let fig = figure ~width:800 ~height:600 () in
  let _ =
    subplot fig
    |> Plotting.plot ~x ~y:y1 ~color:Artist.Color.blue ~label:"sin(x)"
    |> Plotting.plot ~x ~y:y2 ~color:Artist.Color.red ~linestyle:Dashed
         ~label:"cos(x)"
    |> Axes.set_title "Trigonometric Functions"
    |> Axes.set_xlabel "Angle (radians)"
    |> Axes.set_ylabel "Value"
    |> Axes.set_ylim ~min:(-1.2) ~max:1.2
    |> Axes.grid true
  in
  fig
```
