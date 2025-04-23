open Hugin
open Ndarray

let x = linspace float32 0. (2. *. Float.pi) 100
let y1 = map Float.sin x
let y2 = map Float.cos x

let () =
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
  show fig
