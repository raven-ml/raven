open Hugin
open Nx

(* Helper to ensure output directory exists *)
let ensure_dir dir =
  if not (Sys.file_exists dir) then
    Sys.command (Printf.sprintf "mkdir -p %s" dir) |> ignore

(* Gallery output directory *)
let output_dir = "www/site/docs/hugin/gallery_images"
let () = ensure_dir output_dir

(* Example 1: Line Plot *)
let generate_line_plot () =
  let x = linspace float32 0. (2. *. Float.pi) 100 in
  let y1 = sin x in
  let y2 = cos x in

  let fig = figure ~width:800 ~height:600 () in
  let _ =
    subplot fig
    |> Plotting.plot ~x ~y:y1 ~color:Artist.Color.blue ~label:"sin(x)"
    |> Plotting.plot ~x ~y:y2 ~color:Artist.Color.red ~linestyle:Dashed
         ~label:"cos(x)"
    |> Axes.set_title "Trigonometric Functions"
    |> Axes.set_xlabel "x" |> Axes.set_ylabel "y" |> Axes.legend true
    |> Axes.grid true
  in
  savefig (output_dir ^ "/line_plot.png") fig

(* Example 2: Scatter Plot *)
let generate_scatter_plot () =
  let key = Rng.key 42 in
  let n = 100 in
  let x_data = rand float32 ~key [| n |] in
  let x = mul_s x_data 10. in
  let key = (Rng.split key).(0) in
  let noise = rand float32 ~key [| n |] in
  let y = add x (sub_s (mul_s noise 2.) 1.) in

  let fig = figure ~width:800 ~height:600 () in
  let _ =
    subplot fig
    |> Plotting.scatter ~x ~y ~s:50. ~c:Artist.Color.blue
    |> Axes.set_title "Random Scatter"
    |> Axes.set_xlabel "x" |> Axes.set_ylabel "y" |> Axes.grid true
  in
  savefig (output_dir ^ "/scatter_plot.png") fig

(* Example 3: Bar Chart *)
let generate_bar_chart () =
  let x = linspace float32 0. 4. 5 in
  let height = create float32 [| 5 |] [| 23.; 45.; 56.; 78.; 32. |] in

  let fig = figure ~width:800 ~height:600 () in
  let _ =
    subplot fig
    |> Plotting.bar ~x ~height ~width:0.8 ~color:Artist.Color.green
    |> Axes.set_title "Sales by Category"
    |> Axes.set_xlabel "Category" |> Axes.set_ylabel "Sales"
    |> Axes.set_xticks [ 0.; 1.; 2.; 3.; 4. ]
  in
  savefig (output_dir ^ "/bar_chart.png") fig

(* Example 4: Histogram *)
let generate_histogram () =
  Random.init 42;
  let data_array =
    Array.init 1000 (fun _ ->
        let rec normal () =
          let u1 = Random.float 1. in
          let u2 = Random.float 1. in
          if u1 = 0. then normal ()
          else
            Float.sqrt (-2. *. Float.log u1) *. Float.cos (2. *. Float.pi *. u2)
        in
        (normal () *. 2.) +. 5.)
  in
  let data = create float32 [| 1000 |] data_array in

  let fig = figure ~width:800 ~height:600 () in
  let _ =
    subplot fig
    |> Plotting.hist ~x:data ~bins:(`Num 30) ~density:true
         ~color:Artist.Color.orange
    |> Axes.set_title "Normal Distribution (μ=5, σ=2)"
    |> Axes.set_xlabel "Value" |> Axes.set_ylabel "Density" |> Axes.grid true
  in
  savefig (output_dir ^ "/histogram.png") fig

(* Example 5: Subplots *)
let generate_subplots () =
  let fig = figure ~width:1200 ~height:900 () in

  (* Top left: Line plot *)
  let ax1 = Figure.add_subplot ~nrows:2 ~ncols:2 ~index:1 fig in
  let x = linspace float32 0. 10. 50 in
  let _ =
    ax1
    |> Plotting.plot ~x ~y:(sin x) ~color:Artist.Color.blue
    |> Axes.set_title "Sine Wave"
  in

  (* Top right: Scatter plot *)
  let ax2 = Figure.add_subplot ~nrows:2 ~ncols:2 ~index:2 fig in
  let y_scatter = map_item (fun xi -> xi +. Random.float 1.) x in
  let _ =
    ax2
    |> Plotting.scatter ~x ~y:y_scatter ~c:Artist.Color.red
    |> Axes.set_title "Scatter"
  in

  (* Bottom left: Bar chart *)
  let ax3 = Figure.add_subplot ~nrows:2 ~ncols:2 ~index:3 fig in
  let bar_x = create float32 [| 5 |] [| 0.; 1.; 2.; 3.; 4. |] in
  let bar_y = create float32 [| 5 |] [| 2.; 4.; 3.; 5.; 1. |] in
  let _ =
    ax3
    |> Plotting.bar ~x:bar_x ~height:bar_y ~color:Artist.Color.green
    |> Axes.set_title "Bars"
  in

  (* Bottom right: Filled area *)
  let ax4 = Figure.add_subplot ~nrows:2 ~ncols:2 ~index:4 fig in
  let y1 = map_item (fun xi -> Float.sin xi *. 0.5) x in
  let y2 = map_item (fun xi -> (Float.sin xi *. 0.5) +. 0.5) x in
  let _ =
    ax4
    |> Plotting.fill_between ~x ~y1 ~y2 ~color:Artist.Color.cyan
    |> Axes.set_title "Filled Area"
  in

  savefig (output_dir ^ "/subplots.png") fig

(* Example 6: 3D Plot *)
let generate_3d_plot () =
  let n = 100 in
  let t = linspace float32 0. (4. *. Float.pi) n in
  let x = cos t in
  let y = sin t in
  let z = map_item (fun ti -> ti /. 4.) t in

  let fig =
    plot3d ~title:"3D Helix" ~xlabel:"x" ~ylabel:"y" ~zlabel:"z" x y z
  in

  savefig (output_dir ^ "/plot3d.png") fig

(* Generate all plots *)
let () =
  Printf.printf "Generating gallery images...\n";
  generate_line_plot ();
  Printf.printf "Generated line_plot.png\n";
  generate_scatter_plot ();
  Printf.printf "Generated scatter_plot.png\n";
  generate_bar_chart ();
  Printf.printf "Generated bar_chart.png\n";
  generate_histogram ();
  Printf.printf "Generated histogram.png\n";
  generate_subplots ();
  Printf.printf "Generated subplots.png\n";
  generate_3d_plot ();
  Printf.printf "Generated plot3d.png\n";
  Printf.printf "All gallery images generated in %s/\n" output_dir
