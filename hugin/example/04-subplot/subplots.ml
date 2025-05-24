(* Hugin Demo: Showcasing various plot types in one figure *)

open Hugin
module A = Artist
module P = Plotting
module N = Nx

(* Helper function to generate linearly spaced data *)
let linspace start stop num = N.linspace N.Float32 ~endpoint:true start stop num

(* Helper function to generate gaussian data *)
let gaussian mean stddev num =
  let data =
    Array.init num (fun _ ->
        let u1 = Random.float 1.0 in
        let u2 = Random.float 1.0 in
        let r = sqrt (-2.0 *. log u1) in
        let theta = 2.0 *. Float.pi *. u2 in
        mean +. (stddev *. r *. cos theta))
  in
  N.create N.Float32 [| num |] data

let () =
  (* --- Figure Setup --- *)
  let fig = Figure.create ~width:1600 ~height:1200 () in
  let nrows, ncols = (4, 3) in

  (* --- 1. Basic Plot (Line2D) --- *)
  let ax1 = Figure.add_subplot ~nrows ~ncols ~index:1 fig in
  let x1 = linspace 0.0 (2. *. Float.pi) 100 in
  let y1 = N.sin x1 in
  let y1_cos = N.cos x1 in
  let _ax1 =
    ax1
    |> P.plot ~x:x1 ~y:y1 ~color:A.Color.blue ~linestyle:A.Solid
         ~marker:A.Circle ~label:"Sine"
    |> P.plot ~x:x1 ~y:y1_cos ~color:A.Color.red ~linestyle:A.Dashed
         ~label:"Cosine"
    |> Axes.set_title "1. Basic Plot (plot)"
    |> Axes.set_xlabel "Radians" |> Axes.set_ylabel "Value" |> Axes.grid true
    |> P.text ~x:(Float.pi /. 2.0) ~y:0.5 "Annotation" ~color:A.Color.darkgray
  in

  (* --- 2. Plot Y only --- *)
  let ax2 = Figure.add_subplot ~nrows ~ncols ~index:2 fig in
  let x_for_y2 = linspace 0.0 20.0 100 in
  let y2 = N.map (fun x -> exp (-.x /. 10.0) *. sin x) x_for_y2 in
  let _ax2 =
    ax2
    |> P.plot_y ~y:y2 ~color:A.Color.green ~marker:A.Point
         ~label:"exp(-x/10)*sin(x)"
    |> Axes.set_title "2. Plot Y-Data Only (plot_y)"
    |> Axes.set_xlabel "Index" |> Axes.set_ylabel "Value" |> Axes.grid true
  in

  (* --- 3. Scatter Plot --- *)
  let ax3 = Figure.add_subplot ~nrows ~ncols ~index:3 fig in
  let x3_rand = N.rand N.Float32 [| 50 |] in
  let x3 = N.mul x3_rand (N.scalar N.Float32 10.0) in
  let y3 = N.map (fun x -> x +. Random.float 2.0 -. 1.0) x3 in
  let _ax3 =
    ax3
    |> P.scatter ~x:x3 ~y:y3 ~s:50.0
         ~c:{ r = 0.8; g = 0.2; b = 0.8; a = 0.6 }
         ~marker:A.Star ~label:"Noisy Data"
    |> Axes.set_title "3. Scatter Plot"
    |> Axes.set_xlabel "X Value" |> Axes.set_ylabel "Y Value" |> Axes.grid true
  in

  (* --- 4. Bar Chart --- *)
  let ax4 = Figure.add_subplot ~nrows ~ncols ~index:4 fig in
  let x4 = N.linspace N.Float32 0.0 4.0 5 in
  let height4_data = Array.init 5 (fun i -> float_of_int (i + 1) *. 1.5) in
  let height4 = N.create N.Float32 [| 5 |] height4_data in
  let _ax4 =
    ax4
    |> P.bar ~x:x4 ~height:height4 ~width:0.8 ~color:A.Color.orange
         ~label:"Categories"
    |> Axes.set_title "4. Bar Chart"
    |> Axes.set_xlabel "Category Index"
    |> Axes.set_ylabel "Height"
    |> Axes.set_xticks (List.init 5 float_of_int)
  in

  (* --- 5. Histogram --- *)
  let ax5 = Figure.add_subplot ~nrows ~ncols ~index:5 fig in
  let data5 = gaussian 0.0 1.0 1000 in
  let _ax5 =
    ax5
    |> P.hist ~x:data5 ~bins:(`Num 20) ~color:A.Color.cyan ~density:true
         ~label:"Normal Distribution"
    |> Axes.set_title "5. Histogram"
    |> Axes.set_xlabel "Value" |> Axes.set_ylabel "Density"
    |> Axes.grid ~axis:`y true
  in

  (* --- 6. Step Plot --- *)
  let ax6 = Figure.add_subplot ~nrows ~ncols ~index:6 fig in
  let x6 = linspace 0.0 10.0 11 in
  let y6 = N.map (fun x -> sin (x /. 2.0)) x6 in
  let y6_shifted = N.map (fun y -> y -. 0.5) y6 in
  let _ax6 =
    ax6
    |> P.step ~x:x6 ~y:y6 ~where:A.Mid ~color:A.Color.magenta
         ~linestyle:A.DashDot ~label:"Mid Step"
    |> P.step ~x:x6 ~y:y6_shifted ~where:A.Post ~color:A.Color.darkgray
         ~label:"Post Step"
    |> Axes.set_title "6. Step Plot"
    |> Axes.set_xlabel "X" |> Axes.set_ylabel "Y" |> Axes.grid true
  in

  (* --- 7. Fill Between --- *)
  let ax7 = Figure.add_subplot ~nrows ~ncols ~index:7 fig in
  let x7 = linspace 0.0 (2. *. Float.pi) 100 in
  let y7_base = N.sin x7 in
  let y7a = N.add y7_base (N.scalar N.Float32 0.5) in
  let y7b = N.sub y7_base (N.scalar N.Float32 0.5) in
  let _ax7 =
    ax7
    |> P.fill_between ~x:x7 ~y1:y7a ~y2:y7b
         ~color:{ r = 0.5; g = 0.8; b = 0.5; a = 0.5 }
         ~interpolate:true ~label:"Filled Sine Band"
    |> P.plot ~x:x7 ~y:y7a ~color:A.Color.black ~linewidth:0.5
    |> P.plot ~x:x7 ~y:y7b ~color:A.Color.black ~linewidth:0.5
    |> Axes.set_title "7. Fill Between"
    |> Axes.set_xlabel "X" |> Axes.set_ylabel "Y"
  in

  (* --- 8. Error Bars --- *)
  let ax8 = Figure.add_subplot ~nrows ~ncols ~index:8 fig in
  let x8 = linspace 0.5 5.5 6 in
  let y8_inv = N.div (N.scalar N.Float32 1.0) x8 in
  let y8 = N.map (fun y -> y +. (Random.float 0.2 -. 0.1)) y8_inv in
  let yerr8_data = Array.init 6 (fun _ -> Random.float 0.1 +. 0.05) in
  let yerr8 = N.create N.Float32 [| 6 |] yerr8_data in
  let xerr8_data = Array.init 6 (fun _ -> Random.float 0.2 +. 0.1) in
  let xerr8 = N.create N.Float32 [| 6 |] xerr8_data in
  let fmt_style = A.plot_style ~color:A.Color.red ~marker:A.Square () in
  let _ax8 =
    ax8
    |> P.errorbar ~x:x8 ~y:y8 ~yerr:yerr8 ~xerr:xerr8 ~fmt:fmt_style
         ~ecolor:A.Color.gray ~capsize:4.0 ~elinewidth:1.0
         ~label:"Data +/- Error"
    |> Axes.set_title "8. Error Bars"
    |> Axes.set_xlabel "X"
    |> Axes.set_ylabel "1/X + Noise"
    |> Axes.grid true
  in

  (* --- 9. Image Show (imshow) --- *)
  let ax9 = Figure.add_subplot ~nrows ~ncols ~index:9 fig in
  let rows9, cols9 = (10, 10) in
  let data9_arr =
    Array.init (rows9 * cols9) (fun i ->
        let r = i / cols9 in
        let c = i mod cols9 in
        let v = float_of_int ((r * cols9) + c) *. 2.55 |> int_of_float in
        min 255 (max 0 v))
  in
  let data9 = N.create N.UInt8 [| rows9; cols9 |] data9_arr in
  let _ax9 =
    ax9
    |> P.imshow ~data:data9 ~cmap:A.Colormap.viridis ~aspect:"equal"
         ~extent:(0.0, float_of_int cols9, 0.0, float_of_int rows9)
    |> Axes.set_title "9. Image Show (imshow)"
    |> Axes.set_xlabel "Column Index"
    |> Axes.set_ylabel "Row Index"
  in

  (* --- 10. Matrix Show (matshow) --- *)
  let ax10 = Figure.add_subplot ~nrows ~ncols ~index:10 fig in
  let rows10, cols10 = (8, 8) in
  let data10_arr =
    Array.init (rows10 * cols10) (fun i ->
        let r = i / cols10 in
        let c = i mod cols10 in
        sin (float_of_int r /. 2.0) *. cos (float_of_int c /. 3.0))
  in
  let data10 = N.create N.Float32 [| rows10; cols10 |] data10_arr in
  let _ax10 =
    ax10
    |> P.matshow ~data:data10 ~cmap:A.Colormap.coolwarm ~origin:`upper
    |> Axes.set_title "10. Matrix Show (matshow)"
  in

  (* --- 11. 3D Plot (Line3D) --- *)
  let ax11 =
    Figure.add_subplot ~nrows ~ncols ~index:11 ~projection:Axes.ThreeD fig
  in
  let t11 = linspace 0.0 (6. *. Float.pi) 200 in
  let pi6 = N.scalar N.Float32 (6. *. Float.pi) in
  let pi2 = N.scalar N.Float32 (2. *. Float.pi) in
  let one = N.scalar N.Float32 1.0 in
  let scale = N.add one (N.div t11 pi6) in
  let x11 = N.mul (N.cos t11) scale in
  let y11 = N.mul (N.sin t11) scale in
  let z11 = N.div t11 pi2 in
  let _ax11 =
    ax11
    |> P.plot3d ~x:x11 ~y:y11 ~z:z11 ~color:A.Color.blue ~linewidth:2.0
         ~label:"Spiral"
    |> Axes.set_title "11. 3D Plot (plot3d)"
    |> Axes.set_xlabel "X" |> Axes.set_ylabel "Y" |> Axes.set_zlabel "Z"
    |> Axes.set_elev 30.0 |> Axes.set_azim 45.0
  in

  (* --- 12. Placeholder / Combined Example --- *)
  let ax12 = Figure.add_subplot ~nrows ~ncols ~index:12 fig in
  let x12 = linspace (-5.0) 5.0 50 in
  let y12_tanh = N.tanh x12 in
  let y12_step = N.map (fun x -> if x > 0.0 then 1.0 else -1.0) x12 in
  let _ax12 =
    ax12
    |> P.plot ~x:x12 ~y:y12_tanh ~color:A.Color.black ~label:"tanh(x)"
    |> P.step ~x:x12 ~y:y12_step ~color:A.Color.red ~where:A.Mid
         ~label:"step(x)"
    |> Axes.set_title "12. Combined Plot/Step"
    |> Axes.set_xlabel "X" |> Axes.set_ylabel "Y"
    |> Axes.set_ylim ~min:(-1.5) ~max:1.5
    |> Axes.grid true
  in

  print_endline "Showing plot window...";
  show fig;

  print_endline "Done."
