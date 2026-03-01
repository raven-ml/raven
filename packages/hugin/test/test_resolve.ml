open Hugin
open Windtrap

(* Helpers *)

let contains s sub =
  let len_s = String.length s and len_sub = String.length sub in
  if len_sub > len_s then false
  else
    let found = ref false in
    for i = 0 to len_s - len_sub do
      if (not !found) && String.sub s i len_sub = sub then found := true
    done;
    !found

let count_substring s sub =
  let len_s = String.length s and len_sub = String.length sub in
  if len_sub > len_s || len_sub = 0 then 0
  else begin
    let count = ref 0 in
    for i = 0 to len_s - len_sub do
      if String.sub s i len_sub = sub then incr count
    done;
    !count
  end

let render ?(width = 400.) ?(height = 300.) spec =
  let tmp = Filename.temp_file "hugin_test" ".svg" in
  Hugin.render_svg ~width ~height tmp spec;
  let ic = open_in tmp in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  Sys.remove tmp;
  s

let sample_x = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let sample_y = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0) *. 2.)
let sample_line () = Hugin.line ~x:sample_x ~y:sample_y ()
let sample_point () = Hugin.point ~x:sample_x ~y:sample_y ()

let sample_bar () =
  Hugin.bar ~x:sample_x
    ~height:(Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0) +. 1.))
    ()

(* basic marks *)

let test_line_resolves () =
  let svg = render (sample_line ()) in
  is_true ~msg:"starts with xml"
    (String.length svg > 5 && String.sub svg 0 5 = "<?xml");
  is_true ~msg:"contains path" (contains svg "<path");
  is_true ~msg:"contains svg" (contains svg "<svg")

let test_point_resolves () =
  let svg = render (sample_point ()) in
  is_true ~msg:"contains marker" (contains svg "<path d=\"M")

let test_bar_resolves () =
  let svg = render (sample_bar ()) in
  is_true ~msg:"contains closed path" (contains svg " Z\"")

let test_hist_resolves () =
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0) /. 10.) in
  let svg = render (Hugin.hist ~x:data ()) in
  is_true ~msg:"contains path" (contains svg "<path")

let test_text_mark_resolves () =
  let svg = render (Hugin.text ~x:1. ~y:1. "hello" ()) in
  is_true ~msg:"contains text" (contains svg ">hello<")

let test_hline_resolves () =
  let spec = Hugin.layers [ sample_line (); Hugin.hline ~y:3. () ] in
  let svg = render spec in
  (* hline adds a horizontal path; 2 paths from line+hline vs 1 without *)
  let path_count = count_substring svg "<path" in
  is_true ~msg:"at least 2 paths" (path_count >= 2)

let test_vline_resolves () =
  let spec = Hugin.layers [ sample_line (); Hugin.vline ~x:2. () ] in
  let svg = render spec in
  let path_count = count_substring svg "<path" in
  is_true ~msg:"at least 2 paths" (path_count >= 2)

let test_empty_layers () =
  let svg = render (Hugin.layers []) in
  is_true ~msg:"valid svg" (contains svg "<svg")

(* decorations *)

let test_title_appears () =
  let svg = render (sample_line () |> Hugin.title "My Title") in
  is_true ~msg:"title in svg" (contains svg ">My Title<")

let test_xlabel_appears () =
  let svg = render (sample_line () |> Hugin.xlabel "X Axis") in
  is_true ~msg:"xlabel in svg" (contains svg ">X Axis<")

let test_ylabel_appears () =
  let svg = render (sample_line () |> Hugin.ylabel "Y Axis") in
  is_true ~msg:"ylabel in svg" (contains svg ">Y Axis<")

let test_outermost_title_wins () =
  (* decorate prepends to the decoration list, and apply_decoration keeps the
     first-seen title. So the outermost (last-applied) title wins. *)
  let svg =
    render (sample_line () |> Hugin.title "Inner" |> Hugin.title "Outer")
  in
  is_true ~msg:"outer title present" (contains svg ">Outer<");
  is_false ~msg:"inner title absent" (contains svg ">Inner<")

(* histogram normalization *)

let test_hist_bins () =
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.hist ~x:data ~bins:(`Num 5) ()) in
  is_true ~msg:"produces paths" (contains svg "<path")

let test_hist_density () =
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.hist ~x:data ~bins:(`Num 5) ~density:true ()) in
  (* density normalization should produce bars; 5 bins = 5 closed paths *)
  let z_count = count_substring svg " Z\"" in
  is_true ~msg:"has 5 bars" (z_count >= 5)

let test_hist_edges () =
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.hist ~x:data ~bins:(`Edges [| 0.; 50.; 100. |]) ()) in
  (* 2 bins from 3 edges = 2 closed paths *)
  let z_count = count_substring svg " Z\"" in
  is_true ~msg:"has 2 bars" (z_count >= 2)

(* auto coloring *)

let test_auto_color_different () =
  let line1 = Hugin.line ~x:sample_x ~y:sample_y () in
  let y2 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0) *. 3.) in
  let line2 = Hugin.line ~x:sample_x ~y:y2 () in
  let svg = render (Hugin.layers [ line1; line2 ]) in
  let stroke_count = count_substring svg "stroke=\"rgb(" in
  is_true ~msg:"multiple stroke colors" (stroke_count >= 2)

let test_explicit_color_preserved () =
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ~color:Color.black ()) in
  is_true ~msg:"has black stroke" (contains svg "stroke=\"rgb(0,0,0)\"")

(* grid layout *)

let test_grid_2x2 () =
  let a = sample_line () |> Hugin.title "A" in
  let b = sample_line () |> Hugin.title "B" in
  let c = sample_line () |> Hugin.title "C" in
  let d = sample_line () |> Hugin.title "D" in
  let svg = render (Hugin.grid [ [ a; b ]; [ c; d ] ]) in
  is_true ~msg:"has A" (contains svg ">A<");
  is_true ~msg:"has D" (contains svg ">D<");
  (* 4 panels = 4 clip regions *)
  let clip_count = count_substring svg "<clipPath" in
  is_true ~msg:"4 clips" (clip_count = 4)

let test_grid_empty () =
  let svg = render (Hugin.grid []) in
  is_true ~msg:"valid svg" (contains svg "<svg")

let test_hstack () =
  let a = sample_line () |> Hugin.title "L" in
  let b = sample_line () |> Hugin.title "R" in
  let svg = render (Hugin.hstack [ a; b ]) in
  is_true ~msg:"has L" (contains svg ">L<");
  is_true ~msg:"has R" (contains svg ">R<")

let test_vstack () =
  let a = sample_line () |> Hugin.title "Top" in
  let b = sample_line () |> Hugin.title "Bot" in
  let svg = render (Hugin.vstack [ a; b ]) in
  is_true ~msg:"has Top" (contains svg ">Top<");
  is_true ~msg:"has Bot" (contains svg ">Bot<")

(* themes *)

let test_dark_theme () =
  let svg = render (sample_line () |> Hugin.with_theme Theme.dark) in
  (* dark theme has dark background — rgb values near 0 *)
  is_true ~msg:"has dark fill" (contains svg "fill=\"rgb(");
  is_true ~msg:"has light strokes" (contains svg "stroke=\"rgb(")

let test_minimal_theme () =
  let svg_default = render (sample_line ()) in
  let svg_minimal = render (sample_line () |> Hugin.with_theme Theme.minimal) in
  (* minimal theme has no grid, so fewer paths *)
  let default_paths = count_substring svg_default "<path" in
  let minimal_paths = count_substring svg_minimal "<path" in
  is_true ~msg:"fewer paths than default" (minimal_paths <= default_paths)

(* grid_lines *)

let test_grid_lines_off () =
  let svg_on = render (sample_line () |> Hugin.grid_lines true) in
  let svg_off = render (sample_line () |> Hugin.grid_lines false) in
  let on_paths = count_substring svg_on "<path" in
  let off_paths = count_substring svg_off "<path" in
  is_true ~msg:"fewer paths with grid off" (off_paths < on_paths)

(* legend *)

let test_legend_appears () =
  let line1 = Hugin.line ~x:sample_x ~y:sample_y ~label:"Series A" () in
  let svg = render (line1 |> Hugin.legend) in
  is_true ~msg:"legend text" (contains svg ">Series A<")

(* fill_between *)

let test_fill_between_resolves () =
  let y2 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0) *. 3.) in
  let svg = render (Hugin.fill_between ~x:sample_x ~y1:sample_y ~y2 ()) in
  is_true ~msg:"contains path" (contains svg "<path");
  is_true ~msg:"has fill" (contains svg "fill=")

let test_fill_between_with_label () =
  let y2 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0) *. 3.) in
  let spec =
    Hugin.fill_between ~x:sample_x ~y1:sample_y ~y2 ~label:"band" ()
    |> Hugin.legend
  in
  let svg = render spec in
  is_true ~msg:"legend text" (contains svg ">band<")

(* hspan / vspan *)

let test_hspan_resolves () =
  let svg_base = render (sample_line ()) in
  let spec = Hugin.layers [ sample_line (); Hugin.hspan ~y0:1. ~y1:3. () ] in
  let svg = render spec in
  (* hspan adds a filled rectangle = one extra closed path *)
  let base_z = count_substring svg_base " Z\"" in
  let with_z = count_substring svg " Z\"" in
  is_true ~msg:"more closed paths with hspan" (with_z > base_z)

let test_vspan_resolves () =
  let svg_base = render (sample_line ()) in
  let spec = Hugin.layers [ sample_line (); Hugin.vspan ~x0:1. ~x1:3. () ] in
  let svg = render spec in
  let base_z = count_substring svg_base " Z\"" in
  let with_z = count_substring svg " Z\"" in
  is_true ~msg:"more closed paths with vspan" (with_z > base_z)

(* step line *)

let test_step_post () =
  let svg_normal = render (Hugin.line ~x:sample_x ~y:sample_y ()) in
  let svg_step = render (Hugin.line ~x:sample_x ~y:sample_y ~step:`Post ()) in
  (* step line inserts intermediate points, so more L commands in total *)
  let normal_l = count_substring svg_normal " L" in
  let step_l = count_substring svg_step " L" in
  is_true ~msg:"step has more L commands" (step_l > normal_l)

let test_step_pre () =
  let svg_normal = render (Hugin.line ~x:sample_x ~y:sample_y ()) in
  let svg_step = render (Hugin.line ~x:sample_x ~y:sample_y ~step:`Pre ()) in
  (* pre step also inserts intermediate points *)
  let normal_l = count_substring svg_normal " L" in
  let step_l = count_substring svg_step " L" in
  is_true ~msg:"step has more L commands" (step_l > normal_l)

let test_step_mid () =
  let svg_normal = render (Hugin.line ~x:sample_x ~y:sample_y ()) in
  let svg_step = render (Hugin.line ~x:sample_x ~y:sample_y ~step:`Mid ()) in
  (* mid step inserts 2 intermediate points per segment *)
  let normal_l = count_substring svg_normal " L" in
  let step_l = count_substring svg_step " L" in
  is_true ~msg:"step has more L commands" (step_l > normal_l)

(* errorbar *)

let test_errorbar_symmetric () =
  let err = Nx.init Float32 [| 5 |] (fun _ -> 0.5) in
  let svg =
    render (Hugin.errorbar ~x:sample_x ~y:sample_y ~yerr:(`Symmetric err) ())
  in
  (* 5 points × 3 paths each (stem + 2 caps) = 15 paths *)
  let path_count = count_substring svg "<path" in
  is_true ~msg:"at least 15 paths for 5 error bars" (path_count >= 15)

let test_errorbar_asymmetric () =
  let elo = Nx.init Float32 [| 5 |] (fun _ -> 0.3) in
  let ehi = Nx.init Float32 [| 5 |] (fun _ -> 0.7) in
  let svg =
    render
      (Hugin.errorbar ~x:sample_x ~y:sample_y ~yerr:(`Asymmetric (elo, ehi)) ())
  in
  let path_count = count_substring svg "<path" in
  is_true ~msg:"at least 15 paths for 5 error bars" (path_count >= 15)

let test_errorbar_with_xerr () =
  let yerr = Nx.init Float32 [| 5 |] (fun _ -> 0.5) in
  let xerr = Nx.init Float32 [| 5 |] (fun _ -> 0.2) in
  let svg =
    render
      (Hugin.errorbar ~x:sample_x ~y:sample_y ~yerr:(`Symmetric yerr)
         ~xerr:(`Symmetric xerr) ())
  in
  (* with xerr: 5 points × 6 paths each (yerr stem+2caps + xerr stem+2caps) =
     30 *)
  let svg_yerr_only =
    render (Hugin.errorbar ~x:sample_x ~y:sample_y ~yerr:(`Symmetric yerr) ())
  in
  let yerr_paths = count_substring svg_yerr_only "<path" in
  let both_paths = count_substring svg "<path" in
  is_true ~msg:"xerr adds more paths" (both_paths > yerr_paths)

(* heatmap *)

let test_heatmap_resolves () =
  let data =
    Nx.init Float32 [| 3; 4 |] (fun i -> float_of_int (i.(0) + i.(1)))
  in
  let svg = render (Hugin.heatmap ~data ()) in
  is_true ~msg:"contains paths" (contains svg "<path")

let test_heatmap_annotated () =
  let data =
    Nx.init Float32 [| 2; 2 |] (fun i -> float_of_int (i.(0) + i.(1)))
  in
  let svg = render (Hugin.heatmap ~data ~annotate:true ()) in
  is_true ~msg:"contains text" (contains svg "<text")

let test_heatmap_custom_fmt () =
  let data = Nx.init Float32 [| 2; 2 |] (fun _ -> 0.5) in
  let svg =
    render
      (Hugin.heatmap ~data ~annotate:true
         ~fmt:(fun v -> Printf.sprintf "%.0f%%" (v *. 100.))
         ())
  in
  is_true ~msg:"contains formatted text" (contains svg ">50%<")

(* imshow *)

let imshow_data () =
  Nx.init Float32 [| 4; 6 |] (fun i -> float_of_int i.(0) +. float_of_int i.(1))

let test_imshow_rasterizes_to_image () =
  let svg = render (Hugin.imshow ~data:(imshow_data ()) ()) in
  (* imshow is rasterized to an Image in the Prepared stage — verify the SVG
     backend emits an <image> element with base64 PNG data *)
  is_true ~msg:"contains image element" (contains svg "<image");
  is_true ~msg:"has base64 data" (contains svg "base64,")

let test_imshow_stretches_differ () =
  (* Different stretches must produce different pixel data *)
  let data = imshow_data () in
  let svg_linear = render (Hugin.imshow ~data ~stretch:`Linear ()) in
  let svg_log = render (Hugin.imshow ~data ~stretch:`Log ()) in
  let svg_sqrt = render (Hugin.imshow ~data ~stretch:`Sqrt ()) in
  is_true ~msg:"log differs from linear" (svg_log <> svg_linear);
  is_true ~msg:"sqrt differs from linear" (svg_sqrt <> svg_linear);
  is_true ~msg:"log differs from sqrt" (svg_log <> svg_sqrt)

let test_imshow_cmap_changes_output () =
  let data = imshow_data () in
  let svg_default = render (Hugin.imshow ~data ()) in
  let svg_hot = render (Hugin.imshow ~data ~cmap:Cmap.hot ()) in
  let svg_gray = render (Hugin.imshow ~data ~cmap:Cmap.gray_r ()) in
  is_true ~msg:"hot differs from default" (svg_hot <> svg_default);
  is_true ~msg:"gray_r differs from hot" (svg_gray <> svg_hot)

(* contour *)

let contour_data () =
  (* Concentric circles centered at (4.5, 4.5), values = r² *)
  Nx.init Float32 [| 10; 10 |] (fun i ->
      let x = float_of_int i.(1) -. 4.5 in
      let y = float_of_int i.(0) -. 4.5 in
      (x *. x) +. (y *. y))

let test_contour_unfilled_has_stroked_paths () =
  let svg =
    render
      (Hugin.contour ~data:(contour_data ()) ~x0:0. ~x1:9. ~y0:0. ~y1:9.
         ~levels:(`Num 4) ())
  in
  (* Unfilled contours are stroked paths (stroke=, fill="none") *)
  is_true ~msg:"has stroked paths" (contains svg "stroke=\"rgb(");
  let path_count = count_substring svg "<path" in
  is_true ~msg:"multiple contour paths" (path_count >= 4)

let test_contour_filled_more_paths () =
  let data = contour_data () in
  let svg_unfilled =
    render (Hugin.contour ~data ~x0:0. ~x1:9. ~y0:0. ~y1:9. ~levels:(`Num 4) ())
  in
  let svg_filled =
    render
      (Hugin.contour ~data ~x0:0. ~x1:9. ~y0:0. ~y1:9. ~levels:(`Num 4)
         ~filled:true ())
  in
  (* Filled contours use fill=rgb(...), unfilled use stroke *)
  is_true ~msg:"filled has fill color" (contains svg_filled "fill=\"rgb(");
  (* Filled output differs from unfilled *)
  is_true ~msg:"filled differs from unfilled" (svg_filled <> svg_unfilled)

let test_contour_level_count_affects_paths () =
  let data = contour_data () in
  let svg_few =
    render (Hugin.contour ~data ~x0:0. ~x1:9. ~y0:0. ~y1:9. ~levels:(`Num 2) ())
  in
  let svg_many =
    render (Hugin.contour ~data ~x0:0. ~x1:9. ~y0:0. ~y1:9. ~levels:(`Num 8) ())
  in
  let few_paths = count_substring svg_few "<path" in
  let many_paths = count_substring svg_many "<path" in
  is_true ~msg:"more levels = more paths" (many_paths > few_paths)

let test_contour_legend () =
  let svg =
    render
      (Hugin.contour ~data:(contour_data ()) ~x0:0. ~x1:9. ~y0:0. ~y1:9.
         ~label:"density" ()
      |> Hugin.legend)
  in
  is_true ~msg:"legend text" (contains svg ">density<")

(* inverted axes *)

let test_invert_changes_path_data () =
  (* Inversion reverses the scale mapping, so the path d= attribute must differ
     between normal and inverted rendering of the same data. *)
  let svg_normal = render (sample_line ()) in
  let svg_xinv = render (sample_line () |> Hugin.xinvert) in
  let svg_yinv = render (sample_line () |> Hugin.yinvert) in
  is_true ~msg:"xinvert changes path" (svg_xinv <> svg_normal);
  is_true ~msg:"yinvert changes path" (svg_yinv <> svg_normal)

let test_yinvert_hr_diagram () =
  (* An HR diagram uses yinvert (brighter stars at top) and decorations. *)
  let bv = Nx.create Float32 [| 5 |] [| -0.3; 0.; 0.5; 1.0; 1.5 |] in
  let mag = Nx.create Float32 [| 5 |] [| -5.; 0.; 2.; 5.; 10. |] in
  let svg =
    render
      (Hugin.point ~x:bv ~y:mag ()
      |> Hugin.yinvert |> Hugin.xlabel "B-V" |> Hugin.ylabel "Magnitude")
  in
  is_true ~msg:"xlabel present" (contains svg ">B-V<");
  is_true ~msg:"ylabel present" (contains svg ">Magnitude<");
  is_true ~msg:"has markers" (contains svg "<path d=\"M")

(* tick format *)

let test_xtick_format () =
  let spec =
    sample_line ()
    |> Hugin.xtick_format (fun v -> Printf.sprintf "%.0f%%" (v *. 100.))
  in
  let svg = render spec in
  (* x data is 0..4, so formatted ticks should contain "%" *)
  is_true ~msg:"formatted ticks contain %" (contains svg "%")

let test_ytick_format () =
  let spec =
    sample_line () |> Hugin.ytick_format (fun v -> Printf.sprintf "$%.0f" v)
  in
  let svg = render spec in
  (* y data is 0..8, so formatted ticks should contain "$" *)
  is_true ~msg:"formatted ticks contain $" (contains svg "$")

let () =
  run "Resolve"
    [
      group "basic marks"
        [
          test "line" test_line_resolves;
          test "point" test_point_resolves;
          test "bar" test_bar_resolves;
          test "hist" test_hist_resolves;
          test "text" test_text_mark_resolves;
          test "hline" test_hline_resolves;
          test "vline" test_vline_resolves;
          test "empty layers" test_empty_layers;
        ];
      group "decorations"
        [
          test "title appears" test_title_appears;
          test "xlabel appears" test_xlabel_appears;
          test "ylabel appears" test_ylabel_appears;
          test "outermost title wins" test_outermost_title_wins;
        ];
      group "histogram normalization"
        [
          test "bins" test_hist_bins;
          test "density" test_hist_density;
          test "edges" test_hist_edges;
        ];
      group "auto coloring"
        [
          test "different colors" test_auto_color_different;
          test "explicit color preserved" test_explicit_color_preserved;
        ];
      group "grid layout"
        [
          test "2x2 grid" test_grid_2x2;
          test "empty grid" test_grid_empty;
          test "hstack" test_hstack;
          test "vstack" test_vstack;
        ];
      group "themes"
        [
          test "dark theme" test_dark_theme;
          test "minimal theme" test_minimal_theme;
        ];
      group "grid lines" [ test "grid lines off" test_grid_lines_off ];
      group "legend" [ test "legend appears" test_legend_appears ];
      group "fill_between"
        [
          test "resolves" test_fill_between_resolves;
          test "with label" test_fill_between_with_label;
        ];
      group "hspan/vspan"
        [ test "hspan" test_hspan_resolves; test "vspan" test_vspan_resolves ];
      group "step line"
        [
          test "post" test_step_post;
          test "pre" test_step_pre;
          test "mid" test_step_mid;
        ];
      group "errorbar"
        [
          test "symmetric" test_errorbar_symmetric;
          test "asymmetric" test_errorbar_asymmetric;
          test "with xerr" test_errorbar_with_xerr;
        ];
      group "heatmap"
        [
          test "resolves" test_heatmap_resolves;
          test "annotated" test_heatmap_annotated;
          test "custom fmt" test_heatmap_custom_fmt;
        ];
      group "tick format"
        [
          test "xtick_format" test_xtick_format;
          test "ytick_format" test_ytick_format;
        ];
      group "imshow"
        [
          test "rasterizes to image" test_imshow_rasterizes_to_image;
          test "stretches differ" test_imshow_stretches_differ;
          test "cmap changes output" test_imshow_cmap_changes_output;
        ];
      group "contour"
        [
          test "unfilled has stroked paths"
            test_contour_unfilled_has_stroked_paths;
          test "filled more paths" test_contour_filled_more_paths;
          test "level count affects paths"
            test_contour_level_count_affects_paths;
          test "legend" test_contour_legend;
        ];
      group "inverted axes"
        [
          test "invert changes path data" test_invert_changes_path_data;
          test "yinvert HR diagram" test_yinvert_hr_diagram;
        ];
    ]
