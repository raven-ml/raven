open Hugin
open Windtrap

(* Helpers *)

(* Non-overlapping occurrence count, matching the semantics of windtrap's
   [contains ~count]. Used only where the claim is a *difference* between two
   renders: a delta cancels the chart frame, which lets the assertion state the
   exact number of elements a feature adds rather than "more than before". *)
let count_substring s sub =
  let len_s = String.length s and len_sub = String.length sub in
  if len_sub = 0 || len_sub > len_s then 0
  else begin
    let count = ref 0 and i = ref 0 in
    while !i <= len_s - len_sub do
      if String.sub s !i len_sub = sub then begin
        incr count;
        i := !i + len_sub
      end
      else incr i
    done;
    !count
  end

let render ?(width = 400.) ?(height = 300.) spec =
  let tmp = temp_file ~suffix:".svg" () in
  Hugin.render_svg ~width ~height tmp spec;
  let ic = open_in tmp in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

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
  starts_with ~affix:"<?xml" svg;
  in_order ~subs:[ "<svg"; "<path" ] svg

let test_point_resolves () =
  let svg = render (sample_point ()) in
  contains ~sub:"<path d=\"M" svg

let test_bar_resolves () =
  let svg = render (sample_bar ()) in
  contains ~sub:" Z\"" svg

let test_hist_resolves () =
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0) /. 10.) in
  let svg = render (Hugin.hist ~x:data ()) in
  contains ~sub:"<path" svg

let test_text_mark_resolves () =
  let svg = render (Hugin.text ~x:1. ~y:1. "hello" ()) in
  contains ~sub:">hello<" svg

(* hline/vline each add exactly one path to the layered plot. Asserting the
   delta rather than ">= 2" keeps the claim independent of how many paths the
   frame and grid contribute. *)

let test_hline_resolves () =
  let base = count_substring (render (sample_line ())) "<path" in
  let svg = render (Hugin.layers [ sample_line (); Hugin.hline ~y:3. () ]) in
  equal ~msg:"hline adds one path" int (base + 1) (count_substring svg "<path")

let test_vline_resolves () =
  let base = count_substring (render (sample_line ())) "<path" in
  let svg = render (Hugin.layers [ sample_line (); Hugin.vline ~x:2. () ]) in
  equal ~msg:"vline adds one path" int (base + 1) (count_substring svg "<path")

let test_empty_layers () =
  let svg = render (Hugin.layers []) in
  contains ~sub:"<svg" svg

(* decorations *)

let test_title_appears () =
  let svg = render (sample_line () |> Hugin.title "My Title") in
  contains ~sub:">My Title<" svg

let test_xlabel_appears () =
  let svg = render (sample_line () |> Hugin.xlabel "X Axis") in
  contains ~sub:">X Axis<" svg

let test_ylabel_appears () =
  let svg = render (sample_line () |> Hugin.ylabel "Y Axis") in
  contains ~sub:">Y Axis<" svg

let test_outermost_title_wins () =
  (* decorate prepends to the decoration list, and apply_decoration keeps the
     first-seen title. So the outermost (last-applied) title wins. *)
  let svg =
    render (sample_line () |> Hugin.title "Inner" |> Hugin.title "Outer")
  in
  contains ~sub:">Outer<" svg;
  not_contains ~sub:">Inner<" svg

(* histogram normalization *)

let test_hist_bins () =
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.hist ~x:data ~bins:(`Num 5) ()) in
  contains ~sub:"<path" svg

let test_hist_density () =
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.hist ~x:data ~bins:(`Num 5) ~density:true ()) in
  greater_equal ~msg:"5 bins produce at least 5 closed paths" int ~than:5
    (count_substring svg " Z\"")

let test_hist_edges () =
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.hist ~x:data ~bins:(`Edges [| 0.; 50.; 100. |]) ()) in
  greater_equal ~msg:"3 edges give 2 bins" int ~than:2
    (count_substring svg " Z\"")

let test_hist_bin_count_tracks_bins () =
  (* More bins must mean more bars — the claim `>= n` alone cannot make. *)
  let data = Nx.init Float32 [| 100 |] (fun i -> float_of_int i.(0)) in
  let bars n =
    count_substring (render (Hugin.hist ~x:data ~bins:(`Num n) ())) " Z\""
  in
  greater ~msg:"10 bins beat 3" int ~than:(bars 3) (bars 10)

(* auto coloring *)

let test_auto_color_different () =
  let line1 = Hugin.line ~x:sample_x ~y:sample_y () in
  let y2 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0) *. 3.) in
  let line2 = Hugin.line ~x:sample_x ~y:y2 () in
  let svg = render (Hugin.layers [ line1; line2 ]) in
  greater_equal ~msg:"two series get two strokes" int ~than:2
    (count_substring svg "stroke=\"rgb(")

let test_explicit_color_preserved () =
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ~color:Color.black ()) in
  contains ~sub:"stroke=\"rgb(0,0,0)\"" svg

(* grid layout *)

let test_grid_2x2 () =
  let panel t = sample_line () |> Hugin.title t in
  let svg =
    render (Hugin.grid [ [ panel "A"; panel "B" ]; [ panel "C"; panel "D" ] ])
  in
  (* All four panels render, in row-major order, each with its own clip. *)
  in_order ~subs:[ ">A<"; ">B<"; ">C<"; ">D<" ] svg;
  contains ~msg:"4 panels = 4 clip regions" ~sub:"<clipPath" ~count:4 svg

let test_grid_empty () =
  let svg = render (Hugin.grid []) in
  contains ~sub:"<svg" svg

let test_hstack () =
  let a = sample_line () |> Hugin.title "L" in
  let b = sample_line () |> Hugin.title "R" in
  let svg = render (Hugin.hstack [ a; b ]) in
  in_order ~subs:[ ">L<"; ">R<" ] svg

let test_vstack () =
  let a = sample_line () |> Hugin.title "Top" in
  let b = sample_line () |> Hugin.title "Bot" in
  let svg = render (Hugin.vstack [ a; b ]) in
  in_order ~subs:[ ">Top<"; ">Bot<" ] svg

(* themes *)

let test_dark_theme () =
  let svg = render (sample_line () |> Hugin.with_theme Theme.dark) in
  contains ~sub:"fill=\"rgb(" svg;
  contains ~sub:"stroke=\"rgb(" svg

let test_dark_theme_differs_from_default () =
  (* "has an rgb fill" holds of every theme; the distinguishing claim is that
     the dark theme actually changes the document. *)
  not_equal text
    (render (sample_line ()))
    (render (sample_line () |> Hugin.with_theme Theme.dark))

let test_minimal_theme () =
  let paths spec = count_substring (render spec) "<path" in
  (* minimal theme has no grid, so fewer paths *)
  less_equal ~msg:"minimal draws no more paths than default" int
    ~than:(paths (sample_line ()))
    (paths (sample_line () |> Hugin.with_theme Theme.minimal))

(* grid_lines *)

let test_grid_lines_off () =
  let paths on =
    count_substring (render (sample_line () |> Hugin.grid_lines on)) "<path"
  in
  less ~msg:"grid off draws fewer paths" int ~than:(paths true) (paths false)

(* legend *)

let test_legend_appears () =
  let line1 = Hugin.line ~x:sample_x ~y:sample_y ~label:"Series A" () in
  let svg = render (line1 |> Hugin.legend) in
  contains ~sub:">Series A<" svg

let test_label_alone_shows_legend () =
  (* hugin.mli on [legend]: "The legend is automatically visible when any mark
     has a [~label]" — so the label renders without an explicit [legend]
     call. *)
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ~label:"Series A" ()) in
  contains ~sub:">Series A<" svg

(* fill_between *)

let test_fill_between_resolves () =
  let y2 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0) *. 3.) in
  let svg = render (Hugin.fill_between ~x:sample_x ~y1:sample_y ~y2 ()) in
  contains ~sub:"<path" svg;
  contains ~sub:"fill=" svg

let test_fill_between_with_label () =
  let y2 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0) *. 3.) in
  let spec =
    Hugin.fill_between ~x:sample_x ~y1:sample_y ~y2 ~label:"band" ()
    |> Hugin.legend
  in
  contains ~sub:">band<" (render spec)

(* hspan / vspan — each adds exactly one filled rectangle. *)

let closed_paths spec = count_substring (render spec) " Z\""

let test_hspan_resolves () =
  let base = closed_paths (sample_line ()) in
  equal ~msg:"hspan adds one closed path" int (base + 1)
    (closed_paths
       (Hugin.layers [ sample_line (); Hugin.hspan ~y0:1. ~y1:3. () ]))

let test_vspan_resolves () =
  let base = closed_paths (sample_line ()) in
  equal ~msg:"vspan adds one closed path" int (base + 1)
    (closed_paths
       (Hugin.layers [ sample_line (); Hugin.vspan ~x0:1. ~x1:3. () ]))

(* step line — 5 points is 4 segments. Post and Pre insert one intermediate
   point per segment, Mid inserts two, so the extra "L" commands are exactly 4
   and 8. The delta cancels the frame's own path data. *)

let line_commands ?step () =
  count_substring (render (Hugin.line ~x:sample_x ~y:sample_y ?step ())) " L"

let test_step_post () =
  equal ~msg:"post adds one point per segment" int
    (line_commands () + 4)
    (line_commands ~step:`Post ())

let test_step_pre () =
  equal ~msg:"pre adds one point per segment" int
    (line_commands () + 4)
    (line_commands ~step:`Pre ())

let test_step_mid () =
  equal ~msg:"mid adds two points per segment" int
    (line_commands () + 8)
    (line_commands ~step:`Mid ())

(* errorbar — 5 points, 3 paths each (stem + 2 caps). *)

let err n = Nx.init Float32 [| 5 |] (fun _ -> n)
let paths_of spec = count_substring (render spec) "<path"

let test_errorbar_symmetric () =
  (* 5 points × 3 paths each (stem + 2 caps). This counts the whole document,
     frame included, so it is a floor rather than an equality; the exact
     per-point cost is pinned by the xerr delta below. *)
  greater_equal ~msg:"5 error bars need at least 15 paths" int ~than:15
    (paths_of
       (Hugin.errorbar ~x:sample_x ~y:sample_y ~yerr:(`Symmetric (err 0.5)) ()))

let test_errorbar_asymmetric () =
  greater_equal ~msg:"5 error bars need at least 15 paths" int ~than:15
    (paths_of
       (Hugin.errorbar ~x:sample_x ~y:sample_y
          ~yerr:(`Asymmetric (err 0.3, err 0.7))
          ()))

let test_errorbar_with_xerr () =
  (* xerr contributes its own stem and two caps per point: exactly 15 more paths
     for 5 points, whatever the rest of the chart costs. *)
  let yerr = `Symmetric (err 0.5) in
  let yerr_only = paths_of (Hugin.errorbar ~x:sample_x ~y:sample_y ~yerr ()) in
  let both =
    paths_of
      (Hugin.errorbar ~x:sample_x ~y:sample_y ~yerr
         ~xerr:(`Symmetric (err 0.2))
         ())
  in
  equal ~msg:"xerr adds 3 paths per point" int (yerr_only + 15) both

(* heatmap *)

let test_heatmap_resolves () =
  let data =
    Nx.init Float32 [| 3; 4 |] (fun i -> float_of_int (i.(0) + i.(1)))
  in
  contains ~sub:"<path" (render (Hugin.heatmap ~data ()))

let test_heatmap_annotated () =
  let data =
    Nx.init Float32 [| 2; 2 |] (fun i -> float_of_int (i.(0) + i.(1)))
  in
  contains ~sub:"<text" (render (Hugin.heatmap ~data ~annotate:true ()))

let test_heatmap_custom_fmt () =
  let data = Nx.init Float32 [| 2; 2 |] (fun _ -> 0.5) in
  let svg =
    render
      (Hugin.heatmap ~data ~annotate:true
         ~fmt:(fun v -> Printf.sprintf "%.0f%%" (v *. 100.))
         ())
  in
  (* every one of the 2x2 cells carries the formatted value *)
  contains ~msg:"one annotation per cell" ~sub:">50%<" ~count:4 svg

(* imshow *)

let imshow_data () =
  Nx.init Float32 [| 4; 6 |] (fun i -> float_of_int i.(0) +. float_of_int i.(1))

let test_imshow_rasterizes_to_image () =
  let svg = render (Hugin.imshow ~data:(imshow_data ()) ()) in
  (* imshow is rasterized to an Image in the Prepared stage — verify the SVG
     backend emits an <image> element with base64 PNG data *)
  in_order ~subs:[ "<image"; "base64," ] svg

let test_imshow_stretches_differ () =
  (* Different stretches must produce different pixel data *)
  let data = imshow_data () in
  let stretch s = render (Hugin.imshow ~data ~stretch:s ()) in
  let linear = stretch `Linear
  and log = stretch `Log
  and sqrt = stretch `Sqrt in
  not_equal ~msg:"log differs from linear" text log linear;
  not_equal ~msg:"sqrt differs from linear" text sqrt linear;
  not_equal ~msg:"log differs from sqrt" text log sqrt

let test_imshow_cmap_changes_output () =
  let data = imshow_data () in
  let with_cmap c = render (Hugin.imshow ~data ?cmap:c ()) in
  let default = with_cmap None in
  let hot = with_cmap (Some Cmap.hot) in
  let gray = with_cmap (Some Cmap.gray_r) in
  not_equal ~msg:"hot differs from default" text hot default;
  not_equal ~msg:"gray_r differs from hot" text gray hot

(* contour *)

let contour_data () =
  (* Concentric circles centered at (4.5, 4.5), values = r² *)
  Nx.init Float32 [| 10; 10 |] (fun i ->
      let x = float_of_int i.(1) -. 4.5 in
      let y = float_of_int i.(0) -. 4.5 in
      (x *. x) +. (y *. y))

let contour ?filled ?label ~levels () =
  Hugin.contour ~data:(contour_data ()) ~x0:0. ~x1:9. ~y0:0. ~y1:9. ~levels
    ?filled ?label ()

let test_contour_unfilled_has_stroked_paths () =
  let svg = render (contour ~levels:(`Num 4) ()) in
  (* Unfilled contours are stroked paths (stroke=, fill="none") *)
  contains ~sub:"stroke=\"rgb(" svg;
  greater_equal ~msg:"4 levels give at least 4 paths" int ~than:4
    (count_substring svg "<path")

let test_contour_filled_more_paths () =
  let unfilled = render (contour ~levels:(`Num 4) ()) in
  let filled = render (contour ~filled:true ~levels:(`Num 4) ()) in
  contains ~msg:"filled contours carry a fill colour" ~sub:"fill=\"rgb(" filled;
  not_equal ~msg:"filled differs from unfilled" text filled unfilled

let test_contour_level_count_affects_paths () =
  let paths n =
    count_substring (render (contour ~levels:(`Num n) ())) "<path"
  in
  greater ~msg:"8 levels beat 2" int ~than:(paths 2) (paths 8)

let test_contour_legend () =
  contains ~sub:">density<"
    (render (contour ~label:"density" ~levels:(`Num 4) () |> Hugin.legend))

(* inverted axes *)

let test_invert_changes_path_data () =
  (* Inversion reverses the scale mapping, so the path d= attribute must differ
     between normal and inverted rendering of the same data. *)
  let normal = render (sample_line ()) in
  not_equal ~msg:"xinvert changes path" text
    (render (sample_line () |> Hugin.xinvert))
    normal;
  not_equal ~msg:"yinvert changes path" text
    (render (sample_line () |> Hugin.yinvert))
    normal

let test_yinvert_hr_diagram () =
  (* An HR diagram uses yinvert (brighter stars at top) and decorations. *)
  let bv = Nx.create Float32 [| 5 |] [| -0.3; 0.; 0.5; 1.0; 1.5 |] in
  let mag = Nx.create Float32 [| 5 |] [| -5.; 0.; 2.; 5.; 10. |] in
  let svg =
    render
      (Hugin.point ~x:bv ~y:mag ()
      |> Hugin.yinvert |> Hugin.xlabel "B-V" |> Hugin.ylabel "Magnitude")
  in
  contains ~sub:">B-V<" svg;
  contains ~sub:">Magnitude<" svg;
  contains ~sub:"<path d=\"M" svg

(* tick format *)

let test_xtick_format () =
  let spec =
    sample_line ()
    |> Hugin.xtick_format (fun v -> Printf.sprintf "%.0f%%" (v *. 100.))
  in
  (* x data is 0..4; the formatter multiplies by 100, so 0% and 100% are ticks
     the axis must produce — "contains %" alone would pass on any stray sign. *)
  let svg = render spec in
  contains ~sub:">0%<" svg;
  contains ~sub:">100%<" svg

let test_ytick_format () =
  let spec =
    sample_line () |> Hugin.ytick_format (fun v -> Printf.sprintf "$%.0f" v)
  in
  let svg = render spec in
  contains ~sub:">$0<" svg

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
          test "hline adds a path" test_hline_resolves;
          test "vline adds a path" test_vline_resolves;
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
          test "bar count tracks bin count" test_hist_bin_count_tracks_bins;
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
          test "dark theme differs from default"
            test_dark_theme_differs_from_default;
          test "minimal theme" test_minimal_theme;
        ];
      group "grid lines" [ test "grid lines off" test_grid_lines_off ];
      group "legend"
        [
          test "legend appears" test_legend_appears;
          test "a label alone shows the legend" test_label_alone_shows_legend;
        ];
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
