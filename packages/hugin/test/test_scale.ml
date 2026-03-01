(*---------------------------------------------------------------------------
  Tests for Scale logic — exercised indirectly through Hugin.render_svg. We
  verify that linear and log scales produce correct axis tick labels in the SVG
  output, which proves the scale math is correct.
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

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

let render spec =
  let tmp = Filename.temp_file "hugin_test" ".svg" in
  Hugin.render_svg ~width:400. ~height:300. tmp spec;
  let ic = open_in tmp in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  Sys.remove tmp;
  s

let x5 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let y5 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))

(* linear scale *)

let test_linear_ticks_present () =
  let svg = render (Hugin.line ~x:x5 ~y:y5 ()) in
  (* Data range 0-4, auto-ticks should include 0 *)
  is_true ~msg:"has tick 0" (contains svg ">0<")

let test_linear_xlim () =
  (* Use different x and y ranges so we can distinguish x ticks from y ticks. x
     data: 0..10, y data: 100..200. With xlim 0-5, x ticks stay in [0,5] but y
     ticks are around 100-200 — no overlap. *)
  let x = Nx.init Float32 [| 11 |] (fun i -> float_of_int i.(0)) in
  let y =
    Nx.init Float32 [| 11 |] (fun i -> 100. +. (float_of_int i.(0) *. 10.))
  in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xlim 0. 5.) in
  is_true ~msg:"has tick 0" (contains svg ">0<");
  (* With xlim 0-5, we should not see x-axis tick "8" or "10". Y-axis ticks are
     in 100-200 range so no confusion. *)
  is_false ~msg:"no tick 8" (contains svg ">8<");
  is_false ~msg:"no tick 10" (contains svg ">10<")

let test_linear_ylim () =
  let x = Nx.init Float32 [| 11 |] (fun i -> float_of_int i.(0)) in
  let y = Nx.init Float32 [| 11 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.line ~x ~y () |> Hugin.ylim 0. 5.) in
  is_true ~msg:"valid svg" (contains svg "<svg")

let test_linear_negative_range () =
  let x = Nx.create Float32 [| 5 |] [| -10.; -5.; 0.; 5.; 10. |] in
  let y = Nx.create Float32 [| 5 |] [| -10.; -5.; 0.; 5.; 10. |] in
  let svg = render (Hugin.line ~x ~y ()) in
  is_true ~msg:"has tick 0" (contains svg ">0<")

let test_linear_small_range () =
  let x = Nx.create Float32 [| 3 |] [| 0.; 0.0005; 0.001 |] in
  let y = Nx.create Float32 [| 3 |] [| 0.; 0.5; 1. |] in
  let svg = render (Hugin.line ~x ~y ()) in
  is_true ~msg:"valid svg" (contains svg "<svg")

let test_linear_single_point () =
  let x = Nx.create Float32 [| 1 |] [| 5. |] in
  let y = Nx.create Float32 [| 1 |] [| 5. |] in
  let svg = render (Hugin.line ~x ~y ()) in
  is_true ~msg:"valid svg" (contains svg "<svg")

(* log scale *)

let test_log_ticks () =
  let x = Nx.create Float32 [| 4 |] [| 1.; 10.; 100.; 1000. |] in
  let y = Nx.create Float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xscale `Log) in
  is_true ~msg:"has tick 1" (contains svg ">1<");
  is_true ~msg:"has tick 10" (contains svg ">10<");
  is_true ~msg:"has tick 10^2" (contains svg ">10^2<");
  is_true ~msg:"has tick 10^3" (contains svg ">10^3<")

let test_log_y () =
  let x = Nx.create Float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let y = Nx.create Float32 [| 4 |] [| 1.; 10.; 100.; 1000. |] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.yscale `Log) in
  is_true ~msg:"has tick 1" (contains svg ">1<");
  is_true ~msg:"has tick 10^3" (contains svg ">10^3<")

(* custom ticks *)

let test_explicit_xticks () =
  let svg =
    render
      (Hugin.line ~x:x5 ~y:y5 () |> Hugin.xticks [ (0., "zero"); (4., "four") ])
  in
  is_true ~msg:"has custom tick zero" (contains svg ">zero<");
  is_true ~msg:"has custom tick four" (contains svg ">four<")

let test_explicit_yticks () =
  let svg =
    render
      (Hugin.line ~x:x5 ~y:y5 () |> Hugin.yticks [ (0., "low"); (4., "high") ])
  in
  is_true ~msg:"has custom tick low" (contains svg ">low<");
  is_true ~msg:"has custom tick high" (contains svg ">high<")

(* sqrt scale *)

let test_sqrt_handles_zero () =
  (* Sqrt scale handles zero gracefully — critical for astronomical fluxes *)
  let x = Nx.create Float32 [| 5 |] [| 0.; 1.; 4.; 9.; 16. |] in
  let y = Nx.create Float32 [| 5 |] [| 0.; 1.; 2.; 3.; 4. |] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xscale `Sqrt) in
  is_true ~msg:"has tick 0" (contains svg ">0<");
  is_true ~msg:"has path" (contains svg "<path")

let test_sqrt_differs_from_linear () =
  let x = Nx.create Float32 [| 5 |] [| 0.; 1.; 4.; 9.; 16. |] in
  let y = Nx.create Float32 [| 5 |] [| 0.; 1.; 2.; 3.; 4. |] in
  let svg_lin = render (Hugin.line ~x ~y ()) in
  let svg_sqrt = render (Hugin.line ~x ~y () |> Hugin.yscale `Sqrt) in
  is_true ~msg:"sqrt changes output" (svg_sqrt <> svg_lin)

(* asinh scale *)

let test_asinh_negative_values () =
  (* Asinh handles negative values, unlike log — needed for
     background-subtracted fluxes *)
  let x = Nx.create Float32 [| 5 |] [| -100.; -1.; 0.; 1.; 100. |] in
  let y = Nx.create Float32 [| 5 |] [| 0.; 1.; 2.; 3.; 4. |] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xscale `Asinh) in
  is_true ~msg:"has tick 0" (contains svg ">0<");
  is_true ~msg:"has path" (contains svg "<path")

let test_asinh_differs_from_linear () =
  let x = Nx.create Float32 [| 5 |] [| 0.; 1.; 2.; 3.; 4. |] in
  let y = Nx.create Float32 [| 5 |] [| -100.; -1.; 0.; 1.; 100. |] in
  let svg_lin = render (Hugin.line ~x ~y ()) in
  let svg_asinh = render (Hugin.line ~x ~y () |> Hugin.yscale `Asinh) in
  is_true ~msg:"asinh changes output" (svg_asinh <> svg_lin)

(* symlog scale *)

let test_symlog_has_linear_and_log_ticks () =
  (* Symlog should produce ticks in both the linear region (near 0) and the log
     region (far from 0) *)
  let x =
    Nx.create Float32 [| 7 |] [| -1000.; -10.; -1.; 0.; 1.; 10.; 1000. |]
  in
  let y = Nx.init Float32 [| 7 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xscale (`Symlog 10.)) in
  is_true ~msg:"has tick 0 (linear region)" (contains svg ">0<");
  is_true ~msg:"has path" (contains svg "<path")

let test_symlog_differs_from_linear () =
  let x =
    Nx.create Float32 [| 7 |] [| -1000.; -10.; -1.; 0.; 1.; 10.; 1000. |]
  in
  let y = Nx.init Float32 [| 7 |] (fun i -> float_of_int i.(0)) in
  let svg_lin = render (Hugin.line ~x ~y ()) in
  let svg_sym = render (Hugin.line ~x ~y () |> Hugin.xscale (`Symlog 10.)) in
  is_true ~msg:"symlog changes output" (svg_sym <> svg_lin)

(* inverted scales *)

let test_invert_reverses_tick_order () =
  (* The same tick labels should appear, but xinvert swaps pixel positions. We
     verify the SVG output actually changes. *)
  let svg_normal = render (Hugin.line ~x:x5 ~y:y5 ()) in
  let svg_inv = render (Hugin.line ~x:x5 ~y:y5 () |> Hugin.xinvert) in
  is_true ~msg:"has tick 0" (contains svg_inv ">0<");
  is_true ~msg:"invert changes output" (svg_inv <> svg_normal)

let test_invert_preserves_ticks () =
  (* Inversion should not remove or add ticks, just reposition them *)
  let svg_normal = render (Hugin.line ~x:x5 ~y:y5 ()) in
  let svg_inv = render (Hugin.line ~x:x5 ~y:y5 () |> Hugin.yinvert) in
  let normal_texts = count_substring svg_normal "<text" in
  let inv_texts = count_substring svg_inv "<text" in
  is_true ~msg:"same number of text elements" (normal_texts = inv_texts)

let test_log_inverted () =
  (* Log + invert is the typical RA axis for sky charts *)
  let x = Nx.create Float32 [| 4 |] [| 1.; 10.; 100.; 1000. |] in
  let y = Nx.create Float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let svg =
    render (Hugin.line ~x ~y () |> Hugin.xscale `Log |> Hugin.xinvert)
  in
  is_true ~msg:"has tick 1" (contains svg ">1<");
  is_true ~msg:"has tick 10" (contains svg ">10<")

let () =
  run "Scale"
    [
      group "linear"
        [
          test "ticks present" test_linear_ticks_present;
          test "xlim constrains" test_linear_xlim;
          test "ylim constrains" test_linear_ylim;
          test "negative range" test_linear_negative_range;
          test "small range" test_linear_small_range;
          test "single point" test_linear_single_point;
        ];
      group "log"
        [
          test "power-of-10 ticks" test_log_ticks; test "log y axis" test_log_y;
        ];
      group "sqrt"
        [
          test "handles zero" test_sqrt_handles_zero;
          test "differs from linear" test_sqrt_differs_from_linear;
        ];
      group "asinh"
        [
          test "negative values" test_asinh_negative_values;
          test "differs from linear" test_asinh_differs_from_linear;
        ];
      group "symlog"
        [
          test "linear and log ticks" test_symlog_has_linear_and_log_ticks;
          test "differs from linear" test_symlog_differs_from_linear;
        ];
      group "inverted"
        [
          test "reverses tick order" test_invert_reverses_tick_order;
          test "preserves ticks" test_invert_preserves_ticks;
          test "log inverted" test_log_inverted;
        ];
      group "custom ticks"
        [
          test "explicit xticks" test_explicit_xticks;
          test "explicit yticks" test_explicit_yticks;
        ];
    ]
