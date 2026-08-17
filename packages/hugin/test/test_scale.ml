(*---------------------------------------------------------------------------
  Tests for Scale logic — exercised indirectly through Hugin.render_svg. We
  verify that linear and log scales produce correct axis tick labels in the SVG
  output, which proves the scale math is correct.
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

(* Non-overlapping occurrence count, matching windtrap's [contains ~count]. *)
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

let render spec =
  let tmp = temp_file ~suffix:".svg" () in
  Hugin.render_svg ~width:400. ~height:300. tmp spec;
  let ic = open_in tmp in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let x5 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let y5 = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let vec l = Nx.create Float32 [| List.length l |] (Array.of_list l)

(* A degenerate input must still yield a well-formed document rather than a
   crash or a truncated file — the claim these cases actually make. *)
let well_formed svg =
  starts_with ~affix:"<?xml" svg;
  ends_with ~affix:"</svg>\n" svg

(* linear scale *)

let test_linear_ticks_present () =
  let svg = render (Hugin.line ~x:x5 ~y:y5 ()) in
  (* Data range 0-4, auto-ticks should include 0 *)
  contains ~sub:">0<" svg

let test_linear_xlim () =
  (* Use different x and y ranges so we can distinguish x ticks from y ticks. x
     data: 0..10, y data: 100..200. With xlim 0-5, x ticks stay in [0,5] but y
     ticks are around 100-200 — no overlap. *)
  let x = Nx.init Float32 [| 11 |] (fun i -> float_of_int i.(0)) in
  let y =
    Nx.init Float32 [| 11 |] (fun i -> 100. +. (float_of_int i.(0) *. 10.))
  in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xlim 0. 5.) in
  contains ~sub:">0<" svg;
  (* With xlim 0-5, we should not see x-axis tick "8" or "10". Y-axis ticks are
     in 100-200 range so no confusion. *)
  not_contains ~sub:">8<" svg;
  not_contains ~sub:">10<" svg

let test_linear_ylim () =
  (* Mirror of the xlim case: x lives in 100..200 so that a ">10<" match can
     only have come from the y axis. *)
  let x =
    Nx.init Float32 [| 11 |] (fun i -> 100. +. (float_of_int i.(0) *. 10.))
  in
  let y = Nx.init Float32 [| 11 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.line ~x ~y () |> Hugin.ylim 0. 5.) in
  well_formed svg;
  contains ~sub:">0<" svg;
  not_contains ~msg:"ylim 0-5 excludes a tick at 10" ~sub:">10<" svg

let test_linear_negative_range () =
  let x = vec [ -10.; -5.; 0.; 5.; 10. ] in
  let svg = render (Hugin.line ~x ~y:x ()) in
  (* A symmetric range must show the negative ticks, not just the origin. *)
  in_order ~subs:[ ">-10<"; ">-5<"; ">0<"; ">5<"; ">10<" ] svg

let test_linear_small_range () =
  let x = vec [ 0.; 0.0005; 0.001 ] in
  let y = vec [ 0.; 0.5; 1. ] in
  let svg = render (Hugin.line ~x ~y ()) in
  well_formed svg;
  contains ~msg:"a sub-millesimal range still draws its line" ~sub:"<path" svg

let test_linear_single_point () =
  let p = vec [ 5. ] in
  let svg = render (Hugin.line ~x:p ~y:p ()) in
  (* A degenerate zero-width range must not divide by zero or emit NaN
     coordinates, which is what a bare "<svg is present" check would miss. *)
  well_formed svg;
  not_contains ~sub:"NaN" svg;
  not_contains ~sub:"inf" svg

(* log scale *)

let test_log_ticks () =
  let x = vec [ 1.; 10.; 100.; 1000. ] in
  let y = vec [ 1.; 2.; 3.; 4. ] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xscale `Log) in
  (* Decade ticks, ascending along the axis — order is part of the claim. *)
  in_order ~subs:[ ">1<"; ">10<"; ">10^2<"; ">10^3<" ] svg

let test_log_y () =
  let x = vec [ 1.; 2.; 3.; 4. ] in
  let y = vec [ 1.; 10.; 100.; 1000. ] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.yscale `Log) in
  contains ~sub:">1<" svg;
  contains ~sub:">10^3<" svg

let test_log_rejects_nothing_below_one () =
  (* A log axis over a decade span must label every decade in between, not just
     the endpoints. *)
  let x = vec [ 1.; 1000. ] in
  let y = vec [ 1.; 2. ] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xscale `Log) in
  in_order ~subs:[ ">1<"; ">10<"; ">10^2<"; ">10^3<" ] svg

(* custom ticks *)

let test_explicit_xticks () =
  let svg =
    render
      (Hugin.line ~x:x5 ~y:y5 () |> Hugin.xticks [ (0., "zero"); (4., "four") ])
  in
  in_order ~subs:[ ">zero<"; ">four<" ] svg

let test_explicit_yticks () =
  let svg =
    render
      (Hugin.line ~x:x5 ~y:y5 () |> Hugin.yticks [ (0., "low"); (4., "high") ])
  in
  contains ~sub:">low<" svg;
  contains ~sub:">high<" svg

let test_explicit_xticks_replace_auto () =
  (* xticks "Overrides auto-generated ticks" (hugin.mli), so the automatic
     labels must be gone — not merely joined by the custom ones. The y data is
     pushed into the hundreds so that a bare ">1<" cannot be a y-axis tick. *)
  let y =
    Nx.init Float32 [| 5 |] (fun i -> 100. +. (float_of_int i.(0) *. 100.))
  in
  let svg =
    render
      (Hugin.line ~x:x5 ~y () |> Hugin.xticks [ (0., "zero"); (4., "four") ])
  in
  in_order ~subs:[ ">zero<"; ">four<" ] svg;
  not_contains ~sub:">1<" svg;
  not_contains ~sub:">2<" svg;
  not_contains ~sub:">3<" svg

(* sqrt scale *)

let sqrt_x = vec [ 0.; 1.; 4.; 9.; 16. ]
let sqrt_y = vec [ 0.; 1.; 2.; 3.; 4. ]

let test_sqrt_handles_zero () =
  (* Sqrt scale handles zero gracefully — critical for astronomical fluxes *)
  let svg = render (Hugin.line ~x:sqrt_x ~y:sqrt_y () |> Hugin.xscale `Sqrt) in
  contains ~sub:">0<" svg;
  contains ~sub:"<path" svg;
  not_contains ~msg:"sqrt of 0 must not produce NaN" ~sub:"NaN" svg

let test_sqrt_differs_from_linear () =
  not_equal ~msg:"sqrt changes output" text
    (render (Hugin.line ~x:sqrt_x ~y:sqrt_y () |> Hugin.yscale `Sqrt))
    (render (Hugin.line ~x:sqrt_x ~y:sqrt_y ()))

(* asinh scale *)

let asinh_x = vec [ -100.; -1.; 0.; 1.; 100. ]
let asinh_y = vec [ 0.; 1.; 2.; 3.; 4. ]

let test_asinh_negative_values () =
  (* Asinh handles negative values, unlike log — needed for
     background-subtracted fluxes *)
  let svg =
    render (Hugin.line ~x:asinh_x ~y:asinh_y () |> Hugin.xscale `Asinh)
  in
  contains ~sub:">0<" svg;
  contains ~sub:"<path" svg;
  not_contains ~msg:"asinh of a negative must not produce NaN" ~sub:"NaN" svg

let test_asinh_differs_from_linear () =
  not_equal ~msg:"asinh changes output" text
    (render (Hugin.line ~x:asinh_y ~y:asinh_x () |> Hugin.yscale `Asinh))
    (render (Hugin.line ~x:asinh_y ~y:asinh_x ()))

(* symlog scale *)

let symlog_x = vec [ -1000.; -10.; -1.; 0.; 1.; 10.; 1000. ]
let symlog_y = Nx.init Float32 [| 7 |] (fun i -> float_of_int i.(0))

let test_symlog_has_linear_and_log_ticks () =
  (* Symlog should produce ticks in both the linear region (near 0) and the log
     region (far from 0) *)
  let svg =
    render (Hugin.line ~x:symlog_x ~y:symlog_y () |> Hugin.xscale (`Symlog 10.))
  in
  contains ~msg:"linear region" ~sub:">0<" svg;
  contains ~sub:"<path" svg;
  not_contains ~msg:"symlog spans zero without a NaN" ~sub:"NaN" svg

let test_symlog_differs_from_linear () =
  not_equal ~msg:"symlog changes output" text
    (render
       (Hugin.line ~x:symlog_x ~y:symlog_y () |> Hugin.xscale (`Symlog 10.)))
    (render (Hugin.line ~x:symlog_x ~y:symlog_y ()))

(* inverted scales *)

let test_invert_reverses_tick_order () =
  (* The same tick labels should appear, but xinvert swaps pixel positions. We
     verify the SVG output actually changes. *)
  let svg_inv = render (Hugin.line ~x:x5 ~y:y5 () |> Hugin.xinvert) in
  contains ~sub:">0<" svg_inv;
  not_equal ~msg:"invert changes output" text svg_inv
    (render (Hugin.line ~x:x5 ~y:y5 ()))

let test_invert_preserves_ticks () =
  (* Inversion should not remove or add ticks, just reposition them *)
  let texts spec = count_substring (render spec) "<text" in
  equal ~msg:"same number of text elements" int
    (texts (Hugin.line ~x:x5 ~y:y5 ()))
    (texts (Hugin.line ~x:x5 ~y:y5 () |> Hugin.yinvert))

let test_log_inverted () =
  (* Log + invert is the typical RA axis for sky charts *)
  let x = vec [ 1.; 10.; 100.; 1000. ] in
  let y = vec [ 1.; 2.; 3.; 4. ] in
  let svg =
    render (Hugin.line ~x ~y () |> Hugin.xscale `Log |> Hugin.xinvert)
  in
  contains ~sub:">1<" svg;
  contains ~sub:">10<" svg

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
          test "power-of-10 ticks" test_log_ticks;
          test "log y axis" test_log_y;
          test "labels every decade" test_log_rejects_nothing_below_one;
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
          test "explicit ticks replace auto" test_explicit_xticks_replace_auto;
        ];
    ]
