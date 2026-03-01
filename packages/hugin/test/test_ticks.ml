(*---------------------------------------------------------------------------
  Tests for Ticks — exercised indirectly through SVG output. We verify tick
  label formatting, count, and presence in rendered SVGs.
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

(* linear tick formatting *)

let test_zero_label () =
  let x = Nx.create Float32 [| 5 |] [| -10.; -5.; 0.; 5.; 10. |] in
  let y = Nx.create Float32 [| 5 |] [| -10.; -5.; 0.; 5.; 10. |] in
  let svg = render (Hugin.line ~x ~y ()) in
  (* The zero tick should show "0" not "1e-15" or similar *)
  is_true ~msg:"has zero tick" (contains svg ">0<")

let test_reasonable_count () =
  let x = Nx.init Float32 [| 101 |] (fun i -> float_of_int i.(0)) in
  let y = Nx.init Float32 [| 101 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.line ~x ~y ()) in
  (* Count text elements that look like tick labels. Each tick generates a
     <text> element. A basic line plot should have title-area text + x ticks + y
     ticks. We just check it's not an absurd number. *)
  let text_count = count_substring svg "<text " in
  is_true ~msg:"text count reasonable" (text_count > 2 && text_count < 40)

(* log tick formatting *)

let test_log_tick_labels () =
  let x = Nx.create Float32 [| 5 |] [| 0.01; 0.1; 1.; 10.; 100. |] in
  let y = Nx.create Float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xscale `Log) in
  (* Log ticks should be powers of 10, formatted as 10^k *)
  is_true ~msg:"has 10^-2" (contains svg ">10^-2<");
  is_true ~msg:"has 10^2" (contains svg ">10^2<")

(* large range doesn't explode *)

let test_large_range () =
  let x = Nx.create Float32 [| 2 |] [| 0.; 1e6 |] in
  let y = Nx.create Float32 [| 2 |] [| 0.; 1. |] in
  let svg = render (Hugin.line ~x ~y ()) in
  let text_count = count_substring svg "<text " in
  is_true ~msg:"tick count bounded" (text_count < 40)

(* small fractional range *)

let test_fractional_range () =
  let x = Nx.create Float32 [| 2 |] [| 0.; 0.001 |] in
  let y = Nx.create Float32 [| 2 |] [| 0.; 1. |] in
  let svg = render (Hugin.line ~x ~y ()) in
  is_true ~msg:"valid svg" (contains svg "<svg")

let () =
  run "Ticks"
    [
      group "linear formatting"
        [
          test "zero label" test_zero_label;
          test "reasonable count" test_reasonable_count;
          test "large range bounded" test_large_range;
          test "fractional range" test_fractional_range;
        ];
      group "log formatting" [ test "log tick labels" test_log_tick_labels ];
    ]
