open Hugin
open Windtrap

let check_float msg = equal ~msg (float 0.01)
let bw = Cmap.of_colors [| Color.black; Color.white |]

(* eval *)

let test_eval_at_zero () =
  let c = Cmap.eval bw 0.0 in
  check_float "lightness at 0" (Color.lightness Color.black) (Color.lightness c)

let test_eval_at_one () =
  let c = Cmap.eval bw 1.0 in
  check_float "lightness at 1" (Color.lightness Color.white) (Color.lightness c)

let test_eval_negative_clamped () =
  let c0 = Cmap.eval bw 0.0 in
  let cn = Cmap.eval bw (-0.5) in
  check_float "negative clamped" (Color.lightness c0) (Color.lightness cn)

let test_eval_above_one_clamped () =
  let c1 = Cmap.eval bw 1.0 in
  let ch = Cmap.eval bw 1.5 in
  check_float "above 1 clamped" (Color.lightness c1) (Color.lightness ch)

(* of_colors *)

let test_two_stops_midpoint () =
  let mid = Cmap.eval bw 0.5 in
  let l = Color.lightness mid in
  is_true ~msg:"midpoint lightness near 0.5" (l > 0.4 && l < 0.6)

let test_three_stops_midpoint () =
  let red = Color.rgb ~r:1. ~g:0. ~b:0. () in
  let green = Color.rgb ~r:0. ~g:1. ~b:0. () in
  let blue = Color.rgb ~r:0. ~g:0. ~b:1. () in
  let cm = Cmap.of_colors [| red; green; blue |] in
  let mid = Cmap.eval cm 0.5 in
  let r, g, _, _ = Color.to_rgba mid in
  (* At 0.5, should be near the green stop *)
  is_true ~msg:"green channel high at midpoint" (g > r)

let test_one_stop_raises () =
  raises_invalid_arg "Cmap.of_colors: need at least 2 stops" (fun () ->
      Cmap.of_colors [| Color.black |])

let test_empty_raises () =
  raises_invalid_arg "Cmap.of_colors: need at least 2 stops" (fun () ->
      Cmap.of_colors [||])

(* predefined *)

let test_predefined_no_raise () =
  let cmaps =
    [
      Cmap.viridis;
      Cmap.plasma;
      Cmap.inferno;
      Cmap.magma;
      Cmap.cividis;
      Cmap.coolwarm;
    ]
  in
  List.iter
    (fun cm ->
      let _ = Cmap.eval cm 0.0 in
      let _ = Cmap.eval cm 0.5 in
      let _ = Cmap.eval cm 1.0 in
      ())
    cmaps

let test_viridis_endpoints () =
  let r0, _, b0, _ = Cmap.eval Cmap.viridis 0.0 |> Color.to_rgba in
  let r1, _, _, _ = Cmap.eval Cmap.viridis 1.0 |> Color.to_rgba in
  (* Viridis starts dark purple (low r, high b), ends yellow (high r) *)
  is_true ~msg:"viridis start is dark" (r0 < 0.4);
  is_true ~msg:"viridis start has blue" (b0 > 0.2);
  is_true ~msg:"viridis end is bright" (r1 > 0.8)

let () =
  run "Cmap"
    [
      group "eval"
        [
          test "at 0.0" test_eval_at_zero;
          test "at 1.0" test_eval_at_one;
          test "negative clamped" test_eval_negative_clamped;
          test "above 1.0 clamped" test_eval_above_one_clamped;
        ];
      group "of_colors"
        [
          test "two stops midpoint" test_two_stops_midpoint;
          test "three stops midpoint" test_three_stops_midpoint;
          test "one stop raises" test_one_stop_raises;
          test "empty raises" test_empty_raises;
        ];
      group "predefined"
        [
          test "all evaluate without error" test_predefined_no_raise;
          test "viridis endpoints" test_viridis_endpoints;
        ];
    ]
