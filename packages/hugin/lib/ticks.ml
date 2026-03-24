(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nice tick generation *)

let nice_num v round =
  let exp = Float.floor (Float.log10 v) in
  let frac = v /. Float.pow 10. exp in
  let nice =
    if round then
      begin if frac < 1.5 then 1.
      else if frac < 3. then 2.
      else if frac < 7. then 5.
      else 10.
      end
    else
      begin if frac <= 1. then 1.
      else if frac <= 2. then 2.
      else if frac <= 5. then 5.
      else 10.
      end
  in
  nice *. Float.pow 10. exp

let format_tick v =
  let v = if Float.abs v < 1e-14 then 0. else v in
  Printf.sprintf "%.6g" v

let generate_linear ~lo ~hi ~max_ticks =
  let range = nice_num (hi -. lo) false in
  let step = nice_num (range /. float max_ticks) true in
  let lo' = Float.floor (lo /. step) *. step in
  let acc = ref [] in
  let v = ref lo' in
  while !v <= hi +. (step *. 0.5) do
    if !v >= lo -. (step *. 0.001) && !v <= hi +. (step *. 0.001) then
      acc := (!v, format_tick !v) :: !acc;
    v := !v +. step
  done;
  List.rev !acc

let format_log_tick e =
  let ei = int_of_float e in
  if ei = 0 then "1" else if ei = 1 then "10" else Printf.sprintf "10^%d" ei

let generate_log ~lo ~hi ~max_ticks =
  let lo_exp = Float.floor (Float.log10 (Float.max 1e-300 lo)) in
  let hi_exp = Float.ceil (Float.log10 (Float.max 1e-300 hi)) in
  let n_decades = int_of_float (hi_exp -. lo_exp) in
  let stride = Float.of_int (max 1 ((n_decades + max_ticks - 1) / max_ticks)) in
  let acc = ref [] in
  let e = ref lo_exp in
  while !e <= hi_exp do
    let v = Float.pow 10. !e in
    if v >= lo *. 0.999 && v <= hi *. 1.001 then
      acc := (v, format_log_tick !e) :: !acc;
    e := !e +. stride
  done;
  List.rev !acc

(* Sqrt ticks: generate in data space using nice linear ticks *)

let generate_sqrt ~lo ~hi ~max_ticks =
  let lo = Float.max 0. lo in
  generate_linear ~lo ~hi ~max_ticks

(* Asinh ticks: pick nice values in data space *)

let generate_asinh ~lo ~hi ~max_ticks =
  if lo >= 0. then generate_linear ~lo ~hi ~max_ticks
  else generate_linear ~lo ~hi ~max_ticks

(* Symlog ticks: linear ticks inside linthresh, log ticks outside *)

let generate_symlog ~linthresh ~lo ~hi ~max_ticks =
  let ticks = ref [] in
  (* Linear region *)
  let lin_lo = Float.max lo (-.linthresh) in
  let lin_hi = Float.min hi linthresh in
  if lin_lo < lin_hi then begin
    let lin_ticks =
      generate_linear ~lo:lin_lo ~hi:lin_hi ~max_ticks:(max_ticks / 2)
    in
    ticks := lin_ticks
  end;
  (* Positive log region *)
  if hi > linthresh then begin
    let pos_lo = Float.max linthresh lo in
    let pos_ticks = generate_log ~lo:pos_lo ~hi ~max_ticks:(max_ticks / 3) in
    ticks := !ticks @ pos_ticks
  end;
  (* Negative log region *)
  if lo < -.linthresh then begin
    let neg_hi = Float.min (-.linthresh) hi in
    let neg_lo_abs = Float.abs lo in
    let neg_hi_abs = Float.abs neg_hi in
    let pos_ticks =
      generate_log ~lo:neg_hi_abs ~hi:neg_lo_abs ~max_ticks:(max_ticks / 3)
    in
    let neg_ticks =
      List.rev_map (fun (v, _) -> (-.v, format_tick (-.v))) pos_ticks
    in
    ticks := neg_ticks @ !ticks
  end;
  !ticks

let generate kind ~lo ~hi ?(max_ticks = 8) () =
  match kind with
  | `Linear -> generate_linear ~lo ~hi ~max_ticks
  | `Log -> generate_log ~lo ~hi ~max_ticks
  | `Sqrt -> generate_sqrt ~lo ~hi ~max_ticks
  | `Asinh -> generate_asinh ~lo ~hi ~max_ticks
  | `Symlog linthresh -> generate_symlog ~linthresh ~lo ~hi ~max_ticks
