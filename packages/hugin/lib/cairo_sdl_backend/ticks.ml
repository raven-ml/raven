(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let format_tick_value (v : float) : string =
  let epsilon = 1e-12 in
  if abs_float v < epsilon then "0.0" else Printf.sprintf "%.6g" v

let round_to_nice v =
  if Float.equal v 0. || not (Float.is_finite v) then 1.0
  else
    let exponent = Float.floor (Float.log10 (Float.abs v)) in
    let fraction = Float.abs v /. (10. ** exponent) in
    let nice_fraction =
      if fraction <= 1.5 then 1.0
      else if fraction <= 3.5 then 2.0
      else if fraction <= 7.5 then 5.0
      else 10.0
    in
    nice_fraction *. (10. ** exponent)

let generate_linear_ticks ~min_val ~max_val ~max_ticks =
  if max_ticks <= 0 || min_val >= max_val then []
  else
    let range = Float.max 1e-9 (max_val -. min_val) in
    let rough_step = range /. Float.of_int max_ticks in
    let step = round_to_nice rough_step in

    if Float.equal step 0. then []
    else
      let k_start = Float.ceil (min_val /. step) in
      let start_tick = k_start *. step in

      let ticks = ref [] in
      let current_tick = ref start_tick in
      let count = ref 0 in
      let limit = max_val +. (step *. 1e-9) in

      while !current_tick <= limit && !count < max_ticks * 2 do
        ticks := !current_tick :: !ticks;
        current_tick := !current_tick +. step;
        count := !count + 1
      done;

      List.rev !ticks
