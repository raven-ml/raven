(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Data-to-unit mapping functions *)

type t = {
  to_unit : float -> float;
  from_unit : float -> float;
  lo : float;
  hi : float;
}

let maybe_invert invert to_unit from_unit =
  if invert then ((fun v -> 1. -. to_unit v), fun u -> from_unit (1. -. u))
  else (to_unit, from_unit)

let linear ?(invert = false) ~lo ~hi () =
  let range = hi -. lo in
  let range = if range = 0. then 1. else range in
  let to_unit, from_unit =
    maybe_invert invert
      (fun v -> (v -. lo) /. range)
      (fun u -> lo +. (u *. range))
  in
  { to_unit; from_unit; lo; hi }

let log ?(invert = false) ~lo ~hi () =
  let lo_log = Float.log10 (Float.max 1e-300 lo) in
  let hi_log = Float.log10 (Float.max 1e-300 hi) in
  let range = hi_log -. lo_log in
  let range = if range = 0. then 1. else range in
  let to_unit, from_unit =
    maybe_invert invert
      (fun v ->
        if v <= 0. then Float.nan else (Float.log10 v -. lo_log) /. range)
      (fun u -> Float.pow 10. (lo_log +. (u *. range)))
  in
  { to_unit; from_unit; lo; hi }

let sqrt ?(invert = false) ~lo ~hi () =
  let lo_s = Float.sqrt (Float.max 0. lo) in
  let hi_s = Float.sqrt (Float.max 0. hi) in
  let range = hi_s -. lo_s in
  let range = if range = 0. then 1. else range in
  let to_unit, from_unit =
    maybe_invert invert
      (fun v -> (Float.sqrt (Float.max 0. v) -. lo_s) /. range)
      (fun u ->
        let s = lo_s +. (u *. range) in
        s *. s)
  in
  { to_unit; from_unit; lo; hi }

let asinh ?(invert = false) ~lo ~hi () =
  let lo_a = Float.asinh lo in
  let hi_a = Float.asinh hi in
  let range = hi_a -. lo_a in
  let range = if range = 0. then 1. else range in
  let to_unit, from_unit =
    maybe_invert invert
      (fun v -> (Float.asinh v -. lo_a) /. range)
      (fun u ->
        let a = lo_a +. (u *. range) in
        Float.sinh a)
  in
  { to_unit; from_unit; lo; hi }

let symlog ?(invert = false) ~linthresh ~lo ~hi () =
  let transform v =
    if Float.abs v <= linthresh then v /. linthresh
    else Float.copy_sign (1. +. Float.log10 (Float.abs v /. linthresh)) v
  in
  let inv_transform v =
    if Float.abs v <= 1. then v *. linthresh
    else Float.copy_sign (linthresh *. Float.pow 10. (Float.abs v -. 1.)) v
  in
  let lo_t = transform lo in
  let hi_t = transform hi in
  let range = hi_t -. lo_t in
  let range = if range = 0. then 1. else range in
  let to_unit, from_unit =
    maybe_invert invert
      (fun v -> (transform v -. lo_t) /. range)
      (fun u -> inv_transform (lo_t +. (u *. range)))
  in
  { to_unit; from_unit; lo; hi }

let make ?(invert = false) kind ~lo ~hi () =
  match kind with
  | `Linear -> linear ~invert ~lo ~hi ()
  | `Log -> log ~invert ~lo ~hi ()
  | `Sqrt -> sqrt ~invert ~lo ~hi ()
  | `Asinh -> asinh ~invert ~lo ~hi ()
  | `Symlog linthresh -> symlog ~invert ~linthresh ~lo ~hi ()
