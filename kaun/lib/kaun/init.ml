(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  f :
    'layout.
    Rune.Rng.key ->
    int array ->
    (float, 'layout) Rune.dtype ->
    (float, 'layout) Rune.t;
}

type mode = [ `Fan_in | `Fan_out | `Fan_avg ]
type distribution = [ `Normal | `Truncated_normal | `Uniform ]

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let check_non_negative what value =
  if value < 0.0 then invalid_argf "%s must be >= 0, got %g" what value

let normalize_axis ~rank ~name axis =
  let axis = if axis < 0 then rank + axis else axis in
  if axis < 0 || axis >= rank then
    invalid_argf "invalid %s axis: %d for rank-%d shape" name axis rank;
  axis

(* Fan computation for variance scaling. *)

let compute_fans shape ~in_axis ~out_axis =
  let rank = Array.length shape in
  if rank = 0 then (1, 1)
  else if rank = 1 then
    let total = shape.(0) in
    (total, total)
  else
    let in_axis = normalize_axis ~rank ~name:"in" in_axis in
    let out_axis = normalize_axis ~rank ~name:"out" out_axis in
    let fan_in = shape.(in_axis) in
    let fan_out = shape.(out_axis) in
    let receptive = ref 1 in
    for i = 0 to rank - 1 do
      if i <> in_axis && i <> out_axis then receptive := !receptive * shape.(i)
    done;
    (fan_in * !receptive, fan_out * !receptive)

(* Truncated normal with bounds at +/-2 standard deviations. *)

let truncated_normal ~stddev key shape dtype =
  let z = Rune.Rng.truncated_normal ~key dtype ~lower:(-2.0) ~upper:2.0 shape in
  Rune.mul z (Rune.scalar dtype stddev)

(* Variance scaling — the general framework behind glorot/he/lecun. *)

let variance_scaling ~scale ~mode ~distribution ?(in_axis = -2) ?(out_axis = -1)
    () =
  check_non_negative "scale" scale;
  {
    f =
      (fun key shape dtype ->
        let fan_in, fan_out = compute_fans shape ~in_axis ~out_axis in
        let n =
          match mode with
          | `Fan_in -> float_of_int fan_in
          | `Fan_out -> float_of_int fan_out
          | `Fan_avg -> float_of_int (fan_in + fan_out) /. 2.0
        in
        if n <= 0.0 then
          invalid_argf "non-positive fan: fan_in=%d fan_out=%d" fan_in fan_out;
        let variance = scale /. n in
        match distribution with
        | `Normal ->
            let z = Rune.randn dtype ~key shape in
            Rune.mul z (Rune.scalar dtype (sqrt variance))
        | `Truncated_normal ->
            (* Correct for stddev loss from truncation to [-2, 2]. *)
            truncated_normal
              ~stddev:(sqrt variance /. 0.87962566103423978)
              key shape dtype
        | `Uniform ->
            let limit = sqrt (3.0 *. variance) in
            let u = Rune.rand dtype ~key shape in
            Rune.sub
              (Rune.mul u (Rune.scalar dtype (2.0 *. limit)))
              (Rune.scalar dtype limit));
  }

(* Constant *)

let constant value =
  { f = (fun _key shape dtype -> Rune.full dtype shape value) }

let zeros = constant 0.0
let ones = constant 1.0

(* Random *)

let uniform ?(scale = 0.01) () =
  check_non_negative "scale" scale;
  {
    f =
      (fun key shape dtype ->
        Rune.mul (Rune.rand dtype ~key shape) (Rune.scalar dtype scale));
  }

let normal ?(stddev = 0.01) () =
  check_non_negative "stddev" stddev;
  {
    f =
      (fun key shape dtype ->
        Rune.mul (Rune.randn dtype ~key shape) (Rune.scalar dtype stddev));
  }

(* Glorot / Xavier *)

let glorot_uniform ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:1.0 ~mode:`Fan_avg ~distribution:`Uniform ~in_axis
    ~out_axis ()

let glorot_normal ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:1.0 ~mode:`Fan_avg ~distribution:`Truncated_normal
    ~in_axis ~out_axis ()

(* He / Kaiming *)

let he_uniform ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:2.0 ~mode:`Fan_in ~distribution:`Uniform ~in_axis
    ~out_axis ()

let he_normal ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:2.0 ~mode:`Fan_in ~distribution:`Truncated_normal
    ~in_axis ~out_axis ()

(* LeCun *)

let lecun_uniform ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:1.0 ~mode:`Fan_in ~distribution:`Uniform ~in_axis
    ~out_axis ()

let lecun_normal ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:1.0 ~mode:`Fan_in ~distribution:`Truncated_normal
    ~in_axis ~out_axis ()
