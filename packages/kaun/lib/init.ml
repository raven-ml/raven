(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('b, 'd) t =
  'd Nx.Rng.key ->
  fan_in:int ->
  fan_out:int ->
  (float, 'b) Nx.dtype ->
  int array ->
  (float, 'b, 'd) Nx.t

type mode = [ `Fan_in | `Fan_out | `Fan_avg ]
type distribution = [ `Normal | `Truncated_normal | `Uniform ]

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let check_non_negative what value =
  if value < 0.0 then invalid_argf "%s must be >= 0, got %g" what value

(* Constant *)

(* A key's placement, where its draws lie: a key is whole on each device. *)
let at key = Nx.placement (Nx.Rng.to_tensor key)

let constant value key ~fan_in:_ ~fan_out:_ dtype shape =
  Nx.full_at (at key) dtype shape value

let zeros key ~fan_in ~fan_out dtype shape =
  constant 0.0 key ~fan_in ~fan_out dtype shape

let ones key ~fan_in ~fan_out dtype shape =
  constant 1.0 key ~fan_in ~fan_out dtype shape

(* Random *)

let uniform ~scale =
  check_non_negative "scale" scale;
  fun key ~fan_in:_ ~fan_out:_ dtype shape ->
    Nx.mul_s (Nx.Rng.uniform_at (at key) ~key dtype shape) scale

let normal ~stddev =
  check_non_negative "stddev" stddev;
  fun key ~fan_in:_ ~fan_out:_ dtype shape ->
    Nx.mul_s (Nx.Rng.normal_at (at key) ~key dtype shape) stddev

(* Variance scaling — the general scheme behind glorot/he/lecun. *)

(* Standard deviation of a standard normal truncated to [-2, 2]. *)
let truncated_stddev = 0.87962566103423978

let variance_scaling ~scale ~mode ~distribution =
  check_non_negative "scale" scale;
  fun key ~fan_in ~fan_out dtype shape ->
    if fan_in <= 0 || fan_out <= 0 then
      invalid_argf "fans must be positive, got fan_in=%d fan_out=%d" fan_in
        fan_out;
    let n =
      match mode with
      | `Fan_in -> float_of_int fan_in
      | `Fan_out -> float_of_int fan_out
      | `Fan_avg -> float_of_int (fan_in + fan_out) /. 2.0
    in
    let variance = scale /. n in
    match distribution with
    | `Normal ->
        Nx.mul_s (Nx.Rng.normal_at (at key) ~key dtype shape) (sqrt variance)
    | `Truncated_normal ->
        (* Rescale so the truncated samples reach the target variance. *)
        let stddev = sqrt variance /. truncated_stddev in
        let bound v = Nx.full_at (at key) dtype shape v in
        Nx.mul_s
          (Nx.Rng.truncated_normal ~key (bound (-2.0)) (bound 2.0))
          stddev
    | `Uniform ->
        let limit = sqrt (3.0 *. variance) in
        let u = Nx.Rng.uniform_at (at key) ~key dtype shape in
        Nx.sub_s (Nx.mul_s u (2.0 *. limit)) limit

(* Named families. Eta-expanded so each exports as a polymorphic value. *)

let glorot_uniform key ~fan_in ~fan_out dtype shape =
  variance_scaling ~scale:1.0 ~mode:`Fan_avg ~distribution:`Uniform key ~fan_in
    ~fan_out dtype shape

let glorot_normal key ~fan_in ~fan_out dtype shape =
  variance_scaling ~scale:1.0 ~mode:`Fan_avg ~distribution:`Truncated_normal key
    ~fan_in ~fan_out dtype shape

let he_uniform key ~fan_in ~fan_out dtype shape =
  variance_scaling ~scale:2.0 ~mode:`Fan_in ~distribution:`Uniform key ~fan_in
    ~fan_out dtype shape

let he_normal key ~fan_in ~fan_out dtype shape =
  variance_scaling ~scale:2.0 ~mode:`Fan_in ~distribution:`Truncated_normal key
    ~fan_in ~fan_out dtype shape

let lecun_uniform key ~fan_in ~fan_out dtype shape =
  variance_scaling ~scale:1.0 ~mode:`Fan_in ~distribution:`Uniform key ~fan_in
    ~fan_out dtype shape

let lecun_normal key ~fan_in ~fan_out dtype shape =
  variance_scaling ~scale:1.0 ~mode:`Fan_in ~distribution:`Truncated_normal key
    ~fan_in ~fan_out dtype shape
