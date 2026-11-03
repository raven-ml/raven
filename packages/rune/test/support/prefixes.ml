(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Witness

(* The compiler scans an axis past 16 elements in chunks of 16, and the
   chunks' last elements the same way, so lengths either side of 16 and 256
   change the scheme between a prefix and its axis; 32 is where it changed
   before. *)

type running = Cumsum | Cumprod | Cummax | Cummin

type t =
  | Case : {
      dt : (float, 's) Nx.dtype;
      op : running;
      n : int;
      p : int;
      axis : int;
      values : float array;
    }
      -> t

let name = function
  | Cumsum -> "cumsum"
  | Cumprod -> "cumprod"
  | Cummax -> "cummax"
  | Cummin -> "cummin"

let pp ppf (Case c) =
  Format.fprintf ppf "%s %a, first %d of %d along %d: %a" (name c.op)
    Nx.pp_dtype c.dt c.p c.n c.axis
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf x ->
         Format.fprintf ppf "%h" x))
    (Array.to_list c.values)

let floats =
  Nx.Dtype.
    [
      Any Nx.float64;
      Any Nx.float32;
      Any Nx.float16;
      Any Nx.bfloat16;
      Any Nx.float8_e4m3fn;
      Any Nx.float8_e5m2;
      Any Nx.float4_e2m1fn;
    ]

let lengths = [ 15; 16; 17; 31; 32; 33; 255; 256; 257 ]

let case (Nx.Dtype.Any dt) op n p axis values =
  match Nx.Dtype.kind dt with
  | Float -> Case { dt; op; n; p; axis; values }
  | _ -> invalid_arg "Prefixes.case: not a float dtype"

(* A term of [op]: numbers across 17 binades, so that a sum's order shows in
   its bits, near 1 for a product, so that it stays finite over a few hundred
   terms, zeros of either sign, and now and then an infinity or a NaN. *)
let term op =
  let open Gen in
  let number =
    let* e = int_range (-8) 8 in
    let+ m = float_range (-1.) 1. in
    if op = Cumprod then Float.ldexp (1. +. (m /. 8.)) (e / 8)
    else Float.ldexp m e
  in
  frequency
    [
      (40, number);
      (2, of_list [ 0.; -0. ]);
      (1, of_list [ Float.infinity; Float.neg_infinity; Float.nan ]);
    ]

let gen =
  let open Gen in
  let* dt = of_list floats in
  let* op = of_list [ Cumsum; Cumprod; Cummax; Cummin ] in
  let* n = of_list lengths in
  let* p =
    frequency
      [
        (2, of_list (1 :: List.filter (fun p -> p < n) lengths));
        (1, int_range 0 n);
      ]
  in
  let* axis = int_range 0 1 in
  let+ values = array ~size:(constant (2 * n)) (term op) in
  case dt op n p axis values

(* Terms across 40 binades, of fractions of the golden ratio, so that a sum or
   product in another order rounds otherwise. *)
let golden op k =
  let x =
    Float.ldexp
      (Float.rem (Float.of_int k *. 0.6180339887) 1. -. 0.5)
      (((k * 7) mod 41) - 20)
  in
  if op = Cumprod then 1. +. (x /. 64.) else x

let examples =
  let spread dt op n p axis =
    case dt op n p axis (Array.init (2 * n) (golden op))
  and zeros dt op n p axis z = case dt op n p axis (Array.init (2 * n) z) in
  let signs k = if k mod 3 = 0 then 0. else -0. in
  Nx.Dtype.
    [
      spread (Any Nx.float64) Cumsum 33 17 1;
      spread (Any Nx.float32) Cumsum 257 32 0;
      spread (Any Nx.float32) Cumsum 256 255 1;
      spread (Any Nx.float64) Cumprod 255 33 0;
      spread (Any Nx.float16) Cummax 15 0 1;
      spread (Any Nx.bfloat16) Cummin 32 31 0;
      spread (Any Nx.float8_e4m3fn) Cumsum 31 17 1;
      spread (Any Nx.float8_e5m2) Cumprod 16 15 0;
      spread (Any Nx.float4_e2m1fn) Cummax 257 256 1;
      spread (Any Nx.float32) Cummin 33 32 0;
      spread (Any Nx.float16) Cumsum 17 16 1;
      zeros (Any Nx.float32) Cumsum 33 1 0 (fun _ -> -0.);
      zeros (Any Nx.float64) Cumsum 17 16 1 (fun _ -> -0.);
      zeros (Any Nx.float64) Cummax 33 17 1 signs;
      zeros (Any Nx.float32) Cummin 17 1 0 signs;
    ]

let law (module S : Nx.Devices) (Case { dt; op; n; p; axis; values }) =
  cover "about 16 elements" (n <= 17);
  cover "about 32 elements" (n >= 31 && n <= 33);
  cover "about 256 elements" (n >= 255);
  cover "an empty prefix" (p = 0);
  cover "a prefix of 32 or fewer of an axis past 32" (p <= 32 && n > 32);
  cover "a prefix past 32 of a longer axis" (p > 32 && p < n);
  cover "a sum" (op = Cumsum);
  cover "a product" (op = Cumprod);
  cover "a maximum" (op = Cummax);
  cover "a minimum" (op = Cummin);
  cover "along the first axis" (axis = 0);
  let shape = if axis = 0 then [| n; 2 |] else [| 2; n |] in
  let x = Nx.cast dt (Nx.create Nx.float64 shape values) in
  let f x =
    match op with
    | Cumsum -> Nx.cumsum ~axis x
    | Cumprod -> Nx.cumprod ~axis x
    | Cummax -> Nx.cummax ~axis x
    | Cummin -> Nx.cummin ~axis x
  in
  let first t =
    Nx.slice (if axis = 0 then [ R (0, p) ] else [ A; R (0, p) ]) t
  in
  let on_host t = codes (Nx.place Nx.Host.on t) in
  let before = Rune.evaluations () in
  let of_whole =
    outcome (fun () -> on_host (first (Rune.jit' f (Nx.place S.on x))))
  in
  let of_prefix =
    outcome (fun () -> on_host (Rune.jit' f (Nx.place S.on (first x))))
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal (outcome_t codes_t) of_whole of_prefix
