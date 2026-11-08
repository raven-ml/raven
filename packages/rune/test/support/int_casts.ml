(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Witness

type t =
  | Case : {
      src : ('v, 's) Nx.dtype;
      dst : Nx.Dtype.any;
      xs : int64 array;
    }
      -> t

let pp ppf (Case { src; dst = Nx.Dtype.Any dt; xs }) =
  let values ppf a =
    Format.pp_print_list ~pp_sep:Format.pp_print_space
      (fun ppf v -> Format.fprintf ppf "%Ld" v)
      ppf (Array.to_list a)
  in
  Format.fprintf ppf "%a to %a of [%a]" Nx.pp_dtype src Nx.pp_dtype dt values xs

let dsts =
  List.filter
    (fun (Nx.Dtype.Any dt) ->
      match Nx.Dtype.kind dt with Float | Complex -> true | _ -> false)
    Nx.Dtype.all

let srcs =
  List.filter
    (fun (Nx.Dtype.Any dt) ->
      match Nx.Dtype.kind dt with
      | Signed | Unsigned | Boolean -> true
      | _ -> false)
    Nx.Dtype.all

let width (Nx.Dtype.Any dt) = Nx.Dtype.bits dt
let wide s = width s >= 32
let signed (Nx.Dtype.Any dt) = Nx.Dtype.is Signed dt
let boolean (Nx.Dtype.Any dt) = Nx.Dtype.is Boolean dt

(* The bits of a destination's significand, its implicit bit included. *)
let precision (Nx.Dtype.Any dt) =
  match dt with
  | Nx.Dtype.Float64 | Nx.Dtype.Complex128 -> 53
  | Nx.Dtype.Float32 | Nx.Dtype.Complex64 -> 24
  | Nx.Dtype.Float16 -> 11
  | Nx.Dtype.Bfloat16 -> 8
  | Nx.Dtype.Float8_e4m3fn -> 4
  | Nx.Dtype.Float8_e5m2 -> 3
  | Nx.Dtype.Float4_e2m1fn -> 2
  | _ -> invalid_arg "Int_casts.precision: not a float or complex dtype"

let dst64 (Case { dst; _ }) = precision dst = 53
let pow2 k = Int64.shift_left 1L k

(* [x] as the exact value of [s]'s element: [1] or [0] for a boolean, else
   modulo [s]'s width. The greatest uint64 is [-1L]. *)
let wrap s x =
  let w = width s in
  if boolean s then if x = 0L then 0L else 1L
  else if w = 64 then x
  else
    let m = Int64.logand x (Int64.pred (pow2 w)) in
    if signed s && m >= pow2 (w - 1) then Int64.sub m (pow2 w) else m

(* [s]'s extremes and their neighbours, [-1] to [1], and [2^24] and [2^53],
   where float32 and float64 stop holding every integer, and their neighbours,
   of both signs. *)
let edges s =
  let w = width s in
  let lo, hi =
    if signed s then (Int64.neg (pow2 (w - 1)), Int64.pred (pow2 (w - 1)))
    else (0L, if w = 64 then -1L else Int64.pred (pow2 w))
  in
  let around v = [ Int64.pred v; v; Int64.succ v ] in
  let powers = around (pow2 24) @ around (pow2 53) in
  [ lo; Int64.succ lo; Int64.pred hi; hi; -1L; 0L; 1L ]
  @ powers @ List.map Int64.neg powers

(* At each exponent [e] that [s] holds, the ties of a destination of [p] bits,
   the bit one past its precision set, with the last bit it keeps clear and set,
   and their neighbours, of both signs. *)
let ties s p =
  if boolean s then []
  else
    let top = width s - if signed s then 2 else 1 in
    let at e =
      let tie = Int64.logor (pow2 e) (pow2 (e - p)) in
      let odd = Int64.logor tie (pow2 (e - p + 1)) in
      List.concat_map (fun v -> [ Int64.pred v; v; Int64.succ v ]) [ tie; odd ]
    in
    let pos = List.concat_map at (List.init (max 0 (top - p + 1)) (( + ) p)) in
    pos @ List.map Int64.neg pos

let case (Nx.Dtype.Any src as s) dst xs =
  Case { src; dst; xs = Array.of_list (edges s @ xs) }

let examples =
  let narrow_dsts =
    Nx.Dtype.[ Any Nx.float16; Any Nx.bfloat16; Any Nx.float32 ]
  in
  List.concat_map
    (fun s ->
      let ds = if wide s then dsts else narrow_dsts in
      List.map (fun d -> case s d (ties s (precision d))) ds)
    srcs

(* A sum of up to three integers below 2^12, each shifted anywhere and of either
   sign: sparse bits that put ties and near ties at every exponent. *)
let sparse =
  let open Gen in
  let term =
    let+ k = int_range 1 4095 and+ e = int_range 0 63 and+ neg = bool in
    let v = Int64.shift_left (Int64.of_int k) e in
    if neg then Int64.neg v else v
  in
  let* n = int_range 1 3 in
  let+ ts = list ~size:(constant n) term in
  List.fold_left Int64.add 0L ts

let elements = 64

let gen =
  let open Gen in
  let* s = of_list srcs in
  let* d = of_list dsts in
  let+ xs =
    list ~size:(constant elements)
      (frequency [ (2, of_list (edges s)); (3, sparse); (1, int64) ])
  in
  let (Nx.Dtype.Any src) = s in
  Case { src; dst = d; xs = Array.of_list xs }

(* The positions of the highest and lowest set bits of [m], unsigned bits; [-1]
   for both at [0]. *)
let span m =
  let bit i = Int64.logand (Int64.shift_right_logical m i) 1L = 1L in
  let rec high i = if i < 0 || bit i then i else high (i - 1) in
  let rec low i = if i > 63 || bit i then i else low (i + 1) in
  let h = high 63 in
  if h < 0 then (-1, -1) else (h, low 0)

let held bits m =
  let h, l = span m in
  h - l < bits

(* Whether [m] lies past a tie of a destination of [p] bits by bits only that
   float32 drops: rounded to float32 first, it is the tie, which rounds to even,
   down, where [m] rounds up. *)
let doubly_rounded p m =
  let h, l = span m in
  let bit i = Int64.logand (Int64.shift_right_logical m i) 1L = 1L in
  let clear i = not (bit i) in
  p < 24
  && l < h - 24
  && bit (h - p)
  && clear (h - p + 1)
  && List.for_all clear (List.init (24 - p) (( + ) (h - 24)))

let law (module S : Nx.Devices) (Case { src; dst = Nx.Dtype.Any dt as d; xs }) =
  let s = Nx.Dtype.Any src in
  let values = Array.map (wrap s) xs in
  let mags =
    Array.map (fun v -> if signed s && v < 0L then Int64.neg v else v) values
  in
  List.iter
    (fun (Nx.Dtype.Any e as a) ->
      if wide a then cover (Nx.Dtype.name e) (Nx.Dtype.equal e src))
    srcs;
  List.iter
    (fun (Nx.Dtype.Any e as a) ->
      if precision a <= 24 then cover (Nx.Dtype.name e) (Nx.Dtype.equal e dt))
    dsts;
  cover "a negative element" (signed s && Array.exists (fun v -> v < 0L) values);
  cover "an element float32 does not hold"
    (Array.exists (fun m -> not (held 24 m)) mags);
  cover "an element float64 does not hold"
    (Array.exists (fun m -> not (held 53 m)) mags);
  cover "a tie past which float32 drops bits"
    (Array.exists (doubly_rounded (precision d)) mags);
  let n = Array.length xs in
  let x = Nx.cast src (Nx.create Nx.int64 [| n |] xs) in
  let cast x = Nx.cast dt x in
  let before = Rune.evaluations () in
  let y =
    Rune.jit Nx.Ptree.(tensor @-> returns tensor) cast (Nx.place S.on x)
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal ~msg:"cast" codes_t (codes (cast x)) (codes (Nx.place Nx.Host.on y))
