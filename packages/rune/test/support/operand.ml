(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module D = Nx.Dtype

(* Values *)

type value = F of float | Least | Greatest | C of float * float

let pp_value ppf = function
  | F x -> Format.fprintf ppf "%h" x
  | Least -> Format.pp_print_string ppf "least"
  | Greatest -> Format.pp_print_string ppf "greatest"
  | C (re, im) -> Format.fprintf ppf "%h+i%h" re im

let value (type v s) (dt : (v, s) D.t) w : v =
  let least = w = Least in
  let signed m = if least then -.m else m in
  match w with
  | F x -> D.of_float dt x
  | C (re, im) -> (
      match D.kind dt with
      | Complex -> { Complex.re; im }
      | _ -> D.of_float dt re)
  | Least | Greatest -> (
      match D.kind dt with
      | Float -> signed (D.float_format dt).max_finite
      | Complex ->
          let m =
            if D.equal dt D.Complex64 then (D.float_format D.Float32).max_finite
            else Float.max_float
          in
          { Complex.re = signed m; im = 0. }
      | Signed | Unsigned | Boolean ->
          if least then D.min_value dt else D.max_value dt)

let pool =
  [|
    F 0.;
    F (-0.);
    F 1.;
    F (-1.);
    F 0.5;
    F (-2.5);
    F 3.;
    F 7.25;
    F (-0.125);
    F 1e-3;
    F 1e30;
    F (-1e30);
    F Float.infinity;
    F Float.neg_infinity;
    F Float.nan;
    F 0x1p-1074;
    F (-0x1p-1074);
    F 0x1p-149;
    F Float.max_float;
    F (-.Float.max_float);
    F Float.min_float;
    F (-.Float.min_float);
    Least;
    Greatest;
  |]

(* Shapes *)

let numel s = Array.fold_left ( * ) 1 s

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* Past one kernel block: leading extents of 2 to 4, the last making 300 to 1100
   elements. *)
let large =
  let open Gen in
  let* r = int_range 1 3 in
  let* lead = array ~size:(constant (r - 1)) (int_range 2 4) in
  let p = numel lead in
  let+ last = int_range ((300 + p - 1) / p) (1100 / p) in
  Array.append lead [| last |]

let small =
  let open Gen in
  let* r = int_range 0 3 in
  array ~size:(constant r)
    (frequency [ (1, constant 0); (2, constant 1); (4, int_range 2 4) ])

let shape = Gen.frequency [ (19, small); (1, large) ]

(* Operands *)

type view = Plain | Flipped | Strided | Broadcast | Offset of int | Permuted

let pp_view ppf = function
  | Plain -> Format.pp_print_string ppf "plain"
  | Flipped -> Format.pp_print_string ppf "flipped"
  | Strided -> Format.pp_print_string ppf "strided"
  | Broadcast -> Format.pp_print_string ppf "broadcast"
  | Offset k -> Format.fprintf ppf "offset %d" k
  | Permuted -> Format.pp_print_string ppf "permuted"

type t = { view : view; values : value array }

let draw s =
  let open Gen in
  let* view =
    frequency
      [
        (5, of_list [ Plain; Flipped; Strided; Broadcast; Permuted ]);
        (1, map (fun k -> Offset k) (int_range 1 3));
      ]
  in
  let n =
    match view with
    | Plain | Flipped | Permuted -> numel s
    | Strided -> numel s * 2
    | Broadcast -> if Array.length s = 0 then 1 else numel s / Int.max 1 s.(0)
    | Offset k -> numel s + k
  in
  let element =
    frequency
      [
        (3, of_list (Array.to_list pool));
        (2, map (fun x -> F x) (float_range (-4.) 4.));
      ]
  in
  let+ values = array ~size:(constant n) element in
  { view; values }

let make (type v s) (dt : (v, s) D.t) s o : (v, s, Nx.host) Nx.t =
  let r = Array.length s in
  let create s vs = Nx.create dt s (Array.map (value dt) vs) in
  match o.view with
  | Plain -> create s o.values
  | Flipped -> Nx.flip (create s o.values)
  | Permuted ->
      let reversed = Array.init r (fun i -> s.(r - 1 - i)) in
      Nx.transpose (create reversed o.values)
  | Offset k ->
      let n = numel s in
      Nx.reshape s (Nx.slice [ R (k, k + n) ] (create [| k + n |] o.values))
  | Strided when r = 0 -> create s (Array.sub o.values 0 1)
  | Strided ->
      let s' = Array.copy s in
      s'.(r - 1) <- 2 * s.(r - 1);
      Nx.slice
        (List.init r (fun i -> if i = r - 1 then Nx.Rs (0, s'.(i), 2) else Nx.A))
        (create s' o.values)
  | Broadcast when r = 0 -> create s o.values
  | Broadcast ->
      let s1 = Array.copy s in
      s1.(0) <- 1;
      if s.(0) = 0 then create s [||]
      else Nx.broadcast_to s (create s1 o.values)

(* Bits *)

(* [random_bits dt s st] is a value of [dt] and shape [s] of random bits. *)
let random_bits (type v s) (dt : (v, s) Nx.dtype) s st : (v, s, Nx.host) Nx.t =
  let n = numel s in
  let byte () = Random.State.int st 256 in
  match Nx.Dtype.bits dt with
  | 1 ->
      Nx.bitcast dt
        (Nx.create Nx.bit s (Array.init n (fun _ -> Random.State.bool st)))
  | 4 ->
      Nx.bitcast dt
        (Nx.create Nx.uint4 s (Array.init n (fun _ -> Random.State.int st 16)))
  | 8 -> Nx.bitcast dt (Nx.create Nx.uint8 s (Array.init n (fun _ -> byte ())))
  | b ->
      let w = b / 8 in
      Nx.bitcast dt
        (Nx.create Nx.uint8 (Array.append s [| w |])
           (Array.init (n * w) (fun _ -> byte ())))

(* [viewed v ~offset s make] is a value of shape [s] as the view [v] of one
   [make] gives: its last axis stepped by 2 where strided, its first broadcast,
   its storage's first element skipped where [offset]. *)
let viewed v ~offset s make =
  let r = Array.length s in
  let make s =
    if not offset then make s
    else
      let n = numel s in
      Nx.reshape s (Nx.slice [ Nx.R (1, n + 1) ] (make [| n + 1 |]))
  in
  match v with
  | Flipped -> Nx.flip (make s)
  | Strided when r > 0 ->
      let s' = Array.copy s in
      s'.(r - 1) <- 2 * s.(r - 1);
      let step i = if i = r - 1 then Nx.Rs (0, s'.(i), 2) else Nx.A in
      Nx.slice (List.init r step) (make s')
  | Broadcast when r > 0 && s.(0) > 0 ->
      let s1 = Array.copy s in
      s1.(0) <- 1;
      Nx.broadcast_to s (make s1)
  | Permuted when r > 1 ->
      let rev = Array.init r (fun i -> s.(r - 1 - i)) in
      Nx.transpose (make rev)
  | Offset _ -> invalid_arg "viewed: an offset is [~offset]"
  | Plain | Strided | Broadcast | Permuted -> make s

(* [raw x] is [x]'s shape and each element's bits, read through a bitcast to the
   unsigned integers of its width, a complex128's as two [uint64]s. *)
let raw (type v s d) (x : (v, s, d) Nx.t) =
  let each f y = Array.map f (Nx.to_array y) in
  let elements =
    match Nx.Dtype.bits (Nx.dtype x) with
    | 1 -> each (fun b -> if b then "1" else "0") (Nx.bitcast Nx.bit x)
    | 4 -> each string_of_int (Nx.bitcast Nx.uint4 x)
    | 8 -> each string_of_int (Nx.bitcast Nx.uint8 x)
    | 16 -> each string_of_int (Nx.bitcast Nx.uint16 x)
    | 32 -> each (Printf.sprintf "%lx") (Nx.bitcast Nx.uint32 x)
    | _ -> each (Printf.sprintf "%Lx") (Nx.bitcast Nx.uint64 x)
  in
  (Nx.shape x, elements)

(* Placements *)

module Two =
  (val Nx.devices
         ~kernels:(module Nx_support.Counting)
         [ Nx_support.memory 0; Nx_support.memory 1 ])

type _ where =
  | Host : Nx.host where
  | Device : int -> Two.d where
  | Cut : Two.d where

type any_where = Where : 'd where -> any_where

let pp_where (type d) ppf (w : d where) =
  match w with
  | Host -> Format.pp_print_string ppf "host"
  | Device k -> Format.fprintf ppf "device %d" k
  | Cut -> Format.pp_print_string ppf "cut along axis 0"

let on (type d) (w : d where) : d Nx.Placement.t =
  match w with
  | Host -> Nx.Host.on
  | Device k -> Nx.Placement.device Two.v (Nx_support.memory k)
  | Cut -> Two.split ~axis:0

let where s =
  let cut = if Array.length s = 0 then [] else [ Where Cut ] in
  Gen.of_list ([ Where Host; Where (Device 0); Where (Device 1) ] @ cut)
