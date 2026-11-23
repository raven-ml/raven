(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk

type t = Ops.t

(* Constants *)

let movement = function
  | Op.Reshape | Op.Expand | Op.Permute | Op.Shrink -> true
  | _ -> false

(* A scalar literal: a constant of no axis, or a cast of one. *)
let scalar u = Ops.is_literal u && Shape.shape u = [] && not (Ops.is_invalid u)

(* A constant node's elements, as scalar literals: one, or one per index along
   one axis, as a stack of literals gives each of its lanes, the joined parts of
   a complex constant among them. *)
type held = Uniform of t | Lanes of int * t array

let ints dims =
  List.fold_right
    (fun d acc ->
      match (d, acc) with Ops.Int n, Some ns -> Some (n :: ns) | _ -> None)
    dims (Some [])

let product ns = List.fold_left ( * ) 1 ns
let after a ns = product (List.filteri (fun i _ -> i > a) ns)
let on a ls = if Array.length ls = 1 then Uniform ls.(0) else Lanes (a, ls)

(* [moved u h] is the elements of the movement [u] of a node whose elements are
   [h]: the lanes' axis followed to its place in [u]. A reshape keeps it where
   some axis of [u] has its size and the elements after it number as after it in
   the source, so that each element keeps its lane. *)
let moved u = function
  | Uniform _ as h -> Some h
  | Lanes (a, ls) -> (
      let n = Array.length ls in
      match Shape.marg u with
      | Ops.Permute order ->
          Option.map
            (fun (j, _) -> Lanes (j, ls))
            (List.find_opt
               (fun (_, s) -> s = a)
               (List.mapi (fun j s -> (j, s)) order))
      | Ops.Shrink bounds -> (
          match List.nth bounds a with
          | Ops.Int start, Ops.Int size -> Some (on a (Array.sub ls start size))
          | _ -> None)
      | Ops.Expand _ ->
          Some (Lanes (a + Shape.ndim u - Shape.ndim (Ops.nth u 0), ls))
      | Ops.Reshape _ -> (
          match (ints (Shape.shape (Ops.nth u 0)), ints (Shape.shape u)) with
          | Some src, Some dst ->
              let inner = after a src in
              List.find_mapi
                (fun b d ->
                  if d = n && after b dst = inner then Some (Lanes (b, ls))
                  else None)
                dst
          | _ -> None)
      | Ops.Pad _ -> None)

(* [constant u] is the elements of [u] where [u] is a scalar literal, a stack of
   them or a movement of one. *)
let rec constant u =
  match Ops.op u with
  | op when movement op -> Option.bind (constant (Ops.nth u 0)) (moved u)
  | Op.Stack ->
      let lane s =
        match constant s with
        | Some (Uniform l) when Dtype.equal (Ops.dtype l) (Ops.dtype u) ->
            Some l
        | _ -> None
      in
      let ls = List.map lane (Ops.src u) in
      if List.mem None ls then None
      else Some (on 0 (Array.of_list (List.filter_map Fun.id ls)))
  | _ when scalar u -> Some (Uniform u)
  | _ -> None

let uniform u = match constant u with Some (Uniform l) -> Some l | _ -> None

(* Exclusions *)

(* The value of the scalar literal [u] as its type holds it. *)
let held u =
  match Dtype.const (Ops.dtype u) (Ops.value u) with
  | #Dtype.value as v -> Some v
  | `Invalid -> None

let source_value u i =
  let s = Ops.nth u i in
  if scalar s then held s else None

(* A float converted to an integer type folds only where the type holds its
   truncation. *)
let held_by dt = function
  | `Float f -> (
      Float.is_finite f
      &&
      let t = Bigint.of_float f in
      match (Dtype.min dt, Dtype.max dt) with
      | `Int lo, `Int hi -> Bigint.geq t lo && Bigint.leq t hi
      | _ -> false)
  | _ -> true

(* [kept u] is [true] where the node [u], of a graph of scalars, is left to
   tolk's code generation: see the interface. *)
let kept u =
  let dt = Ops.dtype u in
  match Ops.op u with
  | Op.Exp2 | Op.Log2 | Op.Sin | Op.Pow | Op.Threefry -> true
  | Op.Mulacc -> not (Dtype.equal dt Float32 || Dtype.equal dt Float64)
  | Op.Shl | Op.Shr -> (
      match source_value u 1 with
      | Some (`Int n) ->
          not
            (Dtype.mem dt Dtype.ints && Bigint.geq n Bigint.zero
            && Bigint.lt n (Bigint.of_int (Dtype.bitsize dt)))
      | _ -> true)
  | Op.Truncdiv | Op.Truncmod | Op.Div | Op.Mod -> (
      match source_value u 1 with
      | Some (`Int d) -> Bigint.equal d Bigint.zero
      | Some (`Bool d) -> not d
      | _ -> true)
  | Op.Cast when Dtype.mem dt Dtype.ints -> (
      let s = Ops.nth u 0 in
      if not (Dtype.is_float (Ops.dtype s)) then false
      else
        match source_value u 0 with
        | Some v -> not (held_by dt v)
        | None -> true)
  | Op.Cast | Op.Bitcast -> false
  | op -> not (Ops.is_literal u || Op.Set.mem op Op.Set.alu)

let rec keeps s = (not (scalar s)) && (kept s || List.exists keeps (Ops.src s))

(* Folding *)

(* [rewritten s] is the graph of scalars [s] rewritten by tolk's rules to a
   fixed point, sources first, as {!Ops.graph_rewrite} would: a graph of an Ops
   function is a few nodes, too few for that walk's tables to pay. *)
let rec rewritten s =
  if scalar s then s
  else
    let u = Ops.replace s ~src:(List.map rewritten (Ops.src s)) in
    match Ops.Pattern_matcher.rewrite Shape.symbolic_simple () u with
    | Some v when v != u -> rewritten v
    | _ -> u

(* [fold build args r] is [r], which [build] built over [args], as its value
   moved to its shape where every operand is a constant: [build] of the
   operands' scalar literals, a graph of scalars that tolk's own rules fold to
   the value its code generation would give [r]. *)
let fold build args r =
  if List.memq r args then r
  else
    let ls = List.filter_map uniform args in
    if List.compare_lengths ls args <> 0 then r
    else
      let s = build ls in
      if keeps s then r
      else
        let v = rewritten s in
        if not (scalar v && Dtype.equal (Ops.dtype v) (Ops.dtype r)) then r
        else
          let dims = Shape.shape r in
          Shape.expand
            (Shape.reshape v (List.map (fun _ -> Ops.Int 1) dims))
            dims

let unary f x = fold (function [ x ] -> f x | _ -> assert false) [ x ] (f x)

let binary f x y =
  fold (function [ x; y ] -> f x y | _ -> assert false) [ x; y ] (f x y)

let ternary f x y z =
  fold
    (function [ x; y; z ] -> f x y z | _ -> assert false)
    [ x; y; z ] (f x y z)

(* Elementwise *)

let alu x op rest =
  fold
    (function x :: rest -> Ops.alu x op rest | [] -> assert false)
    (x :: rest) (Ops.alu x op rest)

let cast x dt = unary (fun x -> Ops.cast x dt) x
let bitcast x dt = unary (fun x -> Ops.bitcast x dt) x
let add = binary Ops.add
let mul = binary Ops.mul
let lt = binary Ops.lt
let eq = binary Ops.eq
let maximum = binary Ops.maximum
let bitwise_and = binary Ops.bitwise_and
let bitwise_or = binary Ops.bitwise_or
let bitwise_xor = binary Ops.bitwise_xor
let shl = binary Ops.shl
let shr = binary Ops.shr
let reciprocal = unary Ops.reciprocal
let trunc = unary Ops.trunc
let sqrt = unary Ops.sqrt
let exp2 = unary Ops.exp2
let log2 = unary Ops.log2
let sub = binary Ops.sub
let neg = unary Ops.neg
let div ?rounding x y = binary (Ops.div ?rounding) x y
let mod_ = binary Ops.mod_
let fmod = binary Ops.fmod
let floor = unary Ops.floor
let pow = binary Ops.pow
let gt = binary Ops.gt
let le = binary Ops.le
let ge = binary Ops.ge
let ne = binary Ops.ne
let logical_not = unary Ops.logical_not
let bitwise_not = unary Ops.bitwise_not
let where = ternary Ops.where
let minimum = binary Ops.minimum

module O = struct
  let int = Ops.O.int
  let float = Ops.O.float
  let bool = Ops.O.bool
  let ( + ) = add
  let ( - ) = sub
  let ( * ) = mul
  let ( / ) x y = div x y
  let ( // ) = binary Ops.O.( // )
  let ( % ) = mod_
  let ( ~- ) = neg
  let ( < ) = lt
  let ( > ) = gt
  let ( <= ) = le
  let ( >= ) = ge
  let ( <> ) = ne
  let ( land ) = bitwise_and
  let ( lor ) = bitwise_or
  let ( lxor ) = bitwise_xor
  let lnot = bitwise_not
  let ( lsl ) = shl
  let ( lsr ) = shr
end
