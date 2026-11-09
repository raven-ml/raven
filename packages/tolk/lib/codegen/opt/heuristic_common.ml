(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

module K = Postrange.Scheduler

let debug () = Setting.value Setting.debug

(* A split the heuristic has checked applies. *)
let split ?(top = false) k axis amount target =
  match K.apply_opt k (Opt.Split { axis; amount; target; top }) with
  | Ok axes -> axes
  | Error msg -> invalid_arg msg

(* A split that may not apply: whether it did. *)
let try_split ?(top = false) k axis amount target =
  Result.is_ok (K.apply_opt k (Opt.Split { axis; amount; target; top }))

let divisible s n =
  match known_size s with
  | Some s -> Bigint.(equal (rem s (of_int n)) zero)
  | None -> false

let shape_at k i = List.nth (K.full_shape k) i
let holds = function Sint.Known b -> b | Sint.Cond u -> to_bool u
let size_at k i = Option.get (known_size (shape_at k i))

let idx b = get_idx (nth b 1)
let indexes r b = Nodes.mem r (backward_slice ~calls:Skip (idx b))
let reads r b = Nodes.mem r (backward_slice_with_self ~calls:Skip (idx b))

(* The addresses [k] stores to. *)
let stored_addresses k =
  List.filter_map
    (fun u -> if op u = Op.Store then Some (nth u 0) else None)
    (toposort ~calls:Skip (K.ast k))

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* The run of [o] along the range [r]: [d] if [o] reads [r] only as [r / d], 1
   otherwise. *)
let run o r =
  let readers =
    List.filter
      (fun u -> List.memq r (src u))
      (Nodes.to_list (backward_slice_with_self ~calls:Skip o))
  in
  let divisor u =
    if op u = Op.Div && is_literal (nth u 1) then
      known_size (Sym (nth u 1))
    else None
  in
  match List.map divisor readers with
  | Some d :: rest when List.for_all (Option.equal Bigint.equal (Some d)) rest
    ->
      d
  | _ -> Bigint.one

(* The most values an upcast of one axis takes (apply_opt's bound). *)
let upcast_max = 16

(* The stride of the range [r] in the index [i]: 1 for a term [r], [c] for a
   term [r * c]. *)
let stride r i =
  List.find_map
    (fun t ->
      if t == r then Some Bigint.one
      else if op t = Op.Mul && nth t 0 == r && is_literal (nth t 1) then
        match value (nth t 1) with `Int c -> Some c | _ -> None
      else None)
    (split_uop i Op.Add)

(* [n]'s greatest common divisor with the positive [a]. *)
let gcd_with a n = gcd a (Bigint.to_int (Bigint.rem n (Bigint.of_int a)))
