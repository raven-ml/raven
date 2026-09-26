(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/late/gater.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

(* Zero alternative for a gated load: a zero constant stacked to the load's
   own lane count. Vector values must be [Stack]s of scalars so C-style
   renderers emit a vector constructor. *)
let vzero_like node =
  U.broadcast (U.const (Const.zero (U.dtype node))) (U.max_numel node)

let is_invalid_const u =
  match U.op u, U.arg u with
  | Ops.Const, U.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let invalid_where u =
  match U.op u, U.src u with
  | Ops.Where, [| gate; idx; invalid |] when is_invalid_const invalid ->
      Some (gate, idx)
  | _ -> None

(* Image load/store indexes are always float-typed; ungating the coordinates
   must keep the index float so the image-float rules and the renderer treat
   it as an image access. *)
let indexed_two_invalid_gate mop =
  match U.op mop, U.src mop with
  | Ops.Index, [| _; y; x |] -> (
      match invalid_where y, invalid_where x with
      | Some (yg, yi), Some (xg, xi) when U.equal yg xg ->
          let src = Array.copy (U.src mop) in
          src.(1) <- yi;
          src.(2) <- xi;
          Some (yg, U.replace mop ~src ())
      | _ -> None)
  | _ -> None

let indexed_invalid_gate mop =
  match U.op mop, U.src mop with
  | (Ops.Index | Ops.Shrink), src when Array.length src >= 2 -> (
      match invalid_where src.(1) with
      | Some (gate, idx) -> Some (gate, idx)
      | None -> None)
  | _ -> None

let gated_mop mop idx =
  let src = Array.copy (U.src mop) in
  src.(1) <- idx;
  U.replace mop ~src ()

(* Whether constant [c] comes back bit for bit through [dtype], converted as
   a cast converts it. An integer outside [dtype]'s range would wrap, which
   the constant conversion does not model, so it does not come back. *)
let survives c dtype =
  match Const.of_view dtype (Const.view c) with
  | exception (Z.Overflow | Invalid_argument _) -> false
  | there ->
      let in_range =
        match (Const.view there, Dtype.min dtype, Dtype.max dtype) with
        | Const.Int n, `Int lo, `Int hi -> Z.leq lo n && Z.leq n hi
        | _ -> true
      in
      in_range
      && Const.equal (Const.of_view (Const.dtype c) (Const.view there)) c

(* The alternative the load falls back to when the gate is false, in the
   load's dtype. An Invalid alternative carries no value and is bool-typed, so
   it cannot be cast into the load's dtype; it becomes a zero of the load's own
   width. A value of another dtype must come back unchanged through the load's
   dtype: it is a cast from the load's dtype, or a constant that comes back
   bit for bit. Converting any other value can change it (a bfloat16 load
   rounds a float64 alternative, a conversion quiets a signalling NaN), and
   then the select stays. The tinygrad counterpart casts every
   alternative. *)
let alt_in_load_dtype load alt =
  let dtype = U.dtype load in
  if is_invalid_const alt then Some (vzero_like load)
  else if Dtype.equal (U.dtype alt) dtype then Some alt
  else
    match U.op alt, U.src alt, U.as_const alt with
    | Ops.Cast, [| inner |], _ when Dtype.equal (U.dtype inner) dtype ->
        Some inner
    | _, _, Some c when survives c dtype -> Some (U.cast ~src:alt ~dtype)
    | _ -> None

let load_node u =
  match U.op u, U.src u with
  | Ops.Cast, [| load |] ->
      if Option.is_some (U.as_load load) then Some load else None
  | _ -> if Option.is_some (U.as_load u) then Some u else None

let rebuild_load load alt target_dtype =
  match U.as_load load, alt_in_load_dtype load alt with
  | Some { src; alt = Some _; gate = Some gate }, Some alt ->
      let load = U.replace load ~src:[| src; alt; gate |] () in
      Some (U.cast ~src:load ~dtype:target_dtype)
  | _ -> None

let fold_gated_load value gate alt target_dtype =
  match load_node value with
  | Some load -> (
      match U.as_load load with
      | Some { gate = Some load_gate; _ }
        when Dtype.equal (U.dtype load_gate) Dtype.bool && U.equal load_gate gate
        ->
          rebuild_load load alt target_dtype
      | _ -> None)
  | None -> None

let move_gates_from_index_rule node =
  match U.as_load node, U.as_store node with
  | Some { src; alt = None; gate = None }, _ -> (
      match indexed_two_invalid_gate src with
      | Some (gate, gated_src) ->
          Some (U.replace node ~src:[| gated_src; vzero_like node; gate |] ())
      | None -> (
      match indexed_invalid_gate src with
      | None -> None
      | Some (gate, idx) ->
          let mop = gated_mop src idx in
          Some (U.replace node ~src:[| mop; vzero_like node; gate |] ())))
  | Some _, _ -> None
  | None, Some { dst; value; gate = None } -> (
      match indexed_two_invalid_gate dst with
      | Some (gate, gated_dst) ->
          Some (U.replace node ~src:[| gated_dst; value; gate |] ())
      | None -> (
      match indexed_invalid_gate dst with
      | None -> None
      | Some (gate, idx) ->
          Some
            (U.replace node ~src:[| gated_mop dst idx; value; gate |] ())))
  | None, Some _ -> None
  | None, None -> (
      match U.op node, U.src node with
      (* The folded load takes the dtype of the WHERE it replaces, not of
         the branch it absorbs: an Invalid branch is bool. *)
      | Ops.Where, [| gate; value; alt |] -> (
          match fold_gated_load value gate alt (U.dtype node) with
          | Some folded -> Some folded
          | None -> fold_gated_load alt (U.Promoting.not_ gate) value (U.dtype node))
      | _ -> None)

let pm_move_gates_from_index sink =
  U.graph_rewrite ~name:"move gates from index" move_gates_from_index_rule sink
