(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/late/gater.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

let is_invalid_const u =
  match U.op u, U.arg u with
  | Ops.Const, U.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let invalid_where u =
  match U.op u, U.src u with
  | Ops.Where, [| gate; idx; invalid |] when is_invalid_const invalid ->
      Some (gate, idx)
  | _ -> None

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
  | Ops.Index, [| _; coord |] when U.op coord = Ops.Stack -> (
      match U.src coord with
      | [| y; x |] -> (
          match invalid_where y, invalid_where x with
          | Some (yg, yi), Some (xg, xi) when U.equal yg xg ->
              let coord = U.replace coord ~src:[| yi; xi |] () in
              let src = Array.copy (U.src mop) in
              src.(1) <- coord;
              Some (yg, U.replace mop ~src ())
          | _ -> None)
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

let strip_alt_cast load alt =
  match U.op alt, U.src alt with
  | Ops.Cast, [| inner |] when Dtype.equal (U.dtype inner) (U.dtype load) ->
      inner
  | _ ->
      if Dtype.equal (U.dtype alt) (U.dtype load) then alt
      else U.cast ~src:alt ~dtype:(U.dtype load)

let load_node u =
  match U.op u, U.src u with
  | Ops.Cast, [| load |] ->
      if Option.is_some (U.as_load load) then Some load else None
  | _ -> if Option.is_some (U.as_load u) then Some u else None

let rebuild_load load alt target_dtype =
  match U.as_load load with
  | Some { src; alt = Some _; gate = Some gate } ->
      let load =
        U.replace load ~src:[| src; strip_alt_cast load alt; gate |] ()
      in
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
          Some
            (U.replace node
               ~src:[| gated_src; U.zero_like node; gate |] ())
      | None -> (
      match indexed_invalid_gate src with
      | None -> None
      | Some (gate, idx) ->
          Some
            (U.replace node
               ~src:[| gated_mop src idx; U.zero_like node; gate |] ())))
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
      | Ops.Where, [| gate; value; alt |] -> (
          match fold_gated_load value gate alt (U.dtype alt) with
          | Some folded -> Some folded
          | None -> fold_gated_load alt (U.O.not_ gate) value (U.dtype value))
      | _ -> None)

let pm_move_gates_from_index sink =
  U.graph_rewrite ~name:"move gates from index" move_gates_from_index_rule sink
