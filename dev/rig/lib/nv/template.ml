(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_nv_abi

(* rig_nv_stubs.c's operations. *)
let op_add = 0L
let op_shift = 1L

(* The value [t] reads and its operations, innermost first. *)
let rec ops acc : 'v Packet.term -> 'v * (int64 * int64) list = function
  | Value v -> (v, acc)
  | Add (t, n) -> ops ((op_add, n) :: acc) t
  | Shift (t, n) -> ops ((op_shift, Int64.of_int n) :: acc) t

let add b v = Buffer.add_int64_le b v

let add_ops b ops =
  add b (Int64.of_int (List.length ops));
  List.iter
    (fun (op, k) ->
      add b op;
      add b k)
    ops

let hole b (at, (w : int Packet.word)) =
  let t, n =
    match w with
    | W32 t -> (t, 1)
    | W64 t -> (t, 2)
    | Dword _ -> assert false (* a hole holds a term *)
  in
  let slot, ops = ops [] t in
  add b (Int64.of_int at);
  add b (Int64.of_int slot);
  add b (Int64.of_int n);
  add_ops b ops

let flatten known p =
  let words, holes = Packet.template known p in
  let b = Buffer.create 256 in
  List.iter (hole b) holes;
  (words, Buffer.contents b)

(* Structures *)

type operand = Known of int | Slot of int

let structure (s : operand Structure.t) =
  let value = function Known n -> Int64.of_int n | Slot _ -> 0L in
  let b = Buffer.create 256 in
  let field (h : operand Structure.hole) =
    match ops [] h.value with
    | Known _, _ -> ()
    | Slot slot, ops ->
        add b (Int64.of_int h.at);
        add b (Int64.of_int h.bits);
        add b (Int64.of_int slot);
        add_ops b ops
  in
  List.iter field s.holes;
  (Structure.encode value s, Buffer.contents b)
