(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_amd_abi

(* rig_amd_stubs.h's operations. *)
let op_add = 0L
let op_shift = 1L
let op_or = 2L

(* The argument [t] reads and its operations, innermost first. *)
let rec ops acc : int Packet.term -> int * (int64 * int64) list = function
  | Value arg -> (arg, acc)
  | Add (t, n) -> ops ((op_add, n) :: acc) t
  | Or (t, n) -> ops ((op_or, n) :: acc) t
  | Shift (t, n) -> ops ((op_shift, Int64.of_int n) :: acc) t

let hole b (at, (w : int Packet.word)) =
  let t, n =
    match w with
    | W32 t -> (t, 1)
    | W64 t -> (t, 2)
    | Dword _ -> assert false (* a hole holds a term *)
  in
  let arg, ops = ops [] t in
  let add v = Buffer.add_int64_le b v in
  add (Int64.of_int at);
  add (Int64.of_int arg);
  add (Int64.of_int n);
  add (Int64.of_int (List.length ops));
  List.iter
    (fun (op, k) ->
      add op;
      add k)
    ops

let flatten p =
  let words, holes = Packet.template (fun _ -> None) p in
  let b = Buffer.create 256 in
  List.iter (hole b) holes;
  (words, Buffer.contents b)
