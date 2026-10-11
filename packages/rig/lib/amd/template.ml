(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_amd_abi

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type form = Packet | Launch

(* Each form's name and rig_amd_stubs.h's bounds, which C refuses past with a
   fixed message: its words, its holes and the arguments a hole reads. *)
let name = function
  | Packet -> "Rig_amd.make: a packet template"
  | Launch -> "Rig_amd.entry: a launch"

let max_words = function Packet -> 16 | Launch -> 64
let max_holes = function Packet -> 4 | Launch -> 12
let max_args = function Packet -> 3 | Launch -> 11

(* A hole's most operations. *)
let hole_ops = 3

(* rig_amd_stubs.h's operations. *)
let op_add = 0L
let op_shift = 1L
let op_or = 2L

(* The value [t] reads and its operations, innermost first. *)
let rec ops acc : 'v Packet.term -> 'v * (int64 * int64) list = function
  | Value v -> (v, acc)
  | Add (t, n) -> ops ((op_add, n) :: acc) t
  | Or (t, n) -> ops ((op_or, n) :: acc) t
  | Shift (t, n) -> ops ((op_shift, Int64.of_int n) :: acc) t

let hole form nwords arg b (at, (w : _ Packet.word)) =
  let t, n =
    match w with
    | W32 t -> (t, 1)
    | W64 t -> (t, 2)
    | Dword _ -> assert false (* a hole holds a term *)
  in
  let v, ops = ops [] t in
  let arg = arg v in
  let refuse why = invalid_argf "%s's hole %s" (name form) why in
  if arg < 0 || arg >= max_args form then refuse "reads an unknown argument";
  if at < 0 || at + n > nwords then refuse "lies outside its words";
  if List.length ops > hole_ops then refuse "takes too many operations";
  if List.exists (fun (op, k) -> op = op_shift && (k < 0L || k > 63L)) ops
  then refuse "shifts outside 0 to 63";
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

let flatten form known arg p =
  let words, holes = Packet.template known p in
  if String.length words > 4 * max_words form then
    invalid_argf "%s exceeds %d words" (name form) (max_words form);
  if List.length holes > max_holes form then
    invalid_argf "%s has more than %d holes" (name form) (max_holes form);
  let b = Buffer.create 256 in
  List.iter (hole form (String.length words / 4) arg b) holes;
  (words, Buffer.contents b)
