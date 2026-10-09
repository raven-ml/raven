(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_amd_abi

(* rig_amd_stubs.h's RIG_AMD_TEMPLATE_WORDS, RIG_AMD_TEMPLATE_HOLES and
   RIG_AMD_HOLE_OPS, the words of a hole's record, and its operations. *)
let max_words = 16
let max_holes = 4
let max_ops = 3
let fields = 4 + (2 * max_ops)
let op_add = 0L
let op_shift = 1L
let op_or = 2L

let refuse fmt =
  Printf.ksprintf
    (fun s -> invalid_arg ("Rig_amd.make: a packet template " ^ s))
    fmt

(* The argument [t] reads and its operations, innermost first. *)
let rec ops acc : int Packet.term -> int * (int64 * int64) list = function
  | Value arg -> (arg, acc)
  | Add (t, n) -> ops ((op_add, n) :: acc) t
  | Or (t, n) -> ops ((op_or, n) :: acc) t
  | Shift (t, n) ->
      if n < 0 || n > 63 then refuse "hole's shift is outside 0 to 63";
      ops ((op_shift, Int64.of_int n) :: acc) t

let flatten p =
  let words, holes = Packet.template (fun _ -> None) p in
  if String.length words > 4 * max_words then
    refuse "exceeds %d words" max_words;
  if List.length holes > max_holes then
    refuse "has more than %d holes" max_holes;
  let b = Bytes.make (8 * fields * List.length holes) '\000' in
  let hole i (at, (w : int Packet.word)) =
    let set j v = Bytes.set_int64_le b (8 * ((fields * i) + j)) v in
    let t, n =
      match w with
      | W32 t -> (t, 1)
      | W64 t -> (t, 2)
      | Dword _ -> assert false (* a hole holds a term *)
    in
    let arg, ops = ops [] t in
    if List.length ops > max_ops then refuse "hole takes too many operations";
    if arg < 0 || arg > 2 then refuse "hole reads an argument outside 0 to 2";
    set 0 (Int64.of_int at);
    set 1 (Int64.of_int arg);
    set 2 (Int64.of_int n);
    set 3 (Int64.of_int (List.length ops));
    List.iteri
      (fun j (op, k) ->
        set (4 + (2 * j)) op;
        set (5 + (2 * j)) k)
      ops
  in
  List.iteri hole holes;
  (words, Bytes.unsafe_to_string b)
