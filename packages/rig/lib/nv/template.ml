(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_nv_abi

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* rig_nv_stubs.h's bounds, which C refuses past with a fixed message: a
   template's words and holes, a hole's operations, a structure's bytes and
   holes, and the values of a launch. *)
let template_words = 16
let template_holes = 6
let hole_ops = 3
let structure_bytes = 1024
let structure_fields = 16
let values = 9

(* rig_nv_stubs.c's operations. *)
let op_add = 0L
let op_shift = 1L

(* The value [t] reads and its operations, innermost first. *)
let rec ops acc : 'v Packet.term -> 'v * (int64 * int64) list = function
  | Value v -> (v, acc)
  | Add (t, n) -> ops ((op_add, n) :: acc) t
  | Shift (t, n) -> ops ((op_shift, Int64.of_int n) :: acc) t

let add b v = Buffer.add_int64_le b v

(* Refuses, as [what], more than [hole_ops] operations or a shift outside 0 to
   63. *)
let add_ops what b ops =
  if List.length ops > hole_ops then
    invalid_argf "%s hole takes more than %d operations" what hole_ops;
  add b (Int64.of_int (List.length ops));
  List.iter
    (fun (op, k) ->
      if op = op_shift && (k < 0L || k > 63L) then
        invalid_argf "%s hole's shift is outside 0 to 63" what;
      add b op;
      add b k)
    ops

let template = "Rig_nv.make: a template's"

let hole b nwords (at, (w : int Packet.word)) =
  let t, n =
    match w with
    | W32 t -> (t, 1)
    | W64 t -> (t, 2)
    | Dword _ -> assert false (* a hole holds a term *)
  in
  let slot, ops = ops [] t in
  if slot < 0 || slot > 2 then
    invalid_argf "%s hole reads a value outside 0 to 2" template;
  if at < 0 || at + n > nwords then
    invalid_argf "%s hole lies outside its words" template;
  add b (Int64.of_int at);
  add b (Int64.of_int slot);
  add b (Int64.of_int n);
  add_ops template b ops

let flatten known p =
  let words, holes = Packet.template known p in
  if String.length words > 4 * template_words then
    invalid_argf "Rig_nv.make: a template exceeds %d words" template_words;
  if List.length holes > template_holes then
    invalid_argf "Rig_nv.make: a template has more than %d holes"
      template_holes;
  let b = Buffer.create 256 in
  List.iter (hole b (String.length words / 4)) holes;
  (words, Buffer.contents b)

(* Structures *)

type operand = Known of int | Slot of int

let structure_hole = "Rig_nv.entry: a structure's"

(* The bytes of the narrowest word of 1, 2, 4 or 8 bytes that holds [bits]. *)
let width bits =
  if bits <= 8 then 1 else if bits <= 16 then 2 else if bits <= 32 then 4 else 8

let structure (s : operand Structure.t) =
  let value = function Known n -> Int64.of_int n | Slot _ -> 0L in
  let bytes = Structure.encode value s in
  if String.length bytes > structure_bytes then
    invalid_argf "Rig_nv.entry: a structure exceeds %d bytes" structure_bytes;
  let b = Buffer.create 256 and n = ref 0 in
  let field (h : operand Structure.hole) =
    match ops [] h.value with
    | Known _, _ -> ()
    | Slot slot, ops ->
        incr n;
        if !n > structure_fields then
          invalid_argf "Rig_nv.entry: a structure has more than %d holes"
            structure_fields;
        if slot < 0 || slot >= values then
          invalid_argf "%s hole reads no launch value" structure_hole;
        if h.bits < 1 || h.bits > 64 then
          invalid_argf "%s hole's bits are not 1 to 64" structure_hole;
        if h.at < 0 || h.at + width h.bits > String.length bytes then
          invalid_argf "%s hole lies outside its bytes" structure_hole;
        add b (Int64.of_int h.at);
        add b (Int64.of_int h.bits);
        add b (Int64.of_int slot);
        add_ops structure_hole b ops
  in
  List.iter field s.holes;
  (bytes, Buffer.contents b)
