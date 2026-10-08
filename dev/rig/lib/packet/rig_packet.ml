(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type 'v term =
  | Value of 'v
  | Add of 'v term * int64
  | Shift of 'v term * int
  | Or of 'v term * int64

type 'v word = Dword of int | W32 of 'v term | W64 of 'v term
type 'v t = 'v word list

let words = function Dword _ | W32 _ -> 1 | W64 _ -> 2
let size p = List.fold_left (fun n w -> n + words w) 0 p

let check_shift fn n =
  if n < 0 || n > 63 then invalid_argf "%s: shift by %d, expected 0 to 63" fn n

(* Evaluation *)

let rec eval fn value = function
  | Value v -> value v
  | Add (t, n) -> Int64.add (eval fn value t) n
  | Or (t, n) -> Int64.logor (eval fn value t) n
  | Shift (t, n) ->
      let v = eval fn value t in
      check_shift fn n;
      Int64.shift_right_logical v n

(* Writes [w] at byte [at] of [b], little-endian, and is the byte after it. *)
let put fn value b at = function
  | Dword n ->
      Bytes.set_int32_le b at (Int32.of_int n);
      at + 4
  | W32 t ->
      Bytes.set_int32_le b at (Int64.to_int32 (eval fn value t));
      at + 4
  | W64 t ->
      Bytes.set_int64_le b at (eval fn value t);
      at + 8

let encode value p =
  let b = Bytes.create (4 * size p) in
  ignore (List.fold_left (put "Rig_packet.encode" value b) 0 p);
  Bytes.unsafe_to_string b

(* Templates *)

(* A value [known] leaves for later: the word that holds it is a hole. *)
exception Hole

let template known p =
  let value v = match known v with Some n -> n | None -> raise_notrace Hole in
  let b = Bytes.make (4 * size p) '\000' in
  let word (at, holes) w =
    match put "Rig_packet.template" value b at w with
    | next -> (next, holes)
    | exception Hole -> (at + (4 * words w), (at / 4, w) :: holes)
  in
  let _, holes = List.fold_left word (0, []) p in
  (Bytes.unsafe_to_string b, List.rev holes)

(* C templates: rig_packet.h's bounds and operations. *)

let max_words = 16
let max_holes = 6
let max_ops = 3
let args = 3
let op_add = 0
let op_shift = 1
let op_or = 2

(* A hole as caml_rig_packet_load reads it: its word index, argument and width,
   then its operations and their constants in the order they apply. *)
type hole = int * int * bool * int array * int64 array

external set : nativeint -> string -> hole array -> unit
  = "caml_rig_packet_load"
[@@noalloc]

let hole at ~wide t =
  let rec flatten ops : int term -> _ = function
    | Value i ->
        if i < 0 || i >= args then
          invalid_argf "Rig_packet.load: argument %d, expected 0 to %d" i
            (args - 1);
        (i, ops)
    | Add (t, k) -> flatten ((op_add, k) :: ops) t
    | Shift (t, n) ->
        check_shift "Rig_packet.load" n;
        flatten ((op_shift, Int64.of_int n) :: ops) t
    | Or (t, k) -> flatten ((op_or, k) :: ops) t
  in
  let arg, ops = flatten [] t in
  if List.length ops > max_ops then
    invalid_argf "Rig_packet.load: a term of %d operations, expected at most %d"
      (List.length ops) max_ops;
  let ops, ks = List.split ops in
  ((at, arg, wide, Array.of_list ops, Array.of_list ks) : hole)

let load at p =
  let n = size p in
  if n > max_words then
    invalid_argf "Rig_packet.load: %d words, expected at most %d" n max_words;
  let holes, _ =
    List.fold_left
      (fun (holes, i) w ->
        match w with
        | Dword _ -> (holes, i + 1)
        | W32 t -> (hole i ~wide:false t :: holes, i + 1)
        | W64 t -> (hole i ~wide:true t :: holes, i + 2))
      ([], 0) p
  in
  let holes = Array.of_list (List.rev holes) in
  if Array.length holes > max_holes then
    invalid_argf "Rig_packet.load: %d holes, expected at most %d"
      (Array.length holes) max_holes;
  set at (fst (template (fun _ -> None) p)) holes
