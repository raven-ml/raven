(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'v term =
  | Value of 'v
  | Add of 'v term * int64
  | Shift of 'v term * int
  | Or of 'v term * int64

type 'v word = Dword of int | W32 of 'v term | W64 of 'v term
type 'v t = 'v word list
type comparison = Equal | Greater_equal
type scope = Agent | System

let words = function Dword _ | W32 _ -> 1 | W64 _ -> 2
let size p = List.fold_left (fun n w -> n + words w) 0 p

(* Evaluation *)

let rec eval fn value = function
  | Value v -> value v
  | Add (t, n) -> Int64.add (eval fn value t) n
  | Or (t, n) -> Int64.logor (eval fn value t) n
  | Shift (t, n) ->
      let v = eval fn value t in
      if n < 0 || n > 63 then
        invalid_arg (Printf.sprintf "%s: shift by %d, expected 0 to 63" fn n);
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
  ignore (List.fold_left (put "Packet.encode" value b) 0 p);
  Bytes.unsafe_to_string b

(* Templates *)

(* A value [known] leaves for later: the word that holds it is a hole. *)
exception Hole

let template known p =
  let value v = match known v with Some n -> n | None -> raise_notrace Hole in
  let b = Bytes.make (4 * size p) '\000' in
  let word (at, holes) w =
    match put "Packet.template" value b at w with
    | next -> (next, holes)
    | exception Hole -> (at + (4 * words w), (at / 4, w) :: holes)
  in
  let _, holes = List.fold_left word (0, []) p in
  (Bytes.unsafe_to_string b, List.rev holes)
