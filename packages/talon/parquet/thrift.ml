(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

exception Error of int * string

let error pos fmt = Printf.ksprintf (fun msg -> raise (Error (pos, msg))) fmt

(* Readers *)

type t = { b : bigbytes; mutable pos : int; limit : int; mutable depth : int }

let make b ~pos ~limit =
  if not (0 <= pos && pos <= limit && limit <= Bigarray.Array1.dim b) then
    invalid_arg
      (Printf.sprintf "Thrift.make: range %d-%d of %d bytes" pos limit
         (Bigarray.Array1.dim b));
  { b; pos; limit; depth = 0 }

let pos r = r.pos
let left r = r.limit - r.pos

let advance r n =
  if n > left r then error r.pos "%d bytes past the end of the data" (n - left r);
  r.pos <- r.pos + n

let byte r =
  if r.pos >= r.limit then error r.pos "unexpected end of the data";
  let c = Bigarray.Array1.unsafe_get r.b r.pos in
  r.pos <- r.pos + 1;
  c

(* [varint r max] reads an unsigned varint of at most [max] bytes into the 63
   bits of an [int]. *)
let varint r max =
  let start = r.pos in
  let rec loop acc shift =
    let c = byte r in
    let acc = acc lor ((c land 0x7f) lsl shift) in
    if c < 0x80 then acc
    else if shift + 7 >= 7 * max then
      error start "a varint longer than %d bytes" max
    else if shift + 7 >= 63 then error start "an integer outside OCaml's int"
    else loop acc (shift + 7)
  in
  loop 0 0

let zigzag u = (u lsr 1) lxor -(u land 1)

let nest r f =
  if r.depth = 64 then error r.pos "values nested deeper than 64 levels";
  r.depth <- r.depth + 1;
  let v = f () in
  r.depth <- r.depth - 1;
  v

(* Values *)

type ty = int

(* Wire types, and [bool_elt] for a boolean element of a list, a set or a map,
   which is a byte where a field's boolean is its wire type. *)
let t_true = 1
let t_false = 2
let t_i8 = 3
let t_i16 = 4
let t_i32 = 5
let t_i64 = 6
let t_double = 7
let t_binary = 8
let t_list = 9
let t_set = 10
let t_map = 11
let t_struct = 12
let t_uuid = 13
let bool_elt = 14
let element ty = if ty = t_true || ty = t_false then bool_elt else ty

let expect r ty t name =
  if ty <> t then error r.pos "expected %s, found wire type %d" name ty

let fields r f =
  nest r @@ fun () ->
  let rec loop last =
    let h = byte r in
    if h <> 0 then begin
      let delta = h lsr 4 in
      let id = if delta = 0 then zigzag (varint r 3) else last + delta in
      f id (h land 0xf);
      loop id
    end
  in
  loop 0

let structure r ty f =
  expect r ty t_struct "a structure";
  fields r f

let bool r ty =
  if ty = t_true then true
  else if ty = t_false then false
  else if ty = bool_elt then byte r = 1
  else error r.pos "expected a boolean, found wire type %d" ty

let i8 r ty =
  expect r ty t_i8 "an i8";
  let c = byte r in
  if c >= 128 then c - 256 else c

let i32 r ty =
  expect r ty t_i32 "an i32";
  let pos = r.pos in
  let v = zigzag (varint r 5) in
  if v < -0x8000_0000 || v > 0x7FFF_FFFF then
    error pos "an i32 outside its range, %d" v;
  v

let i64 r ty =
  expect r ty t_i64 "an i64";
  zigzag (varint r 10)

let length r =
  let pos = r.pos in
  let n = varint r 5 in
  if n > left r then error pos "a length of %d bytes past the end of the data" n;
  n

let binary r ty =
  expect r ty t_binary "a binary";
  let n = length r in
  let s = String.init n (fun i -> Char.unsafe_chr r.b.{r.pos + i}) in
  r.pos <- r.pos + n;
  s

let list r ty elt =
  if ty <> t_list && ty <> t_set then
    error r.pos "expected a list, found wire type %d" ty;
  let pos = r.pos in
  let h = byte r in
  let n = if h lsr 4 = 15 then varint r 5 else h lsr 4 in
  if n > left r then
    error pos "a list of %d elements past the end of the data" n;
  let ty = element (h land 0xf) in
  nest r @@ fun () -> List.init n (fun _ -> elt r ty)

let skip_varint r max =
  let start = r.pos in
  let rec loop i =
    if byte r >= 0x80 then
      if i = max then error start "a varint longer than %d bytes" max
      else loop (i + 1)
  in
  loop 1

let rec skip r ty =
  if ty = t_true || ty = t_false then ()
  else if ty = t_i8 || ty = bool_elt then advance r 1
  else if ty = t_i16 then skip_varint r 3
  else if ty = t_i32 then skip_varint r 5
  else if ty = t_i64 then skip_varint r 10
  else if ty = t_double then advance r 8
  else if ty = t_binary then advance r (length r)
  else if ty = t_uuid then advance r 16
  else if ty = t_list || ty = t_set then ignore (list r ty skip)
  else if ty = t_map then skip_map r
  else if ty = t_struct then fields r (fun _ ty -> skip r ty)
  else error r.pos "an unknown wire type %d" ty

and skip_map r =
  let pos = r.pos in
  let n = varint r 5 in
  if n > 0 then begin
    if 2 * n > left r then
      error pos "a map of %d entries past the end of the data" n;
    let h = byte r in
    let kt = element (h lsr 4) and vt = element (h land 0xf) in
    nest r @@ fun () ->
    for _ = 1 to n do
      skip r kt;
      skip r vt
    done
  end
