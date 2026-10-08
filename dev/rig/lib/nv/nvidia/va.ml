(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The free ranges, as (start, length), by increasing start. *)
type t = { lock : Mutex.t; mutable free : (int * int) list }

let make ~base n = { lock = Mutex.create (); free = [ (base, n) ] }
let align_up x a = (x + a - 1) land lnot (a - 1)

let alloc r ~align n =
  Mutex.protect r.lock @@ fun () ->
  let rec take before = function
    | [] -> None
    | ((s, l) as hole) :: after ->
        let a = align_up s align in
        if a + n > s + l then take (hole :: before) after
        else
          let left = if a > s then [ (s, a - s) ] else [] in
          let right =
            if a + n < s + l then [ (a + n, s + l - a - n) ] else []
          in
          r.free <- List.rev_append before (left @ right @ after);
          Some a
  in
  take [] r.free

let free r a n =
  Mutex.protect r.lock @@ fun () ->
  let rec put = function
    | (s, l) :: rest when s + l = a -> merge (s, l + n) rest
    | (s, _) :: _ as rest when a < s -> merge (a, n) rest
    | hole :: rest -> hole :: put rest
    | [] -> [ (a, n) ]
  and merge (s, l) = function
    | (s', l') :: rest when s + l = s' -> (s, l + l') :: rest
    | rest -> (s, l) :: rest
  in
  r.free <- put r.free
