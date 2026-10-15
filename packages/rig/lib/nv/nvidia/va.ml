(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The free ranges, as (start, length), by increasing start. Each change
   swaps in a list made from the one it read, and starts again if another
   call swapped first: no lock, so a signal handler, finaliser or Gc.Memprof
   callback that runs inside a change can make one of its own. *)
type t = (int * int) list Atomic.t

let make ~base n = Atomic.make [ (base, n) ]
let align_up x a = (x + a - 1) land lnot (a - 1)

(* [change r f] swaps [r]'s list [l] for the list of [f l], and is its
   answer. *)
let rec change r f =
  let l = Atomic.get r in
  let l', x = f l in
  if l' == l || Atomic.compare_and_set r l l' then x else change r f

let alloc r ~align n =
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
          Some (List.rev_append before (left @ right @ after), a)
  in
  change r @@ fun free ->
  match take [] free with None -> (free, None) | Some (l, a) -> (l, Some a)

let free r a n =
  let rec put = function
    | (s, l) :: rest when s + l = a -> merge (s, l + n) rest
    | (s, _) :: _ as rest when a < s -> merge (a, n) rest
    | hole :: rest -> hole :: put rest
    | [] -> [ (a, n) ]
  and merge (s, l) = function
    | (s', l') :: rest when s + l = s' -> (s, l + l') :: rest
    | rest -> (s, l) :: rest
  in
  change r (fun free -> (put free, ()))
