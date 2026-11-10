(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('k, 'v) t = ('k * 'v) list Atomic.t

let make () = Atomic.make []
let find c key = List.assoc_opt key (Atomic.get c)

let rec first_by f x = function
  | [] -> None
  | (k, v) :: rest -> if f k x then Some v else first_by f x rest

let find_by f c x = first_by f x (Atomic.get c)

(* The first of [es], at most [count] whose [size]s sum to at most [room],
   [es] itself where it has no more. *)
let rec first count room size es =
  match es with
  | [] -> es
  | ((_, v) as e) :: rest ->
      let room = room - size v in
      if count <= 0 || room < 0 then []
      else
        let r = first (count - 1) room size rest in
        if r == rest then es else e :: r

let no_size _ = 0

let keep ?(count = max_int) ?bytes c key v =
  let most, size =
    match bytes with None -> (max_int, no_size) | Some b -> b
  in
  let room = most - size v in
  let rec add () =
    let es = Atomic.get c in
    let kept = (key, v) :: first (count - 1) room size es in
    if not (Atomic.compare_and_set c es kept) then add ()
  in
  if room >= 0 then add ();
  v
