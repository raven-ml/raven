(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type axis = { extent : int; a : bool; b : bool; step : int }
type box = { lo : int array; len : int array }
type t = { boxes : box array; a_bytes : int; b_bytes : int }

let lines varies axes x =
  let n = ref 1 in
  Array.iteri (fun i ax -> if varies ax then n := !n * x.len.(i)) axes;
  !n

let bytes axes ~a ~b x =
  let cast line varies = if line = 0 then 0 else line * lines varies axes x in
  (cast a (fun ax -> ax.a), cast b (fun ax -> ax.b))

(* A range of an operand's innermost axis is a multiple of the tiles'
   columns. *)
let tile = 64

(* The largest [c] in [1, hi] that [fits], for [fits] holding below any count it
   holds of; [0] where [fits 1] fails. *)
let largest fits hi =
  let rec go lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi + 1) / 2 in
      if fits mid then go mid hi else go lo (mid - 1)
  in
  if fits 1 then go 1 hi else 0

(* The boxes of [x] whose axes [ix] are cut at [ix.(j)] in ranges of [c] from
   [x]'s start: one index of each axis before it, [x]'s range of each axis after
   it. *)
let enumerate ix x j c =
  let out = ref [] in
  let lo = Array.copy x.lo and len = Array.copy x.len in
  let rec go q =
    let i = ix.(q) in
    let hi = x.lo.(i) + x.len.(i) in
    if q < j then
      for v = x.lo.(i) to hi - 1 do
        lo.(i) <- v;
        len.(i) <- 1;
        go (q + 1)
      done
    else
      let s = ref x.lo.(i) in
      while !s < hi do
        lo.(i) <- !s;
        len.(i) <- Int.min c (hi - !s);
        out := { lo = Array.copy lo; len = Array.copy len } :: !out;
        s := !s + c
      done
  in
  go 0;
  List.rev !out

(* [x]'s boxes along [ix], the coarsest of which each [fits]; [None] where one
   index of each does not. A range of [x]'s along an axis of [ix] of more than
   one index starts at a multiple of its step. *)
let split ?(round = true) axes ix x ~fits =
  let n = Array.length ix in
  if n = 0 then if fits x then Some [ x ] else None
  else
    let shaped j c =
      let len = Array.copy x.len in
      Array.iteri
        (fun q i ->
          len.(i) <- (if q < j then 1 else if q = j then c else x.len.(i)))
        ix;
      { x with len }
    in
    let rec first j =
      if j >= n then None
      else if fits (shaped j 1) then Some j
      else first (j + 1)
    in
    Option.map
      (fun j ->
        let i = ix.(j) in
        let ax = axes.(i) and whole = x.len.(i) in
        let c = largest (fun c -> fits (shaped j c)) whole in
        let c =
          if round && j = n - 1 && c >= tile && c < whole then c - (c mod tile)
          else c
        in
        let c =
          if c > 1 && ax.step > 1 && c < whole then
            if c >= ax.step then c - (c mod ax.step) else 1
          else c
        in
        enumerate ix x j c)
      (first 0)

(* [x]'s boxes of one index along each of [ix]. *)
let lines_of ix x =
  let n = Array.length ix in
  if n = 0 then [ x ] else enumerate ix x (n - 1) 1

let cuts axes ix x ~fits =
  match split axes ix x ~fits with Some l -> l | None -> lines_of ix x

let cut ~w ~batch ~rows axes ~a ~b =
  let r = Array.length axes in
  let whole =
    { lo = Array.make r 0; len = Array.map (fun ax -> ax.extent) axes }
  in
  let range lo n = Array.init n (fun i -> lo + i) in
  let bi = range 0 batch in
  let ai = range batch rows in
  let ci = range (batch + rows) (r - batch - rows) in
  let a_of x = fst (bytes axes ~a ~b x) and b_of x = snd (bytes axes ~a ~b x) in
  let total x = a_of x + b_of x in
  let within budget f x = f x <= budget in
  (* One batch element: the smaller cast whole where it takes half [w] at most,
     else both cut in halves, the larger outside. *)
  let element s =
    let sa = a_of s and sb = b_of s in
    if Int.min sa sb <= w / 2 then
      if sa <= sb then cuts axes ci s ~fits:(within (w - sa) b_of)
      else cuts axes ai s ~fits:(within (w - sb) a_of)
    else
      let (oi, outer), (ii, inner) =
        if sa >= sb then ((ai, a_of), (ci, b_of)) else ((ci, b_of), (ai, a_of))
      in
      List.concat_map
        (fun o -> cuts axes ii o ~fits:(within (w / 2) inner))
        (cuts axes oi s ~fits:(within (w / 2) outer))
  in
  let boxes =
    if total whole <= w then [ whole ]
    else
      match split ~round:false axes bi whole ~fits:(within w total) with
      | Some bs -> bs
      | None -> List.concat_map element (lines_of bi whole)
  in
  let most f = List.fold_left (fun m x -> Int.max m (f x)) 0 boxes in
  { boxes = Array.of_list boxes; a_bytes = most a_of; b_bytes = most b_of }

let refine ~most axes x =
  let fits x = Array.fold_left ( * ) 1 x.len <= most in
  cuts axes (Array.init (Array.length axes) Fun.id) x ~fits
