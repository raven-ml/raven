(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_vg

(* Device-space polylines, the form the stroker and the coverage pass work
   on. *)

type t = { xs : float array; ys : float array; closed : bool }

(* Growable point buffer *)

type buf = {
  mutable bx : float array;
  mutable by : float array;
  mutable n : int;
}

let buf_create () = { bx = Array.make 16 0.; by = Array.make 16 0.; n = 0 }

let push b x y =
  if b.n = Array.length b.bx then begin
    let grow a = Array.append a (Array.make (Array.length a) 0.) in
    b.bx <- grow b.bx;
    b.by <- grow b.by
  end;
  Array.unsafe_set b.bx b.n x;
  Array.unsafe_set b.by b.n y;
  b.n <- b.n + 1

let last_x b = b.bx.(b.n - 1)
let last_y b = b.by.(b.n - 1)

let flush b ~closed acc =
  let out =
    if b.n = 0 then acc
    else { xs = Array.sub b.bx 0 b.n; ys = Array.sub b.by 0 b.n; closed } :: acc
  in
  b.n <- 0;
  out

(* [of_path m p] is [p] under [m] as polylines in drawing order. *)
let of_path m p =
  let b = buf_create () in
  let acc =
    Path.flatten m
      ~move:(fun acc x y ->
        let acc = flush b ~closed:false acc in
        push b x y;
        acc)
      ~line:(fun acc x y ->
        push b x y;
        acc)
      ~close:(fun acc -> flush b ~closed:true acc)
      [] p
  in
  List.rev (flush b ~closed:false acc)

(* [bounds polys] is the box enclosing [polys], or [None] without points. *)
let bounds polys =
  let x0 = ref infinity and y0 = ref infinity in
  let x1 = ref neg_infinity and y1 = ref neg_infinity in
  List.iter
    (fun { xs; ys; _ } ->
      for i = 0 to Array.length xs - 1 do
        let x = Array.unsafe_get xs i and y = Array.unsafe_get ys i in
        if x < !x0 then x0 := x;
        if x > !x1 then x1 := x;
        if y < !y0 then y0 := y;
        if y > !y1 then y1 := y
      done)
    polys;
  if !x0 > !x1 then None else Some (Box.v !x0 !y0 !x1 !y1)
