(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  crs : Crs.t;
  points : Nx.host Nx.float64_t;
  rings : int array;
  features : int array;
}

let strf = Printf.sprintf

let offsets name a ~last =
  let n = Array.length a in
  if n = 0 || a.(0) <> 0 || a.(n - 1) <> last then
    invalid_arg (strf "Jord.Polygons.v: %s must run from 0 to %d" name last);
  for k = 1 to n - 1 do
    if a.(k) < a.(k - 1) then
      invalid_arg (strf "Jord.Polygons.v: %s decreases at %d" name k)
  done

let v crs points ~rings ~features =
  (match Nx.shape points with
  | [| _; 2 |] -> ()
  | s ->
      invalid_arg
        (strf "Jord.Polygons.v: points of shape [%s] are not [n; 2]"
           (String.concat "; " (List.map string_of_int (Array.to_list s)))));
  if not (Array.for_all Float.is_finite (Nx.to_array points)) then
    invalid_arg "Jord.Polygons.v: a point is not finite";
  offsets "rings" rings ~last:(Nx.dim 0 points);
  offsets "features" features ~last:(Array.length rings - 1);
  { crs; points; rings; features }

let count p = Array.length p.features - 1

let to_crs crs p =
  { p with crs; points = Crs.transform ~src:p.crs ~dst:crs p.points }

(* Each feature by Slab over its own span, coordinates less its first point:
   the sum over its windows of the sign times the window's width times the
   mean of its edge's heights at its ends.

   Bound, (94 + K) u D W + (4E + 36X) u D² (polygons.mli). This is
   Rasterize.areas' arithmetic with heights up to D and unclipped. A height
   Slab computes is within 11 u D, a key within 24 u D and a height
   difference within 24 u D, so two edges ranked in the wrong order within
   rounding of each other cost at most 80 u D per unit of width: 48 where
   they do not cross in the slab, 72 for a crossing missed at a span's end,
   80 for one placed by interpolation. A window's mean height is within
   12 u D, and its width and product add 2 u D, 14 in all; K bounds the sum.
   4E bounds an edge's ends moved by their subtraction, which a steep edge
   turns into a large height error but which costs at most its rise times
   the move, and 36X a crossing placed within rounding, a strip at most
   9 u D wide whose heights differ by at most 2D. *)
let area p =
  (match p.crs.projection with
  | Some _ -> ()
  | None ->
      invalid_arg
        (Format.asprintf
           "Jord.Polygons.area: %a is geographic, its coordinates are not a \
            plane's"
           Crs.pp p.crs));
  let pts = Nx.to_array p.points and s = Slab.create () in
  let e = ref [||] in
  let one f =
    let first = p.rings.(p.features.(f))
    and last = p.rings.(p.features.(f + 1)) in
    if first = last then 0.
    else begin
      let n = last - first in
      if Array.length !e < 4 * n then e := Array.make (8 * n) 0.;
      let e = !e and x0 = pts.(2 * first) and y0 = pts.((2 * first) + 1) in
      let lo = ref Float.infinity and hi = ref Float.neg_infinity in
      let k = ref 0 in
      for ring = p.features.(f) to p.features.(f + 1) - 1 do
        let a = p.rings.(ring) and b = p.rings.(ring + 1) in
        for m = a to b - 1 do
          let q = if m + 1 = b then a else m + 1 in
          let x = pts.(2 * m) -. x0 in
          e.(4 * !k) <- x;
          e.((4 * !k) + 1) <- pts.((2 * m) + 1) -. y0;
          e.((4 * !k) + 2) <- pts.(2 * q) -. x0;
          e.((4 * !k) + 3) <- pts.((2 * q) + 1) -. y0;
          if x < !lo then lo := x;
          if x > !hi then hi := x;
          incr k
        done
      done;
      let nw = Slab.windows s e n ~lo:!lo ~hi:!hi in
      let ends = Slab.ends s and signs = Slab.signs s in
      let sum = ref 0. in
      for v = 0 to nw - 1 do
        let a = ends.(4 * v) and b = ends.((4 * v) + 1) in
        let mean = (ends.((4 * v) + 2) +. ends.((4 * v) + 3)) *. 0.5 in
        sum := !sum +. (signs.(v) *. (b -. a) *. mean)
      done;
      Float.max 0. !sum
    end
  in
  Nx.create Nx.float64 [| count p |] (Array.init (count p) one)
