(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* ocamlopt on arm64, power and riscv contracts [a *. b +. c] into one
   fused multiply-add, where pycocotools' x86-64 code rounds the product
   and the sum apart. Hiding the product from instruction selection keeps
   two roundings on every target. *)
let mul a b = Sys.opaque_identity (a *. b)

(* The grid has five points a pixel. *)
let scale = 5.

(* C's (int) of a vertex is undefined past an int's range. *)
let int_range = 2147483648.

(* Growable int arrays, for the crossings. *)
type ints = { mutable a : int array; mutable n : int }

let push b x =
  if b.n = Array.length b.a then begin
    let a = Array.make (2 * b.n) 0 in
    Array.blit b.a 0 a 0 b.n;
    b.a <- a
  end;
  b.a.(b.n) <- x;
  b.n <- b.n + 1

(* [grid fn v] is the grid coordinate of [v]: (int) (scale * v + .5),
   truncated toward zero (maskApi.c, rleFrPoly). *)
let grid fn v =
  if not (Float.is_finite v) then
    invalid_argf "%s: a polygon coordinate is %h" fn v;
  let g = mul scale v +. 0.5 in
  if Float.abs g >= int_range then
    invalid_argf "%s: a polygon coordinate %g lies past the grid" fn v;
  truncate g

(* Sorted crossing indices [j h + i], ended by [h w], as differences with
   zero counts merged into their neighbours, as rleFrPoly ends. *)
let counts ~height:h ~width:w pts =
  push pts (h * w);
  let a = Array.sub pts.a 0 pts.n in
  Array.sort Int.compare a;
  let p = ref 0 in
  Array.iteri
    (fun i t ->
      a.(i) <- t - !p;
      p := t)
    a;
  let k = Array.length a in
  let b = Array.make k 0 in
  b.(0) <- a.(0);
  let m = ref 1 and j = ref 1 in
  while !j < k do
    if a.(!j) > 0 then begin
      b.(!m) <- a.(!j);
      incr m;
      incr j
    end
    else begin
      incr j;
      if !j < k then begin
        b.(!m - 1) <- b.(!m - 1) + a.(!j);
        incr j
      end
    end
  done;
  Array.sub b 0 !m

(* rleFrPoly (pycocotools 2.0.11, common/maskApi.c), step for step, except
   that each grid point is tested against the one before as it is made
   instead of being stored, so memory grows with the crossings inside the
   image and not with the length of the path. *)
let polygon fn ~height:h ~width:w xy =
  let n = Array.length xy in
  if n land 1 = 1 then
    invalid_argf "%s: a polygon has an odd number of coordinates, %d" fn n;
  let k = n / 2 in
  let gx = Array.init k (fun j -> grid fn xy.(2 * j)) in
  let gy = Array.init k (fun j -> grid fn xy.((2 * j) + 1)) in
  let pts = { a = Array.make 16 0; n = 0 } in
  (* A polygon of fewer than three vertices encloses nothing: its path
     crosses each column an even number of times at the same rows. *)
  if k >= 3 then begin
    let started = ref false and pu = ref 0 and pv = ref 0 in
    let last_col = Float.of_int (w - 1) and fh = Float.of_int h in
    (* A step from [(pu, pv)] to [(u, v)] that crosses between grid columns
       [5j + 2] and [5j + 3] adds pixel [j]'s crossing at the row its upper
       end rounds up to. *)
    let point u v =
      if !started && u <> !pu then begin
        let xd = Float.of_int (if u < !pu then u else u - 1) in
        let xd = ((xd +. 0.5) /. scale) -. 0.5 in
        if Float.floor xd = xd && xd >= 0. && xd <= last_col then begin
          let yd = Float.of_int (if v < !pv then v else !pv) in
          let yd = ((yd +. 0.5) /. scale) -. 0.5 in
          let yd = if yd < 0. then 0. else if yd > fh then fh else yd in
          push pts ((truncate xd * h) + truncate (Float.ceil yd))
        end
      end;
      started := true;
      pu := u;
      pv := v
    in
    for j = 0 to k - 1 do
      let j' = if j = k - 1 then 0 else j + 1 in
      let xs = gx.(j) and xe = gx.(j') and ys = gy.(j) and ye = gy.(j') in
      let dx = abs (xe - xs) and dy = abs (ys - ye) in
      let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye) in
      let xs, xe, ys, ye =
        if flip then (xe, xs, ye, ys) else (xs, xe, ys, ye)
      in
      if dx = 0 && dy = 0 then
        (* The C reads (int) (0/0) for this point's row; its column equals
           its neighbours', so the row is never read. *)
        point xs ys
      else if dx >= dy then begin
        let s = Float.of_int (ye - ys) /. Float.of_int dx in
        let c = Float.of_int ys in
        for d = 0 to dx do
          let t = if flip then dx - d else d in
          point (t + xs) (truncate (c +. mul s (Float.of_int t) +. 0.5))
        done
      end
      else begin
        let s = Float.of_int (xe - xs) /. Float.of_int dy in
        let c = Float.of_int xs in
        for d = 0 to dy do
          let t = if flip then dy - d else d in
          point (truncate (c +. mul s (Float.of_int t) +. 0.5)) (t + ys)
        done
      end
    done
  end;
  counts ~height:h ~width:w pts

(* rleMerge with intersect = 0 (maskApi.c), for two masks: a walk over both
   runs that ends a run where the union changes value. *)
let union a b =
  let ka = Array.length a and kb = Array.length b in
  let out = Array.make (ka + kb + 1) 0 in
  let ca = ref a.(0) and cb = ref b.(0) in
  let ia = ref 1 and ib = ref 1 in
  let va = ref false and vb = ref false and v = ref false in
  let m = ref 0 and cc = ref 0 and ct = ref 1 in
  while !ct > 0 do
    let c = Int.min !ca !cb in
    cc := !cc + c;
    ca := !ca - c;
    if !ca = 0 && !ia < ka then begin
      ca := a.(!ia);
      incr ia;
      va := not !va
    end;
    cb := !cb - c;
    if !cb = 0 && !ib < kb then begin
      cb := b.(!ib);
      incr ib;
      vb := not !vb
    end;
    ct := !ca + !cb;
    let vp = !v in
    v := !va || !vb;
    if !v <> vp || !ct = 0 then begin
      out.(!m) <- !cc;
      incr m;
      cc := 0
    end
  done;
  Array.sub out 0 !m
