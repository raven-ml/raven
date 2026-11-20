(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shewchuk's exact sum ("Adaptive Precision Floating-Point Arithmetic and
   Fast Robust Geometric Predicates", 1997) as CPython's math.fsum runs it
   (Modules/mathmodule.c): the sum is kept as partials of increasing
   magnitude that do not overlap, each addition exact, and the partials are
   added from the largest with one correction for half-even rounding. Only
   additions, so no product can fuse. *)
let sum a n =
  let p = Array.make 32 0. in
  let p = ref p and np = ref 0 in
  for k = 0 to n - 1 do
    let x = ref a.(k) and i = ref 0 in
    for j = 0 to !np - 1 do
      let y = !p.(j) in
      let x', y = if Float.abs !x < Float.abs y then (y, !x) else (!x, y) in
      let hi = x' +. y in
      let lo = y -. (hi -. x') in
      if lo <> 0. then begin
        !p.(!i) <- lo;
        incr i
      end;
      x := hi
    done;
    np := !i;
    if !x <> 0. then begin
      if !np = Array.length !p then begin
        let q = Array.make (2 * !np) 0. in
        Array.blit !p 0 q 0 !np;
        p := q
      end;
      !p.(!np) <- !x;
      incr np
    end
  done;
  let p = !p and n = ref !np in
  if !n = 0 then 0.
  else begin
    decr n;
    let hi = ref p.(!n) and lo = ref 0. and exact = ref true in
    while !exact && !n > 0 do
      let x = !hi in
      decr n;
      let y = p.(!n) in
      hi := x +. y;
      lo := y -. (!hi -. x);
      if !lo <> 0. then exact := false
    done;
    if !n > 0 && ((!lo < 0. && p.(!n - 1) < 0.) || (!lo > 0. && p.(!n - 1) > 0.))
    then begin
      let y = !lo +. !lo in
      let x = !hi +. y in
      if y = x -. !hi then hi := x
    end;
    !hi
  end

let mean a n = if n = 0 then Float.nan else sum a n /. Float.of_int n
