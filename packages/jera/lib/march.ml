(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let check fn ~steps at =
  if steps < 1 then
    invalid_arg (Printf.sprintf "%s: steps = %d is not positive" fn steps);
  if Nx.ndim at <> 1 || Nx.dim 0 at = 0 then
    invalid_arg
      (Printf.sprintf "%s: at must hold at least one time, got shape %s" fn
         (Num.shape (Nx.shape at)));
  let n = Nx.dim 0 at in
  if n > 1 then begin
    let prev = Nx.slice [ Nx.R (0, n - 1) ] at
    and next = Nx.slice [ Nx.R (1, n) ] at in
    let d = Nx.sub next prev in
    (* Each gap's sign against the first's: a product of two gaps of 1e-162
       would underflow to zero. A zero first gap fails at [1]. *)
    let up = Nx.greater_s (Nx.slice [ Nx.R (0, 1) ] d) 0. in
    let same_sign = Nx.where up (Nx.greater_s d 0.) (Nx.less_s d 0.) in
    Nx.check
      Nx.Ptree.(pair tensor tensor)
      same_sign (prev, next)
      (fun i (prev, next) ->
        Invalid_argument
          (Printf.sprintf "%s: at is not strictly monotone at [%d]: %g after %g"
             fn
             (i.(0) + 1)
             (Nx.item [] next) (Nx.item [] prev)))
  end

let stack1 y v = Nx.Ptree.map y (fun _ x -> Nx.unsqueeze ~axes:[ 0 ] x) v

let prepend y v vs =
  Nx.Ptree.map2 y
    (fun _ x xs -> Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] x; xs ])
    v vs

let run c y ~at ~interval ~state init =
  let n = Nx.dim 0 at in
  let y0 = state init in
  if n = 1 then stack1 y y0
  else if n = 2 then
    let t0 = Nx.get [ 0 ] at and t1 = Nx.get [ 1 ] at in
    prepend y y0 (stack1 y (state (interval t0 t1 init)))
  else
    (* Reverse mode keeps each interval's start and runs the interval again
       while it reverses it. *)
    let interval =
      Rune.remat Nx.Ptree.(tensor @-> tensor @-> c @-> returns c) interval
    in
    let starts = Nx.slice [ Nx.R (0, n - 1) ] at
    and ends = Nx.slice [ Nx.R (1, n) ] at in
    let _, ys =
      Rune.scan c
        Nx.Ptree.(pair tensor tensor)
        y
        (fun carry (t0, t1) ->
          let carry = interval t0 t1 carry in
          (carry, state carry))
        ~init (starts, ends)
    in
    prepend y y0 ys

let grid ~steps t0 t1 =
  let n = float steps in
  let h = Nx.div_s (Nx.sub t1 t0) n in
  let k = Nx.arange_f_at (Nx.beside t0) (Nx.dtype t0) 0. n 1. in
  let x = Nx.add t0 (Nx.mul k h) in
  (* With h below half an ulp of the times, t0 + k h can round past t1:
     [0; 1.5e-323] in 5 steps of one subnormal unit reaches 4 units. *)
  let past = Nx.where (Nx.greater t1 t0) (Nx.greater x t1) (Nx.less x t1) in
  (h, Nx.where past (Nx.broadcast_to [| steps |] t1) x)

let steps c x n f init xs =
  if n = 0 then init
  else
    let xs = Nx.Ptree.map x (fun _ v -> Nx.slice [ Nx.R (0, n) ] v) xs in
    if n = 1 then f (Nx.Ptree.map x (fun _ v -> Nx.get [ 0 ] v) xs) init
    else
      fst
        (Rune.scan c x Nx.Ptree.unit (fun carry r -> (f r carry, ())) ~init xs)
