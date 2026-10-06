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
    let first = Nx.slice [ Nx.R (0, 1) ] d in
    let same_sign = Nx.greater (Nx.mul d first) (Nx.zeros_like d) in
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
    let t0 = Nx.slice [ Nx.I 0 ] at and t1 = Nx.slice [ Nx.I 1 ] at in
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
        ~f:(fun carry (t0, t1) ->
          let carry = interval t0 t1 carry in
          (carry, state carry))
        ~init (starts, ends)
    in
    prepend y y0 ys

let steps c dtype n f init =
  if n = 0 then init
  else if n = 1 then f (Nx.zeros dtype [||]) init
  else
    fst
      (Rune.scan c Nx.Ptree.tensor Nx.Ptree.unit
         ~f:(fun carry j -> (f j carry, ()))
         ~init
         (Nx.arange_f dtype 0. (float n) 1.))
