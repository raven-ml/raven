(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let running = -1l
let searching st = Nx.equal_s st running

let settle st cond s =
  Nx.where
    (Nx.logical_and (searching st) cond)
    (Nx.full_like st (Answer.code s))
    st

(* The derivative's system is diagonal for an elementwise residual: its diagonal
   is [op 1], and [op u = b] holds to rounding at the quotient. A non-finite
   quotient, at a singular derivative, is the derivative's. *)
let diagonal fn op b =
  let d = op (Nx.ones_like b) in
  let u = Nx.div b d in
  let r = op u in
  let eps = Num.eps (Nx.dtype b) in
  let bound = Nx.mul_s (Nx.add (Nx.abs (Nx.mul d u)) (Nx.abs b)) (8. *. eps) in
  let fine =
    Nx.logical_or
      (Nx.logical_not (Nx.isfinite u))
      (Nx.less_equal (Nx.abs (Nx.sub r b)) bound)
  in
  Nx.check Nx.Ptree.unit fine () (fun i () ->
      Failure
        (Printf.sprintf
           "%s: the derivative at [%s] is not elementwise: the function reads \
            other elements than its argument's own"
           fn
           (String.concat ", " (Array.to_list (Array.map string_of_int i)))));
  u

let state fn ~ok r x =
  Rune.root Nx.Ptree.tensor ~linear_solve:(diagonal fn)
    ~residual:(fun v -> Nx.where ok (r v) (Nx.sub v x))
    (fun () -> x)

let accepted tol ~e ~y = Nx.less_equal_s (Tolerance.ratio tol ~e ~y) 1.
