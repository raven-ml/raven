(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Scaled of { rel : float; abs : float } | Ulps of float

let fail fn fmt = Printf.ksprintf (fun m -> invalid_arg (fn ^ ": " ^ m)) fmt

let v ~rel ~abs =
  let check name x =
    if not (Float.is_finite x && x >= 0.) then
      fail "Jera.Tol.v" "%s = %g is not finite and non-negative" name x
  in
  check "rel" rel;
  check "abs" abs;
  if rel = 0. && abs = 0. then fail "Jera.Tol.v" "rel and abs are both 0";
  Scaled { rel; abs }

let rel r = v ~rel:r ~abs:0.
let abs a = v ~rel:0. ~abs:a

let ulps k =
  if not (Float.is_finite k && k > 0.) then
    fail "Jera.Tol.ulps" "k = %g is not finite and positive" k;
  Ulps k

let pp ppf = function
  | Scaled { rel; abs } -> Format.fprintf ppf "rel %g abs %g" rel abs
  | Ulps k -> Format.fprintf ppf "ulps %g" k

let scale t y =
  match t with
  | Scaled { rel; abs } -> Nx.add_s (Nx.mul_s (Nx.abs y) rel) abs
  | Ulps k -> Nx.mul_s (Nx.abs y) (k *. Num.eps (Nx.dtype y))

let ratio t ~e ~y =
  let s = scale t y in
  let zero = Nx.zeros_like e in
  let r = Nx.div e (Nx.where (Nx.equal s zero) (Nx.ones_like s) s) in
  Nx.where (Nx.equal e zero) zero
    (Nx.where (Nx.equal s zero) (Nx.full_like e Float.infinity) (Nx.abs r))
