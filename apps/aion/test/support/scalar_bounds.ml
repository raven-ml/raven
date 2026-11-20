(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let forward (tr : Aion.Scalar.transform) x =
  match tr with
  | Identity -> (x, x)
  | Log m -> Bounds.around 1 (Float.log (Float.max x (Bounds.f32 m)))
  | Asinh_asinh ->
      Bounds.through 3 Float.asinh (Bounds.around 3 (Float.asinh x))

let inverse (tr : Aion.Scalar.transform) e =
  match tr with
  | Identity -> (e, e)
  | Log _ -> Bounds.around 2 (Float.exp e)
  | Asinh_asinh -> Bounds.through 3 Float.sinh (Bounds.around 3 (Float.sinh e))

let crosses tr edges x a b =
  a = b
  ||
  let lo = Int.min a b and hi = Int.max a b in
  let v = edges.(lo) in
  let one = ref true in
  for k = lo to hi - 1 do
    if not (Float.equal edges.(k) v) then one := false
  done;
  tr <> Aion.Scalar.Identity && !one && Bounds.inside (forward tr x) v
