(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Move = Nx_array.Move

type axis = { first : int; count : int; step : int; back : bool }

let axis (r : Move.range) =
  if r.count <= 1 then
    { first = r.start; count = r.count; step = 1; back = false }
  else
    let step = abs r.step and back = r.step < 0 in
    let first = if back then r.start + ((r.count - 1) * r.step) else r.start in
    { first; count = r.count; step; back }

let axes (rs : Move.range iarray) = Array.map axis (Iarray.to_array rs)
let extent a = ((a.count - 1) * a.step) + 1

let rec gcd a b = if b = 0 then abs a else gcd b (a mod b)

let meet b c =
  Array.for_all2
    (fun x y ->
      x.count > 0 && y.count > 0
      && x.first < y.first + extent y
      && y.first < x.first + extent x
      && (y.first - x.first) mod gcd x.step y.step = 0)
    b c
