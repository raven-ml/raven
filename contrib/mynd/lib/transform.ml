(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

let entry m i j =
  Nx.slice (List.init (Nx.ndim m - 2) (fun _ -> Nx.A) @ [ Nx.I i; Nx.I j ]) m

let apply m p =
  let fn = "Mynd.Transform.apply" in
  if Nx.ndim m < 2 || Nx.dim (-1) m <> 3 || Nx.dim (-2) m <> 3 then
    invalid_argf "%s: a transform has axes [b...; 3; 3]" fn;
  let batch = Nx.ndim m - 2 in
  let pb, q, p = points fn ~batch p in
  let b = broadcast fn (Array.sub (Nx.shape m) 0 batch) pb in
  let e i j = Nx.unsqueeze ~axes:[ -1 ] (entry m i j) in
  let x = coordinate p 0 and y = coordinate p 1 in
  let row i = Nx.add (Nx.add (Nx.mul (e i 0) x) (Nx.mul (e i 1) y)) (e i 2) in
  let d = row 2 in
  let r = Nx.stack ~axis:(-1) [ Nx.div (row 0) d; Nx.div (row 1) d ] in
  Nx.reshape (Array.concat [ b; q; [| 2 |] ]) r

let matrix rows =
  Nx.stack ~axis:(-2) (List.map (fun r -> Nx.stack ~axis:(-1) r) rows)

let pair fn t =
  if Nx.ndim t < 1 || Nx.dim (-1) t <> 2 then
    invalid_argf "%s: parameters have axes [...; 2]" fn;
  (coordinate t 0, coordinate t 1)

let translation t =
  let tx, ty = pair "Mynd.Transform.translation" t in
  let o = Nx.ones_like tx and z = Nx.zeros_like tx in
  matrix [ [ o; z; tx ]; [ z; o; ty ]; [ z; z; o ] ]

let scaling s =
  let sx, sy = pair "Mynd.Transform.scaling" s in
  let o = Nx.ones_like sx and z = Nx.zeros_like sx in
  matrix [ [ sx; z; z ]; [ z; sy; z ]; [ z; z; o ] ]

let rotation a =
  let c = Nx.cos a and s = Nx.sin a in
  let o = Nx.ones_like a and z = Nx.zeros_like a in
  matrix [ [ c; Nx.neg s; z ]; [ s; c; z ]; [ z; z; o ] ]

let shear s =
  let sx, sy = pair "Mynd.Transform.shear" s in
  let o = Nx.ones_like sx and z = Nx.zeros_like sx in
  matrix [ [ o; sx; z ]; [ sy; o; z ]; [ z; z; o ] ]
