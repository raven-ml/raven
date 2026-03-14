(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64

let jacfwd f x =
  let y0 = f x in
  let n = Nx.numel x and m = Nx.numel y0 in
  let cols =
    List.init n (fun i ->
        let ei = Nx.zeros f64 [| n |] in
        Nx.set_item [ i ] 1.0 ei;
        let _, col =
          Jvp.jvp
            (fun x -> Nx.reshape [| m |] (f (Nx.reshape [| n |] x)))
            (Nx.reshape [| n |] x) ei
        in
        col)
  in
  Nx.stack ~axis:1 cols

let jacrev f x =
  let y0 = f x in
  let n = Nx.numel x and m = Nx.numel y0 in
  let rows =
    List.init m (fun i ->
        let ei = Nx.zeros f64 [| m |] in
        Nx.set_item [ i ] 1.0 ei;
        let _, row =
          Vjp.vjp
            (fun x -> Nx.reshape [| m |] (f (Nx.reshape [| n |] x)))
            (Nx.reshape [| n |] x) ei
        in
        row)
  in
  Nx.stack ~axis:0 rows
