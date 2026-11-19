(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let sum p g f params gs =
  let trip sum group =
    let v, grads, aux =
      Rune.value_and_grad_aux p Nx.Ptree.tensor (f group) params
    in
    (Nx.Ptree.axpy p (Nx.scalar_like v 1.) grads sum, aux)
  in
  let zeros = Nx.Ptree.map p (fun _ x -> Nx.zeros_like x) params in
  Rune.scan p g Nx.Ptree.tensor trip ~init:zeros gs
