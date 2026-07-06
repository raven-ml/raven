(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { w : Nx.float32_t; b : Nx.float32_t }

let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b } =
  { w = f w; b = f b }

let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
  { w = f p.w q.w; b = f p.b q.b }

let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b } =
  f w;
  f b

(* Glorot-uniform weights, zero bias. *)
let init ~inputs ~outputs =
  let limit = Stdlib.sqrt (6.0 /. float_of_int (inputs + outputs)) in
  let w =
    Nx.mul_s
      (Nx.sub_s (Nx.mul_s (Nx.rand Nx.float32 [| inputs; outputs |]) 2.0) 1.0)
      limit
  in
  { w; b = Nx.zeros Nx.float32 [| outputs |] }

let apply p x = Nx.add (Nx.matmul x p.w) p.b
