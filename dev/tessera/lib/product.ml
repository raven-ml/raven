(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let matrix = Nx.Pattern.v "n i, i o -> n o | i"

(* The bias is the accumulator's initial value, so the sum rounds once. *)
let apply (p : ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t) x =
  let dt = Nx.dtype p.w in
  let shape = Nx.shape x in
  let i = Nx.dim (-1) x and o = Nx.dim 1 p.w in
  let n = Nx.numel x / max i 1 in
  let x = Nx.reshape [| n; i |] (Nx.cast dt x) in
  let y =
    match p.b with
    | None -> Nx.contract dt matrix x p.w
    | Some b -> Nx.contract dt ~init:(Nx.broadcast_to [| n; o |] b) matrix x p.w
  in
  shape.(Array.length shape - 1) <- o;
  Nx.reshape shape y
