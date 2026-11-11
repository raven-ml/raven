(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, _) t = { w : 'a; b : 'a option }

let walk c { w; b } =
  let open Nx.Ptree.Walk in
  let w = field c "w" leaf w in
  let b = field c "b" (option leaf) b in
  { w; b }

let make ?(w_init = Init.glorot_uniform) ?(bias_init = Init.zeros)
    ?(bias = true) ~key ~inputs ~outputs dtype =
  if inputs <= 0 || outputs <= 0 then
    Printf.ksprintf invalid_arg
      "Linear.make: inputs and outputs must be positive, got inputs=%d \
       outputs=%d"
      inputs outputs;
  let keys = Nx.Rng.split key in
  let w =
    w_init keys.(0) ~fan_in:inputs ~fan_out:outputs dtype [| inputs; outputs |]
  in
  let b =
    if bias then
      Some
        (bias_init keys.(1) ~fan_in:inputs ~fan_out:outputs dtype [| outputs |])
    else None
  in
  { w; b }

let init ~key ~inputs ~outputs = make ~key ~inputs ~outputs Nx.float32

(* The product of an [x] of [lead + 1] axes by [w], [x]'s leading axes
   named [x0], [x1], …: "x0 i, i o -> x0 o | i" for one. *)
let product_of lead =
  let names = List.init lead (Printf.sprintf "x%d") in
  let side last = String.concat " " (names @ [ last ]) in
  Nx.Pattern.v (Printf.sprintf "%s, i o -> %s | i" (side "i") (side "o"))

(* The products of the ranks layers take most, parsed once. *)
let products = Array.init 4 product_of
let product lead = if lead < 4 then products.(lead) else product_of lead

let apply ?sum p x =
  match p.b with
  | None -> Nx.matmul ?sum x p.w
  | Some b ->
      let out = Array.copy (Nx.shape x) in
      let lead = Array.length out - 1 in
      out.(lead) <- Nx.dim 1 p.w;
      let init = Nx.broadcast_to out b in
      Nx.contract (Nx.dtype p.w) ?sum ~init (product lead) x p.w
