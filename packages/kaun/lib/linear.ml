(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = { w : 'a; b : 'a option }

let walk c { w; b } =
  let open Nx.Ptree.Walk in
  let w = field c "w" leaf w in
  let b = field c "b" (option leaf) b in
  { w; b }

let make ?(w_init = Init.glorot_uniform) ?(bias_init = Init.zeros)
    ?(bias = true) ~inputs ~outputs dtype =
  if inputs <= 0 || outputs <= 0 then
    Printf.ksprintf invalid_arg
      "Linear.make: inputs and outputs must be positive, got inputs=%d \
       outputs=%d"
      inputs outputs;
  let w = w_init ~fan_in:inputs ~fan_out:outputs dtype [| inputs; outputs |] in
  let b =
    if bias then
      Some (bias_init ~fan_in:inputs ~fan_out:outputs dtype [| outputs |])
    else None
  in
  { w; b }

let init ~inputs ~outputs = make ~inputs ~outputs Nx.float32

let apply p x =
  let y = Nx.matmul x p.w in
  match p.b with None -> y | Some b -> Nx.add y b
