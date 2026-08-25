(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = { w : 'a; b : 'a option }

let map f { w; b } =
  let w = f w in
  let b = match b with None -> None | Some b -> Some (f b) in
  { w; b }

let map2 f p q =
  let w = f p.w q.w in
  let b =
    match (p.b, q.b) with
    | Some pb, Some qb -> Some (f pb qb)
    | None, None -> None
    | Some _, None | None, Some _ -> invalid_arg "Linear.map2: bias mismatch"
  in
  { w; b }

let iter f { w; b } =
  f w;
  match b with None -> () | Some b -> f b

let fold f acc { w; b } =
  let acc = f "w" acc w in
  match b with None -> acc | Some b -> f "b" acc b

let fold2 f acc p q =
  let acc = f "w" acc p.w q.w in
  match (p.b, q.b) with
  | Some pb, Some qb -> f "b" acc pb qb
  | None, None -> acc
  | Some _, None | None, Some _ -> invalid_arg "Linear.fold2: bias mismatch"

let names p =
  { w = "w"; b = (match p.b with None -> None | Some _ -> Some "b") }

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
