(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'b params = { w : (float, 'b) Nx.t; b : (float, 'b) Nx.t option }
type t = Nx.float32_elt params

let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { w; b } =
  { w = f w; b = (match b with None -> None | Some b -> Some (f b)) }

let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) p q =
  let b =
    match (p.b, q.b) with
    | Some pb, Some qb -> Some (f pb qb)
    | None, None -> None
    | Some _, None | None, Some _ -> invalid_arg "Linear.map2: bias mismatch"
  in
  { w = f p.w q.w; b }

let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { w; b } =
  f w;
  match b with None -> () | Some b -> f b

let astype dt { w; b } =
  {
    w = Nx.cast dt w;
    b = (match b with None -> None | Some b -> Some (Nx.cast dt b));
  }

let names p = match p.b with None -> [ "w" ] | Some _ -> [ "w"; "b" ]

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
