(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let split fn p =
  let shape = Nx.shape p in
  let rank = Array.length shape in
  if rank = 0 || shape.(rank - 1) <> 2 then
    invalid_arg
      (Printf.sprintf "%s: the points' last axis is not 2 (shape [%s])" fn
         (String.concat "; " (List.map string_of_int (Array.to_list shape))));
  let batch = Array.sub shape 0 (rank - 1) in
  let flat = Nx.reshape [| Array.fold_left ( * ) 1 batch; 2 |] p in
  (batch, Nx.slice [ A; I 0 ] flat, Nx.slice [ A; I 1 ] flat)

let join batch a b =
  Nx.reshape (Array.append batch [| 2 |]) (Nx.stack ~axis:1 [ a; b ])
