(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = { table : 'a }

let map f { table } = { table = f table }
let map2 f p q = { table = f p.table q.table }
let iter f { table } = f table
let fold f acc { table } = f "table" acc table
let fold2 f acc p q = f "table" acc p.table q.table
let names _ = { table = "table" }

let make ?init ~vocab ~dim dtype =
  if vocab <= 0 || dim <= 0 then
    Printf.ksprintf invalid_arg
      "Embedding.make: vocab and dim must be positive, got vocab=%d dim=%d"
      vocab dim;
  let init =
    match init with Some init -> init | None -> Init.normal ~stddev:1.0
  in
  { table = init ~fan_in:dim ~fan_out:dim dtype [| vocab; dim |] }

let init ~vocab ~dim = make ~vocab ~dim Nx.float32

let apply p indices =
  let dim = (Nx.shape p.table).(1) in
  (* [take] flattens the indices along the gathered axis; restore their shape in
     front of the row dimension. *)
  let rows = Nx.take ~axis:0 ~indices p.table in
  Nx.reshape (Array.append (Nx.shape indices) [| dim |]) rows
