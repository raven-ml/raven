(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = { table : 'a }

let walk c { table } = { table = Nx.Ptree.Walk.(field c "table" leaf table) }

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
let apply p indices = Nx.take ~axis:0 ~indices p.table
