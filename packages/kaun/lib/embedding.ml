(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'b params = { table : (float, 'b) Nx.t }
type t = Nx.float32_elt params

let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { table } =
  { table = f table }

let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) p q =
  { table = f p.table q.table }

let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { table } = f table
let names _ = [ "table" ]

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
  let rows = Nx.take ~axis:0 indices p.table in
  Nx.reshape (Array.append (Nx.shape indices) [| dim |]) rows
