(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type blocks = (int, Nx.uint8_elt) Nx.t
type scales = (int, Nx.uint8_elt) Nx.t

let group_bytes = 16

let code_values =
  [|
    0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6.; -0.; -0.5; -1.; -1.5; -2.; -3.; -4.; -6.;
  |]

(* nx has no ldexp: the 256 powers of two are a table, exact in every float
   dtype with float32's exponent range. *)
let scale_values =
  Array.init 256 (fun e ->
      if e = 255 then Float.nan else Float.ldexp 1.0 (e - 127))

let lookup table dt codes =
  let table = Nx.create dt [| Array.length table |] table in
  let indices = Nx.reshape [| -1 |] (Nx.cast Nx.int32 codes) in
  Nx.reshape (Nx.shape codes) (Nx.take ~indices table)

let dequant blocks scales dt =
  let shape = Nx.shape blocks in
  let rank = Array.length shape in
  if rank < 2 || shape.(rank - 1) <> group_bytes then
    invalid_arg "Mxfp4.dequant: blocks must have shape [...; groups; 16]";
  if Nx.shape scales <> Array.sub shape 0 (rank - 1) then
    invalid_arg
      "Mxfp4.dequant: scales must have the shape of blocks without its last \
       axis";
  let low = Nx.bitwise_and blocks (Nx.scalar Nx.uint8 15) in
  let high = Nx.rshift blocks 4 in
  let values = lookup code_values dt (Nx.stack ~axis:(-1) [ low; high ]) in
  let scale =
    Nx.reshape
      (Array.append (Nx.shape scales) [| 1; 1 |])
      (lookup scale_values dt scales)
  in
  let out = Array.sub shape 0 (rank - 1) in
  out.(rank - 2) <- shape.(rank - 2) * group_bytes * 2;
  Nx.reshape out (Nx.mul values scale)

let take_rows ids t =
  let rows = Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] ids) t in
  let rest = Array.sub (Nx.shape t) 1 (Nx.ndim t - 1) in
  Nx.reshape (Array.append (Nx.shape ids) rest) rows

let dequant_rows blocks scales ids dt =
  dequant (take_rows ids blocks) (take_rows ids scales) dt
